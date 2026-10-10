#ifndef NET_HPP
#define NET_HPP
#include <memory>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <functional>
#include "tensor.hpp"
#include "ilayer.h"
#include "weightio.hpp"
#include "optimize.h"   /* GradClipMode (Net::RMSPropMode; 见 rl/optimize.h 顶部) */

namespace RL {

/*
 * ---- 两个"内存当流"的小工具 (2026-10 对弈期间不落盘, 见 weightio.hpp) ----
 *
 * StringOutBuf: 把写进 ostream 的字节**追加**到 std::string 上 (不拷贝第二遍;
 *   std::ostringstream 要先攒在它自己的缓冲里再 str() 拷一次, 146 MB 的权重上很明显)。
 * ConstMemBuf : 把一块已有的内存当成只读 istream 的数据源 (**零拷贝**)。
 *   为什么不用 std::istringstream(buffer): 它的构造函数会**拷贝整块** buffer, 而权重
 *   文件到 146 MB —— 那份拷贝没有意义 (buffer 本来就已经在内存里了)。
 * 两者都只服务权重序列化这一条路径, 所以放在这里而不是单开一个头文件。
 */
class StringOutBuf : public std::streambuf
{
public:
    explicit StringOutBuf(std::string &out) : m_out(out) {}
protected:
    std::streamsize xsputn(const char *s, std::streamsize n) override
    {
        m_out.append(s, (std::size_t)n);
        return n;
    }
    int overflow(int c) override
    {
        if (c != EOF) {
            m_out.push_back((char)c);
        }
        return (c == EOF) ? EOF : c;
    }
private:
    std::string &m_out;
};

class ConstMemBuf : public std::streambuf
{
public:
    ConstMemBuf(const char *data, std::size_t size)
    {
        char *p = const_cast<char *>(data);
        setg(p, p, p + size);
    }
};

class Net
{
public:
    using FnLoss = std::function<Tensor(const Tensor&, const Tensor&)>;
    using Layers = std::vector<iLayer::sptr>;
    float alpha_;
    float beta_;
    /* ∂L/∂x of the most recent backward(). Exposed so callers that need to
       propagate through the network input (e.g. VAE's decoder) can read it
       instead of having to re-run a layer backward after its cache was cleared. */
    Tensor inputGrad;
protected:
    Layers layers;
public:
    Net():alpha_(1),beta_(1){}
    virtual ~Net(){}
    template<typename ...TLayer>
    explicit Net(TLayer&&...layer)
        :alpha_(1),beta_(1),layers({layer...}){}
    explicit Net(const Layers &layers_)
        :alpha_(1),beta_(1),layers(layers_){}
    Net(const Net &r)
        :alpha_(1),beta_(1),layers(r.layers){}

    inline Tensor& output() {return layers.back()->o;}

    inline iLayer* operator[](std::size_t i) {return layers.at(i).get();}

    inline std::size_t size() const {return layers.size();}

    /* 参数量 (只读诊断): 各层 paramCount 之和。见 iLayer::paramCount */
    long long paramCount() const
    {
        long long total = 0;
        for (std::size_t i = 0; i < layers.size(); i++) {
            total += layers[i]->paramCount();
        }
        return total;
    }

    Tensor &forward(const Tensor &x, bool inference=false)
    {
        layers[0]->forward(x, inference);
        for (std::size_t i = 1; i < layers.size(); i++) {
            Tensor &out = layers[i - 1]->o;
            if ((layers[i - 1]->type == iLayer::LAYER_CONV2D ||
                 layers[i - 1]->type == iLayer::LAYER_MAXPOOLING ||
                 layers[i - 1]->type == iLayer::LAYER_AVGPOOLING)&&
                    layers[i]->type == iLayer::LAYER_FC) {
                layers[i]->forward(out.flatten(), inference);
            } else {
                layers[i]->forward(out, inference);
            }
        }
        return layers.back()->o;
    }

    /* 层 i 的输出喂给层 i+1 时需不需要先压平 (conv/pool -> FC 才需要) */
    bool needsFlatten(std::size_t preIndex) const
    {
        if (preIndex + 1 >= layers.size()) {
            return false;
        }
        const int pre = layers[preIndex]->type;
        return (pre == iLayer::LAYER_CONV2D ||
                pre == iLayer::LAYER_MAXPOOLING ||
                pre == iLayer::LAYER_AVGPOOLING) &&
               layers[preIndex + 1]->type == iLayer::LAYER_FC;
    }

    /*
        ============================================================
         R1 (2026-09): 部分前向 + 稀疏输出头
        ============================================================
        forwardTrunk 跑到**倒数第二层为止**并返回它的输出 —— 也就是"输出头将要吃到的
        那个张量" (actor 里就是 Tanh(h) 那 64 维)。配合 sparseLogits 就得到
        "只算合法列"的推理路径 (语义等价的推导见 iLayer::sparseLogits)。

        注意: 它**不写**最后一层的输出, 所以调用之后 output()/layers.back()->o 里
        是上一次全量前向的残留值 —— 不要读它。
    */
    Tensor &forwardTrunk(const Tensor &x, bool inference=false)
    {
        layers[0]->forward(x, inference);
        for (std::size_t i = 1; i + 1 < layers.size(); i++) {
            Tensor &out = layers[i - 1]->o;
            if (needsFlatten(i - 1)) {
                layers[i]->forward(out.flatten(), inference);
            } else {
                layers[i]->forward(out, inference);
            }
        }
        return (layers.size() >= 2) ? layers[layers.size() - 2]->o
                                    : layers.back()->o;
    }

    /*
        R1: 只算最后一层在 idx 上的 logits (激活前)。

        `h` 必须是 forwardTrunk() 的返回值 (倒数第二层的输出); conv/pool -> FC 的
        压平规则与 forward() 里一致, 所以两种写法喂给输出头的张量相同。
        返回 false 表示"这个网络/这次调用走不了稀疏路径"。
    */
    bool sparseLogits(const Tensor &h, const std::vector<int> &idx,
                      std::vector<float> &out) const
    {
        if (layers.empty()) {
            return false;
        }
        if (layers.size() >= 2 && needsFlatten(layers.size() - 2)) {
            Tensor flat = h.flatten();
            return layers.back()->sparseLogits(flat, idx, out);
        }
        return layers.back()->sparseLogits(h, idx, out);
    }

    /*
        输出头能不能走"稀疏列 + 子集重新归一"这条捷径。
        两个条件缺一不可: 头支持 sparseLogits, 且它的激活是整向量 softmax 型
        (只有 softmax 才有"子集归一 = 全量后归一"这条性质)。
    */
    bool sparseOutputSupported() const
    {
        if (layers.empty()) {
            return false;
        }
        return layers.back()->supportsSparseLogits() &&
               layers.back()->subsetSoftmax();
    }

    void backward(const Tensor &x, const Tensor &loss)
    {
        layers[layers.size() - 1]->e = loss;
        backwardFrom(layers.size() - 1, x);
        return;
    }

    /*
       ============================================================
        从第 startIndex 层往回传到第 0 层 (R2, 2026-09)
       ============================================================
       `startIndex` 那一层的 `e` 必须由调用方先设好 —— 用途是"输出头自己用稀疏路径算完
       了梯度"的情形: R2 里训练前向只在**合法列**上做 softmax, 头的权重梯度与往下传的
       梯度都是稀疏算出来的 (见 PPO::accumulateGradSparse), 于是反向从**倒数第二层**
       开始, 头那一层不再走通用的 backward。

       语义与 backward() 完全一致 (backward 现在就是它的薄封装), 所以非稀疏路径一行
       都没变。
    */
    void backwardFrom(std::size_t startIndex, const Tensor &x)
    {
        for (int i = (int)startIndex; i > 0; i--) {
            iLayer::sptr layer = layers[i];
            iLayer::sptr preLayer = layers[i - 1];
            if ((preLayer->type == iLayer::LAYER_CONV2D ||
                 preLayer->type == iLayer::LAYER_MAXPOOLING ||
                 preLayer->type == iLayer::LAYER_AVGPOOLING)&&
                    layer->type == iLayer::LAYER_FC) {
                Tensor e(preLayer->e.totalSize, 1);
                layer->backward(preLayer->o.flatten(), e);
                preLayer->e.val = e.val;
            } else if (preLayer->type == iLayer::LAYER_LSTM) {
                /* LSTM uses BPTT via cacheError/cacheX cache */
                Tensor e(preLayer->o.totalSize, 1);
                layer->backward(preLayer->o, e);
                preLayer->cacheError(e);
            } else if (preLayer->type == iLayer::LAYER_SSM) {
                /* SSM uses the same BPTT pattern as LSTM */
                Tensor e(preLayer->o.totalSize, 1);
                layer->backward(preLayer->o, e);
                preLayer->cacheError(e);
            } else if (preLayer->type == iLayer::LAYER_MAMBA) {
                /* MambaLayer uses the same BPTT pattern as SSM/LSTM */
                Tensor e(preLayer->o.totalSize, 1);
                layer->backward(preLayer->o, e);
                preLayer->cacheError(e);
            }
            /*
               [2026-10] 这里原来还有一条
                 `(type == LAYER_ATTENTION || LAYER_MHA || LAYER_SCALEDCONCAT || LAYER_MOE)
                  && layer->type == LAYER_FC`
               的分支 —— 但它的函数体与下面的 `else` **一字不差**, 是"看起来有特殊处理、
               实际什么都没做"的死分支 (看代码的人会以为这几层走了别的路, 而 LAYER_ATTENTION
               那条线本身是死代码, 见 ilayer.h 的 `requireWired` 与
               docs/tb_expert_training_2026_10.md §9.3)。删掉它不改变任何行为。
            */
            else {
                layer->backward(preLayer->o, preLayer->e);
            }
        }
        /* Also backprop through layer[0] so compound layers
           (TransformerBlock, MHA, etc.) execute their internal
           backward logic, compute LN/MHA gradients, and clear caches.
           The buffer copies the SHAPE OF THE NETWORK INPUT x, not a flat
           (totalSize, 1):
             * sizing it by layer[0]'s OUTPUT made the kikj() inside
               Layer<Fn>::backward index w out of bounds whenever
               inputDim != outputDim;
             * sizing it flat broke conv-first networks entirely — Conv2d::
               backward indexes ei.shape[1] and ei.shape[2], and a 2-D {N,1}
               shape has no index 2, so every training step of ConvPG/ConvDQN
               read past the end of the shape vector (AddressSanitizer:
               heap-buffer-overflow at conv2d.hpp:223).
           The result is kept in inputGrad rather than discarded. */
        inputGrad = Tensor(x.shape);
        layers[0]->backward(x, inputGrad);
        return;
    }

    /*
       clipGrad 以前是**写死 true** 的, 外面没法关。而 Optimize::RMSProp 里的
       clipGrad 做的是 `dw /= dw.norm2()` —— 对每个张量各做一次全量范数 + 全量除法。
       两点代价:
         * 性能: 每步对全部参数多做一遍读 + 一遍读写 (实测占了一步 trainStep 的大头)
         * 语义: 按范数归一化会把每层的梯度**缩成单位长度**, 梯度的大小信息被抹掉 ——
           而 RMSProp 本身已经在做逐参数的尺度归一, 再叠一层全局归一之后, 每层的
           有效步长恒等于 lr, 与真实梯度大小无关。
       默认仍是 true (保持既有行为不变), 想关掉/做对比实验的调用方可以显式传 false。
    */
    void RMSProp(float lr, float rho=0.9, float decay=0, bool clipGrad=true)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->RMSProp(lr, rho, decay, clipGrad);
        }
        return;
    }

    /*
       ============================================================
        [2026-10] 按"裁剪口径"跑一次 RMSProp (见 rl/optimize.h 的 GradClipMode)
       ============================================================
       三种模式的语义差别、以及"为什么它对 RMSProp 几乎是空操作"都在 `GradClipMode`
       的注释里; 这里只说实现:
         * `PerTensorUnitNorm` → 直接走上面那个老函数 (逐位不变, 不额外遍历);
         * `GlobalNorm`        → 先 `gradNorm2()` 求和, 超限时 `scaleGrad(m/n)` **就地**
                                 缩小每层梯度, 再走 `clipGrad=false` 的 RMSProp;
         * `None`              → 直接 `clipGrad=false`。
       为什么"缩放就地把梯度改掉"可以接受: 梯度缓冲本来就是**每一步用完即清**
       (各层的 RMSProp 末尾都 `g.zero()`), 所以缩放不会污染下一步; 而且它省掉了
       "再开一批全尺寸临时张量"的内存 (TB 骨干下这是 GB 级的差别)。
       `maxNorm <= 0` 视为不裁剪, 免得调用方误传 0 把梯度清成 0。
    */
    void RMSPropMode(float lr, float rho, float decay, int mode, float maxNorm)
    {
        if (mode == GRAD_CLIP_PER_TENSOR_UNIT_NORM) {
            RMSProp(lr, rho, decay, true);
            return;
        }
        if (mode == GRAD_CLIP_GLOBAL_NORM && maxNorm > 0.0f) {
            const double n2 = gradNorm2();
            if (n2 > 0.0) {
                const double n = std::sqrt(n2);
                if (n > (double)maxNorm) {
                    scaleGrad((float)((double)maxNorm / n));
                }
            }
        }
        RMSProp(lr, rho, decay, false);
    }

    /*
       全网梯度范数² (只统计实现了 `gradNorm2()` 的层; 见 ilayer.h 的说明)。
       用途: ① `GlobalNorm` 裁剪; ② 面板/报告里的"这一步的信号有多大"读数 ——
       本工程此前**没有**这个数, 于是"学习率要不要跟"只能靠试。
    */
    double gradNorm2() const
    {
        double s = 0.0;
        for (std::size_t i = 0; i < layers.size(); i++) {
            s += layers[i]->gradNorm2();
        }
        return s;
    }
    double gradNorm() const { return std::sqrt(gradNorm2()); }

    /* 把全网所有层的梯度张量乘 s (全局裁剪用; 未实现的层原样不动) */
    void scaleGrad(float s)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->scaleGrad(s);
        }
    }

    /* 逐层梯度范数 (只读诊断): 输出 "层号:范数" 的文本, 给报告/面板用 */
    std::string gradNormReport() const
    {
        std::ostringstream os;
        for (std::size_t i = 0; i < layers.size(); i++) {
            const double n2 = layers[i]->gradNorm2();
            os << (i == 0 ? "" : " ") << i << ":" << std::sqrt(n2);
        }
        return os.str();
    }

    void Adam(float lr, float alpha=0.99, float beta=0.9, float decay=0)
    {
        alpha_ *= alpha;
        beta_ *= beta;
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->Adam(lr, alpha, beta, alpha_, beta_, decay, true);
        }
        return;
    }

    void clamp(float c0, float cn)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->clamp(c0, cn);
        }
        return;
    }

    void copyTo(Net& dstNet)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->copyTo(dstNet.layers[i].get());
        }
        return;
    }
    void softUpdateTo(Net& dstNet, float alpha)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->softUpdateTo(dstNet.layers[i].get(), alpha);
        }
        return;
    }

    /*
       ============================================================
        权重文件: 带校验头的原子写入 (v2)
       ============================================================

       格式:
           CHWGT2 <层数> <类型指纹>
           <第 0 层的一行/若干行张量>
           ...
       头一行是**自描述**的: 载入时先比层数与"每层的 layer type 序列"算出来的
       FNV-1a 指纹, 不匹配就直接失败 —— 以前把某个 agent 的权重喂给另一个结构
       不同的 agent 时, 没有任何检查, 结果是一堆形状错乱的张量被静默载入, 直到
       某次前向才崩 (或者更糟: 不崩但输出全是垃圾)。

       写入是**原子**的: 先写 `<path>.tmp`, 成功后用 std::filesystem::rename 覆盖
       原文件。这样"存到一半崩了/被 kill 了"不会把上一次训练好的模型毁掉 ——
       后台训练每轮都在存盘, 而关窗时正好可能打断它。

       张量本身用 Tensor::toString 的 v2 编码 (base64 原始数据, 无损, 见 tensor.hpp)。
       载入时同时兼容 v1 的纯文本格式 (没有 CHWGT2 头), 所以旧权重文件照样能读。
       ============================================================
    */
    static constexpr const char *kWeightMagic = "CHWGT2";
    static constexpr const char *kWeightTmpSuffix = ".tmp";

    /* 每层类型序列的指纹 (不需要密码学强度, 只要"结构不同就一定不同"的常见情形) */
    std::uint64_t structureFingerprint() const
    {
        std::uint64_t h = 1469598103934665603ULL;   /* FNV-1a offset basis */
        for (std::size_t i = 0; i < layers.size(); i++) {
            const std::uint64_t t = (std::uint64_t)(int)layers[i]->type;
            h ^= t;
            h *= 1099511628211ULL;
        }
        return h;
    }

    /*
       ---- payload 的写出: 磁盘与内存**共用这一份** (2026-10) ----
       两份实现迟早漂移, 而漂移的表现是"内存版与磁盘版写出的字节不同" —— 那是那种
       查起来最贵的 bug (同一个网络, 两条路读回来的东西不一样)。所以格式只写在这里,
       `save()` 只负责"把这一串字节送到哪儿去"。
    */
    void writePayload(std::ostream &os) const
    {
        os << kWeightMagic << " " << layers.size() << " "
           << structureFingerprint() << "\n";
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->write(os);
        }
    }

    int save(const std::string &fileName) const
    {
        /*
           [2026-10 用户口径 "对弈期间不更新保存模型权重"] 临时权重路径在对弈期间**改走
           内存** (见 weightio.hpp): 写出的字节与磁盘版逐字节相同, 只是不落盘 ——
           后台训练那一轮的往返因此照常跑完, 而 weights/ 目录里一个临时文件都不多。
           返回 true = 已经接管, 这里**故意不碰磁盘** (读侧用的是同一个判据)。
        */
        if (WeightIO::shouldDivert(fileName)) {
            std::string payload;
            {
                StringOutBuf buf(payload);
                std::ostream os(&buf);
                writePayload(os);
                os.flush();
                if (!os.good()) {
                    return -1;
                }
            }
            return WeightIO::divertWrite(fileName, std::move(payload)) ? 0 : -1;
        }

        const std::string tmp = fileName + kWeightTmpSuffix;
        {
            std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
            if (!file.is_open()) {
                return -1;
            }
            writePayload(file);
            file.flush();
            if (!file.good()) {
                /* 磁盘满 / 权限问题: 删掉半截的临时文件, 保留原文件 */
                file.close();
                std::remove(tmp.c_str());
                return -1;
            }
        }
        /* 原子替换。rename 在 Windows 上也允许覆盖已存在的目标 (MOVEFILE_REPLACE_EXISTING) */
        std::error_code ec;
        std::filesystem::rename(tmp, fileName, ec);
        if (ec) {
            /* 跨设备等罕见情况下退化成"拷贝 + 删除" */
            std::filesystem::copy_file(tmp, fileName,
                                       std::filesystem::copy_options::overwrite_existing, ec);
            std::remove(tmp.c_str());
            if (ec) {
                return -1;
            }
        }
        return 0;
    }

    int load(const std::string &fileName)
    {
        /*
           ---- 取字节: **内存优先, 磁盘回落** (2026-10 对弈期间的临时权重不落盘) ----
           内存里有这一份 = 它是对弈期间写出来的临时权重 (见上面 save 的说明); 没有就按
           原来的方式读磁盘 —— "改道关着"时这条路径与改动前逐字节相同。
        */
        std::string buffer;
        const bool fromMemory = WeightIO::divertLookup(fileName, buffer);
        if (!fromMemory) {
            std::ifstream in(fileName, std::ios::binary | std::ios::ate);
            if (!in.is_open()) {
                std::cerr << "[weights] " << fileName << ": 打不开" << std::endl;
                return -1;
            }
            const std::streamoff size = in.tellg();
            if (size <= 0) {
                std::cerr << "[weights] " << fileName << ": 空文件" << std::endl;
                return -1;
            }
            in.seekg(0, std::ios::beg);
            buffer.resize((std::size_t)size);
            in.read(&buffer[0], size);
            if (!in.good() && !in.eof()) {
                std::cerr << "[weights] " << fileName << ": 读入失败" << std::endl;
                return -1;
            }
        }
        if (buffer.empty()) {
            std::cerr << "[weights] " << fileName << ": 空文件" << std::endl;
            return -1;
        }

        /*
           第一行 (v2 的头, 或者 v1 的第一个张量) 从 **buffer 上切** —— 不再从 ifstream
           上 getline。这样"内存"与"磁盘"两个来源走的是同一段解析代码, 不会因为两条来源
           各自的实现而漂移。
        */
        const std::size_t firstEol = buffer.find('\n');
        const std::size_t firstLen =
            (firstEol == std::string::npos) ? buffer.size() : firstEol;
        const std::string firstLine = buffer.substr(0, firstLen);

        /*
           先看第一行是不是 v2 的头。不是的话 (老文件, 第一行就是第一个张量) 要从
           **开头**按 v1 逐层读 —— 这样老权重文件不需要任何转换。
        */
        const std::size_t magicLen = std::strlen(kWeightMagic);
        const bool isV2 = (firstLine.compare(0, magicLen, kWeightMagic) == 0);
        std::size_t payloadStart = 0;
        if (isV2) {
            std::istringstream hs(firstLine);
            std::string magic;
            std::size_t count = 0;
            std::uint64_t fp = 0;
            hs >> magic >> count >> fp;
            if (count != layers.size()) {
                std::cerr << "[weights] " << fileName << ": 层数不匹配 (文件 "
                          << count << ", 当前网络 " << layers.size() << ")" << std::endl;
                return -1;
            }
            if (fp != structureFingerprint()) {
                std::cerr << "[weights] " << fileName << ": 结构指纹不匹配 (文件 "
                          << fp << ", 当前网络 " << structureFingerprint()
                          << ") —— 这个权重文件不是给这个网络的" << std::endl;
                return -1;
            }
            payloadStart = (firstEol == std::string::npos) ? buffer.size() : firstEol + 1;
        }

        /*
           ============================================================
            第一步: 预校验 payload 的**每一行**(但不改动网络)
           ============================================================
           为什么值得多校验一遍: 载入失败时**绝不能**把网络改成半成品。各 agent 的
           loadModel() 在失败时只返回 false (调用方看一眼就接着用), 如果这时某些
           张量已经被写成了空张量 (解码失败时 fromString 返回的就是空张量), 下一次
           前向就是越界读 —— 一个坏文件能把"载入失败"升级成段错误。

           每一行的编码都是自校验的: 形状要能解析、base64 长度要等于形状的乘积 ×
           sizeof(T)、CRC32 要对得上。所以"所有行都能解码"等价于"这个文件完整"。

           **在内存里扫**: 权重文件一个可以到 146 MB, 实测
             整块读进来           1075 MB/s
             ifstream + getline 逐行 143 MB/s      <- 慢 7 倍, 而且下面"真正载入"那一遍
                                                      还要再来一次
           所以字节块要么来自**内存** (对弈期间改道的临时权重, 见本函数开头),
           要么整块读进内存 (几十毫秒), 预校验直接在内存上用 memchr 切行 ——
           省掉一次磁盘读和一整轮逐行流式读取。稀疏 MoE 那 3 个 146 MB 的文件因此从
           ~15 秒降到 ~6 秒 (启动 19 s -> 10 s 量级)。
        */
        std::size_t lineNo = 0;
        /*
           v2 文件的第一行是头 (CHWGT2 <层数> <指纹>), 它不是张量, 跳过;
           老格式 (v1) 的第一行就是第一个张量, 所以从 0 开始 ——
           这两件事已经在上面算成了 payloadStart (内存/磁盘两个来源走**同一段**代码)。
           (第一版忘了跳过头, 于是每个文件都在"第 1 个张量校验失败" —— 而这个
            错误又恰好被"载入失败不影响网络"那条保证掩盖住了: 网络是好的,
            只是什么都没载入。test_weights 立刻抓到了。)
        */
        std::size_t scan = payloadStart;
        /*
           **元素总数校验** (维度守卫)。
           为什么必须有: v2 的"结构指纹"只哈希**层的类型序列**, 不含任何维度 ——
           于是"同一套层结构、但输入维度不同"的文件指纹完全一样, 载入会被放行, 而
           `Layer::read` 是 `w = Tensor::fromString(...)` (整块替换), 网络的张量就被
           **静默换成文件里的形状**, 直到某次前向才崩 (或者更糟: 不崩但输出全是垃圾)。
           这正是 CHWGT2 想防的那类失效, 只是漏掉了维度这一维。
           这里的判据是"文件里所有张量的元素总数 == 当前网络的 paramCount()":
           任何维度变化 (输入维、隐层宽、专家数…只要总参数变了) 都会被拦下,
           而且**在改动网络之前** (仍守"载入失败时网络保持不变"的保证)。
           场景来源: 给某个 agent 的状态编码加平面 (1440 -> 1710) 之后, 旧权重文件必须
           是"被明确拒绝", 而不是"被静默读成另一个形状"。
        */
        long long fileElements = 0;
        while (scan < buffer.size()) {
            const std::size_t lineStart = scan;
            const std::size_t eol = buffer.find('\n', scan);
            std::size_t len = 0;
            if (eol == std::string::npos) {
                len = buffer.size() - lineStart;
                scan = buffer.size();
            } else {
                len = eol - lineStart;
                scan = eol + 1;
            }
            if (len == 0) {
                continue;   /* 末尾多一个换行是正常的 */
            }
            lineNo++;
            long long lineElements = 0;
            if (!Tensor::validateEncodedCount(buffer.data() + lineStart, len, lineElements)) {
                std::cerr << "[weights] " << fileName << ": 第 " << lineNo
                          << " 个张量校验失败 (文件被截断/损坏? ), "
                             "已放弃这次载入, 网络保持不变" << std::endl;
                return -1;
            }
            fileElements += lineElements;
        }
        if (lineNo == 0) {
            std::cerr << "[weights] " << fileName << ": 没有张量数据" << std::endl;
            return -1;
        }
        if (fileElements != paramCount()) {
            std::cerr << "[weights] " << fileName << ": 参数量不匹配 (文件 "
                      << fileElements << " 个元素, 当前网络 " << paramCount()
                      << " 个) —— 层类型一样但**维度不同**, 典型的来源是状态编码改了"
                         " (例如加了规则上下文平面)。拒绝载入, 网络保持不变。"
                      << std::endl;
            return -1;
        }

        /*
           第二步: 真正载入 (从 payload 开头重新读一遍)。
           数据源是**内存里的 buffer** (零拷贝的 ConstMemBuf), 不再是 ifstream ——
           这样"内存里那份临时权重"与"磁盘上的文件"走的是同一条载入代码, 不会出现
           "其中一种来源能载入、另一种不行"的分叉。
        */
        ConstMemBuf payloadBuf(buffer.data() + payloadStart,
                               buffer.size() - payloadStart);
        std::istream payload(&payloadBuf);
        Tensor::clearDecodeFailed();
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->read(payload);
        }
        /*
           payload.fail() 用来抓"在行边界上被截断"的文件: 这种情况下每一行都是完整的
           (预校验会通过), 但行数不够, 最后几层的 getline 会失败。
        */
        if (payload.fail() || Tensor::lastDecodeFailed()) {
            std::cerr << "[weights] " << fileName
                      << ": 张量数量不足 (文件被截断?), 这次载入不可信" << std::endl;
            return -1;
        }
        return 0;
    }
};

}
#endif // NET_HPP
