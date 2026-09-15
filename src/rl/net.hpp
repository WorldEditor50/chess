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

namespace RL {

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
            } else if ((preLayer->type == iLayer::LAYER_ATTENTION ||
                      preLayer->type == iLayer::LAYER_MHA ||
                      preLayer->type == iLayer::LAYER_SCALEDCONCAT ||
                      preLayer->type == iLayer::LAYER_MOE) &&
                      layer->type == iLayer::LAYER_FC) {
                layer->backward(preLayer->o, preLayer->e);
            } else {
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

    int save(const std::string &fileName) const
    {
        const std::string tmp = fileName + kWeightTmpSuffix;
        {
            std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
            if (!file.is_open()) {
                return -1;
            }
            file << kWeightMagic << " " << layers.size() << " "
                 << structureFingerprint() << "\n";
            for (std::size_t i = 0; i < layers.size(); i++) {
                layers[i]->write(file);
            }
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
        std::ifstream file(fileName, std::ios::binary);
        if (!file.is_open()) {
            return -1;
        }

        std::string firstLine;
        if (!std::getline(file, firstLine)) {
            std::cerr << "[weights] " << fileName << ": 空文件" << std::endl;
            return -1;
        }

        /*
           先看第一行是不是 v2 的头。不是的话 (老文件, 第一行就是第一个张量) 要
           **倒回文件开头**, 再按 v1 逐层读 —— 这样老权重文件不需要任何转换。
        */
        const std::size_t magicLen = std::strlen(kWeightMagic);
        const bool isV2 = (firstLine.compare(0, magicLen, kWeightMagic) == 0);
        std::streamoff payloadStart = 0;
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
            payloadStart = (std::streamoff)file.tellg();
        } else {
            file.clear();
            file.seekg(0, std::ios::beg);
        }

        /*
           ============================================================
            第一步: 预校验 paylaod 的**每一行**(但不改动网络)
           ============================================================
           为什么值得多读一遍: 载入失败时**绝不能**把网络改成半成品。各 agent 的
           loadModel() 在失败时只返回 false (调用方看一眼就接着用), 如果这时某些
           张量已经被写成了空张量 (解码失败时 fromString 返回的就是空张量), 下一次
           前向就是越界读 —— 一个坏文件能把程序从"载入失败"升级成段错误。

           每一行的编码都是自校验的: 形状要能解析、base64 长度要等于形状的乘积 ×
           sizeof(T)、CRC32 要对得上。所以"所有行都能解码"等价于"这个文件完整"。
        */
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
           所以先把文件整块读进内存 (几十毫秒), 预校验直接在内存上用 memchr 切行 ——
           省掉一次磁盘读和一整轮逐行流式读取。稀疏 MoE 那 3 个 146 MB 的文件因此从
           ~15 秒降到 ~6 秒 (启动 19 s -> 10 s 量级)。
        */
        std::string buffer;
        {
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

        std::size_t lineNo = 0;
        /*
           v2 文件的第一行是头 (CHWGT2 <层数> <指纹>), 它不是张量, 跳过;
           老格式 (v1) 的第一行就是第一个张量, 所以从 0 开始。
           (第一版忘了跳过头, 于是每个文件都在"第 1 个张量校验失败" —— 而这个
            错误又恰好被"载入失败不影响网络"那条保证掩盖住了: 网络是好的,
            只是什么都没载入。test_weights 立刻抓到了。)
        */
        std::size_t scan = 0;
        if (isV2) {
            const std::size_t hdrEnd = buffer.find('\n');
            scan = (hdrEnd == std::string::npos) ? buffer.size() : hdrEnd + 1;
        }
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
            if (!Tensor::validateEncoded(buffer.data() + lineStart, len)) {
                std::cerr << "[weights] " << fileName << ": 第 " << lineNo
                          << " 个张量校验失败 (文件被截断/损坏? ), "
                             "已放弃这次载入, 网络保持不变" << std::endl;
                return -1;
            }
        }
        if (lineNo == 0) {
            std::cerr << "[weights] " << fileName << ": 没有张量数据" << std::endl;
            return -1;
        }

        /* 第二步: 真正载入 (从 payload 开头重新读一遍; 数据已经在系统缓存里) */
        file.clear();
        file.seekg(payloadStart, std::ios::beg);
        Tensor::clearDecodeFailed();
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->read(file);
        }
        /*
           file.fail() 用来抓"在行边界上被截断"的文件: 这种情况下每一行都是完整的
           (预校验会通过), 但行数不够, 最后几层的 getline 会失败。
        */
        if (file.fail() || Tensor::lastDecodeFailed()) {
            std::cerr << "[weights] " << fileName
                      << ": 张量数量不足 (文件被截断?), 这次载入不可信" << std::endl;
            return -1;
        }
        return 0;
    }
};

}
#endif // NET_HPP
