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

    void backward(const Tensor &x, const Tensor &loss)
    {
        std::size_t outputIndex = layers.size() - 1;
        layers[outputIndex]->e = loss;
        for (int i = layers.size() - 1; i > 0; i--) {
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

    void RMSProp(float lr, float rho=0.9, float decay=0)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->RMSProp(lr, rho, decay, true);
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
        {
            std::size_t lineNo = 0;
            std::string line;
            while (std::getline(file, line)) {
                lineNo++;
                if (!Tensor::validateEncoded(line)) {
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
