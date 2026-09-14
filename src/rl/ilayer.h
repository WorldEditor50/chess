#ifndef ILAYER_H
#define ILAYER_H
#include "tensor.hpp"
#include <memory>

namespace RL {

class iLayer
{
public:
    enum Type {
        LAYER_FC = 0,
        LAYER_LSTM,
        LAYER_CONCAT,
        LAYER_SCALEDCONCAT,
        LAYER_CONV2D,
        LAYER_MAXPOOLING,
        LAYER_AVGPOOLING,
        LAYER_ATTENTION,
        LAYER_SCALEDDOTPRODUCT,
        LAYER_MHA,
        LAYER_TRANSFORMERBLOCK,
        LAYER_MOE,
        LAYER_SSM,
        LAYER_MAMBA
    };
    using sptr = std::shared_ptr<iLayer>;
public:
    int type;
    Tensor o;
    Tensor e;
public:
    iLayer(){}
    virtual ~iLayer(){}
    virtual void initParams(){}
    virtual Tensor& forward(const Tensor& x, bool inference=false)
    {
        return o;
    }
    virtual void backward(const Tensor& x, Tensor &ei){}
    virtual void cacheError(const Tensor &e){}
    virtual void SGD(float lr){}
    virtual void RMSProp(float lr, float rho, float decay, bool clipGrad){}
    virtual void Adam(float lr, float alpha, float beta,
                      float alpha_, float beta_,
                      float decay, bool clipGrad){}
    virtual void clamp(float c0, float cn){}
    virtual void copyTo(iLayer* layer){}
    virtual void softUpdateTo(iLayer* layer, float alpha){}
    /*
       参数量 (只读诊断)。
       层自己知道它有多少个可训练标量, 上层就不必对每种层做 dynamic_cast 去拼凑 ——
       骨干每换一次就得重算一遍参数量的话, 没人会去维护那个数字。
       默认 0; 容器型层 (SparseMoE / MlpExpert) 自己重载并累加子层。
    */
    virtual long long paramCount() const { return 0; }
    virtual void write(std::ofstream &file){}
    virtual void read(std::ifstream &file){}
};

}
#endif // ILAYER_H
