/* 一次性编译/实例化自检 (scratch) —— 把所有新模板组合都实例化一遍 */
#include <cstdio>
#include "rl/tensor.hpp"
#include "rl/layer.h"
#include "rl/concat.hpp"
#include "rl/moe.hpp"
#include "rl/sparse_moe.hpp"
#include "rl/expert.hpp"
#include "rl/net.hpp"
#include "rl/loss.h"

using namespace RL;

int main()
{
    const int D = 32;
    Tensor x(D, 1);
    Random::uniform(x, -1.0f, 1.0f);

    /* ScaledConcat: 三种专家 + 保维/残差 */
    {
        Net net(ScaledConcat<MlpExpert, 4, 2>::_(D, true, 8));
        Tensor &o = net.forward(x, true);
        Tensor t(D, 1);
        Random::uniform(t, -0.5f, 0.5f);
        Tensor e = Loss::MSE::df(o, t);
        net.backward(x, e);
        std::printf("ScaledConcat<MlpExpert,4,2>            out=%zu param=%lld\n",
                    (std::size_t)o.totalSize, net.paramCount());
    }
    {
        Net net(ScaledConcat<TransformerBlock<4>, 2, 3>::_(D, true));
        std::printf("ScaledConcat<TransformerBlock<4>,2,3> out=%zu param=%lld\n",
                    (std::size_t)net.forward(x, true).totalSize, net.paramCount());
    }
    {
        Net net(ScaledConcat<Layer<Gelu>, 8, 4>::_(D, true));
        std::printf("ScaledConcat<Layer<Gelu>,8,4>         out=%zu param=%lld\n",
                    (std::size_t)net.forward(x, true).totalSize, net.paramCount());
    }
    {
        /* 保维 + 残差: 输出宽度 == 输入宽度 */
        Net net(ScaledConcat<MlpExpert, 4, 2, D>::_(D, true, 8, true));
        std::printf("ScaledConcat<MlpExpert,4,2,Out=32>    out=%zu param=%lld\n",
                    (std::size_t)net.forward(x, true).totalSize, net.paramCount());
        Net net2(ScaledConcat<MlpExpert, 4, 2, D>::_(D, false, 8, false, 2.0f, false));
        net2.forward(x, true);
    }
    /* MOE: 老写法必须还是老行为 */
    {
        Net net(MOE<3, 4>::_(D, true));
        std::printf("MOE<3,4> (默认 TB 专家)               out=%zu param=%lld\n",
                    (std::size_t)net.forward(x, true).totalSize, net.paramCount());
    }
    {
        Net net(MOE<4, 4, MlpExpert>::_(D, true, 16));
        std::printf("MOE<4,4,MlpExpert> hidden=16           out=%zu param=%lld\n",
                    (std::size_t)net.forward(x, true).totalSize, net.paramCount());
    }
    {
        Net net(MOE<4, 4, Layer<Gelu> >::_(D, true));
        std::printf("MOE<4,4,Layer<Gelu>>                  out=%zu param=%lld\n",
                    (std::size_t)net.forward(x, true).totalSize, net.paramCount());
    }
    {
        Net net(MOE<4, 4, MlpExpert>::_(D, true, 16, true));
        std::printf("MOE<4,4,MlpExpert> + scaleExperts      param=%lld\n", net.paramCount());
    }
    /* SparseMoE 一个字没改, 但要确认抽头文件之后还能编 */
    {
        Net net(std::make_shared<SparseMoE<MlpExpert, 4, 2> >(D, true, 8));
        std::printf("SparseMoE<MlpExpert,4,2>               out=%zu param=%lld\n",
                    (std::size_t)net.forward(x, true).totalSize, net.paramCount());
    }
    std::printf("OK\n");
    return 0;
}
