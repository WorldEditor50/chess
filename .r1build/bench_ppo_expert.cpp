/*
   一次性 scratch (不入库到 CMake): 量 "PPO 骨干换专家" 的代价。

   对照的是 rl/ppo.cpp 里那条骨干
       state -> SparseMoE<Expert, E, K> -> Tanh(d -> h) -> Linear(h -> out)
   只是把 Expert / E / K 换掉, 用同一份代码量参数量与前向+反向耗时。

   编译见 .r1build/bench_ppo_expert.bat。
*/
#include <cstdio>
#include <chrono>
#include <memory>
#include <vector>
#include "net.hpp"
#include "sparse_moe.hpp"
#include "expert.hpp"
#include "transformer.hpp"
#include "layer.h"
#include "util.hpp"

using namespace RL;

template<typename E, int N, int K>
static void bench(const char *name, int d, int h, int a, int iters, bool withGrad)
{
    Net::Layers ls;
    ls.push_back(std::make_shared<SparseMoE<E, N, K> >(d, withGrad, 64));
    ls.push_back(Layer<Tanh>::_(d, h, true, withGrad));
    ls.push_back(Layer<Linear>::_(h, a, true, withGrad));
    Net net(ls);

    Tensor x(d, 1);
    x.zero();
    for (int i = 0; i < d; i += 7) { x[i] = 1.0f; }
    Tensor loss(a, 1);
    loss.fill(1.0f / (float)a);

    net.forward(x);                 /* warm up */

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) { net.forward(x); }
    const auto t1 = std::chrono::high_resolution_clock::now();
    double fwd = 0.0, fwb = 0.0;
    if (withGrad) {
        for (int it = 0; it < iters; it++) { net.forward(x); net.backward(x, loss); }
        const auto t2 = std::chrono::high_resolution_clock::now();
        fwd = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        fwb = std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
    } else {
        fwd = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    }

    std::printf("  %-40s params=%11lld   fwd=%8.3f ms   fwd+bwd=%8.3f ms\n",
                name, net.paramCount(), fwd, fwb);
    std::fflush(stdout);
}

int main()
{
    Random::setSeed(12345u);

    const int D = 1440;   /* PPOMCTSAgent::STATE_DIM */
    const int H = 64;     /* hiddenDim: 压到 64 再进策略头 */
    const int A = 8100;   /* ACTION_DIM */

    std::printf("PPO 骨干 (d=%d, h=%d, head=%d), 单网络, 20 次平均\n", D, H, A);
    std::printf("\n[A] 现在的配置: MlpExpert (E=8, top-2)\n");
    bench<MlpExpert, 8, 2>("MlpExpert          E=8 top=2", D, H, A, 20, true);

    std::printf("\n[B] 专家换成 TransformerBlock<16,360> (E=8, top-2)\n");
    bench<TransformerBlock<16, 360>, 8, 2>("TB<16,360>         E=8 top=2", D, H, A, 10, true);

    std::printf("\n[C] 专家换成 TransformerBlock<16,360> (E=4, top-1)\n");
    bench<TransformerBlock<16, 360>, 4, 1>("TB<16,360>         E=4 top=1", D, H, A, 20, true);

    std::printf("\n[D] 专家换成 TransformerBlock<15,315> (E=4, top-1, 与 SACAZ 同一配置)\n");
    bench<TransformerBlock<15, 315>, 4, 1>("TB<15,315>         E=4 top=1", D, H, A, 20, true);

    std::printf("\n[E] 专家换成 TransformerBlock<16,360> (E=2, top-1)\n");
    bench<TransformerBlock<16, 360>, 2, 1>("TB<16,360>         E=2 top=1", D, H, A, 20, true);

    std::printf("\n[F] 只推理形态 (withGrad=false, worker 用), E=4 top=1\n");
    bench<TransformerBlock<16, 360>, 4, 1>("TB<16,360>         E=4 top=1 (nograd)", D, H, A, 20, false);

    return 0;
}
