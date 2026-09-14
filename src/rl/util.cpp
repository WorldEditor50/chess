#include "util.hpp"
#include <functional>
#include <thread>

/* ============================================================
 *  RL::Random 的线程安全实现 (设计说明见 util.hpp 里 Random 的注释)
 * ============================================================ */
namespace {

/*
   纯函数: 把两个整数混成一个种子。**不含任何静态对象、不访问任何 thread_local、
   不用 std::call_once / std::random_device** —— 这一点是硬性的: 它会出现在
   thread_local 的初始化式里, 而 UCRT 的 _Init_thread_header/_Init_thread_footer
   之间要跑这段初始化代码, 那把锁不可重入, 期间再触发一次静态/TLS 初始化就死锁。
   实测症状: test_sparse_moe 在 main 之前挂死, 且**间歇**发作 (4 次里挂 3 次)。
*/
unsigned mixSeed(unsigned a, unsigned b)
{
    unsigned x = a * 0x9E3779B9u + b * 0x85EBCA6Bu + 0x165667B1u;
    x ^= x >> 16;  x *= 0x7FEB352Du;
    x ^= x >> 15;  x *= 0x846CA68Bu;
    x ^= x >> 16;
    return (x != 0u) ? x : 1u;
}

/*
   本线程的"盐": 让不同线程默认拿到不同的随机流, 于是 worker 就算忘了显式播种,
   也不会所有线程跑出完全一样的对局。用 OS 线程 id 而不是"thread_local 计数器" ——
   后者本身是另一个 thread_local, 会在初始化式里嵌套 TLS 初始化 (上面说的死锁)。

   注意线程 id 只保证"同时存活的两条线程不同", 跨次运行会重复 —— 所以要整轮
   可复现仍然必须显式调 seedCurrentThread(index)。
*/
unsigned threadSalt()
{
    const std::size_t h = std::hash<std::thread::id>()(std::this_thread::get_id());
    return (unsigned)(h ^ (h >> 32));
}

} // namespace

std::atomic<unsigned> RL::Random::baseSeed{ 0u };

/*
   初始化式只有纯函数 + threadSalt() —— 不读 baseSeed、不碰任何静态量, 因此
   不可能在这里嵌套出第二次静态/TLS 初始化。

   baseSeed 的默认值是 0, 表示"调用方没显式播种", 此时种子只由线程身份决定
   (跨次运行随机, 与原实现用 random_device 的效果一致); 一旦调用 setSeed(),
   所有显式播种路径都走 (baseSeed, index) 的确定性混合, 整轮可复现。
*/
thread_local std::default_random_engine RL::Random::engine(mixSeed(threadSalt(), 1u));
thread_local std::mt19937 RL::Random::generator(mixSeed(threadSalt(), 2u));

void RL::Random::seedCurrentThread(unsigned index)
{
    const unsigned b = baseSeed.load();
    engine.seed(mixSeed(b, index * 2u + 1u));
    generator.seed(mixSeed(b, index * 2u + 2u));
}

void RL::Random::setSeed(unsigned s)
{
    baseSeed.store(s);
    seedCurrentThread(0u);
}

void RL::zscore(Tensor &x)
{
    /* sigma */
    float u = x.mean();
    float sigma = std::sqrt(x.variance(u) + 1e-9);
    for (std::size_t i = 0 ; i < x.size(); i++) {
        x[i] = (x[i] - u)/sigma;
    }
    return;
}

void RL::normalize(Tensor &x)
{
    float minValue = x[0];
    float maxValue = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (minValue > x[i]) {
            minValue = x[i];
        }
        if (maxValue < x[i]) {
            maxValue = x[i];
        }
    }
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = (x[i] - minValue) / (maxValue - minValue);
    }
    return;
}

float RL::variance(const Tensor &x, float u)
{
    float sigma = 0;
    /* sigma */
    for (std::size_t i = 0 ; i < x.size(); i++) {
        sigma += (x[i] - u) * (x[i] - u);
    }
    return sigma / float(x.size());
}

float RL::covariance(const Tensor &x1, const Tensor &x2)
{
    float u = x1.mean();
    float v = x2.mean();
    float covar = 0;
    for (std::size_t i = 0; i < x1.size(); i++) {
         covar += (x1[i] - u) * (x2[i] - v);
    }
    return covar;
}

float RL::clip(float x, float lo, float hi)
{
    float y = x;
    if (x < lo) {
        y = lo;
    } else if (x > hi) {
        y = hi;
    }
    return y;
}

float RL::hmean(const RL::Tensor &x)
{
    float s = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        s += 1/x[i];
    }
    return float(x.size())/s;
}

float RL::gmean(const RL::Tensor &x)
{
    float s = 1;
    for (std::size_t i = 0; i < x.size(); i++) {
        s *= x[i];
    }
    return std::pow(s, 1.0/x.size());
}

float RL::gaussian(float x, float u, float sigma)
{
    /* Standard normal pdf: 1/sqrt(2*pi*sigma^2) * exp(-(x-u)^2/(2*sigma^2)).
       The previous version used sqrt(2*pi*sigma) and divided by sigma instead
       of 2*sigma^2, so it was not a Gaussian at all. */
    float s2 = sigma*sigma;
    return std::exp(-0.5f*(x - u)*(x - u)/s2)/std::sqrt(2.0f*pi*s2);
}
