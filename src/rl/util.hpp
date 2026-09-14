#ifndef UTIL_HPP
#define UTIL_HPP
#include <atomic>
#include <vector>
#include <random>
#include <ctime>
#include "tensor.hpp"

namespace RL {

constexpr static float pi = 3.1415926535898;

struct Random {
    /*
       线程安全: 引擎改成**每条线程一份**。

       以前 engine / generator 是进程级全局静态量, 而 uniform()/categorical()/normal()
       会被多处并发调用 —— 至少包括 GUI 的 m_bgTrainThread 与主线程 (chessboard.cpp
       的自对弈/后台训练), 以及本项目新增的多线程分身自对弈 worker。std::minstd_rand
       与 std::mt19937 都不是线程安全的, 并发使用是数据竞争 (UB), 症状是偶发的权重
       初始化错乱、采样序列异常, 而且极难复现。这是**修一个既有缺陷**, 不只是为多线程
       让路。

       单线程行为完全不变: 所有访问都在同一条线程, 拿到的还是同一个引擎、同一条序列。
       多线程下每条线程有独立引擎; 默认种子由线程身份派生, 所以两条线程**不会**退化成
       同一条随机序列 (worker 忘了播种也不会跑出一样的棋); 需要整轮可复现时, worker
       入口显式调 seedCurrentThread(index) —— index 由调用方给, 不依赖线程创建顺序,
       因此并行跑也完全可复现。

       **thread_local 的初始化式必须是"纯函数"** —— 这是踩过坑的硬约束:
       初始化的值只能来自 threadSalt() 这种普通函数调用, 不能涉及任何静态对象、
       std::call_once 或 std::random_device。原因写在 util.cpp 里 threadSalt() 的注释:
       UCRT 的 _Init_thread_header/_Init_thread_footer 之间跑用户初始化代码, 而那把锁
       不可重入, 在初始化式里再触发一次静态/TLS 初始化就会死锁。实测症状是
       test_sparse_moe 在 main 之前挂死 (CPU 不再增长、线程全部 Wait), 而且**间歇**发作。
    */
    static std::atomic<unsigned> baseSeed;
    static thread_local std::default_random_engine engine;
    static thread_local std::mt19937 generator;

    /* 设定进程级基种子并立即播种当前线程。用它替代 Random::engine.seed(x) */
    static void setSeed(unsigned s);
    /* worker 线程入口调用: 用 (baseSeed, index) 确定性地播种本线程 */
    static void seedCurrentThread(unsigned index);

    inline static int categorical(const Tensor& p)
    {
        std::discrete_distribution<int> distribution(p.begin(), p.end());
        return distribution(Random::generator);
    }
    inline static void uniform(Tensor &x, float x1, float x2)
    {
        std::uniform_real_distribution<float> distribution(x1, x2);
        for (std::size_t i = 0; i < x.size(); i++) {
            x[i] = distribution(Random::engine);
        }
        return;
    }
    inline static void bernoulli(Tensor &x, float p)
    {
        std::bernoulli_distribution distribution(p);
        for (std::size_t i = 0; i <x.size(); i++) {
            x[i] = distribution(Random::engine) / (1 - p);
        }
        return;
    }
    inline static void normal(Tensor &x, float u, float sigma)
    {
        std::normal_distribution<float> distribution(u, sigma);
        for (std::size_t i = 0; i <x.size(); i++) {
            x[i] = distribution(Random::generator);
        }
        return;
    }
};

namespace Norm {
    inline float l1(const Tensor &x1, const Tensor& x2)
    {
        float s = 0;
        for (std::size_t i = 0; i < x1.size(); i++) {
            float d = std::abs(x1[i] - x2[i]);
            s += d;
        }
        return s;
    }
    inline float l2(const Tensor &x1, const Tensor& x2)
    {
        float s = 0;
        for (std::size_t i = 0; i < x1.size(); i++) {
            float d = x1[i] - x2[i];
            s += d*d;
        }
        return std::sqrt(s);
    }
    inline float lp(const Tensor &x1, const Tensor& x2, float p)
    {
        float s = 0;
        for (std::size_t i = 0; i < x1.size(); i++) {
            float d = x1[i] - x2[i];
            s += std::pow(d, p);
        }
        return std::pow(s, 1.0/p);
    }
    inline float l8(const Tensor &x1, const Tensor& x2)
    {
        float s = x1[0] - x2[0];
        for (std::size_t i = 1; i < x1.size(); i++) {
            float d = x1[i] - x2[i];
            if (d > s) {
                s = d;
            }
        }
        return s;
    }
}
inline float sigmoid(float x)
{
    return 1.0/(1 + std::exp(-x));
}
inline Tensor& sqrt(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::sqrt(x[i]);
    }
    return x;
}

inline Tensor& exp(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::exp(x[i]);
    }
    return x;
}

inline Tensor& log(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::log(x[i]);
    }
    return x;
}

inline Tensor& tanh(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::tanh(x[i]);
    }
    return x;
}

inline Tensor& sin(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::sin(x[i]);
    }
    return x;
}

inline Tensor& cos(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::cos(x[i]);
    }
    return x;
}

inline Tensor sqrt(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::sqrt(x[i]);
    }
    return y;
}

inline Tensor exp(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::exp(x[i]);
    }
    return y;
}

inline Tensor log(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::log(x[i]);
    }
    return y;
}

inline Tensor tanh(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::tanh(x[i]);
    }
    return y;
}

inline Tensor sin(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::sin(x[i]);
    }
    return y;
}

inline Tensor cos(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::cos(x[i]);
    }
    return y;
}

inline Tensor upTriangle(int rows, int cols)
{
    Tensor x(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = i + 1; j < cols; j++) {
            x(i, j) = 1;
        }
    }
    return x;
}

inline Tensor lowTriangle(int rows, int cols)
{
    Tensor x(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < i; j++) {
            x(i, j) = 1;
        }
    }
    return x;
}

/* exponential moving average */
inline void lerp(Tensor &x, const Tensor xi, float r)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = (1 - r) * x[i] + r * xi[i];
    }
    return;
}

inline Tensor onehot(const Tensor &xi)
{
    Tensor xo(xi.shape);
    int k = xi.argmax();
    xo[k] = 1;
    return xo;
}

float gaussian(float x, float u, float sigma);
/* Clamp x into [lo, hi]. The parameters used to be named (sup, inf), which
   read as the opposite bounds of what they actually are. */
float clip(float x, float lo, float hi);
float hmean(const Tensor &x);
float gmean(const Tensor &x);
float variance(const Tensor &x, float u);
float covariance(const Tensor& x1, const Tensor& x2);
void zscore(Tensor &x);
void normalize(Tensor &x);

inline float entropy(float p)
{
    return -p*std::log(p + 1e-7);
}
namespace Metrics {
/* Kullback Leibler Divergence: KL(p||q) = Σ p*log(p/q) */
inline float KL(float p, float q)
{
    return p*std::log(p/q);
}

/* Jensen-Shannon */
inline float JS(float p, float q)
{
    float r = (p + q)/2;
    return (KL(p, r) + KL(q, r))/2;
}

}


inline float M3(const Tensor &x, float u)
{
    float s = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        float d = x[i] - u;
        s += d*d*d;
    }
    return s/float(x.size());
}

inline float M4(const Tensor &x, float u)
{
    float s = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        float d = x[i] - u;
        s += d*d*d*d;
    }
    return s/float(x.size());
}

inline void clamp(Tensor &x, float x1, float x2)
{
    std::uniform_real_distribution<float> uniform(x1, x2);
    for (std::size_t i = 0; i < x.size(); i++) {
        float xi = x[i];
        x[i] = xi < x1 ? x1 : xi;
        x[i] = xi > x2 ? x2 : xi;
    }
    return;
}

template<typename T>
inline void uniformRand(T &x, float x1, float x2)
{
    std::uniform_real_distribution<float> uniform(x1, x2);
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = uniform(Random::engine);
    }
    return;
}

inline Tensor& eGreedy(Tensor& x, float exploringRate, bool hard)
{
    std::uniform_real_distribution<float> uniformReal(0, 1);
    float p = uniformReal(Random::engine);
    if (p < exploringRate) {
        if (hard) {
            x.zero();
        }
        std::uniform_int_distribution<int> uniform(0, x.size() - 1);
        int index = uniform(Random::engine);
        x[index] = 1;
    }
    return x;
}

inline Tensor& noise(Tensor& x)
{
    Tensor epsilon(x.shape);
    Random::uniform(epsilon, 0, 2);
    x += epsilon;
    float m = x.max();
    /* Guard the normalization: dividing by a zero max produced inf/NaN. */
    if (std::fabs(m) > 1e-12f) {
        x /= m;
    }
    return x;
}

inline Tensor& noise(Tensor& x, float exploringRate)
{
    std::uniform_real_distribution<float> uniform(0, 1);
    float p = uniform(Random::engine);
    if (p < exploringRate) {
        Tensor epsilon(x.shape);
        Random::uniform(epsilon, 0, 2);
        x += epsilon;
        float m = x.max();
        if (std::fabs(m) > 1e-12f) {
            x /= m;
        }
    }
    return x;
}

inline Tensor& softmax(Tensor &x)
{
    float s = 0;
    float maxValue = x.max();
    x -= maxValue;
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::exp(x[i]);
        s += x[i];
    }
    x /= s;
    return x;
}

inline Tensor& gumbelSoftmax(Tensor &x, float tau)
{
    Tensor epsilon(x.shape);
    Random::uniform(epsilon, 0, 1);
    for (std::size_t i = 0; i < epsilon.size(); i++) {
        epsilon[i] = -std::log(-std::log(epsilon[i] + 1e-7) + 1e-7);
    }
    x += epsilon;
    x /= tau;
    x = softmax(x);
    return x;
}

inline Tensor& gumbelSoftmax(Tensor &x, const Tensor& tau)
{
    Tensor epsilon(x.shape);
    Random::uniform(epsilon, 0, 1);
    for (std::size_t i = 0; i < epsilon.size(); i++) {
        epsilon[i] = -std::log(-std::log(epsilon[i] + 1e-7) + 1e-7);
    }
    x += epsilon;
    x /= tau;
    softmax(x);
    return x;
}

}
#endif // UTIL_HPP
