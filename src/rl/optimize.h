#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <cctype>
#include <string>
#include "tensor.hpp"
#include "util.hpp"

namespace RL {

/*
 * ============================================================
 *  [2026-10] 梯度裁剪口径 (原来只有"一个 bool", 而那个 bool 的名字是错的)
 * ============================================================
 * 历史: `Optimize::RMSProp/SGD/Adam` 的第 7 个参数叫 `clipGrad`, 默认 true, 而它做的是
 *
 *     dw /= dw.norm2() + 1e-8;
 *
 * —— 这**不是裁剪**, 而是"把整张梯度张量归一化到单位长度"。方向保留, 幅度信息被丢掉。
 *
 * ⚠ 但必须把下面这句写在最前面, 否则会得出一个错的结论:
 *   **对 RMSProp 来说, 任何"整张量等比缩放"都会被它自己的逐坐标归一抵消** ——
 *   `v[i] = ρv[i] + (1−ρ)dw[i]²` 与 `w[i] -= lr·dw[i]/(sqrt(v[i])+1e-9)`,
 *   `dw` 乘 s 时 `sqrt(v)` 也乘 s, 商不变。所以
 *     `PER_TENSOR_UNIT_NORM` / `GLOBAL_NORM` / `NONE` 三种模式在**所有实用梯度尺度上**
 *     给出**同一条轨迹** (实测: 尺度 1 与 1e6 的一步 |Δw| 完全相同, 相对差 < 1e-6)。
 *   ⇒ **这个开关不是训练问题的解药**。TB 专家"90% 参数是惰性的却与有效参数等速移动"
 *     这件事的根因在 RMSProp 的逐坐标归一本身 (每坐标 ±lr), 与 clipGrad 无关。
 *   三种模式**唯一**的差别在极端小梯度上 (实测):
 *     |g| ≳ 1e-8 : 三者相同;
 *     1e-16 ≲ |g| ≲ 3e-9 : legacy 把 dw 放大 1e8 倍后仍在 RMSProp 的工作区间里 ⇒ 照常
 *                          给出完整一步; 而 NONE/GLOBAL_NORM 的 `sqrt(v) < 1e-9` 被那个
 *                          epsilon 顶住 ⇒ 更新≈0 (**这两种模式反而更早"冻结"**);
 *     |g| ≲ 1e-17 : 三者都不动。
 *
 * 那为什么还要这三个模式: ① 把语义写清楚 (名字叫 clip 却是归一化, 已经骗过至少一次
 * "换个裁剪就能修"的判断); ② 让"关掉裁剪/换成全局裁剪"的**对照实验**可以只靠命令行跑,
 * 而不是改源码重编 (本工程踩过"改了源码但二进制没重编"); ③ `NONE` 那一档配合
 * `Net::gradNorm()` 才能读出"这一步真实的信号有多大"。
 * 断言把上面这些性质钉在 `test_ppo_backbone` 的 [4] 节。
 */
enum GradClipMode {
    GRAD_CLIP_PER_TENSOR_UNIT_NORM = 0,   /* 旧行为: dw /= |dw| + 1e-8 (默认) */
    GRAD_CLIP_GLOBAL_NORM,                /* 先算全网范数, 超限时整体等比缩小 */
    GRAD_CLIP_NONE                        /* 不裁剪 */
};

inline const char *gradClipModeName(int m)
{
    switch (m) {
    case GRAD_CLIP_PER_TENSOR_UNIT_NORM: return "per-tensor-unit-norm(legacy)";
    case GRAD_CLIP_GLOBAL_NORM:          return "global-norm";
    case GRAD_CLIP_NONE:                 return "none";
    default:                             return "?";
    }
}

/* 解析 "legacy"/"global"/"none"(大小写不敏感, 另有等价别名); 失败返回 false */
inline bool parseGradClipMode(const char *s, int &out)
{
    if (s == nullptr) { return false; }
    std::string v(s);
    for (std::size_t i = 0; i < v.size(); i++) {
        v[i] = (char)std::tolower((unsigned char)v[i]);
    }
    if (v == "legacy" || v == "unit" || v == "per-tensor" || v == "1") {
        out = GRAD_CLIP_PER_TENSOR_UNIT_NORM; return true;
    }
    if (v == "global" || v == "global-norm" || v == "clip") {
        out = GRAD_CLIP_GLOBAL_NORM; return true;
    }
    if (v == "none" || v == "off" || v == "0") {
        out = GRAD_CLIP_NONE; return true;
    }
    return false;
}

namespace Optimize {

inline void SGD(Tensor &w, Tensor &dw,
                float lr, float gamma=0, bool clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2() + 1e-8;
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        w[i] = (1 - gamma)*w[i] - lr*dw[i];
    }
    return;
}

inline void SGDM(Tensor &w, Tensor &m, Tensor &dw,
                 float lr, float alpha,
                 float gamma=0, float clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2() + 1e-8;
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        m[i] = (1 - gamma)*m[i] - dw[i]*alpha;
        w[i] -= lr * m[i];
    }
    return;
}

inline void Adagrad(Tensor &w, Tensor &r, Tensor &dw,
                    float lr, float gamma=0, float clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2() + 1e-8;
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        r[i] += dw[i]*dw[i];
        w[i] = (1 - gamma)*w[i] - lr*dw[i]/(std::sqrt(r[i]) + 1e-9);
    }
    return;
}

inline void AdaDelta(Tensor &w, Tensor &v, Tensor &delta, Tensor &dwPrime, Tensor &dw,
                     float lr, float rho, float gamma = 0, bool clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2() + 1e-8;
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        v[i] = rho * v[i] + (1 - rho) * dw[i] * dw[i];
        delta[i] = rho*delta[i] + (1 - rho)*dwPrime[i]*dwPrime[i];
        dwPrime[i] = std::sqrt((delta[i] + 1e-9)/(v[i] + 1e-9));
        w[i] = (1 - gamma)*w[i] - lr*dwPrime[i];
    }
    return;
}

inline void RMSProp(Tensor &w, Tensor &v, Tensor &dw,
                    float lr, float rho=0.9,
                    float gamma = 0, bool clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2() + 1e-8;
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        v[i] = rho*v[i] + (1 - rho)*dw[i]*dw[i];
        w[i] = (1 - gamma)*w[i] - lr*dw[i]/(std::sqrt(v[i]) + 1e-9);
    }
    return;
}

inline void Adam(Tensor &w, Tensor &v, Tensor &m, Tensor &dw,
                 float alpha_, float beta_,
                 float lr, float alpha, float beta,
                 float gamma = 0, bool clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2() + 1e-8;
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        m[i] = alpha*m[i] + (1 - alpha)*dw[i];
        v[i] = beta*v[i] + (1 - beta)*dw[i]*dw[i];
        float m_ = m[i]/(1 - alpha_);
        float v_ = v[i]/(1 - beta_);
        w[i] = (1 - gamma)*w[i] - lr*m_/(std::sqrt(v_) + 1e-9);
    }
    return;
}

inline void clamp(Tensor &w, float c0, float cn)
{
    std::uniform_real_distribution<float> uniform(c0, cn);
    for (std::size_t i = 0; i < w.totalSize; i++) {
        if (w[i] > cn || w[i] < c0) {
            w[i] = uniform(RL::Random::engine);
        }
    }
    return;
}

} // optimizer

} // RL
#endif // OPTIMIZER_H
