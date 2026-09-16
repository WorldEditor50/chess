/*
 * test_weights_main.cpp - 权重文件格式 (Net::save / Net::load) 的验证
 * ============================================================================
 *
 * 为什么值得单独一个测试: 权重文件是**训练成果的唯一载体**, 而它原来有三个问题,
 * 每一个都能静默地毁掉训练结果:
 *
 *   1. **有损**。老格式是十进制文本 (`std::ostream << float`, 默认 6 位有效数字)。
 *      存一次读回来权重就漂移 ~1e-6 相对。而 ChessBoard::backgroundTrainLoop 每轮
 *      都在 save -> load, 于是这个漂移会被反复注入。→ 现在必须是**逐比特无损**。
 *   2. **无校验**。文件被截断 / 翻了一个字节 / 把别的 agent 的权重喂进来, 都会
 *      静默载入一堆形状错乱或数值不对的张量。→ 现在要有: 结构指纹 + 每张量的
 *      CRC32 + 长度检查, 任何一项不对都必须返回失败。
 *   3. **非原子**。写一半崩掉/被 kill 就把上一次的模型毁了 (关窗正好会打断后台
 *      训练)。→ 现在先写 .tmp 再 rename。
 *
 * 这个测试逐条盯上面三件事, 另外确认**老格式的文件仍然能读**(兼容性)。
 */
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

#include "rl/tensor.hpp"
#include "rl/layer.h"
#include "rl/net.hpp"
#include "rl/loss.h"
#include "rl/transformer.hpp"

using namespace RL;

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                               \
            ++g_failed;                                              \
            std::printf("  [FAIL] %s\n", (msg));                     \
        }                                                            \
    } while (0)

static long long fileSize(const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        return -1;
    }
    return (long long)f.tellg();
}

static bool readWholeFile(const std::string &path, std::string &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        return false;
    }
    out.assign((std::istreambuf_iterator<char>(f)),
               std::istreambuf_iterator<char>());
    return true;
}

static bool writeWholeFile(const std::string &path, const std::string &data)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.good()) {
        return false;
    }
    f.write(data.data(), (std::streamsize)data.size());
    return f.good();
}

/* 一个确定性的小网络, 覆盖三种层 (Layer<Tanh> / Layer<Linear> / TransformerBlock) */
static Net makeNet(bool withGrad = true)
{
    return Net(Layer<Tanh>::_(24, 8, true, withGrad),
               TransformerBlock<4, 12>::_(8, withGrad),
               Layer<Linear>::_(8, 5, true, withGrad));
}

static Tensor makeInput()
{
    Tensor x(24, 1);
    for (int i = 0; i < 24; i++) {
        x[i] = 0.3f * std::sin(1.7f * (float)i);
    }
    return x;
}

/* 逐比特比较两个张量 (权重无损检查用) */
static bool sameBits(const Tensor &a, const Tensor &b)
{
    if (a.size() != b.size()) {
        return false;
    }
    return std::memcmp(a.val.data(), b.val.data(), a.size() * sizeof(float)) == 0;
}

/* 把网络里所有 iFcLayer 的 w/b 收集起来 (逐比特比较用) */
static void collectFc(Net &net, std::vector<const Tensor *> &out)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        const iFcLayer *fc = dynamic_cast<const iFcLayer *>(net[i]);
        if (fc != nullptr) {
            out.push_back(&fc->w);
            out.push_back(&fc->b);
        }
        /* TransformerBlock 里的子层: 它不是 iFcLayer, 但注意力/FFN 是 */
        const TransformerBlock<4, 12> *tb =
            dynamic_cast<const TransformerBlock<4, 12> *>(net[i]);
        if (tb != nullptr) {
            out.push_back(&tb->gamma1);
            out.push_back(&tb->beta1);
            out.push_back(&tb->gamma2);
            out.push_back(&tb->beta2);
            out.push_back(&tb->ffn_up.w);
            out.push_back(&tb->ffn_up.b);
            out.push_back(&tb->ffn_down.w);
            out.push_back(&tb->ffn_down.b);
        }
    }
}

/* ============================================================
 *  [1] 无损: 存 -> 读 -> 再存, 两次的文件必须逐字节相同
 * ============================================================ */
static void part1()
{
    std::printf("\n[1] 无损往返 (存 -> 读 -> 再存, 文件必须逐字节相同)\n");
    const std::string p1 = "test_weights_a.wgt";
    const std::string p2 = "test_weights_b.wgt";

    Net net = makeNet();
    Tensor x = makeInput();
    Tensor y0 = net.forward(x, true);

    CHECK(net.save(p1) == 0, "save 返回成功");

    Net net2 = makeNet();
    CHECK(net2.load(p1) == 0, "load 返回成功");
    Tensor y1 = net2.forward(x, true);

    /* 前向输出 */
    double dOut = 0;
    for (std::size_t i = 0; i < y0.size(); i++) {
        dOut = std::fmax(dOut, std::fabs((double)y0[i] - (double)y1[i]));
    }
    /* 参数本身 (比输出更严格: 输出可能被饱和的激活掩盖差异) */
    std::vector<const Tensor *> a;
    std::vector<const Tensor *> b;
    collectFc(net, a);
    collectFc(net2, b);
    CHECK(a.size() == b.size() && !a.empty(), "收集到同样数量的张量");
    bool allSame = true;
    long long total = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); i++) {
        if (!sameBits(*a[i], *b[i])) {
            allSame = false;
            std::printf("      [差异] 第 %zu 个张量不是逐比特相同\n", i);
        }
        total += (long long)a[i]->size();
    }
    std::printf("    前向输出最大差 = %.3e; 逐比特比较了 %lld 个参数\n", dOut, total);
    CHECK(dOut == 0.0, "载入后前向输出逐位相同");
    CHECK(allSame, "载入后所有参数逐比特相同 (无损)");

    CHECK(net2.save(p2) == 0, "第二次 save 成功");
    std::string f1;
    std::string f2;
    CHECK(readWholeFile(p1, f1) && readWholeFile(p2, f2), "两个文件都能读回来");
    CHECK(f1 == f2, "两次存出来的文件逐字节相同 (说明编码里没有任何有损/随机成分)");

    std::remove(p1.c_str());
    std::remove(p2.c_str());
}

/* ============================================================
 *  [2] 体积与速度 (与老格式对比)
 * ============================================================ */
static void part2()
{
    std::printf("\n[2] 体积与读写速度 (对比老的十进制文本格式)\n");
    const std::string p = "test_weights_size.wgt";

    /* 用一个稍大的网络, 让数字有代表性 */
    Net net(Layer<Tanh>::_(1260, 64, true, true),
            Layer<Tanh>::_(64, 64, true, true),
            Layer<Linear>::_(64, 128, true, true));

    auto t0 = std::chrono::steady_clock::now();
    CHECK(net.save(p) == 0, "save 成功");
    auto t1 = std::chrono::steady_clock::now();
    Net other(Layer<Tanh>::_(1260, 64, true, true),
              Layer<Tanh>::_(64, 64, true, true),
              Layer<Linear>::_(64, 128, true, true));
    CHECK(other.load(p) == 0, "load 成功");
    auto t2 = std::chrono::steady_clock::now();

    const double saveMs =
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    const double loadMs =
        (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

    /* 老格式的体积: 每行 `shape|v1,v2,...` (用 toDebugString 生成同样的文本) */
    long long v2Bytes = fileSize(p);
    long long v1Bytes = 0;
    for (std::size_t i = 0; i < net.size(); i++) {
        const iFcLayer *fc = dynamic_cast<const iFcLayer *>(net[i]);
        if (fc == nullptr) {
            continue;
        }
        v1Bytes += (long long)fc->w.toDebugString().size() + 1;
        v1Bytes += (long long)fc->b.toDebugString().size() + 1;
    }
    std::printf("    v2 (base64+CRC32): %lld 字节, save %.1f ms, load %.1f ms\n",
                v2Bytes, saveMs, loadMs);
    std::printf("    v1 (十进制文本)   : %lld 字节 (按同一组张量算出来)\n", v1Bytes);
    std::printf("    -> 体积 %.2fx, 每个 float %.2f 字节 (base64 的理论下限是 5.33)\n",
                v1Bytes > 0 ? (double)v1Bytes / (double)v2Bytes : 0.0,
                v2Bytes > 0 ? (double)v2Bytes / (double)(1260 * 64 + 64 + 64 * 64 + 64 + 64 * 128 + 128) : 0.0);
    CHECK(v2Bytes > 0, "v2 文件大小非零");
    CHECK(v1Bytes > v2Bytes, "v2 比 v1 小");
    CHECK((double)v1Bytes / (double)v2Bytes > 1.5, "v2 至少比 v1 小 1.5 倍");
    CHECK(saveMs < 200.0 && loadMs < 200.0, "读写都在 200 ms 以内 (93k 参数的小网络)");

    std::remove(p.c_str());
}

/* ============================================================
 *  [3] 老格式 (v1 纯文本) 仍然能读
 * ============================================================ */
static void part3()
{
    std::printf("\n[3] 兼容性: 老格式 (十进制文本, 没有 CHWGT2 头) 仍然能读\n");
    const std::string p = "test_weights_legacy.wgt";

    /* 单层网络: 老格式就是 "w 一行, b 一行" */
    Net net(Layer<Tanh>::_(24, 8, true, true), Layer<Linear>::_(8, 5, true, true));
    Tensor x = makeInput();
    Tensor y0 = net.forward(x, true);

    const iFcLayer *l0 = dynamic_cast<const iFcLayer *>(net[0]);
    const iFcLayer *l1 = dynamic_cast<const iFcLayer *>(net[1]);
    std::string legacy;
    legacy += l0->w.toDebugString() + "\n";
    legacy += l0->b.toDebugString() + "\n";
    legacy += l1->w.toDebugString() + "\n";
    legacy += l1->b.toDebugString() + "\n";
    CHECK(writeWholeFile(p, legacy), "写出一个 v1 格式的文件");

    Net other(Layer<Tanh>::_(24, 8, true, true), Layer<Linear>::_(8, 5, true, true));
    const int rc = other.load(p);
    CHECK(rc == 0, "老格式的文件载入成功 (没有 CHWGT2 头时走兼容分支)");
    Tensor y1 = other.forward(x, true);
    double d = 0;
    for (std::size_t i = 0; i < y0.size(); i++) {
        d = std::fmax(d, std::fabs((double)y0[i] - (double)y1[i]));
    }
    std::printf("    老格式载入后前向输出最大差 = %.3e (v1 是有损的, 所以不是 0)\n", d);
    CHECK(d < 1e-4, "老格式载入后的输出与保存前一致 (容差按 v1 的 6 位有效数字)");
    std::remove(p.c_str());
}

/* ============================================================
 *  [4] 损坏 / 截断 / 结构不匹配都必须**报错**, 不能静默载入
 * ============================================================ */
static void part4()
{
    std::printf("\n[4] 损坏/截断/结构不匹配 -> load 必须失败\n");
    const std::string p = "test_weights_bad.wgt";

    Net net = makeNet();
    CHECK(net.save(p) == 0, "先存一个正常文件");
    std::string good;
    CHECK(readWholeFile(p, good), "读回正常文件");

    /* (a) 截断: 砍掉最后 40% */
    {
        const std::string cut = good.substr(0, good.size() * 6 / 10);
        CHECK(writeWholeFile(p, cut), "写出截断文件");
        Net other = makeNet();
        const int rc = other.load(p);
        std::printf("    截断文件 -> load 返回 %d\n", rc);
        CHECK(rc != 0, "截断的文件载入失败 (不会静默载入半个模型)");
    }

    /* (b) 头部被改坏 (指纹不匹配) */
    {
        std::string bad = good;
        const std::size_t pos = bad.find("CHWGT2");
        CHECK(pos != std::string::npos, "文件里有 CHWGT2 头");
        /* 把指纹改成别的数 */
        const std::size_t sp = bad.find(' ', pos + 7);
        if (sp != std::string::npos && sp + 2 < bad.size()) {
            bad[sp + 2] = (bad[sp + 2] == '9') ? '8' : '9';
        }
        CHECK(writeWholeFile(p, bad), "写出头部被改坏的文件");
        Net other = makeNet();
        const int rc = other.load(p);
        std::printf("    指纹不匹配 -> load 返回 %d\n", rc);
        CHECK(rc != 0, "结构指纹不匹配时载入失败");
    }

    /* (c) 张量数据里翻一个 base64 字符 (CRC32 要抓住它) */
    {
        std::string bad = good;
        /* 找一个 payload 里的字母改成另一个合法 base64 字符 */
        std::size_t at = bad.find("b64:");
        CHECK(at != std::string::npos, "文件里有 b64 段");
        at += 4 + 8 + 1;   /* 跳过 "b64:" + crc + ':' */
        if (at < bad.size()) {
            char &c = bad[at];
            c = (c == 'A') ? 'B' : 'A';
        }
        CHECK(writeWholeFile(p, bad), "写出被翻了一位数据的文件");
        Net other = makeNet();
        const int rc = other.load(p);
        std::printf("    数据被改一位 -> load 返回 %d\n", rc);
        CHECK(rc != 0, "数据损坏 (CRC32 不匹配) 时载入失败");
    }

    /* (d) 结构完全不同: 把 TransformerBlock 网络的权重喂给一个纯 MLP */
    {
        CHECK(writeWholeFile(p, good), "恢复正常文件");
        Net mlp(Layer<Tanh>::_(24, 8, true, true),
                Layer<Tanh>::_(8, 8, true, true),
                Layer<Linear>::_(8, 5, true, true));
        const int rc = mlp.load(p);
        std::printf("    结构不同的网络 -> load 返回 %d\n", rc);
        CHECK(rc != 0, "层数与类型都对不上时载入失败");
    }

    /* (e) 文件不存在 */
    {
        Net other = makeNet();
        CHECK(other.load("no_such_weight_file_xyz.wgt") != 0, "文件不存在时载入失败");
    }

    /*
       (g) **层类型完全相同、只是维度不同** (本轮新增的"维度守卫")
       这是最阴的一种失效: v2 的结构指纹只哈希层的**类型序列**, 1440 维与 1710 维的
       同一套网络指纹**一模一样**, 载入会被放行 —— 而 `Layer::read` 是整块替换
       (`w = Tensor::fromString(...)`), 于是网络的张量被静默换成文件里的形状, 直到
       某次前向才崩, 或者更糟: 不崩但输出全是垃圾。
       触发场景就是本轮做的事: 给状态编码加平面之后, 旧权重文件必须被**明确拒绝**。
    */
    {
        Net dim1440(Layer<Tanh>::_(1440, 64, true, false),
                    Layer<Tanh>::_(64, 64, true, false),
                    Layer<Linear>::_(64, 8, true, false));
        Net dim1710(Layer<Tanh>::_(1710, 64, true, false),
                    Layer<Tanh>::_(64, 64, true, false),
                    Layer<Linear>::_(64, 8, true, false));
        CHECK(dim1440.structureFingerprint() == dim1710.structureFingerprint(),
              "同一套层结构、不同输入维度 -> 结构指纹**相同** (所以光靠指纹拦不住)");
        CHECK(dim1440.paramCount() != dim1710.paramCount(),
              "但参数总量不同 (维度守卫的判据)");
        const std::string pd = "test_weights_dim.wgt";
        CHECK(dim1440.save(pd) == 0, "存一个 1440 维的权重文件");
        const int rc = dim1710.load(pd);
        std::printf("    1440 维文件载入 1710 维网络 -> load 返回 %d\n", rc);
        CHECK(rc != 0, "维度不同必须被拒绝 (不会把张量静默换成文件里的形状)");
        /* 正对照: 同一个文件载入**同维度**的网络必须成功 */
        Net same(Layer<Tanh>::_(1440, 64, true, false),
                 Layer<Tanh>::_(64, 64, true, false),
                 Layer<Linear>::_(64, 8, true, false));
        CHECK(same.load(pd) == 0, "正对照: 同维度的网络仍然能载入这个文件");
        std::remove(pd.c_str());
    }

    /* (f) 在**行边界**上截断 (每一行本身都是完整的, 只是行数不够) */
    {
        std::string lines;
        std::size_t pos = 0;
        int kept = 0;
        while (pos < good.size() && kept < 6) {
            const std::size_t nl = good.find('\n', pos);
            if (nl == std::string::npos) {
                break;
            }
            lines += good.substr(pos, nl - pos + 1);
            pos = nl + 1;
            kept++;
        }
        CHECK(writeWholeFile(p, lines), "写出一个刚好砍在行边界上的文件");
        Net other = makeNet();
        const int rc = other.load(p);
        std::printf("    行边界截断 -> load 返回 %d\n", rc);
        CHECK(rc != 0, "行边界截断 (行数不够) 也会被拒绝");
    }

    /* (g) 载入失败时**网络必须保持原样** (这是"先预校验再载入"的意义) */
    {
        CHECK(writeWholeFile(p, good.substr(0, good.size() * 6 / 10)),
              "再写一个坏文件");
        Net other = makeNet();
        Tensor x = makeInput();
        Tensor before = other.forward(x, true);
        const int rc = other.load(p);
        CHECK(rc != 0, "坏文件载入失败");
        Tensor after = other.forward(x, true);
        double d = 0;
        for (std::size_t i = 0; i < before.size(); i++) {
            d = std::fmax(d, std::fabs((double)before[i] - (double)after[i]));
        }
        std::printf("    载入失败后网络的变化 = %.3e (必须是 0)\n", d);
        CHECK(d == 0.0, "载入失败不会把网络改成半成品 (张量没有变成空张量)");
        /* 顺带确认每个张量还是满的: 空张量在前向里会越界 */
        std::vector<const Tensor *> ts;
        collectFc(other, ts);
        bool nonEmpty = !ts.empty();
        for (std::size_t i = 0; i < ts.size(); i++) {
            if (ts[i]->size() == 0) {
                nonEmpty = false;
            }
        }
        CHECK(nonEmpty, "载入失败后所有张量仍然非空");
    }

    std::remove(p.c_str());
}

/* ============================================================
 *  [5] 原子写入: 不留 .tmp, 失败不破坏原文件
 * ============================================================ */
static void part5()
{
    std::printf("\n[5] 原子写入\n");
    const std::string p = "test_weights_atomic.wgt";
    const std::string tmp = p + ".tmp";

    Net net = makeNet();
    CHECK(net.save(p) == 0, "第一次 save 成功");
    std::string first;
    CHECK(readWholeFile(p, first), "读回第一次的内容");

    /* 覆盖写: 改一下权重再存, 旧内容应当被整体替换 (rename 语义) */
    net.clamp(-0.01f, 0.01f);
    CHECK(net.save(p) == 0, "第二次 save 成功 (目标已存在时也能覆盖)");
    std::string second;
    CHECK(readWholeFile(p, second), "读回第二次的内容");
    CHECK(first != second, "覆盖写之后内容确实变了");
    CHECK(fileSize(tmp) < 0, "没有残留 .tmp 文件");

    /* 写到不可写的路径 -> 返回失败, 且不会留下 .tmp */
    Net n2 = makeNet();
    const int rc = n2.save("no_such_dir_xyz/weights.wgt");
    std::printf("    写不可达路径 -> save 返回 %d\n", rc);
    CHECK(rc != 0, "写不进去时 save 返回失败");
    CHECK(fileSize("no_such_dir_xyz/weights.wgt.tmp") < 0, "失败时没有留下临时文件");

    std::remove(p.c_str());
}

int main()
{
    std::printf("=== 权重文件格式测试 (Net::save / Net::load) ===\n");
    part1();
    part2();
    part3();
    part4();
    part5();
    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
