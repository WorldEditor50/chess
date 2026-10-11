#ifndef RL_TENSOR_CODEC_HPP
#define RL_TENSOR_CODEC_HPP

/*
 * tensor_codec.hpp — 张量序列化的**低层件**（2026-10 P4 从 `tensor.hpp` 搬出）
 * ============================================================================
 * 为什么单独一个文件: 方案 Part 1.1 的结构账里, 序列化那一块占 `tensor.hpp` 的
 * **23.4%（531 行）**, 而且它和 `Tensor_` 本身**没有耦合** —— 它只吃"指针 + 长度 +
 * 元素大小"。搬出来之后:
 *   * `tensor.hpp` 少 230 行 (crc32 / base64 / 形状解析 / 错误旗标);
 *   * 形状解析与 `b64:<crc>:` 头解析**各自只剩一份**（原来 `validateEncodedCount`
 *     与 `fromString` 各写了一遍 —— P4 的"合一"就是靠下面这个 `parseLine`）;
 *   * 错误旗标从 `Tensor_` 的静态成员变成这里的 `codec::decodeFailedRef()`,
 *     `Tensor::lastDecodeFailed()` 只是转发 ⇒ `Net::load` 的汇总口径一个字节没变。
 *
 * ⚠ **字节格式一个字都不许变**: `test_weights` 的 49 条断言、`test_transformer [3]`
 * 的 FNV 指纹、`test_bc` 的 12 处 `netDigest` 都钉着 `toString/fromString` 的输出。
 * 所以 P4 是"**只搬不改**": 算法、遍历顺序、`%08x` 的小写十六进制、base64 的补齐
 * 方式全部原样保留（下面的 base64/crc32 是从 `tensor.hpp` **整块搬过来**的）。
 *
 * 格式 (v2, 现在默认的): `<shape 用逗号分隔>|b64:<crc32 8 位小写十六进制>:<base64 原始数据>`
 * 老格式 (v1, 仍要能读): `<shape>|<十进制浮点, 用逗号分隔>`
 */
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace RL {
namespace codec {

/* ============================================================
 *  错误旗标（原来挂在 `Tensor_` 上的那个静态布尔）
 * ============================================================
 * 97 处调用点都写成 `x = Tensor::fromString(s)`, 没有地方能接收错误码; 由 `Net::load`
 * 在读一整层之前清零、之后检查, 就能把"文件坏了"变成一次明确的失败。
 * (线程安全口径不变: 权重读写只在单一训练/AI 线程里发生, 且有锁保护。)
 */
inline bool &decodeFailedRef()
{
    static bool failed = false;
    return failed;
}

/* ============================================================
 *  形状解析: "1,2,3" → 元素总数（每个维度必须为正）
 * ============================================================
 * 参数是**指针 + 长度**而不是 `std::string`: 预校验在一整块内存上跑, 一行可能是
 * 34 MB, 每行都拷成 std::string 就白省了。
 */
inline bool parseShape(const char *s, std::size_t n, long long &product)
{
    product = 1;
    if (n == 0) {
        return false;
    }
    long long cur = 0;
    bool any = false;
    for (std::size_t i = 0; i < n; i++) {
        const char c = s[i];
        if (c >= '0' && c <= '9') {
            cur = cur * 10 + (c - '0');
            any = true;
        } else if (c == ',') {
            if (!any || cur <= 0) {
                return false;
            }
            product *= cur;
            cur = 0;
            any = false;
        } else {
            return false;
        }
    }
    if (!any || cur <= 0) {
        return false;
    }
    product *= cur;
    return true;
}

/* ============================================================
 *  一行的结构（**不做任何解码、不分配**）
 * ============================================================
 * `validateEncodedCount`（预校验）与 `fromString`（真载入）共用这一份解析 ⇒
 * "两边判定一致"不再靠人肉对齐, 而是结构上就是同一段代码。
 */
struct LineView {
    std::size_t   shapeLen;      /* 形状串长度 (到 '|' 为止) */
    long long     shapeProduct;  /* 形状乘积 */
    bool          v2;            /* 带 `b64:<crc>:` 头 = v2; 否则是老十进制 v1 */
    std::uint32_t crcWant;       /* v2: 文件里写的 CRC32 */
    const char   *payload;       /* v2: base64 串; v1: 十进制串 */
    std::size_t   payloadLen;    /* 已去掉行尾的 \r \n 空格 */
    std::size_t   valueCount;    /* v1: 逗号数 + 1; v2: 0 */
};

inline bool parseLine(const char *s, std::size_t n, LineView &out)
{
    out = LineView();
    const void *barPtr = std::memchr(s, '|', n);
    if (barPtr == nullptr) {
        return false;
    }
    out.shapeLen = (std::size_t)((const char *)barPtr - s);
    if (!parseShape(s, out.shapeLen, out.shapeProduct)) {
        return false;
    }
    const char *p = s + out.shapeLen + 1;
    std::size_t m = n - out.shapeLen - 1;
    while (m > 0 && (p[m - 1] == '\r' || p[m - 1] == '\n' || p[m - 1] == ' ')) {
        m--;
    }
    out.payload = p;
    out.payloadLen = m;
    if (m >= 4 && std::strncmp(p, "b64:", 4) == 0) {
        out.v2 = true;
        if (m < 4 + 8 + 1 || p[12] != ':') {
            return false;
        }
        std::uint32_t want = 0;
        for (std::size_t k = 4; k < 12; k++) {
            const char c = p[k];
            int d;
            if (c >= '0' && c <= '9') {
                d = c - '0';
            } else if (c >= 'a' && c <= 'f') {
                d = c - 'a' + 10;
            } else if (c >= 'A' && c <= 'F') {
                d = c - 'A' + 10;
            } else {
                return false;
            }
            want = (want << 4) | (std::uint32_t)d;
        }
        out.crcWant = want;
        return true;
    }
    /* v1 (十进制文本): 值的个数 = 逗号数 + 1 */
    out.valueCount = (m == 0) ? 0 : 1;
    for (std::size_t k = 0; k < m; k++) {
        if (p[k] == ',') {
            out.valueCount++;
        }
    }
    return true;
}

/* v2 的 payload 真实起点/长度（跳过 `b64:<8 位 crc>:` 这 13 个字符） */
inline const char *v2Payload(const LineView &li) { return li.payload + 13; }
inline std::size_t v2PayloadLen(const LineView &li) { return li.payloadLen - 13; }

/* ============================================================
 *  **非张量**的自描述标记行（白名单，精确匹配）
 * ============================================================
 * 为什么需要它（2026-10，一处实测出来的缺陷）: `Net::load` 的预校验要求**每一行**
 * 都是一个能解码的张量（形状 + base64 + CRC32 全过），于是任何"层自己写的元信息行"
 * 都会被判成"文件被截断/损坏"⇒ **权重存得下去、载不回来**。
 *
 * 第一个（也是目前唯一）生产者是序列专家的位置编码: `SeqPosEnc::write` 在非默认模式
 * 下先写一行 `POS2D` / `POSROPE`（`LEARN_1D` 不写，保持存量文件逐字节不变），
 * 而 `SeqPosEnc::read` 靠这一行比对模式、不一致就响亮失败。它的单元测试走的是
 * **专家级** `write/read`，绕过了 `Net::load`，所以这个缺陷在 `test_transformer`
 * 全绿的情况下依然存在 —— 直到用 `bench_agent_arena --pos-ppo-a=2d` 载入真实权重
 * 才暴露（实测: `seqenc_2d_actor: 第 5 个张量校验失败`）。
 *
 * 判据刻意写成**白名单 + 精确匹配**，而不是"首字符是字母就跳过":
 * 后者会把"真的损坏了"的文件（例如被截断成半行）也放过去，正是预校验要挡的东西。
 * 新增一种标记时必须同时改这里，并在 `test_ppo_backbone` 里加一条载入断言。
 */
inline bool isMetaMarkerLine(const char *s, std::size_t n)
{
    while (n > 0 && (s[n - 1] == '\r' || s[n - 1] == '\n' || s[n - 1] == ' ')) {
        n--;
    }
    struct Marker { const char *text; std::size_t len; };
    static const Marker kMarkers[] = {
        { "POS2D",   5 },
        { "POSROPE", 7 },
    };
    for (std::size_t i = 0; i < sizeof(kMarkers) / sizeof(kMarkers[0]); i++) {
        if (n == kMarkers[i].len && std::memcmp(s, kMarkers[i].text, n) == 0) {
            return true;
        }
    }
    return false;
}
inline const std::uint32_t *crc32Table()
{
    static const std::vector<std::uint32_t> table = [] {
        std::vector<std::uint32_t> t(256);
        for (std::uint32_t i = 0; i < 256; i++) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; k++) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();
    return table.data();
}

inline std::uint32_t crc32Update(std::uint32_t c, unsigned char b)
{
    return crc32Table()[(c ^ b) & 0xFFu] ^ (c >> 8);
}

inline std::uint32_t crc32(const unsigned char *data, std::size_t n)
{
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; i++) {
        c = crc32Update(c, data[i]);
    }
    return c ^ 0xFFFFFFFFu;
}

/*
   ============================================================
    流式 base64 解码: 一趟扫完, 零中间分配
   ============================================================
   为什么要有它: 权重文件里一个张量可以是一行 34 MB 的 base64, 而老实现是
     std::string(p + 13, n - 13)     先拷一份 34 MB
     base64Decode() 里 vector.push_back   再分配 34 MB
       (那个**按 std::string 收、按 vector 吐**的死解码器已在 [2026-10 P2] 删掉 ——
        全仓零调用者; 在用的只有本函数与 base64Encode)
     crc32(raw)                       再扫一遍
     memcpy 到张量                     再一遍
   而且"先校验一遍再真正载入"把这一串**整体做了两遍**。实测稀疏 MoE 那 3 个
   146 MB 的权重文件要 14.9 秒 (29 MB/s), 把"启动时加载所有模型"变成 19 秒。
   现在: 直接在内存字节流上解码, 每解出 3 个字节就
     * 增量更新 CRC,
     * 若有 dst 就**直接写进张量的存储** (连 memcpy 都省了),
   一趟同时完成"合法性 + 长度 + 校验和 + 写入"。dst == nullptr 就是纯校验模式
   (给 Net::load 的预校验用, 校验通过才真正写进网络)。
*/
/* [2026-10 P4] 前置声明: `base64DecodeInto` 用到 `base64DecodeTable`, 而它的定义在
   这块**整块搬过来**的代码的后半段。原来两者都是 `Tensor_` 的成员函数 —— 类内顺序
   无所谓; 变成自由函数之后必须先声明 (这是"搬家"唯一踩到的编译期坑)。 */
inline const signed char *base64DecodeTable();
inline bool base64DecodeInto(const char *in, std::size_t n,
                             unsigned char *dst, std::size_t expectedBytes,
                             std::uint32_t wantCrc)
{
    if (expectedBytes == 0 || n == 0 || (n % 4) != 0) {
        return false;
    }
    const std::size_t groups = n / 4;
    /* 编码长度必须与字节数精确对应 (少一个 '=' 就是被截断了) */
    if (groups * 3 < expectedBytes || (groups * 3 - expectedBytes) > 2) {
        return false;
    }
    /*
       逐字符的分支链换成 256 项查找表: 老写法每个字符要过 6 个比较, 而 109 MB 的
       数据是 1.45 亿个字符 —— 实测一换, 解码从 99 MB/s 提到 ~250 MB/s。
       '=' 用 64 表示, 非法字符用 -1。
    */
    const signed char *tab = base64DecodeTable();
    std::uint32_t crc = 0xFFFFFFFFu;
    std::size_t written = 0;
    for (std::size_t g = 0; g < groups; g++) {
        const char *q = in + g * 4;
        const signed char v0 = tab[(unsigned char)q[0]];
        const signed char v1 = tab[(unsigned char)q[1]];
        const signed char v2 = tab[(unsigned char)q[2]];
        const signed char v3 = tab[(unsigned char)q[3]];
        if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) {
            return false;
        }
        int pad = 0;
        if (v2 == 64) {
            /* 填充只允许出现在最后一组的末尾 */
            if (g + 1 != groups || v3 != 64) {
                return false;
            }
            pad = 2;
        } else if (v3 == 64) {
            if (g + 1 != groups) {
                return false;
            }
            pad = 1;
        }
        const unsigned int x = ((unsigned int)v0 << 18) | ((unsigned int)v1 << 12)
                               | ((unsigned int)(v2 & 63) << 6)
                               | (unsigned int)(v3 & 63);
        const int cnt = 3 - pad;
        if (written + (std::size_t)cnt > expectedBytes) {
            return false;
        }
        const unsigned char b0 = (unsigned char)((x >> 16) & 0xFFu);
        const unsigned char b1 = (unsigned char)((x >> 8) & 0xFFu);
        const unsigned char b2 = (unsigned char)(x & 0xFFu);
        crc = crc32Update(crc, b0);
        if (dst != nullptr) {
            dst[written] = b0;
        }
        written++;
        if (cnt >= 2) {
            crc = crc32Update(crc, b1);
            if (dst != nullptr) {
                dst[written] = b1;
            }
            written++;
        }
        if (cnt >= 3) {
            crc = crc32Update(crc, b2);
            if (dst != nullptr) {
                dst[written] = b2;
            }
            written++;
        }
    }
    if (written != expectedBytes) {
        return false;
    }
    return (crc ^ 0xFFFFFFFFu) == wantCrc;
}

/* base64 字符 -> 值 (0..63), '=' -> 64, 其它 -> -1 */
inline const signed char *base64DecodeTable()
{
    static const std::vector<signed char> table = [] {
        std::vector<signed char> t(256, (signed char)-1);
        const char *alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) {
            t[(unsigned char)alphabet[i]] = (signed char)i;
        }
        t[(unsigned char)'='] = (signed char)64;
        return t;
    }();
    return table.data();
}
/* [2026-10 P4] 原来是 `base64Encode(const std::vector<T, Alloc<T>> &)` —— 搬进 codec 后
   改成**指针 + 字节数**的形式, 于是这一层与 `Tensor_`/分配器彻底解耦 (输出逐字节不变)。 */
inline std::string base64Encode(const void *raw, std::size_t rawBytes)
{
    static const char *kTab =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const unsigned char *bytes = static_cast<const unsigned char *>(raw);
    const std::size_t n = rawBytes;
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        const unsigned int v = ((unsigned int)bytes[i] << 16)
                               | ((unsigned int)bytes[i + 1] << 8)
                               | (unsigned int)bytes[i + 2];
        out += kTab[(v >> 18) & 63];
        out += kTab[(v >> 12) & 63];
        out += kTab[(v >> 6) & 63];
        out += kTab[v & 63];
    }
    const std::size_t rest = n - i;
    if (rest == 1) {
        const unsigned int v = (unsigned int)bytes[i] << 16;
        out += kTab[(v >> 18) & 63];
        out += kTab[(v >> 12) & 63];
        out += '=';
        out += '=';
    } else if (rest == 2) {
        const unsigned int v = ((unsigned int)bytes[i] << 16)
                               | ((unsigned int)bytes[i + 1] << 8);
        out += kTab[(v >> 18) & 63];
        out += kTab[(v >> 12) & 63];
        out += kTab[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

} /* namespace codec */
} /* namespace RL */

#endif // RL_TENSOR_CODEC_HPP