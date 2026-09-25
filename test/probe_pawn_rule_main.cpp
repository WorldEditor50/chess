/*
 * probe_pawn_rule_main.cpp - 红兵过河后能否横走 (2026-09 用户报障的最小验证)
 * ============================================================================
 * 报障: "红方中央的兵走过河后, 吃掉黑方中间的卒后不能左右行走"。
 * 这个探针只做一件事: 直接问引擎 `Bing::tryMoveTo` —— 把"规则对不对"钉死。
 * 规则对而玩家仍走不动 ⇒ 问题在点击/选中/状态 (见 chessboard.cpp 的 [dbg] 日志),
 * 不在规则。规则不对 ⇒ 才轮到改 stone.h。
 *
 * 用法: probe_pawn_rule
 */
#include <cstdio>
#include <vector>

#include "chess.h"

namespace {
const char *colorName(int c) { return c == Stone::COLOR_RED ? "红" : "黑"; }
}

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    Chess c;
    c.reset();

    int failed = 0;
    auto check = [&failed](bool ok, const char *what) {
        std::printf("  [%s] %s\n", ok ? "OK" : "!!", what);
        if (!ok) { failed++; }
    };

    /* 找红方中兵 (初始 x=6, y=4) */
    Stone *bing = nullptr;
    for (int i = Stone::ID_RED; i < Stone::ID_RED_END; i++) {
        Stone *s = c.m_children[i];
        if (s != nullptr && s->alive && s->type == Stone::TYPE_BING
            && s->pos.x == 6 && s->pos.y == 4) {
            bing = s;
        }
    }
    if (bing == nullptr) {
        std::printf("找不到红方中兵 (x=6,y=4)\n");
        return 1;
    }
    std::printf("红方中兵: 类型=%d 位置=(%d,%d) %s\n",
                (int)bing->type, bing->pos.x, bing->pos.y, colorName(bing->color));

    std::printf("\n[1] 未过河 (x=6):\n");
    check(bing->tryMoveTo(Pos(5, 4)), "前进 (6,4)->(5,4) 合法");
    check(!bing->tryMoveTo(Pos(6, 3)), "横走 (6,4)->(6,3) **不**合法 (未过河)");
    check(!bing->tryMoveTo(Pos(7, 4)), "后退 (6,4)->(7,4) **不**合法");

    /* 摆到过河位置: 报障现场 = 吃掉黑卒后停在 (4,4) */
    c.m_map[bing->pos] = nullptr;
    bing->pos = Pos(4, 4);
    c.m_map[bing->pos] = bing;
    std::printf("\n[2] 已过河 (x=4, 报障现场):\n");
    check(bing->tryMoveTo(Pos(3, 4)), "前进 (4,4)->(3,4) 合法");
    check(bing->tryMoveTo(Pos(4, 3)), "横走 (4,4)->(4,3) **应当**合法  <-- 报障核心");
    check(bing->tryMoveTo(Pos(4, 5)), "横走 (4,4)->(4,5) **应当**合法  <-- 报障核心");
    check(!bing->tryMoveTo(Pos(5, 4)), "后退 (4,4)->(5,4) **不**合法");

    std::printf("  该子 getPossibleSteps 里形状可达的目标:");
    std::vector<Step *> cand;
    bing->getPossibleSteps(cand);
    int reach = 0;
    for (Step *s : cand) {
        if (s != nullptr && bing->tryMoveTo(s->nextPos)) {
            std::printf(" (%d,%d)", s->nextPos.x, s->nextPos.y);
            reach++;
        }
    }
    Steps::instance().put(cand);
    std::printf("  共 %d 个\n", reach);

    /* 底线 (x=0): 只能横走, 这是象棋规则 */
    c.m_map[bing->pos] = nullptr;
    bing->pos = Pos(0, 4);
    c.m_map[bing->pos] = bing;
    std::printf("\n[3] 到底线 (x=0):\n");
    /*
       注意: `tryMoveTo` **不做棋盘边界校验** (那是 sample()/getPossibleSteps 那一层的事),
       所以这里不能拿 Pos(-1,4) 当"再前进"来测 —— 它 x 方向没变反而小, 不构成"后退",
       delta 又恰好是 1, 于是返回 true。第一版就是这么误报的 (假失败)。
       要测"到底线后不能再过河", 应该看**形状上**它还能去哪: 只剩横走。
    */
    check(!bing->tryMoveTo(Pos(1, 4)), "到底线后不能往回走 (x+1) —— 那是后退");
    check(bing->tryMoveTo(Pos(0, 3)) && bing->tryMoveTo(Pos(0, 5)),
          "到底线后**只能**横走 (这是象棋规则, 不是 bug)");

    std::printf("\n结论: %s (失败 %d 项)\n",
                failed == 0 ? "规则正确 —— 走不动的原因不在规则"
                            : "规则有问题, 需要改 stone.h 的 Bing::tryMoveTo",
                failed);

    /* ================================================================
     *  [4] 自由走子 (调试开关): "被将军时我希望能移动所有棋子"
     * ================================================================
     * 用户口径: 希望**任意棋子**都能被移动。形状规则 (兵只能前进/过河横走…) 会挡住它,
     * 所以开关打开后必须能走"形状上不合法"的一步 —— 这一节就是那条判据:
     *   * 关着: 中兵走"马步"(6,4)->(4,3) 必须被拒;
     *   * 开着: 同一步必须被允许, 而且棋盘真的变了 (pos 与吃子结算都走同一条路)。
     */
    std::printf("\n[4] 自由走子开关 (任意棋子任意格):\n");
    {
        Chess g;
        g.reset();
        Stone *b2 = nullptr;
        for (int i = Stone::ID_RED; i < Stone::ID_RED_END; i++) {
            Stone *s = g.m_children[i];
            if (s != nullptr && s->alive && s->type == Stone::TYPE_BING
                && s->pos.x == 6 && s->pos.y == 4) {
                b2 = s;
            }
        }
        const Pos from = b2->pos;
        const Pos weird(4, 3);      /* 马步: 兵在形状上永远走不到 */

        /*
           注意: `Chess::applyMove` 是 private, 所以这里直接用 `Stone::moveTo(pos, freeMove)`
           —— 它就是 applyMove 内部唯一的那次调用 (applyMove 只多做参数/一致性校验),
           所以这条路径与真实落子**同源**。
        */
        const bool movedOff = b2->moveTo(weird, /*freeMove=*/false);
        check(!movedOff, "开关关闭时: 兵走马步被拒 (形状规则生效)");
        check(b2->pos == from, "  被拒之后棋子**没有**移动 (pos 未变)");

        const bool movedOn = b2->moveTo(weird, /*freeMove=*/true);
        check(movedOn, "开关打开时: 兵走马步**被允许** (任意子任意格)");
        check(b2->pos == weird, "  棋子真的走到了 (4,3) —— pos 已更新");
        check(g.m_map[weird] == b2, "  m_map 也更新了 (状态一致, 不是只改了 pos)");
        check(g.m_map[from] == nullptr, "  起点格已清空");
        if (movedOn && b2->pos == weird) {
            std::printf("  -> 结论: 自由走子打开后, 任意棋子可以走到任意一格\n");
        } else {
            std::printf("  -> 结论: 自由走子**没有**放开 (用户要的能力还没实现)\n");
        }
    }

    std::printf("\n总失败项: %d\n", failed);
    return failed == 0 ? 0 : 1;
}
