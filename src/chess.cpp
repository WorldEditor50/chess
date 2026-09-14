#include "chess.h"

/* ============================================================
 * 棋子-位置价值表 (Piece-Square Tables)
 * 红方视角 (黑方使用时应镜像翻转)
 * 数值越大表示该位置对该棋子越有利
 * 中心控制、进攻性位置给予更高权重
 * ============================================================ */

/* 车: 控制开放线, 占据中心 */
const double Chess::chePST[10][9] = {
    {0.00, 0.01, 0.02, 0.03, 0.05, 0.03, 0.02, 0.01, 0.00},
    {0.01, 0.05, 0.06, 0.07, 0.08, 0.07, 0.06, 0.05, 0.01},
    {0.02, 0.06, 0.08, 0.10, 0.13, 0.10, 0.08, 0.06, 0.02},
    {0.03, 0.07, 0.10, 0.12, 0.15, 0.12, 0.10, 0.07, 0.03},
    {0.03, 0.07, 0.10, 0.12, 0.18, 0.12, 0.10, 0.07, 0.03},
    {0.03, 0.07, 0.10, 0.12, 0.15, 0.12, 0.10, 0.07, 0.03},
    {0.02, 0.06, 0.08, 0.10, 0.13, 0.10, 0.08, 0.06, 0.02},
    {0.01, 0.05, 0.06, 0.07, 0.08, 0.07, 0.06, 0.05, 0.01},
    {0.01, 0.03, 0.04, 0.05, 0.06, 0.05, 0.04, 0.03, 0.01},
    {0.00, 0.01, 0.02, 0.03, 0.05, 0.03, 0.02, 0.01, 0.00}
};

/* 马: 中心区域价值高, 边角价值低 */
const double Chess::maPST[10][9] = {
    {0.00, 0.00, 0.01, 0.02, 0.02, 0.02, 0.01, 0.00, 0.00},
    {0.00, 0.02, 0.04, 0.05, 0.05, 0.05, 0.04, 0.02, 0.00},
    {0.01, 0.04, 0.08, 0.10, 0.12, 0.10, 0.08, 0.04, 0.01},
    {0.02, 0.05, 0.10, 0.14, 0.16, 0.14, 0.10, 0.05, 0.02},
    {0.02, 0.05, 0.10, 0.14, 0.18, 0.14, 0.10, 0.05, 0.02},
    {0.02, 0.05, 0.10, 0.14, 0.16, 0.14, 0.10, 0.05, 0.02},
    {0.01, 0.04, 0.08, 0.10, 0.12, 0.10, 0.08, 0.04, 0.01},
    {0.00, 0.02, 0.04, 0.05, 0.05, 0.05, 0.04, 0.02, 0.00},
    {0.00, 0.00, 0.01, 0.02, 0.02, 0.02, 0.01, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}
};

/* 炮: 需要灵活位置, 中心及兵线价值高 */
const double Chess::paoPST[10][9] = {
    {0.00, 0.00, 0.02, 0.03, 0.03, 0.03, 0.02, 0.00, 0.00},
    {0.01, 0.02, 0.04, 0.05, 0.06, 0.05, 0.04, 0.02, 0.01},
    {0.02, 0.04, 0.06, 0.08, 0.10, 0.08, 0.06, 0.04, 0.02},
    {0.03, 0.05, 0.08, 0.10, 0.12, 0.10, 0.08, 0.05, 0.03},
    {0.03, 0.05, 0.08, 0.10, 0.14, 0.10, 0.08, 0.05, 0.03},
    {0.03, 0.05, 0.08, 0.10, 0.12, 0.10, 0.08, 0.05, 0.03},
    {0.02, 0.04, 0.06, 0.08, 0.10, 0.08, 0.06, 0.04, 0.02},
    {0.01, 0.02, 0.04, 0.05, 0.06, 0.05, 0.04, 0.02, 0.01},
    {0.00, 0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.00},
    {0.00, 0.00, 0.02, 0.03, 0.03, 0.03, 0.02, 0.00, 0.00}
};

/* 兵/卒: 过河后价值激增, 中路兵价值更高 */
const double Chess::bingPST[10][9] = {
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.00},
    {0.01, 0.03, 0.06, 0.08, 0.10, 0.08, 0.06, 0.03, 0.01},
    {0.02, 0.05, 0.08, 0.12, 0.15, 0.12, 0.08, 0.05, 0.02},
    {0.03, 0.07, 0.10, 0.14, 0.20, 0.14, 0.10, 0.07, 0.03},
    {0.04, 0.08, 0.12, 0.16, 0.22, 0.16, 0.12, 0.08, 0.04},
    {0.03, 0.06, 0.09, 0.12, 0.15, 0.12, 0.09, 0.06, 0.03},
    {0.02, 0.04, 0.06, 0.08, 0.10, 0.08, 0.06, 0.04, 0.02},
    {0.01, 0.02, 0.03, 0.04, 0.06, 0.04, 0.03, 0.02, 0.01},
    {0.00, 0.00, 0.01, 0.02, 0.03, 0.02, 0.01, 0.00, 0.00}
};

/* 帅/将: 安全优先, 中路较安全 */
const double Chess::jiangPST[10][9] = {
    {0.00, 0.00, 0.00, 0.01, 0.02, 0.01, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.02, 0.03, 0.02, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.03, 0.05, 0.03, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.03, 0.05, 0.03, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.02, 0.03, 0.02, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.01, 0.02, 0.01, 0.00, 0.00, 0.00}
};

/* 仕/士: 保护将/帅, 靠内线价值高 */
const double Chess::shiPST[10][9] = {
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.05, 0.08, 0.05, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.05, 0.08, 0.05, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}
};

/* 相/象: 防守为主, 靠近将/帅区域价值高 */
const double Chess::xiangPST[10][9] = {
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}
};

/* 局面价值项开关 (Phase 5): 定义在这里, 默认开 */
bool Chess::g_positionalEvalEnabled = true;

/* 位置价值函数 */
static double getPositionValue(int type, int color, int x, int y, const double pst[10][9])
{
    if (color == Stone::COLOR_RED) {
        /* 红方: 使用原表 */
        return pst[x][y];
    } else {
        /* 黑方: 上下镜像 (x -> 9-x) */
        return pst[9 - x][y];
    }
}

/* 增强评估函数: 材质 + 位置 */
double Chess::evaluate()
{
    double score = 0.0;
    for (int i = 0; i < 32; i++) {
        Stone *s = m_children[i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        double matVal = s->value;
        double posVal = 0.0;
        switch (s->type) {
        case Stone::TYPE_CHE:   posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, chePST);   break;
        case Stone::TYPE_MA:    posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, maPST);    break;
        case Stone::TYPE_PAO:   posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, paoPST);   break;
        case Stone::TYPE_BING:  posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, bingPST);  break;
        case Stone::TYPE_JIANG: posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, jiangPST); break;
        case Stone::TYPE_SHI:   posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, shiPST);   break;
        case Stone::TYPE_XIANG: posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, xiangPST); break;
        }
        /* 返回值从AI(黑方)视角: 正=AI有利 */
        if (s->color == Stone::COLOR_BLACK) {
            score += matVal + posVal;
        } else {
            score -= matVal + posVal;
        }
    }
    /*
       ---- 局面价值项刻意**不**并进这里 ----
       evaluate() 是 ABAgent 的**叶子评估**, 一步 depth-4 搜索要调约 170 万次。实测
       叠加局面项 (将安全/空间/士象) 之后一次评估从 0.10 us 涨到 0.65 us (**6.5 倍**),
       depth-4 单步从约 0.5 s 涨到 1.27 s —— 在等时间对局里等于把搜索深度吃回去,
       是"评估更准"换"搜得更浅"。而局面价值真正该去的地方是 **RL 的势能 Φ**
       (每手只算两次, 开销可以忽略): 见 evaluatePositional()。
    */
    return score;
}

/* ------------------------------------------------------------------
 *  evaluatePositional: 完整局面价值 = 材质 + PST + 局面项 (供 RL 的势能 Φ 用)
 *
 *  与 evaluate() 的分工:
 *    * evaluate()            给 ABAgent / EVAB 的**叶子评估**: 极便宜 (0.10 us), 不能动;
 *    * evaluatePositional()  给 PPO 的**势能塑形** Φ: 每手两次, 可以算得细一点。
 *  把"将安全 / 空间 / 机动性 / 士象完整度"放在后者, 既拿到了局面信息, 又不拖慢 AB。
 *
 *  开关 g_positionalEvalEnabled 只影响这一路 —— 用来做势能塑形的 A/B。
 * ------------------------------------------------------------------ */
double Chess::evaluatePositional()
{
    return evaluate() + (g_positionalEvalEnabled ? positionalScore() : 0.0);
}

/* ------------------------------------------------------------------
 *  positionalScore: 局面价值项 (黑方视角, 与 evaluate() 同一口径)
 *
 *  为什么原来只有材质 + PST 不够: 材质差要等到吃子才变化, 而象棋里大量局面的差别
 *  在"将有多危险、攻势有多强、士象是否完整"上。诊断 (test_reward_diag) 已经量出
 *  critic 的价值目标在安静局面几乎全 0, 也就是没有任何位置信息可用 —— 这些项正是
 *  势能 Φ 的原料 (Φ = tanh(走子方视角的 evaluate()/SCALE), 见 stone.h)。
 *
 *  四项, 都刻意取得比材质小一个量级 (材质差最大 ±3.5, 这些合计 ≤1.2):
 *    * 被将军        : ±0.50  —— 被将军是失先手的硬信号 (将的机动性被限)
 *    * 九宫受攻格数  : 每格 0.02 (最多 9 格 -> 0.18)
 *    * 攻击对方将格  : 每个子 0.03 —— 多重攻击比单次攻击危险得多
 *    * 缺士/象       : 每个 0.04 —— 象棋里"士象全"是防守完整度的标准项
 *
 *  成本: 18 次 isAttacked (双方九宫) + 一次 32 子扫描。isAttacked 是按棋子类型做
 *  几何判定 (车同线/炮炮架/马腿/兵方向/飞将), 不去重建走法形状, 本来就是为搜索
 *  热路径写的 —— 实测 depth-4 的一步大约多 3~5% (见 docs 的 Phase 5 实测)。
 * ------------------------------------------------------------------ */
double Chess::positionalScore()
{
    const Pos blackJiangPos = blackJiang.pos;
    const Pos redJiangPos   = redJiang.pos;

    double s = 0.0;

    /* ---- 1) 被将军 (将所在格被对方攻击) ---- */
    const bool blackInCheck = (blackJiang.alive && isAttacked(blackJiangPos, Stone::COLOR_RED));
    const bool redInCheck   = (redJiang.alive && isAttacked(redJiangPos, Stone::COLOR_BLACK));
    if (blackInCheck) s -= 0.50;
    if (redInCheck)   s += 0.50;

    /*
       ---- 2) 将周围受攻: 已去掉 ----
       它要给双方各 4 个将邻格做 isAttacked, 而 isAttacked 每次都要扫一遍敌方棋子
       列表 (车/炮还要数中间有几个子)。实测: 这一项 + 被将军 (10 次 isAttacked) 让
       一次 evaluate() 从 0.10 us 涨到 0.80 us (**8 倍**), 而 AB depth-4 一步要调约
       170 万个叶子 —— 单步从约 0.5 s 涨到 1.34 s, 在等时间对局里等于把搜索深度
       吃回去 (用"评估更准"换"搜得更浅")。所以只保留**被将军**这一项: 它是这份信息
       里最关键、也最便宜 (2 次调用) 的, 而且将"周围"受攻与"被将军"高度重复。
       剩下的预算给了 activityScore (纯数组访问, 便宜) —— 也就是"空间"那一项。
    */

    /*
       ---- 3) 攻击对方将格的子数: 已去掉 ----
       这一项与 (1) 被将军、(2) 将周围受攻 高度重复 (都是"将有多危险"), 而它要给
       **双方全部 32 个子**各做一次攻击判定 —— 车/炮的判定还要算"中间有几个子"
       (走线扫描)。实测它是这一整块里最贵的部分: 去掉它 depth-4 单步从 1621 ms 降到
       约 950 ms, 而信息几乎没少。宁可少一项也不能把 AB 的等时间深度吃回去。
    */

    /* ---- 4) 士象完整度 ---- */
    int blackGuards = 0, redGuards = 0;
    for (int i = 0; i < 32; i++) {
        Stone *st = m_children[i];
        if (st == nullptr || st->alive == false) continue;
        if (st->type != Stone::TYPE_SHI && st->type != Stone::TYPE_XIANG) continue;
        if (st->color == Stone::COLOR_BLACK) blackGuards++; else redGuards++;
    }
    s -= 0.04 * (4 - blackGuards);
    s += 0.04 * (4 - redGuards);

    /*
       ---- 5) 空间 / 机动性 ----
       这一项的意义是"不吃子的着法也能改变局面价值": 材质要等吃子才动, 而调子/占位/
       争空间的价值在**未来的选择权**里 —— 未来价值藏在空间位置中。势能 Φ 会把这份
       空间价值提前搬进当前的学习目标 (见 stone.h 的 PBRS 推导)。
    */
    s += activityScore();

    return s;
}

/* ------------------------------------------------------------------
 *  activityScore: 空间 / 机动性 (Phase 5, 黑方视角)
 *
 *  为什么要它: **不吃子的着法也必须能改变局面价值**。材质差要等吃子才动, 而象棋里
 *  绝大多数着法是"调子、占位、争空间"——它们的价值不在当下, 而在**未来的选择权**
 *  (能到达的格子越多, 未来的威胁/防守机会越多)。这正是"未来价值藏在空间位置里"。
 *
 *  做法: 逐子几何地数"可达格数"与"伸进对方半场的格数", 不用 isAttacked —— 后者要
 *  扫棋子列表, 而这里是 AB 的叶子热路径 (depth-4 一步约 1600 个叶子)。车/炮走射线、
 *  马查马腿、兵只前进, 都是 O(1)~O(9) 的便宜操作。
 *
 *  两项 (都取得比材质小一个量级 —— 材质差最大 ±3.5):
 *    * 机动性: 每个可达空格 0.004  —— 一车在开阔线路上约 10 格 -> 0.04
 *    * 空间  : 可达格位于**对方半场**时再 +0.006 —— 伸进对方半场才是真正的空间优势
 *
 *  仕/士 与 相/象 不参与: 它们永远出不了己方半场/九宫, 机动性恒为常数, 对"空间"
 *  没有贡献 (它们的作用已经由 positionalScore 的士象完整度项表达)。
 * ------------------------------------------------------------------ */
double Chess::activityScore()
{
    static const double W_MOBILITY = 0.004;   /* 每个可达格 */
    static const double W_SPACE    = 0.006;   /* 可达格在对方半场时的额外权重 */

    double s = 0.0;

    for (int i = 0; i < 32; i++) {
        Stone *st = m_children[i];
        if (st == nullptr || st->alive == false) continue;

        const int me = st->color;
        const bool isRed = (me == Stone::COLOR_RED);
        /* 对方半场: 红方的对方半场是 x<=4, 黑方的对方半场是 x>=5 */
        auto inEnemyHalf = [isRed](const Pos &p) {
            return isRed ? (p.x <= 4) : (p.x >= 5);
        };

        double pieceScore = 0.0;
        /* 累计"这一步可到达的格子"(含被对方子占据的格 —— 那是可吃/可攻击的格) */
        auto addSquare = [&](const Pos &p) {
            pieceScore += W_MOBILITY;
            if (inEnemyHalf(p)) {
                pieceScore += W_SPACE;
            }
        };

        switch (st->type) {
        case Stone::TYPE_CHE:
        case Stone::TYPE_PAO: {
            /*
               只算**车与炮**的空间: 它们是长程子, 可达格数最多、空间优势最明显;
               马/兵/将/仕/相的活动范围小, 其位置价值已经由 PST 表达 (bingPST 里
               过河兵的溢价就是这一项的替代)。少算 4 类子 = 少掉一大半 map 查找 ——
               而这是 AB 的叶子热路径 (每步要走 ~1600 个叶子)。
            */
            const bool isPao = (st->type == Stone::TYPE_PAO);
            static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
            for (int d = 0; d < 4; d++) {
                int x = st->pos.x, y = st->pos.y;
                bool jumped = false;
                while (true) {
                    x += dirs[d][0];
                    y += dirs[d][1];
                    if (x < 0 || x > 9 || y < 0 || y > 8) break;
                    Stone *occ = m_map[Pos(x, y)];
                    if (!isPao) {
                        /* 车: 沿线路直到被挡住; 空位与可吃的敌子都算可达 */
                        if (occ == nullptr) {
                            addSquare(Pos(x, y));
                            continue;
                        }
                        if (occ->color != me) addSquare(Pos(x, y));
                        break;
                    }
                    /* 炮: 炮架之前可平移, 炮架之后第一个子是可击目标 */
                    if (!jumped) {
                        if (occ == nullptr) {
                            addSquare(Pos(x, y));
                            continue;
                        }
                        jumped = true;
                        continue;
                    }
                    if (occ != nullptr) {
                        if (occ->color != me) addSquare(Pos(x, y));
                        break;
                    }
                }
            }
            break;
        }
        case Stone::TYPE_MA: {
            /* 马的活动范围小, 只给它"有没有被蹩腿"这一个粗信号: 蹩腿 = 机动性损失 */
            static const int off[8][2] = {{2,1},{2,-1},{-2,1},{-2,-1},
                                          {1,2},{1,-2},{-1,2},{-1,-2}};
            for (int k = 0; k < 8; k++) {
                const int x = st->pos.x + off[k][0];
                const int y = st->pos.y + off[k][1];
                if (x < 0 || x > 9 || y < 0 || y > 8) continue;
                Pos leg = st->pos;
                if (std::abs(off[k][0]) == 2) leg.x += (off[k][0] > 0) ? 1 : -1;
                else                          leg.y += (off[k][1] > 0) ? 1 : -1;
                if (m_map[leg] != nullptr) continue;   /* 蹩腿 */
                Stone *occ = m_map[Pos(x, y)];
                if (occ == nullptr || occ->color != me) {
                    pieceScore += 0.002;   /* 只在"能走"这一层给分, 不再分辨远近 */
                }
            }
            break;
        }
        default:
            /* 兵/卒 与 将/帅/仕/士/相/象: 活动范围小或受九宫/半场锁死, 由 PST 表达 */
            break;
        }

        if (isRed) {
            s -= pieceScore;   /* 黑方视角: 红方得空间 -> 对黑不利 */
        } else {
            s += pieceScore;
        }
    }

    return s;
}

/* 单个棋子是否攻击 target (供 positionalScore 的第 3 项用) */bool Chess::isAttackedOne(const Pos &target, const Stone *s)
{
    if (s == nullptr || s->alive == false) return false;
    switch (s->type) {
    case Stone::TYPE_CHE:
        return (s->pos.x == target.x || s->pos.y == target.y)
               && m_map.countStoneOnLine(s->pos, target) == 0;
    case Stone::TYPE_PAO:
        return (s->pos.x == target.x || s->pos.y == target.y)
               && m_map.countStoneOnLine(s->pos, target) == 1;
    case Stone::TYPE_MA: {
        const int dx = std::abs(target.x - s->pos.x);
        const int dy = std::abs(target.y - s->pos.y);
        if (!((dx == 2 && dy == 1) || (dx == 1 && dy == 2))) return false;
        Pos leg = s->pos;
        if (dx == 2) leg.x += (target.x > s->pos.x) ? 1 : -1;
        else         leg.y += (target.y > s->pos.y) ? 1 : -1;
        return m_map[leg] == nullptr;
    }
    case Stone::TYPE_BING:
        if (s->color == Stone::COLOR_RED) {
            if (target.x == s->pos.x - 1 && target.y == s->pos.y) return true;
            return (s->pos.x <= 4 && target.x == s->pos.x
                    && std::abs(target.y - s->pos.y) == 1);
        }
        if (target.x == s->pos.x + 1 && target.y == s->pos.y) return true;
        return (s->pos.x >= 5 && target.x == s->pos.x
                && std::abs(target.y - s->pos.y) == 1);
    case Stone::TYPE_JIANG:
        /* 飞将: 同列且中间无子 */
        return (s->pos.y == target.y && m_map.countStoneOnLine(s->pos, target) == 0);
    default:
        /* 仕/士 与 相/象 永远到不了对方九宫 */
        return false;
    }
}

Chess::Chess():
    stones(m_children),
    redChe1(Stone::ID_RED_CHE1, Stone::COLOR_RED, Pos(9, 0), &m_map, &m_children),
    redMa1(Stone::ID_RED_MA1, Stone::COLOR_RED, Pos(9, 1), &m_map, &m_children),
    redXiang1(Stone::ID_RED_XIANG1, Stone::COLOR_RED, Pos(9, 2), &m_map, &m_children),
    redShi1(Stone::ID_RED_SHI1, Stone::COLOR_RED, Pos(9, 3), &m_map, &m_children),
    redJiang(Stone::ID_RED_JIANG, Stone::COLOR_RED, Pos(9, 4), &m_map, &m_children),
    redShi2(Stone::ID_RED_SHI2, Stone::COLOR_RED, Pos(9, 5), &m_map, &m_children),
    redXiang2(Stone::ID_RED_XIANG2, Stone::COLOR_RED, Pos(9, 6), &m_map, &m_children),
    redMa2(Stone::ID_RED_MA2, Stone::COLOR_RED, Pos(9, 7), &m_map, &m_children),
    redChe2(Stone::ID_RED_CHE2, Stone::COLOR_RED, Pos(9, 8), &m_map, &m_children),
    redPao1(Stone::ID_RED_PAO1, Stone::COLOR_RED, Pos(7, 1), &m_map, &m_children),
    redPao2(Stone::ID_RED_PAO2, Stone::COLOR_RED, Pos(7, 7), &m_map, &m_children),
    redBing1(Stone::ID_RED_BING1, Stone::COLOR_RED, Pos(6, 0), &m_map, &m_children),
    redBing2(Stone::ID_RED_BING2, Stone::COLOR_RED, Pos(6, 2), &m_map, &m_children),
    redBing3(Stone::ID_RED_BING3, Stone::COLOR_RED, Pos(6, 4), &m_map, &m_children),
    redBing4(Stone::ID_RED_BING4, Stone::COLOR_RED, Pos(6, 6), &m_map, &m_children),
    redBing5(Stone::ID_RED_BING5, Stone::COLOR_RED, Pos(6, 8), &m_map, &m_children),
    blackChe1(Stone::ID_BLACK_CHE1, Stone::COLOR_BLACK, Pos(0, 0), &m_map, &m_children),
    blackMa1(Stone::ID_BLACK_MA1, Stone::COLOR_BLACK, Pos(0, 1), &m_map, &m_children),
    blackXiang1(Stone::ID_BLACK_XIANG1, Stone::COLOR_BLACK, Pos(0, 2), &m_map, &m_children),
    blackShi1(Stone::ID_BLACK_SHI1, Stone::COLOR_BLACK, Pos(0, 3), &m_map, &m_children),
    blackJiang(Stone::ID_BLACK_JIANG, Stone::COLOR_BLACK, Pos(0, 4), &m_map, &m_children),
    blackShi2(Stone::ID_BLACK_SHI2, Stone::COLOR_BLACK, Pos(0, 5), &m_map, &m_children),
    blackXiang2(Stone::ID_BLACK_XIANG2, Stone::COLOR_BLACK, Pos(0, 6), &m_map, &m_children),
    blackMa2(Stone::ID_BLACK_MA2, Stone::COLOR_BLACK, Pos(0, 7), &m_map, &m_children),
    blackChe2(Stone::ID_BLACK_CHE2, Stone::COLOR_BLACK, Pos(0, 8), &m_map, &m_children),
    blackPao1(Stone::ID_BLACK_PAO1, Stone::COLOR_BLACK, Pos(2, 1), &m_map, &m_children),
    blackPao2(Stone::ID_BLACK_PAO2, Stone::COLOR_BLACK, Pos(2, 7), &m_map, &m_children),
    blackBing1(Stone::ID_BLACK_BING1, Stone::COLOR_BLACK, Pos(3, 0), &m_map, &m_children),
    blackBing2(Stone::ID_BLACK_BING2, Stone::COLOR_BLACK, Pos(3, 2), &m_map, &m_children),
    blackBing3(Stone::ID_BLACK_BING3, Stone::COLOR_BLACK, Pos(3, 4), &m_map, &m_children),
    blackBing4(Stone::ID_BLACK_BING4, Stone::COLOR_BLACK, Pos(3, 6), &m_map, &m_children),
    blackBing5(Stone::ID_BLACK_BING5, Stone::COLOR_BLACK, Pos(3, 8), &m_map, &m_children)
{
    /*
       必须显式 reset(): StoneMap 的默认构造现在会把整张表清空, 而 32 个棋子
       构造函数只写了各自所在的 32 个格子 —— 不 reset 的话剩余 58 格虽是 nullptr
       (安全), 但棋盘状态没有保证。reset() 同时初始化 sideToMove / halfMoveClock /
       history。
    */
    reset();
}

/* 拷贝构造函数: 深度复制所有棋子的状态 */
Chess::Chess(const Chess &other):
    m_children(),  /* start empty, stones will be filled by stone constructors below */
    stones(m_children),
    redChe1(other.redChe1.id, other.redChe1.color, other.redChe1.pos, &m_map, &m_children),
    redMa1(other.redMa1.id, other.redMa1.color, other.redMa1.pos, &m_map, &m_children),
    redXiang1(other.redXiang1.id, other.redXiang1.color, other.redXiang1.pos, &m_map, &m_children),
    redShi1(other.redShi1.id, other.redShi1.color, other.redShi1.pos, &m_map, &m_children),
    redJiang(other.redJiang.id, other.redJiang.color, other.redJiang.pos, &m_map, &m_children),
    redShi2(other.redShi2.id, other.redShi2.color, other.redShi2.pos, &m_map, &m_children),
    redXiang2(other.redXiang2.id, other.redXiang2.color, other.redXiang2.pos, &m_map, &m_children),
    redMa2(other.redMa2.id, other.redMa2.color, other.redMa2.pos, &m_map, &m_children),
    redChe2(other.redChe2.id, other.redChe2.color, other.redChe2.pos, &m_map, &m_children),
    redPao1(other.redPao1.id, other.redPao1.color, other.redPao1.pos, &m_map, &m_children),
    redPao2(other.redPao2.id, other.redPao2.color, other.redPao2.pos, &m_map, &m_children),
    redBing1(other.redBing1.id, other.redBing1.color, other.redBing1.pos, &m_map, &m_children),
    redBing2(other.redBing2.id, other.redBing2.color, other.redBing2.pos, &m_map, &m_children),
    redBing3(other.redBing3.id, other.redBing3.color, other.redBing3.pos, &m_map, &m_children),
    redBing4(other.redBing4.id, other.redBing4.color, other.redBing4.pos, &m_map, &m_children),
    redBing5(other.redBing5.id, other.redBing5.color, other.redBing5.pos, &m_map, &m_children),
    blackChe1(other.blackChe1.id, other.blackChe1.color, other.blackChe1.pos, &m_map, &m_children),
    blackMa1(other.blackMa1.id, other.blackMa1.color, other.blackMa1.pos, &m_map, &m_children),
    blackXiang1(other.blackXiang1.id, other.blackXiang1.color, other.blackXiang1.pos, &m_map, &m_children),
    blackShi1(other.blackShi1.id, other.blackShi1.color, other.blackShi1.pos, &m_map, &m_children),
    blackJiang(other.blackJiang.id, other.blackJiang.color, other.blackJiang.pos, &m_map, &m_children),
    blackShi2(other.blackShi2.id, other.blackShi2.color, other.blackShi2.pos, &m_map, &m_children),
    blackXiang2(other.blackXiang2.id, other.blackXiang2.color, other.blackXiang2.pos, &m_map, &m_children),
    blackMa2(other.blackMa2.id, other.blackMa2.color, other.blackMa2.pos, &m_map, &m_children),
    blackChe2(other.blackChe2.id, other.blackChe2.color, other.blackChe2.pos, &m_map, &m_children),
    blackPao1(other.blackPao1.id, other.blackPao1.color, other.blackPao1.pos, &m_map, &m_children),
    blackPao2(other.blackPao2.id, other.blackPao2.color, other.blackPao2.pos, &m_map, &m_children),
    blackBing1(other.blackBing1.id, other.blackBing1.color, other.blackBing1.pos, &m_map, &m_children),
    blackBing2(other.blackBing2.id, other.blackBing2.color, other.blackBing2.pos, &m_map, &m_children),
    blackBing3(other.blackBing3.id, other.blackBing3.color, other.blackBing3.pos, &m_map, &m_children),
    blackBing4(other.blackBing4.id, other.blackBing4.color, other.blackBing4.pos, &m_map, &m_children),
    blackBing5(other.blackBing5.id, other.blackBing5.color, other.blackBing5.pos, &m_map, &m_children),
    history(other.history)
{
    sideToMove = other.sideToMove;
    halfMoveClock = other.halfMoveClock;
    /* Sync alive flags from source */
    for (int i = 0; i < 32; i++) {
        Stone *src = other.m_children[i];
        Stone *dst = m_children[i];
        if (src && dst) {
            dst->alive = src->alive;
        }
    }
    /* Rebuild m_map from only alive stones — dead stones may have left
     * stale entries during construction that overlap with alive stones. */
    m_map.clear();
    for (int i = 0; i < 32; i++) {
        Stone *s = m_children[i];
        if (s && s->alive) {
            m_map[s->pos] = s;
        }
    }
}

Chess& Chess::operator=(const Chess &other)
{
    if (this == &other) return *this;
    /* Copy state: positions, alive flags */
    for (int i = 0; i < 32; i++) {
        Stone *src = other.m_children[i];
        Stone *dst = m_children[i];
        if (src && dst) {
            dst->pos = src->pos;
            dst->alive = src->alive;
        }
    }
    /* Rebuild m_map */
    m_map.clear();
    for (int i = 0; i < 32; i++) {
        Stone *s = m_children[i];
        if (s && s->alive) {
            m_map[s->pos] = s;
        }
    }
    history = other.history;
    sideToMove = other.sideToMove;
    halfMoveClock = other.halfMoveClock;
    return *this;
}

Chess::~Chess()
{
}

Stone *Chess::get(int id)
{
    return m_children[id];
}

void Chess::reset()
{
    m_map.clear();
    history.clear();
    sideToMove = Stone::COLOR_RED;
    halfMoveClock = 0;
    /* 红方 */
    std::vector<Pos> redGroupPos = {{9, 0}, {9, 1}, {9, 2}, {9, 3},
                                    {9, 4}, {9, 5}, {9, 6}, {9, 7},
                                    {9, 8}, {7, 1}, {7, 7}, {6, 0},
                                    {6, 2}, {6, 4}, {6, 6}, {6, 8}};

    for (std::size_t i = 0; i < redGroupPos.size(); i++) {
        const Pos &pos = redGroupPos[i];
        Stone *stone = m_children[i + Stone::ID_RED];
        stone->pos = pos;
        stone->alive = 1;
        m_map[pos] = stone;
    }
    /* 黑方 */
    std::vector<Pos> blackGroupPos = {{0, 0}, {0, 1}, {0, 2}, {0, 3},
                                      {0, 4}, {0, 5}, {0, 6}, {0, 7},
                                      {0, 8}, {2, 1}, {2, 7}, {3, 0},
                                      {3, 2}, {3, 4}, {3, 6}, {3, 8}};
    for (std::size_t i = 0; i < blackGroupPos.size(); i++) {
        Pos &pos = blackGroupPos[i];
        Stone *stone = m_children[i + Stone::ID_BLACK];
        stone->pos = pos;
        stone->alive = 1;
        m_map[pos] = stone;
    }
    return;
}

/* ============================================================
 *  落子 / 回退
 * ============================================================ */

/*
 * applyMove: 只改棋盘 (m_map + 棋子位置 + 被吃子 alive), 不做任何记账。
 *   返回 false 表示这一步在棋盘上根本无法执行 (棋子已死 / 目标格有己方子),
 *   此时棋盘保持原样。moveForward 与 isLegalMove 共用它, 避免两份"落子"逻辑
 *   走偏。
 */
bool Chess::applyMove(const Step *s)
{
    /*
       先做边界校验, 再看棋子。原来直接 `m_children[s->id]` / `moveTo(s->nextPos)`:
       id 和坐标都是**外部传进来**的 (搜索返回的 Step、回放记录、数据库行), 越界时
       std::array 和 StoneMap::data[10][9] 都是不检查的 —— 一次越界写就落在 Chess
       对象的相邻成员上, 表现出来是"AI 明明有棋可走却说无合法走法"甚至段错误。
       这里把"这一步根本没法在棋盘上执行"变成返回 false (调用方本来就要处理 false)。
    */
    if (s == nullptr || !s->valid) {
        return false;
    }
    if (s->id < 0 || s->id >= (int)m_children.size()) {
        return false;
    }
    if (!m_map.isInner(s->pos) || !m_map.isInner(s->nextPos)) {
        return false;
    }
    Stone *stone = m_children[s->id];
    if (stone == nullptr || stone->alive == false) {
        return false;
    }
    if (!(stone->pos == s->pos)) {
        /* Step 里的起点和棋子当前位置不一致 -> 这是一个过期的走法 */
        return false;
    }
    /* moveTo() 自带 tryMoveTo() 形状校验, 并会在目标格是己方子时返回 false */
    return stone->moveTo(s->nextPos);
}

void Chess::undoMove(const Step *s)
{
    if (s == nullptr || s->id < 0 || s->id >= (int)m_children.size()) {
        return;
    }
    if (!m_map.isInner(s->pos) || !m_map.isInner(s->nextPos)) {
        return;
    }
    Stone *stone = m_children[s->id];
    if (stone == nullptr) {
        return;
    }
    m_map[s->pos] = stone;
    stone->pos = s->pos;
    if (s->nextId != Stone::ID_NONE) {
        Stone *dst = m_children[s->nextId];
        if (dst == nullptr) {
            return;
        }
        m_map[s->nextPos] = dst;
        dst->pos = s->nextPos;
        dst->alive = true;
    } else {
        m_map[s->nextPos] = nullptr;
    }
}

void Chess::moveForward(const Step *s, double &totalReward)
{
    if (s == nullptr) {
        return;
    }
    Stone *stone = m_children[s->id];
    if (stone == nullptr) {
        return;
    }
    const int mover = stone->color;
    /* 落子失败时不做任何记账: 以前的实现无视返回值, 结果 moveBack 会把
     * 一个从未被吃掉的棋子"复活"到终点格, 把棋盘写坏。 */
    if (applyMove(s) == false) {
        return;
    }
    bool captured = (s->nextId != Stone::ID_NONE);
    if (captured) {
        Stone *dst = m_children[s->nextId];
        if (dst->color == Stone::COLOR_RED) {
            totalReward += dst->value;
        } else {
            totalReward -= dst->value;
        }
    }
    sideToMove = (mover == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    /*
       先在"更新计数之前"记录, HistoryRecord.halfMoveClock 保存的就是走这一步
       **之前**的值, moveBack 才能精确恢复 (否则回退后计数会停在走完后的值上)。
    */
    pushHistory();
    /* 60 回合自然限着: 吃子或走兵/卒才重置计数 */
    if (captured || stone->type == Stone::TYPE_BING) {
        halfMoveClock = 0;
    } else {
        halfMoveClock++;
    }
    return;
}

void Chess::moveBack(const Step *s, double &totalReward)
{
    if (s == nullptr) {
        return;
    }
    Stone *stone = m_children[s->id];
    if (stone == nullptr) {
        return;
    }
    /* 该步没有被真正执行过 (applyMove 失败) -> 什么都不用回退 */
    if (!(stone->pos == s->nextPos)) {
        return;
    }
    undoMove(s);
    if (s->nextId != Stone::ID_NONE) {
        Stone *dst = m_children[s->nextId];
        if (dst->color == Stone::COLOR_RED) {
            totalReward -= dst->value;
        } else {
            totalReward += dst->value;
        }
    }
    sideToMove = stone->color;
    if (!history.empty()) {
        /* 恢复到走这一步之前的计数 */
        halfMoveClock = history.back().halfMoveClock;
        history.pop_back();
    }
    return;
}

/* ============================================================
 *  走法生成
 * ============================================================ */

void Chess::samplePseudo(int color, std::vector<Step *> &steps)
{
    int i0 = Stone::ID_RED;
    int in = Stone::ID_RED_END;
    if (color == Stone::COLOR_BLACK) {
        i0 = Stone::ID_BLACK;
        in = Stone::ID_BLACK_END;
    }
    for (int i = i0; i < in; i++) {
        if (m_children[i] != nullptr && m_children[i]->alive) {
            m_children[i]->getPossibleSteps(steps);
        }
    }
    return;
}

/*
 * sample: 伪合法走法 + 合法性过滤。
 *
 * 过滤掉的是"走后自家将/帅被攻击"的走法 —— 包括被将军时不应将、以及走成
 * 两将照面。这是中国象棋的核心规则, 以前完全缺失 (isInCheck() 写好了却从没
 * 被调用过), 于是 AI 和玩家都能自杀 / 不应将 / 主动走出照面。
 */
void Chess::sample(int color, std::vector<Step *> &steps)
{
    std::vector<Step *> pseudo;
    samplePseudo(color, pseudo);
    /* "当前是否被将" 与具体走法无关, 整批只算一次 */
    const bool inCheck = isInCheck(color);
    /* 被拒绝的走法攒起来一次性还回对象池 (逐个 put 会为每个走法分配一次临时
     * vector, 而这是搜索热路径) */
    std::vector<Step *> rejected;
    rejected.reserve(pseudo.size());
    for (std::size_t i = 0; i < pseudo.size(); i++) {
        Step *s = pseudo[i];
        if (isLegalMoveInternal(color, s, inCheck)) {
            steps.push_back(s);
        } else {
            rejected.push_back(s);
        }
    }
    if (!rejected.empty()) {
        Steps::instance().put(rejected);
    }
    return;
}

/* ============================================================
 *  攻击 / 将军 / 合法性
 * ============================================================ */

/*
 * isAttacked: target 是否被 byColor 方的棋子攻击。
 *
 * 按棋子类型做几何判定, 而不是对 16 个敌方棋子逐个调 tryMoveTo() ——
 * 后者每次都要重建走法形状, 而本函数处在一个会被搜索热点反复调用的位置。
 * 仕/士 与 相/象 永远到不了对方九宫, 直接跳过。
 */
bool Chess::isAttacked(const Pos &target, int byColor)
{
    int i0 = (byColor == Stone::COLOR_RED) ? Stone::ID_RED : Stone::ID_BLACK;
    int in = (byColor == Stone::COLOR_RED) ? Stone::ID_RED_END : Stone::ID_BLACK_END;

    for (int i = i0; i < in; i++) {
        Stone *s = m_children[i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        switch (s->type) {
        case Stone::TYPE_CHE:
            /* 车: 同线且中间无子 */
            if ((s->pos.x == target.x || s->pos.y == target.y)
                    && m_map.countStoneOnLine(s->pos, target) == 0) {
                return true;
            }
            break;
        case Stone::TYPE_PAO:
            /* 炮: 同线且中间恰有一个炮架 */
            if ((s->pos.x == target.x || s->pos.y == target.y)
                    && m_map.countStoneOnLine(s->pos, target) == 1) {
                return true;
            }
            break;
        case Stone::TYPE_MA: {
            /* 马: 日字且马腿无子 */
            int dx = std::abs(target.x - s->pos.x);
            int dy = std::abs(target.y - s->pos.y);
            if ((dx == 2 && dy == 1) || (dx == 1 && dy == 2)) {
                Pos leg = s->pos;
                if (dx == 2) {
                    leg.x += (target.x > s->pos.x) ? 1 : -1;
                } else {
                    leg.y += (target.y > s->pos.y) ? 1 : -1;
                }
                if (m_map[leg] == nullptr) {
                    return true;
                }
            }
            break;
        }
        case Stone::TYPE_BING:
            /* 兵/卒: 只能攻击它自己走得到的那一格 */
            if (byColor == Stone::COLOR_RED) {
                if (target.x == s->pos.x - 1 && target.y == s->pos.y) {
                    return true;
                }
                if (s->pos.x <= 4 && target.x == s->pos.x
                        && std::abs(target.y - s->pos.y) == 1) {
                    return true;
                }
            } else {
                if (target.x == s->pos.x + 1 && target.y == s->pos.y) {
                    return true;
                }
                if (s->pos.x >= 5 && target.x == s->pos.x
                        && std::abs(target.y - s->pos.y) == 1) {
                    return true;
                }
            }
            break;
        case Stone::TYPE_JIANG: {
            /* 飞将: 只有当目标格上是对方将/帅时才是"攻击" */
            Stone *opp = m_map[target];
            if (opp != nullptr && opp->type == Stone::TYPE_JIANG
                    && opp->color != s->color
                    && opp->pos.y == s->pos.y
                    && m_map.countStoneOnLine(s->pos, target) == 0) {
                return true;
            }
            /* 将/帅: 对方九宫内相邻一格 */
            if (std::abs(target.x - s->pos.x) + std::abs(target.y - s->pos.y) == 1
                    && target.y >= 3 && target.y <= 5) {
                if (byColor == Stone::COLOR_RED) {
                    if (target.x >= 7 && target.x <= 9) {
                        return true;
                    }
                } else {
                    if (target.x >= 0 && target.x <= 2) {
                        return true;
                    }
                }
            }
            break;
        }
        default:
            /* 仕/士, 相/象: 活动范围限制在己方, 无法攻击对方将/帅 */
            break;
        }
    }
    return false;
}

int Chess::isGameOver()
{
    if (redJiang.alive == false) {
        return Stone::COLOR_BLACK;
    }
    if (blackJiang.alive == false) {
        return Stone::COLOR_RED;
    }
    return Stone::COLOR_NONE;
}

bool Chess::isInCheck(int color)
{
    Stone *jiang = (color == Stone::COLOR_RED) ? &redJiang : &blackJiang;
    if (jiang->alive == false) {
        return false;
    }
    int enemyColor = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    return isAttacked(jiang->pos, enemyColor);
}

bool Chess::isLegalMove(int color, const Step *s)
{
    return isLegalMoveInternal(color, s, isInCheck(color));
}

/*
 * isLegalMoveInternal - isLegalMove() 的内部版本。
 *
 * inCheck 由调用方预先算好: 同一次 sample() 里几十个候选走法共享同一个"当前是否
 * 被将"的结论 (它只取决于走之前的局面), 于是热路径上每个走法的成本从"落子 +
 * isAttacked + 回退"降到几次坐标比较。
 *
 * 快速路径的判据: 当"移动的不是将/帅"且"当前没有被将"时, 这一步只可能通过下面
 * 两条途径改变己方将/帅的安全, 其余情况必然合法 ——
 *   (a) 改变某条经过己方将/帅的横线/竖线上的阻挡 (敌方车、炮、以及飞将规则);
 *   (b) 打开或封住某个正对己方将/帅的敌方马的蹩马腿。
 * 敌方兵与敌方将的攻击只取决于几何位置和将/帅自身所在格, 与其它子的阻挡无关,
 * 因此在"没动将/帅"的前提下不会因为这一步而改变。
 */
bool Chess::isLegalMoveInternal(int color, const Step *s, bool inCheck)
{
    if (s == nullptr || s->valid == false) {
        return false;
    }
    Stone *mover = m_children[s->id];
    if (mover == nullptr || mover->alive == false || mover->color != color) {
        return false;
    }
    /* 形状校验 (不改变棋盘) */
    if (mover->tryMoveTo(s->nextPos) == false) {
        return false;
    }
    Stone *dst = m_map[s->nextPos];
    if (dst != nullptr && dst->color == mover->color) {
        return false;
    }

    Stone *jiang = (color == Stone::COLOR_RED) ? &redJiang : &blackJiang;
    if (jiang->alive && mover->type != Stone::TYPE_JIANG && !inCheck) {
        const Pos g = jiang->pos;
        bool affects = (s->pos.x == g.x || s->pos.y == g.y
                        || s->nextPos.x == g.x || s->nextPos.y == g.y);
        if (!affects) {
            const int enemy = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                          : Stone::COLOR_RED;
            const int i0 = (enemy == Stone::COLOR_RED) ? Stone::ID_RED : Stone::ID_BLACK;
            const int in = (enemy == Stone::COLOR_RED) ? Stone::ID_RED_END
                                                       : Stone::ID_BLACK_END;
            for (int i = i0; i < in; i++) {
                Stone *e = m_children[i];
                if (e == nullptr || e->alive == false || e->type != Stone::TYPE_MA) {
                    continue;
                }
                const int dx = std::abs(g.x - e->pos.x);
                const int dy = std::abs(g.y - e->pos.y);
                if (!((dx == 2 && dy == 1) || (dx == 1 && dy == 2))) {
                    continue;
                }
                Pos leg = e->pos;
                if (dx == 2) {
                    leg.x += (g.x > e->pos.x) ? 1 : -1;
                } else {
                    leg.y += (g.y > e->pos.y) ? 1 : -1;
                }
                if (leg == s->pos || leg == s->nextPos) {
                    affects = true;
                    break;
                }
            }
        }
        if (!affects) {
            return true;
        }
    }

    if (applyMove(s) == false) {
        return false;
    }
    bool bad = isInCheck(color);
    undoMove(s);
    return !bad;
}

bool Chess::hasLegalMoves(int color)
{
    std::vector<Step *> steps;
    sample(color, steps);
    bool any = !steps.empty();
    Steps::instance().put(steps);
    return any;
}

bool Chess::isRepetition()
{
    unsigned long long h = computeHash();
    int count = 0;
    /*
       只需要看最近 halfMoveClock 步: 局面只有在"这段没有吃子、没有走兵"的可逆
       区间内才可能重复。全量扫描在长对局里会把 O(history) 带进每个搜索节点。
    */
    int window = halfMoveClock;
    if (window > (int)history.size()) {
        window = (int)history.size();
    }
    int begin = (int)history.size() - window;
    for (int i = (int)history.size() - 1; i >= begin; i--) {
        if (history[i].hash == h) {
            count++;
        }
    }
    /* 同一局面出现3次判和 */
    return count >= 3;
}

bool Chess::isDraw()
{
    if (isGameOver() != Stone::COLOR_NONE) {
        return false;
    }
    /* 三次重复局面 */
    if (isRepetition()) {
        return true;
    }
    /* 60 回合 (120 半回合) 内双方都没有吃子 */
    if (halfMoveClock >= 120) {
        return true;
    }
    return false;
}

int Chess::getResult(int colorToMove)
{
    int win = isGameOver();
    if (win != Stone::COLOR_NONE) {
        return (win == Stone::COLOR_RED) ? RESULT_RED_WIN : RESULT_BLACK_WIN;
    }
    if (isDraw()) {
        return RESULT_DRAW;
    }
    /*
       将杀 / 困毙: 轮到走的一方没有任何合法走法。中国象棋里困毙同样判负,
       所以不需要区分两者。注意 isGameOver() 保持"便宜"的语义 (只看将帅是否
       存活), 这个较贵的判定只给 GUI / 训练循环的终局判断用。
    */
    if (!hasLegalMoves(colorToMove)) {
        return (colorToMove == Stone::COLOR_RED) ? RESULT_BLACK_WIN : RESULT_RED_WIN;
    }
    return RESULT_ONGOING;
}

/*
 * Zobrist 表: 32 个棋子 x 90 个格子 + 1 个"轮到黑方"键。
 *
 * 原来的 computeHash() 是把 (棋子id | x<<8 | y<<12) 直接异或起来, 结构化的键
 * 之间相关性很强 —— 用作重复局面判定勉强够, 但拿去当**置换表**的索引就会频繁
 * 碰撞: 碰撞后 TT 会返回别的局面的分数, 搜索直接出错。
 * 这里换成标准的 Zobrist 随机键 (splitmix64 + 固定种子, 保证每次运行一致)。
 */
const std::array<unsigned long long, 32*90 + 1> &Chess::zobrist()
{
    static const std::array<unsigned long long, 32*90 + 1> table = []() {
        std::array<unsigned long long, 32*90 + 1> t{};
        unsigned long long x = 0x9E3779B97F4A7C15ULL;
        for (std::size_t i = 0; i < t.size(); i++) {
            x += 0x9E3779B97F4A7C15ULL;
            unsigned long long z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            t[i] = z ^ (z >> 31);
        }
        return t;
    }();
    return table;
}

unsigned long long Chess::computeHash()
{
    const std::array<unsigned long long, 32*90 + 1> &z = zobrist();
    unsigned long long h = 0;
    for (int i = 0; i < 32; i++) {
        Stone *s = m_children[i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        h ^= z[(std::size_t)i*90 + (std::size_t)(s->pos.x*9 + s->pos.y)];
    }
    if (sideToMove == Stone::COLOR_BLACK) {
        h ^= z[32*90];
    }
    return h;
}

void Chess::pushHistory()
{
    HistoryRecord r;
    r.hash = computeHash();
    /* 记录当前 (即这一步**之前**) 的计数, 供 moveBack 精确恢复 */
    r.halfMoveClock = halfMoveClock;
    history.push_back(r);
}
