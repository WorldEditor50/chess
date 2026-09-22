#include "abagent.h"
#include <algorithm>
#include <cstdio>   /* selfCheckReport 的 snprintf */
#include <cstdlib>
#include <string>

/*
 * ABAgent - Alpha-Beta Pruning Chess Agent
 *
 * Full alpha-beta pruning search implementation, migrated from Chess.
 * Uses chess.evaluate() for leaf evaluation, chess.sample() for move
 * generation, and chess.moveForward()/moveBack() for board state.
 */

/* ============================================================
 *  辅助函数 (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
 * ============================================================ */

/* 棋子价值排序 (值越大越有价值被吃) */
static int victimValue(int type)
{
    switch (type) {
    case Stone::TYPE_JIANG: return 1000;
    case Stone::TYPE_CHE:   return 500;
    case Stone::TYPE_MA:    return 300;
    case Stone::TYPE_PAO:   return 300;
    case Stone::TYPE_SHI:   return 200;
    case Stone::TYPE_XIANG: return 200;
    case Stone::TYPE_BING:  return 100;
    default:                return 0;
    }
}

/* 攻击方价值排序 (LV: 用小棋子吃大棋子应优先搜索) */
static int attackerValue(int type)
{
    switch (type) {
    case Stone::TYPE_JIANG: return 0;    /* 将/帅吃子 */
    case Stone::TYPE_BING:  return 1;    /* 兵吃子 */
    case Stone::TYPE_SHI:   return 2;
    case Stone::TYPE_XIANG: return 3;
    case Stone::TYPE_MA:    return 4;
    case Stone::TYPE_PAO:   return 4;
    case Stone::TYPE_CHE:   return 5;    /* 车是最"贵"的攻击方，应后搜 */
    default:                return 10;
    }
}

/* ============================================================
 *  Constructor
 * ============================================================ */
ABAgent::ABAgent(Chess &chess_, int depth)
    : AgentBase(), chess(chess_), maxDepth(depth)
{
}

/* ============================================================
 *  AgentBase interface
 * ============================================================ */
Step ABAgent::getBestMove(int color)
{
    return findBestMove(color);
}

Step ABAgent::getBestMove(int color, int depth)
{
    int savedDepth = maxDepth;
    maxDepth = depth;
    Step result = findBestMove(color);
    maxDepth = savedDepth;
    return result;
}

std::string ABAgent::getName() const
{
    return "Alpha-Beta Pruning (depth=" + std::to_string(maxDepth) + ")";
}

Step ABAgent::findBestMove(int color)
{
    std::vector<Step*> steps;
    chess.sample(color, steps);

    /* 优化: 走法排序 */
    if (steps.size() > 1) {
        orderMoves(steps);
    }

    double totalReward = 0;
    Step* best = nullptr;

    /*
     * 根据走棋方正确选择根节点类型:
     *   - 黑方 (AI): 根为 MAX, 最大化 black-perspective 得分
     *   - 红方:     根为 MIN, 最小化 black-perspective 得分
     */
    if (color == Stone::COLOR_BLACK) {
        /* 黑方 = MAX 节点 */
        double beta = -Stone::value_infi;
        for (Step *s : steps) {
            chess.moveForward(s, totalReward);
            double r = minimizeAlpha(Stone::COLOR_RED, maxDepth - 1, beta, totalReward);
            chess.moveBack(s, totalReward);
            /*
               `best == nullptr ||` 不是冗余判断, 它修的是 2026-09 用户报的那个 bug
               ("[arena] agent(0) 返回无效走法 ... valid=0 id=0 pos=(0,0)->(0,0)"):
                 若**每一步都必输**, 所有 r 都等于 -value_infi (子节点里对方无合法走法,
                minimizeAlpha 直接返回 ±value_infi), 而这里的初值 beta 也是 -value_infi,
                于是严格不等号 `r > beta` 一次都不成立 -> best 保持 nullptr -> 函数返回
                默认构造的 Step (valid=false, id=0, pos=(0,0))。
               调用方把 valid=false 读成"这一步没棋可走" -> 明明还有合法走法却被判负。
               现在第一个走法无条件成为候选, "全负" 时退化成"返回排序后的第一手"
               (走法已按 MVV-LVA 排序, 所以是"先看吃子"), 棋理上无差别 (都输), 但
               契约上必须返回一步**合法**走法。
            */
            if (best == nullptr || r > beta) {
                best = s;
                beta = r;
            }
        }
        /*
           记下根分值供价值头蒸馏用 (Step 2)。黑方是 MAX 节点, 所以最终的 beta 就是
           "这个局面黑方有多好"; 红方那一支取 alpha, 两者**同一口径** (black-perspective)。
        */
        m_lastScore = beta;
    } else {
        /* 红方 = MIN 节点 */
        double alpha = Stone::value_infi;
        for (Step *s : steps) {
            chess.moveForward(s, totalReward);
            double r = maximizeBeta(Stone::COLOR_BLACK, maxDepth - 1, alpha, totalReward);
            chess.moveBack(s, totalReward);
            /* 同样的兜底, 对称的那一半: 红方每一步都必输时所有 r 都是 +value_infi,
               而初值 alpha 也是 +value_infi, 严格不等号一次都不成立 (见上面黑方分支
               的长注释: 那正是 arena 里 "agent(0) 返回无效走法" 的来源)。 */
            if (best == nullptr || r < alpha) {
                best = s;
                alpha = r;
            }
        }
        m_lastScore = alpha;
    }

    Step step;
    if (best != nullptr) {
        step = *best;
    }
    /*
       走到这里 best == nullptr **只可能**是因为 steps 为空 —— 即真的无合法走法
       (将杀/困毙), 此时返回默认构造的 Step (valid=false) 才是正确语义。上面两个
       分支已经保证"有走法就一定有候选", 所以这个默认值不会再被误用。
    */
    /*
       没有合法走法时不记分数 (getScoreValid() = false): 那种局面的"分值"是 ±value_infi
       量级, 拿去做回归目标会把 critic 一带带偏。
    */
    m_scoreValid = (best != nullptr) && (std::fabs(m_lastScore) < Stone::value_infi * 0.5);
    Steps::instance().put(steps);
    return step;
}

/* ============================================================
 *  orderMoves - 走法排序 (MVV-LVA)
 * ============================================================ */
void ABAgent::orderMoves(std::vector<Step*> &steps)
{
    std::vector<int> scores(steps.size(), 0);
    for (std::size_t i = 0; i < steps.size(); i++) {
        Step *s = steps[i];
        if (s->nextId != Stone::ID_NONE) {
            Stone *victim = chess.stones[s->nextId];
            Stone *attacker = chess.stones[s->id];
            if (victim != nullptr && attacker != nullptr) {
                scores[i] = victimValue(victim->type) * 100 - attackerValue(attacker->type);
            }
        }
    }
    for (std::size_t i = 0; i + 1 < steps.size(); i++) {
        for (std::size_t j = i + 1; j < steps.size(); j++) {
            if (scores[j] > scores[i]) {
                std::swap(steps[i], steps[j]);
                std::swap(scores[i], scores[j]);
            }
        }
    }
}

/* ============================================================
 *  quiescenceSearch - 静态搜索 (缓解水平线效应)
 *
 *  只搜索吃子走法, 稳定局面评估。
 *  注意: chess.evaluate() 始终返回黑方(AI)视角的评估值。
 *
 *  外层搜索保证:
 *    - MAXIMIZE 节点 (黑方走) 调用时 color=BLACK, 查找让黑方得分更高的吃子
 *    - MINIMIZE 节点 (红方走) 调用时 color=RED,   查找让黑方得分更低的吃子
 *
 *  返回值: 始终是黑方(AI)视角的评估值 (正值=黑方有利)
 * ============================================================ */
double ABAgent::quiescenceSearch(int color, double alpha, double beta, int depth)
{
    /* 静态评估: chess.evaluate() 返回黑方视角, 需要转换到当前走棋方视角 */
    double standPat = chess.evaluate();
    if (color == Stone::COLOR_RED) {
        standPat = -standPat;  /* RED 要走, 评估从 RED 视角看是 neg(黑方视角) */
    }

    if (depth <= 0) {
        return standPat;
    }

    /* Negamax 标准剪枝 */
    if (standPat >= beta) {
        return beta;
    }
    if (standPat > alpha) {
        alpha = standPat;
    }

    /* 生成当前方的所有走法, 只保留吃子走法 */
    std::vector<Step*> allSteps;
    /*
     * 这里用 samplePseudo(): sample() 会对每个走法做一次"走后是否被将"的完整
     * 校验, 而静态搜索只用得上吃子走法 —— 对几十个安静走法做校验纯属浪费。
     * 吃子走法在下面逐个用 isLegalMove() 校验, 被拒绝的还回对象池。
     */
    chess.samplePseudo(color, allSteps);

    std::vector<Step*> captures;
    /* 安静走法攒起来一次还回对象池 (热路径上不要逐个分配临时 vector) */
    std::vector<Step*> quiet;
    quiet.reserve(allSteps.size());
    for (Step *s : allSteps) {
        if (s->nextId != Stone::ID_NONE) {
            captures.push_back(s);
        } else {
            quiet.push_back(s);
        }
    }
    Steps::instance().put(quiet);

    /* 过滤掉不合法的吃子 (例如吃子后自己被将) */
    {
        const bool inCheck = chess.isInCheck(color);
        std::vector<Step*> legalCaptures;
        legalCaptures.reserve(captures.size());
        std::vector<Step*> rejected;
        for (Step *s : captures) {
            if (chess.isLegalMoveInternal(color, s, inCheck)) {
                legalCaptures.push_back(s);
            } else {
                rejected.push_back(s);
            }
        }
        if (!rejected.empty()) {
            Steps::instance().put(rejected);
        }
        captures.swap(legalCaptures);
    }

    /* MVV-LVA 排序 */
    if (captures.size() > 1) {
        std::vector<int> scores(captures.size(), 0);
        for (std::size_t i = 0; i < captures.size(); i++) {
            Stone *victim = chess.stones[captures[i]->nextId];
            Stone *attacker = chess.stones[captures[i]->id];
            if (victim != nullptr && attacker != nullptr) {
                scores[i] = victimValue(victim->type) * 100 - attackerValue(attacker->type);
            }
        }
        for (std::size_t i = 0; i + 1 < captures.size(); i++) {
            for (std::size_t j = i + 1; j < captures.size(); j++) {
                if (scores[j] > scores[i]) {
                    std::swap(captures[i], captures[j]);
                    std::swap(scores[i], scores[j]);
                }
            }
        }
    }

    double totalReward = 0;
    int color_ = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    for (Step *s : captures) {
        chess.moveForward(s, totalReward);

        Stone *victim = (s->nextId != Stone::ID_NONE) ? chess.stones[s->nextId] : nullptr;
        if (victim != nullptr && victim->type == Stone::TYPE_JIANG) {
            chess.moveBack(s, totalReward);
            Steps::instance().put(captures);
            /* 吃将: 当前走棋方视角下的最大值 */
            return Stone::value_infi;
        }

        /* Negamax: 递归到对方节点, 取负交换窗口 */
        double r = -quiescenceSearch(color_, -beta, -alpha, depth - 1);
        chess.moveBack(s, totalReward);

        if (r >= beta) {
            Steps::instance().put(captures);
            return beta;
        }
        if (r > alpha) {
            alpha = r;
        }
    }

    Steps::instance().put(captures);
    return alpha;
}

/*
 * quiescenceBlackView: 把 negamax 静态搜索的结果换算成"黑方视角"分值。
 *   lo / hi 为黑方视角窗口; 轮到红方走时窗口需要上下翻转 (-hi, -lo)。
 */
double ABAgent::quiescenceBlackView(int color, double lo, double hi)
{
    double a = (color == Stone::COLOR_BLACK) ? lo : -hi;
    double b = (color == Stone::COLOR_BLACK) ? hi : -lo;
    double q = quiescenceSearch(color, a, b, 3);
    return (color == Stone::COLOR_RED) ? -q : q;
}

/* ============================================================
 *  minimizeAlpha - MIN 节点
 *
 *  当前走棋方试图最小化评估值.
 *  返回值: 从黑方 (AI) 视角的评估值 (正值=对黑方有利)
 * ============================================================ */
double ABAgent::minimizeAlpha(int color, int depth, double beta, double &totalReward)
{
    int gameResult = chess.isGameOver();
    if (gameResult == Stone::COLOR_BLACK) {
        return Stone::value_infi;   /* 黑方赢 -> AI(黑方)有利 */
    }
    if (gameResult == Stone::COLOR_RED) {
        return -Stone::value_infi;  /* 红方赢 -> AI(黑方)不利 */
    }

    /* 三次重复局面判和: 黑方视角的和棋分值是 0 */
    if (chess.isRepetition()) {
        return 0.0;
    }

    /*
       叶节点同样要进静态搜索 —— 以前只有 maximizeBeta (MAX/黑方节点) 会进,
       MIN 节点直接返回 evaluate()。结果是奇数深度和偶数深度的评估口径不同
       (一方有 3 层吃子延伸、另一方没有), 同一个局面在不同奇偶深度会得到系统性
       偏差, 而且红方的吃子序列完全看不见。

       窗口必须传**本节点自己的**黑方视角窗口。MIN 节点收到的 `beta` 参数其实是
       父 MAX 节点的当前最优值 (见函数注释: 参数名有误导性), MIN 节点的窗口就是
       (beta, +infi) —— 注意上界是 +infi, 不是 beta 本身: 若把 beta 当上界,
       根节点初始的 beta = -value_infi 会退化成零宽窗口 (alpha=beta=-infi),
       静态搜索直接返回边界值, 于是所有走法得分相同、根节点选不出走法。
    */
    if (depth == 0) {
        return quiescenceBlackView(color, beta, Stone::value_infi);
    }

    std::vector<Step*> steps;
    chess.sample(color, steps);

    if (steps.empty()) {
        /* 当前方无合法走法 → 对方获胜 */
        return (color == Stone::COLOR_RED) ? Stone::value_infi : -Stone::value_infi;
    }

    if (steps.size() > 1) {
        orderMoves(steps);
    }

    double alpha = Stone::value_infi;
    for (Step *s : steps) {
        chess.moveForward(s, totalReward);
        int color_ = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        double r = maximizeBeta(color_, depth - 1, alpha, totalReward);
        chess.moveBack(s, totalReward);

        if (r < alpha) {
            alpha = r;
        }
        if (alpha <= beta) {
            break;
        }
    }
    Steps::instance().put(steps);
    return alpha;
}

/* ============================================================
 *  maximizeBeta - MAX 节点
 *
 *  当前走棋方试图最大化评估值.
 * ============================================================ */
double ABAgent::maximizeBeta(int color, int depth, double alpha, double &totalReward)
{
    int gameResult = chess.isGameOver();
    if (gameResult == Stone::COLOR_BLACK) {
        return Stone::value_infi;   /* 黑方赢 -> AI(黑方)有利 */
    }
    if (gameResult == Stone::COLOR_RED) {
        return -Stone::value_infi;  /* 红方赢 -> AI(黑方)不利 */
    }

    /* 三次重复局面判和 */
    if (chess.isRepetition()) {
        return 0.0;
    }

    /*
       叶节点进入静态搜索。以前这里固定用完全开放窗口 (-infi, +infi), 于是父节点
       传下来的边界被丢掉, quiescenceSearch 里的 standPat >= beta 永远不成立 ——
       从"错误剪枝"变成了"完全不剪枝", 每个叶节点都要展开全部吃子序列。
       现在把父边界真正传进去。MAX 节点收到的 `alpha` 参数是父 MIN 节点的当前
       最优值, 所以本节点的黑方视角窗口是 (-infi, alpha)。
    */
    if (depth == 0) {
        return quiescenceBlackView(color, -Stone::value_infi, alpha);
    }

    std::vector<Step*> steps;
    chess.sample(color, steps);

    if (steps.empty()) {
        /* 当前方无合法走法 → 对方获胜 */
        return (color == Stone::COLOR_RED) ? Stone::value_infi : -Stone::value_infi;
    }

    if (steps.size() > 1) {
        orderMoves(steps);
    }

    double beta = -Stone::value_infi;
    for (Step *s : steps) {
        chess.moveForward(s, totalReward);
        int color_ = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        double r = minimizeAlpha(color_, depth - 1, beta, totalReward);
        chess.moveBack(s, totalReward);

        if (r > beta) {
            beta = r;
        }
        if (beta >= alpha) {
            break;
        }
    }
    Steps::instance().put(steps);
    return beta;
}

/* ============================================================
 *  自检报告
 * ============================================================ */

/*
 * stepInMoveList: s 是否就是 legal 里的某一步 (决策合法性自检用)。
 *
 * 比对 (起点棋子 id, 起点格, 终点格) 三项, 不比对整个对象: 走法生成器给同一个走法
 * 填的 reward 与搜索回填的不一定同源, 拿整对象比会出现"明明是这一步却说不相等"。
 * `!s.valid` 直接算不在集合里 —— 默认构造的 Step (id=0, nextId=0, pos=(0,0)) 恰好
 * 可能"等于"某个越界/占位对象, 只有 valid 位能区分"生成器产出的走法"与"没填过的
 * 占位对象" (stone.h 里 valid 字段的存在理由就是这个)。
 */
static bool stepInMoveList(const Step &s, const std::vector<Step *> &legal)
{
    if (!s.valid) {
        return false;
    }
    for (const Step *l : legal) {
        if (l == nullptr || !l->valid) {
            continue;
        }
        if (l->id == s.id && l->nextId == s.nextId
            && l->pos.x == s.pos.x && l->pos.y == s.pos.y
            && l->nextPos.x == s.nextPos.x && l->nextPos.y == s.nextPos.y) {
            return true;
        }
    }
    return false;
}

/* ------------------------------------------------------------------
 *  selfCheckReport —— 界面"模型自检"面板的数据源
 *
 *  这里报的全是**结构 / 口径**事实, 一条棋力都没有。判读写在行尾, 因为面板的读者
 *  是看训练曲线的人, 而这四行恰好回答四个"曲线答不了"的问题:
 *
 *    (1) 这个 agent 到底有没有可训练参数? ABAgent 没有 (叶子是手工 evaluate()):
 *        所以它的损失曲线空着是**正确**状态, 不是训练挂了。
 *    (2) **它还能不能返回合法走法**? 这是回归指示器, 钉的是 2026-09 用户报的 bug:
 *        根节点上"每一步都必输"时, 所有根走法分值都是 ±Stone::value_infi, 而根节点
 *        的窗口初值也是 ±value_infi, 于是严格不等号 (`r > beta` / `r < alpha`) 一次都
 *        不成立 -> best 保持 nullptr -> findBestMove 返回默认构造的 Step (valid=false,
 *        id=0, pos=(0,0))。arena 那边记的是
 *          "[arena] agent(0) 返回无效走法 ... valid=0 id=0 pos=(0,0)->(0,0),
 *           仍有 1 个合法走法, 已兜底"
 *        也就是"还有棋可下却被读成无棋可走" -> GUI 可能直接判负。abagent.cpp 的
 *        findBestMove 现在用 `best == nullptr || ...` 兜底修掉了, 这一行就是那笔修复
 *        的**验收读数**: 一旦又打印"非法", 说明兜底被谁改回去了。
 *    (3) 标准开局的两个确定性读数 (合法走法数 / evaluate()) —— 与训练进度、与当前
 *        对局无关, 一打开面板就能看, 改走法生成或评估口径时会动, 当尺子用。
 *    (4) 最近一次搜索的根分值 (黑方视角, 供价值头蒸馏) 和它有没有分数。
 *
 *  **只读**: 全程用 chess 的副本 + 一个临时局部 agent, 不碰 this->chess (GUI 线程
 *  调用它时搜索线程可能正在用同一个棋盘), 不改任何成员 (本函数是 const)。
 *  临时 agent 换来的是两件事: findBestMove() 非 const 也能跑; 而且它的搜索**不会**
 *  覆盖本 agent 供蒸馏用的 m_lastScore —— 这正是下面第 4 行要读的那个值。
 *  代价只有一次深度 1 搜索: 根上约 44 个走法 (确切值由报告里那一行打出), 每个叶子
 *  进深度 3 静态搜索 (只搜吃子),
 *  比界面上那一步 depth=4 便宜得多 (chessboard.cpp 里记的实测是 depth 3 = 63 ms,
 *  depth 4 = 169 ms, depth 1 的根与叶子开销都在这之下一个数量级), 而且面板是
 *  **每一手棋**刷新一次 (见 mainwindow 的 updateSelfCheckPanel 调用点), 不在搜索
 *  热路径上调它, 所以这点开销可以接受。
 * ------------------------------------------------------------------ */
std::string ABAgent::selfCheckReport() const
{
    char buf[512];
    std::string out;

    /* ---- 1. 搜索配置与评估来源 (为什么"没有损失曲线") ---- */
    std::snprintf(buf, sizeof(buf),
                  "搜索配置: 深度 %d (界面 AB_DEPTH = 4, 见 chessboard.cpp) | "
                  "叶子评估 Chess::evaluate() = 材质 + 子力位置表 (手工, 无学习)\n",
                  maxDepth);
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "可学习权重: 0 个 (纯搜索 agent) -> getLastTrainLoss() 恒为 NaN, "
                  "训练损失曲线对它永远是空的\n");
    out += buf;

    /* ---- 2+3. 决策合法性自检 + 标准开局"尺子"读数 ---- */
    {
        /* 棋盘副本 + 临时 agent: 见函数头注释 (不能碰 this->chess, 也不能改本 agent) */
        Chess probe(chess);
        probe.reset();                    /* 标准开局, sideToMove = 红 */
        ABAgent tmp(probe, 1);            /* 深度 1: 只为验证"返回值在合法集里" */
        Step s = tmp.findBestMove(Stone::COLOR_RED);
        /*
           评估读数取自同一个副本 (它已被 tmp 搜索过后完整回退, 仍是标准开局)。
           另外用第二个副本取合法走法集: 走法集是 sample() 现生成的, 与 s 的来源无关,
           两份副本互不影响。
        */
        const double evalBlack = probe.evaluate();

        Chess probe2(chess);
        probe2.reset();
        std::vector<Step *> legal;
        probe2.sample(Stone::COLOR_RED, legal);
        const int legalCount = (int)legal.size();
        const bool legalDecision = stepInMoveList(s, legal);

        if (legalDecision) {
            std::snprintf(buf, sizeof(buf),
                          "决策合法性自检: 合法 (走法 (%d,%d)->(%d,%d))\n",
                          s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y);
            out += buf;
        } else {
            /*
               走到这里说明兜底失效了 (或走法生成/合法集口径被改坏): 复现的是
               "[arena] agent(0) 返回无效走法" 那类误判。把 valid 位也打出来,
               因为 valid=false 与 "valid=true 但不在合法集里" 是两个不同的坏法。
            */
            std::snprintf(buf, sizeof(buf),
                          "决策合法性自检: 非法 (返回了 valid=%d 的 Step: "
                          "(%d,%d)->(%d,%d), 开局仍有 %d 个合法走法) "
                          "<-- 回归! 见 findBestMove 的 best==nullptr 兜底\n",
                          s.valid ? 1 : 0, s.pos.x, s.pos.y,
                          s.nextPos.x, s.nextPos.y, legalCount);
            out += buf;
        }

        std::snprintf(buf, sizeof(buf),
                      "标准开局(红先): 合法走法 %d 个 | evaluate() = %.1f "
                      "(黑方视角, 正 = 黑优)\n",
                      legalCount,
                      /* -0.0 会打成 "-0.0", 面板上看着像 bug: 标准开局是对称局面,
                         分值本来就是 0, 所以把 "-0" 归一成 "0" */
                      (evalBlack == 0.0) ? 0.0 : evalBlack);
        out += buf;

        /* 用完必须还回对象池, 否则池会碎片化 (sample 每次都从这里取) */
        Steps::instance().put(legal);
    }

    /* ---- 4. 最近一次搜索的根分值 (价值头蒸馏的监督源) ---- */
    if (getScoreValid()) {
        std::snprintf(buf, sizeof(buf),
                      "最近一次搜索根分值: %.1f (黑方视角, getScoreValid()=true)\n",
                      getLastScore());
        out += buf;
    } else {
        /*
           面板上"0.0"与"没有分数"必须分开写: 默认构造的 m_lastScore 就是 0.0,
           直接打 0 会被当成"这个局面绝对均势"。getScoreValid()=false 的语义是
           那次搜索没有合法走法 (无分数可言), 或本对象还没搜过。
        */
        std::snprintf(buf, sizeof(buf),
                      "最近一次搜索根分值: 无 (getScoreValid()=false —— 该次搜索无"
                      "合法走法, 或本 agent 还没搜过)\n");
        out += buf;
    }

    out += "以上是表示/口径事实, **不是棋力**; 棋力请用 bench_anchor 的锚点对局 "
           "(给出 Elo 差与 95% 置信区间)\n";
    return out;
}
