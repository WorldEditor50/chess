#ifndef MOELOADVIEW_H
#define MOELOADVIEW_H

#include <QDialog>
#include <QSize>
#include <QString>
#include <QTimer>
#include <QVector>
#include <QWidget>

#include "chessboard.h"   /* MoeLoadSnapshot */

class QLabel;

/*
 * ============================================================================
 *  MoeLoadView —— 稀疏 MoE 专家负载均衡状态的小控件 (2026-10)
 * ============================================================================
 *
 * 为什么要有它
 * ------------
 * 稀疏 MoE 的失败模式几乎全是**静默**的: 路由坍缩之后, 前向照跑、loss 照降、
 * 权重照存, 界面上一点异常都没有 —— 只有把"每个专家分到多少流量"画出来才能看见。
 * 本工程实测过这条路的两个极端:
 *   * 基线 (只靠辅助损失): 训练侧 MaxVio 0.318 ~ 0.963, 最弱专家只有 1.1% 流量;
 *   * 开了无辅助损失偏置均衡: MaxVio 0.006 ~ 0.009 (几乎完美)。
 * 两者在**其它任何读数上都分不出来**。
 *
 * 三个刻意的设计选择 (每一条都对应一次踩过的坑)
 * ------------------------------------------------
 * 1. **按前向来源分开画 (训练侧 / 推理侧)**。
 *    `usageSnapshot` 读的是生命周期累计, 它混着"训练批的前向"和"MCTS 展开/叶子估值
 *    的前向" —— 而后者在数量上通常压倒前者。只看合计会把"搜索访问到的局面分布"
 *    当成"训练批的路由分布", 于是**判不出均衡机制到底有没有生效**。
 *
 * 2. **显示 MaxVio 与"有效专家数", 不显示 max/min**。
 *    `MaxVio = max_i share_i / (1/E) - 1` (0 = 完美均衡, 与 Loss-Free Balancing 论文
 *    同口径); "有效专家" = 份额 > 5% 的专家个数。而 `max/min` 对"最弱专家是 1.1%
 *    还是 2.7%"过于敏感, 读它容易把噪声当趋势。
 *
 * 3. **画一根 1/E 的参考虚线**。
 *    负载这个量没有"越大越好/越小越好", 它的意义完全相对于"完美均衡"那根线。
 *    没有参考线的话, 一排高低不齐的柱子看不出任何东西。
 *
 * 4. **[2026-10] 呼吸灯 = "此刻哪个专家在工作"** (用户口径)。
 *    光晕画在柱子的**后面**, 强度 = 活跃度 × 呼吸 —— 两个因子相乘是有理由的:
 *    只按活跃度则最忙的那个一直最亮 (看不出"在动"), 只按呼吸则所有专家一起闪
 *    (看不出"是谁"); 谷值留 0.35 不归零, 否则"灯灭"会与"这个专家没在干活"
 *    混成同一种视觉。柱顶的小圆点 = 最近**一次**前向选中的 top-k。
 *    ⚠ 这份数据与上面那份**不是同一条路**: 它必须**无锁**读
 *    (整段决策都持着 `m_agentMutex`, 加锁的读数在思考中会一直阻塞 —— 而那正是
 *    唯一想看它的时刻)。见 `ChessBoard::liveMoeRoute` 与 `RL::MoERouteProbe`。
 *
 * 线程与取数
 * ----------
 * 本控件**只接受数据、不取数**。两条来路:
 *   * `setSnapshot` (累计份额): agent 的内部计数器要在 `m_agentMutex` 上读
 *     (后台训练线程可能正在 loadModel), 所以走 MainWindow 那个自检 worker,
 *     算完用队列信号贴回 GUI 线程 (与 selfCheckView 完全同一条路径);
 *   * `setLiveRoute` (呼吸灯): 走 `ChessBoard::liveMoeRoute()`, **不加锁**,
 *     由 MainWindow 一个 66 ms 的定时器直接喂进来。
 * 控件自己不碰 ChessBoard, 也不加锁。
 *
 * 可测性
 * ------
 * 与 CurveChart 同一做法: 每次 setSnapshot() 都把一份**机器可读**的摘要写进
 * AccessibleName/Description, 这样 `tools/verify_*.ps1` (走 Windows UI Automation,
 * 不做像素识别) 能直接断言读数, 不必去点开界面用眼睛看。
 */
class MoeLoadView : public QWidget
{
    Q_OBJECT

public:
    explicit MoeLoadView(QWidget *parent = nullptr);

    /*
       喂一份快照。`snapshot.applicable == false` 表示当前 agent 没有稀疏 MoE 层
       (纯 MLP / AB / MCTS ...), 控件会画成"不适用"而不是画一排 0 —— 后者会被误读成
       "负载是 0"。
    */
    void setSnapshot(const ChessBoard::MoeLoadSnapshot &snapshot);

    /*
       [2026-10] 实时路由 (呼吸高亮的数据源)。**只有把"呼吸高亮"打开时才会被调用**
       (MainWindow 的那个 66 ms 定时器跟着开关启停) —— 关着的时候界面连探针都不读。
       ⚠ 这份数据是**无锁**读来的 (`ChessBoard::liveMoeRoute`), 因为 AI 思考中
       `m_agentMutex` 被整段决策占着 —— 所以它由 MainWindow 用一个**不取锁**的定时器
       按 ~15 Hz 喂进来, 与 `setSnapshot` (走 worker + 锁, 每手棋一次) 是两条路。
    */
    void setLiveRoute(const ChessBoard::MoeLiveRoute &live);
    bool liveActive() const { return m_live.active; }

    /*
       [2026-10 用户口径] **呼吸高亮开关, 默认关**。
       关着的时候这个控件与"加这个功能之前"完全一样 (只画累计柱状图), 而且:
         * 不画高亮、不跑 25 fps 的呼吸动画 (见 syncBreathTimer);
         * **不显示实时读数** —— 标题里没有 `#N`, 末行也没有"此刻 …"。半开着
           (高亮关、序号还在跳) 会更糟: 面板仍然每 66 ms 变一次, 那才叫晃眼。
       所以"关"= 这一整块实时显示都不存在; 开关由界面上的"呼吸高亮"勾选框控制,
       并且**取数定时器也一起停** (没人看的时候不读探针)。
    */
    void setHighlightEnabled(bool on);
    bool highlightEnabled() const { return m_highlight; }

    /* 当前是否显示着一个有稀疏 MoE 的 agent 的快照 */
    bool isApplicable() const { return m_snap.applicable; }

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

    /*
       [2026-10] 放大口径: 弹窗里那一份用这个 (更高、两个方向都可伸展)。
       主面板里那一份是 Fixed 高度 + 96 px 下限 —— 面板只有 ~300 px 宽, 柱状图挤在
       96 px 里只能看个大概; 双击弹窗给的是 560x300 起步。
    */
    void setExpanded(bool on);

signals:
    /* 双击 (界面上用来弹出放大窗口) —— 与 CurveChart::doubleClicked 同一约定 */
    void doubleClicked();

protected:
    void paintEvent(QPaintEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    /* 藏起来就停掉呼吸动画 (没必要为一个看不见的控件每秒重绘 25 次) */
    void hideEvent(QHideEvent *event) override;
    void showEvent(QShowEvent *event) override;

private:
    void updateAccessibility();
    /* 呼吸动画的开关: 只在"有实时数据 + 自己可见"时跑 */
    void syncBreathTimer();
    /* 第 i 个专家"此刻的活跃度" (0..1, 全部专家归一化) —— 呼吸灯的强度 */
    double heatShare(int i) const;
    /* 最近一次前向是否选中了第 i 个专家 (界面上的"就是它"标记) */
    bool pickedNow(int i) const;
    /* "此刻是谁在干活"的文字版 (按热度取前 n 个, "3:41%,7:28%") */
    QString topHeatText(int n) const;

    ChessBoard::MoeLoadSnapshot m_snap;
    ChessBoard::MoeLiveRoute m_live;
    bool m_hasData = false;
    bool m_highlight = false;        /* [2026-10] 呼吸高亮开关 (默认关) */
    QTimer *m_breathTimer = nullptr;
    double m_breathPhase = 0.0;      /* 呼吸相位, [0,1) */
    unsigned m_lastAccSerial = 0;    /* 无障碍摘要的节流 (见 updateAccessibility) */
    bool m_lastAccActive = false;
};

/*
 * ============================================================================
 *  MoeLoadDialog —— 把专家负载放大到独立窗口里看
 * ============================================================================
 * 与 CurveChartDialog 同一套做法: 非模态 (可以一边跑对弈一边看)、WA_DeleteOnClose、
 * 数据由 MainWindow 在每次自检刷新时一起喂进来 (见 applySelfCheckPanel)。
 * 为什么需要它: 主面板那一格只有 ~300x96, 8 个专家 (E=8 那一支) 的柱子在那里
 * 糊成一团; 放大之后柱子、参考虚线、每专家的百分比都读得清。
 */
class MoeLoadDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MoeLoadDialog(QWidget *parent = nullptr);
    void setSnapshot(const ChessBoard::MoeLoadSnapshot &snapshot);
    /* 实时路由 (呼吸高亮) —— 与主面板同一个控件类, 所以放大窗口里也有 */
    void setLiveRoute(const ChessBoard::MoeLiveRoute &live);
    /* 呼吸高亮开关 (同一个, 由 MainWindow 的勾选框一起喂给主面板与放大窗口) */
    void setHighlightEnabled(bool on);

private:
    MoeLoadView *m_view = nullptr;
};

#endif // MOELOADVIEW_H
