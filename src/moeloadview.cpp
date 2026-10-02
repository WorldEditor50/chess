#include "moeloadview.h"

#include <QFontMetrics>
#include <QHideEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPen>
#include <QShowEvent>
#include <QStringList>
#include <QVBoxLayout>

#include "metricsview.h"   /* ChartStyle: 与曲线控件共用的面板配色 */

#include <algorithm>
#include <cmath>
#include <utility>   /* qMakePair */

namespace {

/* ---- 负载的读法 (与 bench_gate_moe / 实验报告同一口径) ---- */

struct LoadStat {
    double maxVio = 0.0;    /* max_i share_i / (1/E) - 1 ; 0 = 完美均衡 */
    double minShare = 0.0;
    int    effective = 0;   /* 份额 > 5% 的专家个数 */
    long long total = 0;
};

LoadStat loadOf(const std::vector<long long> &counts)
{
    LoadStat s;
    const int e = (int)counts.size();
    if (e <= 0) {
        return s;
    }
    for (long long v : counts) {
        s.total += v;
    }
    if (s.total <= 0) {
        s.minShare = 0.0;
        return s;
    }
    double mx = 0.0;
    s.minShare = 1.0;
    for (int i = 0; i < e; i++) {
        const double sh = (double)counts[i] / (double)s.total;
        mx = std::max(mx, sh);
        s.minShare = std::min(s.minShare, sh);
        if (sh > 0.05) {
            s.effective++;
        }
    }
    s.maxVio = mx * (double)e - 1.0;
    return s;
}

const char *kNoDataText = "尚无前向（还没走过子 / 还没训练过）";

/* ---- 呼吸灯 ---- */

/* 呼吸周期 (秒)。1.6 s ≈ 37 次/分钟 —— 比心跳慢一点, 看着不烦、又能一眼看出"在动" */
constexpr double kBreathPeriodSec = 1.6;
/* 动画帧间隔: 与 chessboard.cpp 的"思考中"动画同一口径 (40 ms = 25 fps) */
constexpr int kBreathIntervalMs = 40;

/* 呼吸灯配色: 暖橙 (与"超额专家"的 cOver 同一色系, 但更亮, 免得两种读数分不清) */
const QColor kGlowCore(255, 170, 60);
const QColor kGlowEdge(255, 120, 30);

} /* namespace */

MoeLoadView::MoeLoadView(QWidget *parent)
    : QWidget(parent)
{
    /*
       高度给足: 标题 + 柱状图 + 两行读数。
       **不填自身背景** (setAutoFillBackground 保持默认 false) —— 与 CurveChart 一致:
       只画一块 ChartStyle 的圆角半透明板, 四角露父级底色。第一版这里是
       `setAutoFillBackground(true)` + 一个不透明的 (250,250,252) 调色板, 于是它和
       旁边的 loss/reward 曲线看着是两种东西 (用户报的"背景颜色不一致")。
    */
    setMinimumHeight(96);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    setToolTip(QStringLiteral(
        "稀疏 MoE 的每个专家分到多少流量。（**双击可放大到独立窗口**）\n"
        "两根柱子的高度意义完全相对于虚线（完美均衡 = 1/E）：\n"
        "  MaxVio = max_share/(1/E) − 1，0 就是完美均衡。\n"
        "训练侧与推理侧**分开**画：生命周期累计里混着 MCTS 的叶子估值前向，"
        "而它通常压倒训练前向 —— 只看合计判不出均衡机制有没有生效。\n\n"
        "**呼吸高亮的整根条柱 = 此刻正在干活的专家**（整列都亮，柱身再亮一层；"
        "亮度按最近约 1 秒的前向强度，越忙越亮、一起呼吸）；"
        "柱顶的亮块 = 最近**一次**前向选中的那 top-k 个；"
        "标题里的 #N 是前向序号（一直在跳就说明读数在走）。\n"
        "⚠ 这一块由中间那一列的「**呼吸高亮**」勾选框控制，**默认关**（关着时这里与"
        "没有这个功能时完全一样，连探针都不读）。\n"
        "柱子的**高度与底色仍然只表示累计份额**（蓝色=正常、橙红=超额），呼吸高亮是叠在"
        "上面的一层，两个口径不混。\n"
        "这一部分是**无锁**实时读数，所以 AI 思考中也在动 —— 而累计份额只在选中 agent / "
        "每手棋的探索+预训练之后刷新一次（没走预训练时累计会停在旧读数，此时高亮仍在）。\n\n"
        "要看到读数需要三个前提（缺一个都会显示成说明文字）：\n"
        "  1. 选一个**带稀疏 MoE** 的 agent（选单里带“稀疏MoE”字样的那几个）；\n"
        "  2. 等它把常驻网络建出来（SAC+AZ-MoE 是第一次决策时才建网的）；\n"
        "  3. 让它走一步（没有前向就没有负载可统计）。\n\n"
        "与其它自检读数一样：这是**结构与口径**事实，不是棋力。"));
    /*
       呼吸动画: 只在"有实时数据 + 自己可见"时跑 (见 syncBreathTimer)。
       相位在这里推进而不是用 QElapsedTimer 读时钟 —— 首帧从 0 开始, 于是两次刷新之间
       相位是连续的, 不会因为回调抖动而"跳一下"。
    */
    m_breathTimer = new QTimer(this);
    m_breathTimer->setInterval(kBreathIntervalMs);
    connect(m_breathTimer, &QTimer::timeout, this, [this]() {
        m_breathPhase += (double)kBreathIntervalMs / 1000.0 / kBreathPeriodSec;
        if (m_breathPhase >= 1.0) {
            m_breathPhase -= 1.0;
        }
        if (isVisible()) {
            update();
        }
    });
    updateAccessibility();
}

void MoeLoadView::setSnapshot(const ChessBoard::MoeLoadSnapshot &snapshot)
{
    m_snap = snapshot;
    m_hasData = true;
    /*
       快照换了 agent (或换了实例) -> 上一份实时读数就不再属于这个控件了。
       `experts` 不一致就丢掉它: 否则会在 B 的柱状图上画 A 的呼吸灯 (专家序号还一样,
       肉眼看不出错) —— 这类"残留读数"是本工程反复栽的坑。
    */
    if (m_live.experts != m_snap.experts) {
        m_live = ChessBoard::MoeLiveRoute();
        m_lastAccSerial = 0;
        m_lastAccActive = false;
    }
    syncBreathTimer();
    updateAccessibility();
    update();
}

void MoeLoadView::setLiveRoute(const ChessBoard::MoeLiveRoute &live)
{
    /*
       开关关着时**直接忽略**实时数据: 界面不该出现"关了但还在动"的状态。
       上半开 (高亮关、序号还在跳) 比全关更糟 —— 面板仍会每 66 ms 变一次。
    */
    if (!m_highlight) {
        return;
    }
    m_live = live;
    /*
       只在**读数真的变了**的时候刷新无障碍摘要: 这个函数每秒被调 ~15 次, 每次都
       setAccessibleName 会往 UIA 灌事件 (而 Name 是验证脚本的读法, 见
       updateAccessibility 的说明)。节流到"活跃状态翻转"或"前向序号跨过 32"。
    */
    if (live.active != m_lastAccActive
        || (live.serial >= m_lastAccSerial + 32u)
        || (live.serial < m_lastAccSerial)) {
        updateAccessibility();
    }
    syncBreathTimer();
    update();
}

bool MoeLoadView::pickedNow(int i) const
{
    if (!m_live.available || m_live.pickedCount <= 0) {
        return false;
    }
    for (int k = 0; k < m_live.pickedCount && k < ChessBoard::MoeLiveRoute::kMaxPicked; k++) {
        if (m_live.picked[k] == i) {
            return true;
        }
    }
    return false;
}

double MoeLoadView::heatShare(int i) const
{
    if (!m_live.available || m_live.heatTotal <= 0.0f) {
        return 0.0;
    }
    if (i < 0 || i >= m_live.experts || i >= ChessBoard::MoeLiveRoute::kMaxExperts) {
        return 0.0;
    }
    return (double)m_live.heat[i] / (double)m_live.heatTotal;
}

void MoeLoadView::setHighlightEnabled(bool on)
{
    if (m_highlight == on) {
        return;
    }
    m_highlight = on;
    if (!on) {
        /*
           关掉时**清空实时数据**: 否则最后一帧的高亮会"冻"在面板上 (没有再来的
           setLiveRoute 去覆盖它), 而那个状态看起来正好像"某个专家一直在满负荷工作"
           —— 一个关掉的功能留下一个像读数的东西, 是本工程反复栽的坑。
        */
        m_live = ChessBoard::MoeLiveRoute();
        m_lastAccSerial = 0;
        m_lastAccActive = false;
    }
    syncBreathTimer();
    updateAccessibility();
    update();
}

void MoeLoadView::syncBreathTimer()
{
    if (m_breathTimer == nullptr) {
        return;
    }
    const bool want = m_highlight && m_live.active && m_live.available && m_snap.applicable
                      && m_live.experts == m_snap.experts && isVisible();
    if (want == m_breathTimer->isActive()) {
        return;
    }
    if (want) {
        m_breathTimer->start();
    } else {
        m_breathTimer->stop();
    }
}

void MoeLoadView::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    syncBreathTimer();      /* 藏起来就停动画 */
}

void MoeLoadView::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    syncBreathTimer();      /* 重新出现: 数据还在的话接着呼吸 */
}

QSize MoeLoadView::sizeHint() const
{
    return QSize(320, 104);
}

QSize MoeLoadView::minimumSizeHint() const
{
    return QSize(190, 96);
}

void MoeLoadView::setExpanded(bool on)
{
    if (on) {
        setMinimumHeight(240);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setMinimumWidth(360);
    } else {
        setMinimumHeight(96);
        setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        setMinimumWidth(0);
    }
    updateGeometry();
    update();
}

void MoeLoadView::mouseDoubleClickEvent(QMouseEvent *event)
{
    Q_UNUSED(event);
    emit doubleClicked();
}

MoeLoadDialog::MoeLoadDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("稀疏 MoE 专家负载 (放大)"));
    setAttribute(Qt::WA_DeleteOnClose);
    /* 非模态: 与 CurveChartDialog 同一约定 —— 可以一边跑对弈一边看 */
    setModal(false);
    resize(620, 380);

    auto *lay = new QVBoxLayout(this);
    auto *hint = new QLabel(QStringLiteral(
        "柱高 = 该专家分到的流量份额；虚线 = 完美均衡 1/E；MaxVio = max_share/(1/E) − 1。\n"
        "训练侧与推理侧分开看：偏置/辅助损失只作用在**训练批**上，"
        "而 MCTS 的叶子估值前向在数量上通常压倒训练前向。\n"
        "**呼吸高亮的整根条柱 = 此刻正在干活的专家**（整列都亮，柱身再亮一层；"
        "最近约 1 秒的前向强度，越忙越亮）；柱顶的亮块 = 最近**一次**前向选中的那 top-k 个；"
        "标题里的 #N 是前向序号。\n"
        "⚠ 由中间那一列的「**呼吸高亮**」勾选框控制，**默认关**；关着时这一份与没有这个"
        "功能时完全一样（不读探针、不跑动画、标题里没有 #N）。\n"
        "呼吸高亮是叠在柱子**上面**的一层，柱高/柱色仍然只读作累计份额。\n"
        "这一份是**无锁**实时读数，所以 AI 思考中也在动 —— 而累计份额"
        "只在选中 agent / 每手棋的探索+预训练之后刷新一次。"
        "停止计算约 3 秒后高亮自己灭（热度按时间衰减）。"), this);
    hint->setWordWrap(true);
    lay->addWidget(hint);

    m_view = new MoeLoadView(this);
    m_view->setExpanded(true);
    lay->addWidget(m_view, 1);
}

void MoeLoadDialog::setSnapshot(const ChessBoard::MoeLoadSnapshot &snapshot)
{
    if (m_view != nullptr) {
        m_view->setSnapshot(snapshot);
    }
}

void MoeLoadDialog::setLiveRoute(const ChessBoard::MoeLiveRoute &live)
{
    if (m_view != nullptr) {
        m_view->setLiveRoute(live);
    }
}

void MoeLoadDialog::setHighlightEnabled(bool on)
{
    if (m_view != nullptr) {
        m_view->setHighlightEnabled(on);
    }
}

void MoeLoadView::updateAccessibility()
{
    /*
       与 CurveChart 同一做法: 把状态写成**机器可读**的摘要, 让 tools/verify_*.ps1
       (Windows UI Automation, 不做像素识别) 能直接断言读数。

       ⚠ 摘要在 **accessibleName** 里, 不是 accessibleDescription —— 这是实测出来的:
       本工程用的是 Qt 6.9 的 Windows UIA 桥, 而 `QWidget::accessibleDescription` 走
       UIA 的 HelpText 属性时**读出来是空串** (tools/verify_moe_load_view.ps1 第一版
       就是按 HelpText 读的, 当场失败: Name='MoE专家负载' 而 HelpText='').
       证据不止这一条: tools/ 下**所有**既有验证脚本读的都是 NameProperty, 一个读
       HelpText 的都没有 —— 也就是说这个工程里"能被脚本读到"的通道一直是 Name。
       所以: Name = 中文标题 + " | " + 摘要 (照着 CurveChart 把状态写进 Name 的写法);
       Description 仍然照写一份 (语义正确, 供读屏软件用), 但**不依赖**它。
       格式刻意保持简单稳定 (key=value, 空格分隔), 免得脚本要写正则去啃中文。
    */
    QString title = QStringLiteral("MoE专家负载");
    QString d;
    if (!m_hasData) {
        d = QStringLiteral("state=nodata");
    } else if (!m_snap.applicable && m_snap.capable) {
        /* 类型对, 但常驻实例还没建出来 (SAC+AZ-MoE 是懒建网的) —— 这与"这个 agent
           根本没有 MoE 读数"是两件事, 必须分开报 (实测: 刚选中那一刻就读到过它) */
        d = QStringLiteral("state=noinstance experts=0");
    } else if (!m_snap.applicable) {
        d = QStringLiteral("state=na agents=0");
    } else {
        d = QStringLiteral("state=%1 experts=%2 topk=%3")
                .arg(m_snap.splitReady ? QStringLiteral("ok") : QStringLiteral("totalonly"))
                .arg(m_snap.experts)
                .arg(m_snap.topK);
        auto append = [&d](const char *tag, const std::vector<long long> &v) {
            const LoadStat s = loadOf(v);
            if (s.total <= 0) {
                d += QStringLiteral(" %1=noforward").arg(QLatin1String(tag));
                return;
            }
            d += QStringLiteral(" %1_maxvio=%2 %1_eff=%3 %1_total=%4")
                     .arg(QLatin1String(tag))
                     .arg(s.maxVio, 0, 'f', 3)
                     .arg(s.effective)
                     .arg(s.total);
        };
        if (m_snap.splitReady) {
            append("train", m_snap.train);
            append("infer", m_snap.infer);
        } else {
            append("total", m_snap.total);
        }
        /*
           [2026-10] 实时路由 (呼吸高亮) 也写进摘要。为什么值得写: 这条读数**只在
           "AI 正在思考"时才有意义", 而验证脚本没法靠等去撞那个瞬间 —— 有了
           live_serial 至少能断言"探针在走"、"活跃时 live=on"。
           ⚠ **开关的状态无条件写进去** (`hl=on|off`): 否则"关着"与"探针没数据"
           在摘要里长得一模一样, 脚本没法断言"默认关闭"这条约定。
        */
        d += QStringLiteral(" hl=%1").arg(m_highlight ? QStringLiteral("on")
                                                       : QStringLiteral("off"));
        if (m_highlight && m_live.available) {
            d += QStringLiteral(" live=%1 live_serial=%2 live_picked=%3")
                     .arg(m_live.active ? QStringLiteral("on") : QStringLiteral("off"))
                     .arg(m_live.serial)
                     .arg(m_live.pickedCount);
            if (m_live.active && m_live.heatTotal > 0.0f) {
                d += QStringLiteral(" live_top=%1").arg(topHeatText(2));
            }
        }
    }
    setAccessibleName(title + QStringLiteral(" | ") + d);
    setAccessibleDescription(d);
    m_lastAccSerial = m_live.serial;
    m_lastAccActive = m_live.active;
}

/*
 * topHeatText —— "此刻是谁在干活"的文字版 (按热度排序取前 n 个)。
 * 只在有实时数据时调用; 单位是**份额百分比**, 与柱状图同一口径 (所以两者能互相核对)。
 */
QString MoeLoadView::topHeatText(int n) const
{
    if (!m_live.available || m_live.heatTotal <= 0.0f || n <= 0) {
        return QStringLiteral("none");
    }
    QVector<QPair<double, int>> order;
    for (int i = 0; i < m_live.experts && i < ChessBoard::MoeLiveRoute::kMaxExperts; i++) {
        const double hs = heatShare(i);
        if (hs > 0.0) {
            order.append(qMakePair(hs, i));
        }
    }
    std::sort(order.begin(), order.end(), [](const QPair<double, int> &a,
                                             const QPair<double, int> &b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;      /* 同份额按下标 —— 让输出可复现 */
    });
    QStringList parts;
    for (int k = 0; k < order.size() && k < n; k++) {
        parts << QStringLiteral("%1:%2%")
                     .arg(order[k].second)
                     .arg(order[k].first * 100.0, 0, 'f', 0);
    }
    return parts.isEmpty() ? QStringLiteral("none") : parts.join(QStringLiteral(","));
}

void MoeLoadView::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const int pad = 6;
    const int w = width();
    const int h = height();

    /* 底板: 与 CurveChart 用**同一份**配色 (ChartStyle), 半径与调整量也逐字相同 ——
       "和 loss 窗口背景一致"这件事只有一个实现来源, 不靠两边各自抄颜色 */
    {
        const QRectF panel = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
        p.setBrush(ChartStyle::kPanelFill);
        p.setPen(QPen(ChartStyle::kPanelEdge, 1.0));
        p.drawRoundedRect(panel, 8.0, 8.0);
    }

    QFont f = font();
    if (f.pointSizeF() > 0) {
        f.setPointSizeF(std::max(7.0, f.pointSizeF() - 1.0));
    }
    p.setFont(f);
    const QFontMetrics fm(f);
    const int lineH = fm.height();

    const QColor cText(60, 60, 60);
    const QColor cDim(130, 130, 130);
    const QColor cBar(70, 130, 180);
    const QColor cOver(198, 92, 60);
    const QColor cRef(150, 150, 150);
    const QColor cEmpty(210, 210, 210);

    int y = pad;

    /* ---------- 标题 ---------- */
    if (!m_hasData) {
        p.setPen(cDim);
        p.drawText(QRect(pad, y, w - 2 * pad, lineH), Qt::AlignLeft | Qt::AlignVCenter,
                   QStringLiteral("稀疏 MoE 专家负载：等待读数…"));
        return;
    }
    if (!m_snap.applicable) {
        p.setPen(cText);
        p.drawText(QRect(pad, y, w - 2 * pad, lineH), Qt::AlignLeft | Qt::AlignVCenter,
                   QStringLiteral("稀疏 MoE 专家负载"));
        y += lineH;
        p.setPen(cDim);
        const QString who = m_snap.agentName.empty()
                                ? QStringLiteral("当前 agent")
                                : QString::fromStdString(m_snap.agentName);
        if (m_snap.capable) {
            /*
               类型对、实例还没建 —— SAC+AZ-MoE 的常驻网络是**懒建**的 (第一次决策/
               预热时才建网)。实测: 在界面上刚选中它那一刻, 这里读到过这个状态。
               说清楚"要先让它走一步", 否则用户会以为选错了 agent。
            */
            p.drawText(QRect(pad, y, w - 2 * pad, lineH * 3),
                       Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap,
                       QStringLiteral("「%1」有稀疏 MoE，但实例还没建出来 ——\n"
                                      "它的网络是第一次决策时才建的。让它先走一步\n"
                                      "（或勾上“走子前先探索训练”跑一次），这里就会出现负载读数。")
                           .arg(who));
        } else {
            /*
               文案刻意**短**: 原来那段解释"本控件覆盖哪些 agent / 为什么 PPO 不报"
               的文字有五行, 挤在 312x96 里被截断, 读起来像"这个控件坏了/空着" ——
               用户就是据此报的"负载均衡状态控件并没有显示"。
               现在第一行直接说"你选的 agent 名字 + 没有稀疏 MoE", 细节挪进 tooltip。
            */
            p.drawText(QRect(pad, y, w - 2 * pad, lineH * 2),
                       Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap,
                       QStringLiteral("「%1」没有稀疏 MoE —— 这一栏对它不适用。\n"
                                      "换一个带 MoE 的 agent（选单里带“稀疏MoE”字样的那几个）即可。")
                           .arg(who));
        }
        return;
    }

    p.setPen(cText);
    {
        /*
           标题行: 累计口径 + **实时序号**。#12345 是"前向序号", 每次 forward +1 ——
           它一直在跳就说明探针在走 (而不是"灯卡住了")。空闲时明确写"停", 免得用户
           盯着一个不亮的灯猜是不是坏了。
        */
        QString head = QStringLiteral("稀疏 MoE 专家负载 · E=%1 top-%2%3")
                           .arg(m_snap.experts)
                           .arg(m_snap.topK)
                           .arg(m_snap.splitReady ? QString()
                                                  : QStringLiteral(" · 合计（未拆分）"));
        if (m_live.available) {
            head += m_live.active
                        ? QStringLiteral(" · 实时 #%1").arg(m_live.serial)
                        : QStringLiteral(" · 实时 停");
        }
        p.drawText(QRect(pad, y, w - 2 * pad, lineH), Qt::AlignLeft | Qt::AlignVCenter, head);
    }
    y += lineH;

    /*
       主图用**训练侧** (splitReady 时): 均衡机制 (辅助损失 / 偏置) 作用在训练批上,
       所以"它有没有生效"只能看这一侧; 搜索侧顺手报在读数行里。
    */
    const std::vector<long long> primary = m_snap.splitReady ? m_snap.train : m_snap.total;
    const int E = (int)primary.size();
    const LoadStat ps = loadOf(primary);

    const int bottomTextH = 2 * lineH;
    int chartTop = y;
    int chartH = h - pad - bottomTextH - chartTop;
    if (chartH < 18) {
        chartH = 18;
    }
    const int chartBottom = chartTop + chartH;

    /*
       ---- 先决定"有没有柱子可画"与"有没有实时数据可点灯" ----
       两者**互相独立**: 累计计数可能是空的 (这一场没走"探索+预训练" -> 快照停在
       `noforward`, 而它是走锁的、只在选中 agent / 每手棋的探索之后才刷新), 而实时路由
       一直在动。第一版把"累计为空"当成"什么都没有"直接 return, 于是**恰好用户最可能
       看到的那个场景里, 呼吸灯一个字都画不出来** —— 像素采样当场抓到 (整块控件里
       暖色像素 0 个)。
    */
    const bool liveOk = m_highlight && m_live.available && m_live.active && E > 0
                        && m_live.experts == E
                        && E <= ChessBoard::MoeLiveRoute::kMaxExperts;
    const bool hasBars = (E > 0) && (ps.total > 0);

    if (E <= 0 || (!hasBars && !liveOk)) {
        p.setPen(cDim);
        p.drawText(QRect(pad, chartTop, w - 2 * pad, chartH), Qt::AlignCenter,
                   QLatin1String(kNoDataText));
        /* 说明"为什么还没有数" —— 只写"尚无前向"用户不知道下一步该做什么 */
        p.drawText(QRect(pad, chartTop + chartH, w - 2 * pad, lineH),
                   Qt::AlignHCenter | Qt::AlignTop,
                   QStringLiteral("让它先走一步（每步搜索/训练都会产生前向）"));
        return;
    }

    const double ref = 1.0 / (double)E;
    double shareMax = 0.0;
    QVector<double> share(E, 0.0);
    for (int i = 0; i < E; i++) {
        share[i] = hasBars ? (double)primary[i] / (double)ps.total : 0.0;
        shareMax = std::max(shareMax, share[i]);
    }
    /* 纵轴上限: 至少到 2/E, 保证"完美均衡"那根参考线永远看得见 */
    const double top = std::max(shareMax, 2.0 * ref);

    /* ---------- 参考虚线 (完美均衡 = 1/E): 只有真的画了柱子才有意义 ---------- */
    if (hasBars) {
        const double refY = (double)chartBottom - (ref / top) * (double)chartH;
        QPen pen(cRef);
        pen.setStyle(Qt::DashLine);
        pen.setWidth(1);
        p.setPen(pen);
        p.drawLine(pad, (int)std::lround(refY), w - pad, (int)std::lround(refY));
    }

    /*
       ================================================================
       呼吸高亮: 直接画在柱子上 (用户口径: "直接呼吸高亮 MOE 负载专家的柱状图")
       ================================================================
       几何上把"发光"与"柱子高度"解耦: 每一列都有一个**整列矩形** `colRect`, 因为
       累计计数可能是空的 (界面上这一场没走"走子前先探索+预训练"时就是这样: 快照停在
       `noforward`, 而实时路由一直在动), 那时没有柱子可高亮, 但**必须还看得见谁在工作**
       —— 那正是这个控件存在的理由。
    */
    const int gap = (E > 1) ? std::max(2, std::min(6, (w - 2 * pad) / (E * 6))) : 0;
    const int barW = std::max(2, (w - 2 * pad - gap * (E - 1)) / E);
    QVector<QRect> colRect(E);
    QVector<QRect> barRect(E);
    for (int i = 0, x = pad; i < E; i++, x += barW + gap) {
        colRect[i] = QRect(x, chartTop, barW, chartH);
        const int barH = hasBars ? (int)std::lround((share[i] / top) * (double)chartH) : 0;
        barRect[i] = QRect(x, chartBottom - barH, barW, barH);
    }

    constexpr double kTwoPi = 6.283185307179586;
    const double breath = 0.5 + 0.5 * std::sin(kTwoPi * m_breathPhase);

    if (!hasBars && liveOk) {
        /* 累计没有数、但实时有: 在图表区写一句"这里为什么没有柱子", 免得被读成坏了 */
        p.setPen(cDim);
        p.drawText(QRect(pad, chartTop, w - 2 * pad, lineH), Qt::AlignHCenter | Qt::AlignTop,
                   QStringLiteral("累计计数还没刷新（本场没有走“探索+预训练”）—— 下面是实时路由"));
    }

    for (int i = 0; i < E; i++) {
        if (!hasBars) {
            break;
        }
        const QRect &r = barRect[i];
        const int x = r.left();
        if (primary[i] <= 0) {
            /* 一次都没被选中 —— 画空心框, 与"份额很小"区分开 */
            p.setPen(QPen(cEmpty, 1));
            p.setBrush(Qt::NoBrush);
            p.drawRect(QRect(x, chartTop, barW, chartH - 1));
        } else {
            p.setPen(Qt::NoPen);
            p.setBrush(share[i] > 2.0 * ref ? cOver : cBar);
            p.drawRect(r);
        }
        /* 专家序号 + 百分比 (柱子太窄就不写, 免得糊成一团) */
        if (barW >= 16) {
            /*
               正在干活的那几个: 序号用暖色 (与呼吸灯同色系), 其余保持灰 —— 这样
               即使柱子很矮/光柱很淡, 也还能从文字上认出"是谁"。
            */
            p.setPen(liveOk && heatShare(i) > 0.02 ? kGlowEdge : cDim);
            p.drawText(QRect(x, chartBottom + 1, barW, lineH),
                       Qt::AlignHCenter | Qt::AlignTop,
                       QStringLiteral("%1").arg(i));
        }
    }

    if (!hasBars && liveOk && barW >= 16) {
        /* 没有柱子的时候序号更要写 —— 它旁边就是那根正在发光的光柱 */
        p.setPen(cDim);
        for (int i = 0; i < E; i++) {
            p.setPen(liveOk && heatShare(i) > 0.02 ? kGlowEdge : cDim);
            p.drawText(QRect(colRect[i].left(), chartBottom + 1, barW, lineH),
                       Qt::AlignHCenter | Qt::AlignTop,
                       QStringLiteral("%1").arg(i));
        }
    }

    /* ---------- 呼吸高亮 (画在柱子**之后**, 直接叠在柱子上) ---------- */
    if (liveOk) {
        for (int i = 0; i < E; i++) {
            const double hs = heatShare(i);
            /*
               判据是"有热度 **或** 被最近那一次选中": 后者是瞬时事实, 不能因为它在
               热度上占比很小就被跳过 (那两个读数是分开取的, 会有一帧的时间差)。
            */
            const bool picked = pickedNow(i);
            if (hs <= 0.005 && !picked) {
                continue;
            }
            /*
               强度 = **活跃度 × 呼吸**。为什么是两个因子相乘, 而不是任取一个:
                 * 只按活跃度 -> 最忙的那个一直最亮, 看不出"在动";
                 * 只按呼吸   -> 全部专家一起闪, 看不出"是谁在干活"。
               乘起来才是要的效果: 谁在干活谁亮, 而且亮的那几个**一起呼吸**。
               呼吸的谷值留 0.35 (不是 0): 灯不该完全灭掉, 否则会与"这个专家没在干活"
               混成同一种视觉 (那正是本控件要区分的两件事)。
            */
            const double k = hs * (0.35 + 0.65 * breath);

            /*
               (a) **正题: 高亮整根条柱** —— 先铺一层**整列**光柱 (chartTop..chartBottom),
               再把柱身染得更亮一层。
               为什么"整列"而不是只染柱身那一段 (用户口径: "高亮的时候, 高亮整个条柱"):
               柱高表示的是**累计份额**, 一个刚被选中、累计份额还很小的专家柱子只有几像素,
               只染柱身的话高亮就只剩那一小截, 看着像"没高亮"。铺满整列之后, "这一列此刻
               在工作"这件事与柱高无关, 一眼就能看见。
               为什么柱身还要再叠一层: 整列底色是"这一列活跃", 柱身更亮才是"这根柱子在干活"
               —— 两层叠起来, 既有整列的范围感, 又保留柱子的实体感。
               ⚠ 两层都必须画在柱子**上面**: 第一版把光晕画在柱子后面, 可见部分只剩柱子
               外沿 1~2 px, 等于看不见 (真界面上量过: 整块控件里只有 257 个像素带暖色)。
            */
            QColor band = kGlowCore;
            band.setAlpha((int)std::lround(std::min(135.0, 20.0 + 130.0 * k)));
            p.setPen(Qt::NoPen);
            p.setBrush(band);
            p.drawRoundedRect(colRect[i], 3.0, 3.0);

            if (hasBars && primary[i] > 0 && barRect[i].height() > 0) {
                QColor wash = kGlowCore;
                wash.setAlpha((int)std::lround(std::min(185.0, 55.0 + 190.0 * k)));
                p.setBrush(wash);
                p.drawRect(barRect[i]);
            }

            /* (b) 柱顶/列顶的"灯头": 亮度最高的一块, 标明最近**一次**就是它 */
            if (picked) {
                QColor cap = kGlowEdge;
                cap.setAlpha((int)std::lround(std::min(250.0, 120.0 + 300.0 * k)));
                const int capTop = (hasBars && barRect[i].height() > 0)
                                       ? std::max(chartTop, barRect[i].top() - 3)
                                       : (chartTop + 2);
                p.setBrush(cap);
                p.drawRoundedRect(QRect(colRect[i].left(), capTop, barW, 6), 2.5, 2.5);
            }
        }
    }

    /* ---------- 读数行 ---------- */
    int ty = chartBottom + lineH;
    p.setPen(cText);
    QString l1;
    if (!hasBars) {
        /*
           累计为空、但实时有数据。**不印那一排 0**: "训练侧 MaxVio 0.000 · 有效 0/8"
           会被读成"负载是 0", 而真相是"这份快照还没刷新过"(它走锁, 只在选中 agent /
           每手棋的探索+预训练之后才取一次)。
        */
        l1 = QStringLiteral("累计尚无前向（快照走锁，本场未触发刷新）· 下面是实时路由");
    } else if (m_snap.splitReady) {
        const LoadStat is = loadOf(m_snap.infer);
        l1 = QStringLiteral("训练侧 MaxVio %1 · 有效 %2/%3 · 最小 %4%")
                 .arg(ps.maxVio, 0, 'f', 3)
                 .arg(ps.effective)
                 .arg(E)
                 .arg(ps.minShare * 100.0, 0, 'f', 1);
        if (is.total > 0) {
            l1 += QStringLiteral("   |   推理侧 %1 · 有效 %2/%3")
                      .arg(is.maxVio, 0, 'f', 3)
                      .arg(is.effective)
                      .arg(E);
        }
    } else {
        l1 = QStringLiteral("合计 MaxVio %1 · 有效 %2/%3 · 最小 %4%（该支未接线训练/推理拆分）")
                 .arg(ps.maxVio, 0, 'f', 3)
                 .arg(ps.effective)
                 .arg(E)
                 .arg(ps.minShare * 100.0, 0, 'f', 1);
    }
    p.drawText(QRect(pad, ty, w - 2 * pad, lineH), Qt::AlignLeft | Qt::AlignVCenter, l1);

    ty += lineH;
    p.setPen(cDim);
    {
        QString l2;
        if (!hasBars) {
            l2 = QStringLiteral("光柱 = 最近约 1 秒的前向强度（越忙越亮）· 灯头 = 最近一次选中的专家");
        } else {
            l2 = QStringLiteral("前向 训练 %1 / 推理 %2 · MaxVio 0 = 完美均衡")
                     .arg(ps.total)
                     .arg([&] {
                         long long t = 0;
                         for (qint64 v : m_snap.infer) { t += v; }
                         return t;
                     }());
        }
        /*
           "此刻是谁在干活"的文字版。**放得下才写**: 主面板只有 ~312 px 宽, 硬塞会把
           这一行截断成看不懂的半句 (用户已经报过一次"文字看不清")。放不下时靠呼吸灯
           与序号颜色表达同一件事 —— 读数不丢, 只是这一行不重复它。
        */
        if (liveOk && m_live.heatTotal > 0.0f) {
            const QString extra = QStringLiteral(" · 此刻 %1").arg(topHeatText(2));
            if (fm.horizontalAdvance(l2 + extra) <= w - 2 * pad) {
                l2 += extra;
            }
        }
        p.drawText(QRect(pad, ty, w - 2 * pad, lineH), Qt::AlignLeft | Qt::AlignVCenter, l2);
    }
}
