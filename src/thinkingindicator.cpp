#include "thinkingindicator.h"

#include <QFontInfo>
#include <QFontMetrics>
#include <QPainter>
#include <QPolygonF>
#include <QRadialGradient>
#include <QtMath>

#include <cmath>

namespace {

/* 30 fps: 够顺滑, 又不至于让 CPU 空转 (每帧只重画一个 200x200 的小控件) */
constexpr int kFrameIntervalMs = 33;
constexpr qreal kTwoPi = 6.283185307179586;

/* 动画尺寸 */
constexpr qreal kDialCy = 76.0;    /* 动画圆心 y (控件内坐标) */
constexpr qreal kHaloR = 56.0;     /* 呼吸光晕半径 */
constexpr qreal kRingR = 43.0;     /* 粒子环半径 */
constexpr qreal kSandHalfW = 17.0; /* 沙漏半宽 */
constexpr qreal kSandHalfH = 21.0; /* 沙漏半高 */
constexpr int kParticleCount = 10;

/* 配色 (与 appstyle.qss 的 #fef9e3 米色背景协调) */
const QColor kAccent(47, 127, 214);      /* 思考中: 蓝 */
const QColor kDone(46, 125, 50);         /* 已完成: 绿 */
const QColor kIdleText(150, 140, 110);   /* 未开始: 灰褐 */
const QColor kGlass(150, 140, 110, 210);
const QColor kSand(232, 163, 61, 235);
const QColor kPanelFill(255, 255, 255, 170);
const QColor kPanelEdge(224, 216, 189);

/* 耗时格式化: <1 分钟 -> "12.34 s", 否则 -> "1:05.20" */
QString formatElapsed(long long ms)
{
    if (ms < 0) {
        ms = 0;
    }
    const long long totalSec = ms / 1000;
    const int centi = static_cast<int>((ms % 1000) / 10);
    const long long minutes = totalSec / 60;
    const long long seconds = totalSec % 60;
    if (minutes > 0) {
        return QStringLiteral("%1:%2.%3")
            .arg(minutes)
            .arg(seconds, 2, 10, QChar('0'))
            .arg(centi, 2, 10, QChar('0'));
    }
    return QStringLiteral("%1.%2 s")
        .arg(seconds)
        .arg(centi, 2, 10, QChar('0'));
}

} // namespace

ThinkingIndicator::ThinkingIndicator(QWidget *parent)
    : QWidget(parent)
{
    setMinimumSize(170, 186);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

    m_timer.setInterval(kFrameIntervalMs);
    connect(&m_timer, &QTimer::timeout, this, &ThinkingIndicator::tick);
}

QSize ThinkingIndicator::sizeHint() const
{
    return QSize(200, 200);
}

QSize ThinkingIndicator::minimumSizeHint() const
{
    return QSize(170, 186);
}

void ThinkingIndicator::start(const QString &agentName, const QString &stage)
{
    m_agent = agentName;
    m_stage = stage;
    m_finishedMs = -1;
    m_phase = 0.0;
    m_running = true;
    m_everStarted = true;
    m_clock.start();
    m_timer.start();
    emit elapsedChanged(0);
    update();
}

void ThinkingIndicator::setStage(const QString &stage)
{
    if (m_stage == stage) {
        return;
    }
    m_stage = stage;
    update();
}

void ThinkingIndicator::stop()
{
    if (!m_running) {
        return;
    }
    m_finishedMs = m_clock.isValid() ? m_clock.elapsed() : 0;
    m_running = false;
    m_timer.stop();
    update();
}

void ThinkingIndicator::resetToIdle()
{
    m_timer.stop();
    m_running = false;
    m_everStarted = false;
    m_finishedMs = -1;
    m_phase = 0.0;
    m_agent.clear();
    m_stage.clear();
    update();
}

long long ThinkingIndicator::elapsedMs() const
{
    if (m_running) {
        return m_clock.isValid() ? m_clock.elapsed() : 0;
    }
    return m_finishedMs < 0 ? 0 : m_finishedMs;
}

void ThinkingIndicator::tick()
{
    m_phase += kFrameIntervalMs / 1000.0;
    if (m_phase >= 1.0) {
        m_phase -= 1.0;
    }
    emit elapsedChanged(m_clock.isValid() ? m_clock.elapsed() : 0);
    update();
}

void ThinkingIndicator::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);

    const qreal w = width();
    const qreal h = height();
    const QPointF c(w / 2.0, kDialCy);

    /* ---- 卡片底 ---- */
    const QRectF panel = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
    QColor edge = kPanelEdge;
    if (m_running) {
        /* 边框也参与呼吸: 与光晕同相位 */
        const qreal k = 0.5 + 0.5 * std::sin(kTwoPi * m_phase);
        edge = QColor(kAccent.red(), kAccent.green(), kAccent.blue(),
                      static_cast<int>(70 + 120 * k));
    }
    p.setBrush(kPanelFill);
    p.setPen(QPen(edge, m_running ? 1.6 : 1.0));
    p.drawRoundedRect(panel, 10.0, 10.0);

    /* ---- 标题 ---- */
    QString title;
    QColor titleColor;
    if (m_running) {
        title = QStringLiteral("AI 正在思考…");
        titleColor = QColor(31, 78, 121);
    } else if (m_everStarted) {
        title = QStringLiteral("AI 思考完成");
        titleColor = kDone;
    } else {
        title = QStringLiteral("等待 AI 思考");
        titleColor = kIdleText;
    }
    QFont titleFont = font();
    titleFont.setPointSize(9);
    titleFont.setBold(true);
    p.setFont(titleFont);
    p.setPen(titleColor);
    p.drawText(QRectF(4.0, 4.0, w - 8.0, 18.0), Qt::AlignCenter, title);

    /* ---- 三种动画: 呼吸灯 / 旋转粒子 / 沙漏 ---- */
    drawHalo(p, c);
    drawParticles(p, c);
    drawHourglass(p, c);

    /* ---- 实时耗时 ---- */
    QFont timeFont(QStringLiteral("Consolas"));
    timeFont.setPointSize(15);
    timeFont.setBold(true);
    if (!QFontInfo(timeFont).family().contains(QStringLiteral("Consolas"), Qt::CaseInsensitive)) {
        timeFont = font();
        timeFont.setPointSize(14);
        timeFont.setBold(true);
    }
    p.setFont(timeFont);
    p.setPen(m_running ? QColor(20, 60, 100) : (m_everStarted ? kDone : kIdleText));
    const QString timeText = m_everStarted ? formatElapsed(elapsedMs())
                                           : QStringLiteral("--.-- s");
    p.drawText(QRectF(4.0, h - 66.0, w - 8.0, 28.0), Qt::AlignCenter, timeText);

    /* ---- agent 名 ---- */
    QFont smallFont = font();
    smallFont.setPointSize(8);
    p.setFont(smallFont);
    p.setPen(QColor(120, 110, 80));
    const QFontMetrics smallFm(smallFont);
    const QString agentText = m_agent.isEmpty()
        ? QStringLiteral("(未选择 agent)")
        : smallFm.elidedText(m_agent, Qt::ElideRight, static_cast<int>(w) - 12);
    p.drawText(QRectF(4.0, h - 40.0, w - 8.0, 15.0), Qt::AlignCenter, agentText);

    /* ---- 当前阶段 ---- */
    p.setPen(m_running ? kAccent : kIdleText);
    const QString stageText = m_stage.isEmpty()
        ? QStringLiteral("—")
        : smallFm.elidedText(m_stage, Qt::ElideRight, static_cast<int>(w) - 12);
    p.drawText(QRectF(4.0, h - 24.0, w - 8.0, 15.0), Qt::AlignCenter, stageText);
}

/* 呼吸灯: 半径与透明度同相位缓慢起伏 */
void ThinkingIndicator::drawHalo(QPainter &p, const QPointF &c) const
{
    const qreal wave = 0.5 + 0.5 * std::sin(kTwoPi * m_phase);
    const qreal radius = kHaloR * (0.94 + 0.10 * wave);
    const int alphaMax = m_running ? static_cast<int>(40 + 70 * wave) : 18;

    QRadialGradient halo(c, radius);
    halo.setColorAt(0.0, QColor(kAccent.red(), kAccent.green(), kAccent.blue(),
                                static_cast<int>(alphaMax * 0.35)));
    halo.setColorAt(0.55, QColor(kAccent.red(), kAccent.green(), kAccent.blue(),
                                 static_cast<int>(alphaMax * 0.75)));
    halo.setColorAt(1.0, QColor(kAccent.red(), kAccent.green(), kAccent.blue(), 0));

    p.setPen(Qt::NoPen);
    p.setBrush(halo);
    p.drawEllipse(c, radius, radius);
}

/* 旋转粒子: 一圈固定位置的粒子, 亮度峰值 (彗头) 随时间绕圈跑 */
void ThinkingIndicator::drawParticles(QPainter &p, const QPointF &c) const
{
    /* 参考圆: 让"环"本身可见, 否则只有亮点在飘 */
    p.setBrush(Qt::NoBrush);
    p.setPen(QPen(QColor(170, 165, 140, m_running ? 90 : 45), 1.0));
    p.drawEllipse(c, kRingR, kRingR);

    const qreal headIdx = m_phase * kParticleCount;   /* 彗头在"粒子序号"空间的位置 */
    const qreal breathe = 0.5 + 0.5 * std::sin(kTwoPi * m_phase * 2.0);

    p.setPen(Qt::NoPen);
    for (int i = 0; i < kParticleCount; ++i) {
        /* 粒子 i 距离彗头的序号差 (0 = 正在头部) */
        qreal d = std::fmod(i - headIdx, static_cast<qreal>(kParticleCount));
        if (d < 0.0) {
            d += kParticleCount;
        }
        const qreal head = std::pow(1.0 - d / kParticleCount, 2.0);   /* 1 -> 0 的彗尾 */

        const qreal ang = kTwoPi * (i / static_cast<qreal>(kParticleCount));
        const qreal rr = kRingR + 2.5 * std::sin(kTwoPi * m_phase + i);
        const QPointF pt(c.x() + rr * std::cos(ang), c.y() + rr * std::sin(ang));

        const qreal scale = m_running ? 1.0 : 0.3;
        QColor col(kAccent);
        col.setAlphaF(qBound(0.0, (0.12 + 0.78 * head) * scale, 1.0));
        p.setBrush(col);
        const qreal r = (1.7 + 2.7 * head) * (0.85 + 0.15 * breathe);
        p.drawEllipse(pt, r, r);
    }
}

/*
   沙漏: 上半部的沙子按相位漏到下半部, 漏完复位。

   几何与"流量守恒" (这里原来画反了)
   ---------------------------------
   上下两个腔体都是三角形: 上半部**上宽下尖** (顶点在 mid), 下半部**上尖下宽**。
   所以"剩余多少沙"和"沙面在哪"不是线性的 —— 对一个二维三角形, 从顶点往上到高度 u
   的截面积是

       A(u) = ∫₀^u 2·hw·(t/hh) dt = hw·u²/hh,     A(hh) = hw·hh = 总面积

   于是"剩余比例 f"对应的高度是 `u = hh·√f` (而不是 `u = hh·f`): 沙面下降一开始快、
   接近漏完时慢, 这才是真实沙漏看起来的样子。沙面处的半宽自然是 `hw·u/hh`。

   原实现的两个问题:
     1. 上半部的沙子被画成"贴着腔体顶部、从下往上被吃掉" —— 顶部那条边永远是满宽,
        底边往上收。真实的沙面是**从上往下**落的, 所以减少方向正好反了。
     2. 下半部的沙堆顶半宽用了 `hw·(1−drained)` (应该是 `hw·u/hh`), 除了 drained=0.5
        那一点以外都偏胖。
   现在上下两腔都用同一套 `u = hh·√f` 的换算, 上半部剩 `f = 1−drained`,
   下半部积 `f = drained`, 于是"漏下去的"和"堆起来的"在任何相位都面积相等。
*/
void ThinkingIndicator::drawHourglass(QPainter &p, const QPointF &c) const
{
    const qreal hw = kSandHalfW;
    const qreal hh = kSandHalfH;
    const QPointF top(c.x(), c.y() - hh);
    const QPointF mid(c.x(), c.y());
    const QPointF bot(c.x(), c.y() + hh);

    /*
       drained = 已经从上半部漏下去的比例。
       思考中: 跟着相位走 (漏完自动复位, 变成"一直在漏"的循环感)。
       思考结束: 全部漏到下半部 —— 一眼就能看出"这次算完了"。
       从未思考: 保持在上半部。
    */
    qreal drained;
    if (m_running) {
        drained = m_phase;
    } else {
        drained = m_everStarted ? 1.0 : 0.0;
    }

    /* 玻璃里的一点点底色光, 让它不显得是纯线框 */
    if (m_running) {
        QRadialGradient glow(mid, hh * 1.4);
        glow.setColorAt(0.0, QColor(kSand.red(), kSand.green(), kSand.blue(), 55));
        glow.setColorAt(1.0, QColor(kSand.red(), kSand.green(), kSand.blue(), 0));
        p.setPen(Qt::NoPen);
        p.setBrush(glow);
        p.drawEllipse(mid, hh * 1.4, hh * 1.4);
    }

    /* 面积守恒的液面换算: 高度 u = hh·√f, 该高度处的半宽 hw·u/hh */
    const qreal fUp = 1.0 - drained;             /* 上半部剩余比例 */
    const qreal fDn = drained;                   /* 下半部已积比例 */
    const qreal uUp = hh * std::sqrt(fUp);       /* 上半部沙面高于 mid 的高度 */
    const qreal uDn = hh * std::sqrt(fDn);       /* 下半部沙面低于 mid 的高度 */
    const qreal halfUp = hw * (uUp / hh);
    const qreal halfDn = hw * (uDn / hh);
    const qreal topY = mid.y() - uUp;            /* 上半部沙面 y */
    const qreal botY = mid.y() + uDn;            /* 下半部沙面 y */

    p.setPen(Qt::NoPen);
    p.setBrush(kSand);

    /* 上半部的沙: 从沙面往下收到漏斗口 (三角形) */
    if (uUp > 0.7) {
        QPolygonF poly;
        poly << QPointF(c.x() - halfUp, topY)
             << QPointF(c.x() + halfUp, topY)
             << mid;
        p.drawPolygon(poly);
    }

    /* 下半部的沙堆: 从漏斗口往下张开到沙面 (三角形) */
    if (uDn > 0.7) {
        QPolygonF poly;
        poly << mid
             << QPointF(c.x() - halfDn, botY)
             << QPointF(c.x() + halfDn, botY);
        p.drawPolygon(poly);
    }

    /* 中间下落的一股细沙 (从漏斗口到下方沙面) */
    if (m_running && drained > 0.02 && drained < 0.985 && botY > mid.y() + 1.0) {
        p.setBrush(kSand);
        p.setPen(Qt::NoPen);
        p.drawRect(QRectF(c.x() - 0.8, mid.y() + 1.0, 1.6, botY - mid.y() - 1.0));
    }

    /* 玻璃外框 */
    p.setBrush(Qt::NoBrush);
    p.setPen(QPen(kGlass, 1.6));
    QPolygonF topTri;
    topTri << QPointF(c.x() - hw, top.y()) << QPointF(c.x() + hw, top.y()) << mid;
    QPolygonF botTri;
    botTri << QPointF(c.x() - hw, bot.y()) << QPointF(c.x() + hw, bot.y()) << mid;
    p.drawPolygon(topTri);
    p.drawPolygon(botTri);
}
