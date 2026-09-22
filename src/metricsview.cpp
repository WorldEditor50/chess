#include "metricsview.h"

#include <QFontMetrics>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPolygonF>
#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>

#include <cmath>

namespace {

const QColor kPanelFill(255, 255, 255, 170);
const QColor kPanelEdge(224, 216, 189);
const QColor kGrid(226, 220, 200);
const QColor kZeroLine(180, 170, 145);
const QColor kText(96, 88, 62);
const QColor kTitle(52, 66, 84);

/* 纵轴刻度取"好看"的步长: 1/2/5 × 10^k */
double niceStep(double span, int targetTicks)
{
    if (!(span > 0.0) || targetTicks < 1) {
        return 1.0;
    }
    const double raw = span / (double)targetTicks;
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double norm = raw / mag;
    double step;
    if (norm <= 1.0) {
        step = 1.0;
    } else if (norm <= 2.0) {
        step = 2.0;
    } else if (norm <= 5.0) {
        step = 5.0;
    } else {
        step = 10.0;
    }
    return step * mag;
}

QString fmtValue(double v, const QString &suffix)
{
    const double a = std::fabs(v);
    QString s;
    if (a >= 1000.0 || (a > 0.0 && a < 0.01)) {
        s = QString::number(v, 'e', 2);
    } else if (a >= 100.0) {
        s = QString::number(v, 'f', 0);
    } else if (a >= 1.0) {
        s = QString::number(v, 'f', 2);
    } else {
        s = QString::number(v, 'f', 3);
    }
    return s + suffix;
}

} // namespace

CurveChart::CurveChart(QWidget *parent)
    : QWidget(parent)
{
    setMinimumHeight(96);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    setAutoFillBackground(false);
}

QSize CurveChart::sizeHint() const
{
    return QSize(300, 118);
}

QSize CurveChart::minimumSizeHint() const
{
    return QSize(200, 96);
}

void CurveChart::setTitle(const QString &title)
{
    m_title = title;
    updateAccessibility();
    update();
}

void CurveChart::setValueSuffix(const QString &suffix)
{
    m_suffix = suffix;
    updateAccessibility();
    update();
}

int CurveChart::addSeries(const QString &name, const QColor &color)
{
    Series s;
    s.name = name;
    s.color = color;
    m_series.push_back(s);
    updateAccessibility();
    emit dataChanged();
    update();
    return m_series.size() - 1;
}

void CurveChart::setSeriesAt(int i, const QString &name, const QColor &color,
                             const QVector<double> &values)
{
    if (i < 0 || i >= m_series.size()) {
        return;
    }
    m_series[i].name = name;
    m_series[i].color = color;
    setPoints(i, values);
}

void CurveChart::removeSeriesAt(int i)
{
    if (i < 0 || i >= m_series.size()) {
        return;
    }
    m_series.remove(i);
    recomputeRange();
    updateAccessibility();
    emit dataChanged();
    update();
}
void CurveChart::setWindow(int maxPoints)
{
    m_maxPoints = maxPoints;
    if (m_maxPoints > 0) {
        for (int i = 0; i < m_series.size(); ++i) {
            QVector<double> &v = m_series[i].pts;
            if (v.size() > m_maxPoints) {
                v.remove(0, v.size() - m_maxPoints);
            }
        }
        recomputeRange();
    }
    updateAccessibility();
}

void CurveChart::removeAllSeries()
{
    if (m_series.isEmpty()) {
        return;
    }
    m_series.clear();
    m_hasData = false;
    m_lo = 0.0;
    m_hi = 1.0;
    updateAccessibility();
    emit dataChanged();
    update();
}

void CurveChart::addPoint(int series, double value)
{
    if (series < 0 || series >= m_series.size()) {
        return;
    }
    if (!std::isfinite(value)) {
        /* 非有限值不上图: 一条 NaN 会把整条折线画没, 而且掩盖真正的数值问题 */
        return;
    }
    Series &s = m_series[series];
    s.pts.push_back(value);
    /* 全部采样点另存一份 (不受窗口限制) —— 导出/分析用, 见头文件 Series 的说明 */
    s.history.push_back(value);

    /*
       窗口满时淘汰最老的点。**必须把它从统计量里减掉** ——
       原来只 remove 了数据, 没有维护 s.sum/s.mn/s.mx, 于是:
         * "均值" = (全部历史之和) / (窗口内的点数) -> 越跑越大, 完全失真;
         * mn/mx 还留着已经被淘汰的极值 -> 纵轴范围也是错的。
       (用户看到的"显示表示也是错的"就是这个。)
       淘汰是"每次至多一个", 所以极值在多数情况下没法增量维护, 直接重算这一段
       —— 窗口只有 2000 个点, 重算一次是微秒级。
    */
    bool evicted = false;
    if (m_maxPoints > 0 && s.pts.size() > m_maxPoints) {
        s.sum -= s.pts.front();
        s.pts.remove(0, s.pts.size() - m_maxPoints);
        evicted = true;
    }

    s.last = value;
    s.sum += value;
    if (!s.valid) {
        s.valid = true;
        s.mn = value;
        s.mx = value;
    } else {
        s.mn = std::fmin(s.mn, value);
        s.mx = std::fmax(s.mx, value);
    }
    if (evicted) {
        s.mn = s.pts[0];
        s.mx = s.pts[0];
        for (int i = 1; i < s.pts.size(); ++i) {
            s.mn = std::fmin(s.mn, s.pts[i]);
            s.mx = std::fmax(s.mx, s.pts[i]);
        }
    }
    recomputeRange();
    updateAccessibility();
    emit dataChanged();
    update();
}

/*
   把曲线状态写进无障碍属性。作用是**让画出来的东西可被测试读到**:
   曲线的内容全是 paintEvent 画的, UIA 看不到任何一个数字, 于是自动化脚本
   (tools/verify_match_ui.ps1) 只能靠"看"来判断有没有数据。有了
   AccessibleDescription, 脚本就能断言 "n=4" 这样的确凿信息。
*/
void CurveChart::updateAccessibility()
{
    setAccessibleName(m_title);
    QString d;
    for (int i = 0; i < m_series.size(); ++i) {
        const Series &s = m_series[i];
        if (i > 0) {
            d += QStringLiteral("; ");
        }
        if (s.valid && !s.pts.isEmpty()) {
            d += QStringLiteral("%1 last=%2 mean=%3 count=%4")
                     .arg(s.name)
                     .arg(s.last, 0, 'g', 6)
                     .arg(s.sum / (double)s.pts.size(), 0, 'g', 6)
                     .arg(s.pts.size());
        } else {
            d += QStringLiteral("%1 last=- mean=- count=0").arg(s.name);
        }
    }
    d += QStringLiteral("; n=%1").arg(sampleCount());
    setAccessibleDescription(d);
    /* 让读屏/自动化工具真的能拿到 (见头文件注释) */
    setToolTip(d);
}

void CurveChart::setPoints(int series, const QVector<double> &values)
{
    if (series < 0 || series >= m_series.size()) {
        return;
    }
    Series &s = m_series[series];
    s.pts = values;
    s.history = values;      /* history 也整体替换 (导入/重建曲线的语义) */
    if (m_maxPoints > 0 && s.pts.size() > m_maxPoints) {
        s.pts.remove(0, s.pts.size() - m_maxPoints);
    }
    s.valid = !s.pts.isEmpty();
    s.mn = 0.0;
    s.mx = 0.0;
    s.sum = 0.0;
    s.last = 0.0;
    for (int i = 0; i < s.pts.size(); ++i) {
        const double v = s.pts[i];
        if (!s.valid) {
            continue;
        }
        if (i == 0) {
            s.mn = v;
            s.mx = v;
        } else {
            s.mn = std::fmin(s.mn, v);
            s.mx = std::fmax(s.mx, v);
        }
        s.sum += v;
        s.last = v;
    }
    recomputeRange();
    updateAccessibility();
    emit dataChanged();
    update();
}

void CurveChart::clearData()
{
    for (int i = 0; i < m_series.size(); ++i) {
        m_series[i].pts.clear();
        m_series[i].history.clear();     /* 清空 = 连历史也清 (否则导出会带回旧数据) */
        m_series[i].valid = false;
        m_series[i].last = 0.0;
        m_series[i].mn = 0.0;
        m_series[i].mx = 0.0;
        m_series[i].sum = 0.0;
    }
    m_hasData = false;
    m_lo = 0.0;
    m_hi = 1.0;
    updateAccessibility();
    emit dataChanged();
    update();
}

int CurveChart::sampleCount() const
{
    int n = 0;
    for (int i = 0; i < m_series.size(); ++i) {
        n = qMax(n, m_series[i].pts.size());
    }
    return n;
}

/*
   一行文字读数: "SAC+AZ: 最新 -1, 均值 -1.15, 2 点  |  EVAB: ..."
   界面标签、放大窗口、无障碍描述都用它, 免得三处各写一份格式化 (以前就有这个重复,
   导致"均值"的口径不一致)。
*/
QString CurveChart::toCsv(const QString &comment) const
{
    QString out;
    out += QStringLiteral("# ") + comment + QStringLiteral("\n");
    out += QStringLiteral("sample");
    for (int s = 0; s < m_series.size(); ++s) {
        out += QStringLiteral(",") + m_series[s].name;
    }
    out += QStringLiteral("\n");
    /*
       导出的是 **全部采样点** (history), 不是屏幕窗口 (pts): 屏幕窗口默认只有 2000 点
       (setWindow), 而用户会把导出的文件当成"整场对弈"来分析 —— 那时早期数据其实已经
       被窗口挤掉了 (实测踩过: 一个 75 的损失尖峰在导出文件里根本不存在)。窗口口径只
       影响"画什么/读数怎么算", 不该影响"导出什么"。
    */
    const int maxN = historyCount();
    for (int i = 0; i < maxN; ++i) {
        out += QString::number(i + 1);
        for (int s = 0; s < m_series.size(); ++s) {
            out += QStringLiteral(",");
            const Series &sr = m_series[s];
            /*
               短的那条曲线**留空**, 不补 0: 补 0 会被读成"那个采样点上损失是 0",
               而真相是"这条曲线还没到那么多个点"(不同 agent 的训练次数不同)。
            */
            if (i < sr.history.size()) {
                out += QString::number(sr.history[i], 'g', 8);
            }
        }
        out += QStringLiteral("\n");
    }
    return out;
}

int CurveChart::historyCount() const
{
    int n = 0;
    for (int i = 0; i < m_series.size(); ++i) {
        n = qMax(n, m_series[i].history.size());
    }
    return n;
}

/*
   一行文字读数: "SAC+AZ: 最新 -1, 均值 -1.15, 2 点  |  EVAB: ..."
   界面标签、放大窗口、无障碍描述都用它, 免得三处各写一份格式化 (以前就有这个重复,
   导致"均值"的口径不一致)。
*/
QString CurveChart::readoutText(const QString &prefix) const
{
    if (m_series.isEmpty()) {
        return prefix.isEmpty() ? QStringLiteral("-") : prefix + QStringLiteral(" -");
    }
    QString s;
    for (int i = 0; i < m_series.size(); ++i) {
        const Series &sr = m_series[i];
        if (i > 0) {
            s += QStringLiteral("  |  ");
        }
        if (sr.valid && !sr.pts.isEmpty()) {
            s += QStringLiteral("%1: 最新 %2, 均值 %3, 最小 %4, 最大 %5, %6 点")
                     .arg(sr.name)
                     .arg(sr.last, 0, 'g', 5)
                     .arg(sr.sum / (double)sr.pts.size(), 0, 'g', 5)
                     .arg(sr.mn, 0, 'g', 5)
                     .arg(sr.mx, 0, 'g', 5)
                     .arg(sr.pts.size());
        } else {
            s += QStringLiteral("%1: 暂无").arg(sr.name);
        }
    }
    return prefix.isEmpty() ? s : prefix + QStringLiteral(" ") + s;
}

void CurveChart::recomputeRange()
{
    m_hasData = false;
    double lo = 0.0;
    double hi = 0.0;
    for (int i = 0; i < m_series.size(); ++i) {
        const Series &s = m_series[i];
        if (!s.valid || s.pts.isEmpty()) {
            continue;
        }
        if (!m_hasData) {
            m_hasData = true;
            lo = s.mn;
            hi = s.mx;
        } else {
            lo = std::fmin(lo, s.mn);
            hi = std::fmax(hi, s.mx);
        }
    }
    if (!m_hasData) {
        m_lo = 0.0;
        m_hi = 1.0;
        return;
    }
    /* 0 线尽量包含进来: 损失/奖励的"零"是有意义的参考 */
    lo = std::fmin(lo, 0.0);
    hi = std::fmax(hi, 0.0);
    if (hi - lo < 1e-9) {
        hi = lo + 1.0;
    }
    const double pad = (hi - lo) * 0.08;
    m_lo = lo - pad;
    m_hi = hi + pad;
}

/* 折线区域: 左侧留给纵轴刻度, 顶部留给标题, 底部留给横轴文字 */
QRectF CurveChart::plotRect() const
{
    const qreal left = 46.0;
    const qreal top = 20.0;
    const qreal right = 8.0;
    const qreal bottom = 16.0;
    QRectF r = QRectF(rect()).adjusted(left, top, -right, -bottom);
    if (r.width() < 10.0) {
        r.setWidth(10.0);
    }
    if (r.height() < 10.0) {
        r.setHeight(10.0);
    }
    return r;
}

void CurveChart::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);

    /* ---- 底板 ---- */
    const QRectF panel = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
    p.setBrush(kPanelFill);
    p.setPen(QPen(kPanelEdge, 1.0));
    p.drawRoundedRect(panel, 8.0, 8.0);

    QFont small = font();
    small.setPointSize(8);
    QFont tiny = font();
    tiny.setPointSize(7);

    const QRectF plot = plotRect();

    /* ---- 标题 ---- */
    QFont titleFont = font();
    titleFont.setPointSize(9);
    titleFont.setBold(true);
    p.setFont(titleFont);
    p.setPen(kTitle);
    p.drawText(QRectF(8.0, 3.0, width() - 16.0, 15.0),
               Qt::AlignLeft | Qt::AlignVCenter, m_title);

    /* ---- 纵轴刻度 ---- */
    const double span = m_hi - m_lo;
    const double step = niceStep(span, 4);
    p.setFont(tiny);
    const QFontMetrics tinyFm(tiny);
    const int ticks = (int)std::floor(span / step) + 1;
    for (int i = 0; i <= ticks; ++i) {
        const double v = m_lo + step * i;
        if (v > m_hi + 1e-12) {
            break;
        }
        const qreal y = plot.bottom() - (v - m_lo) / span * plot.height();
        if (y < plot.top() - 0.5 || y > plot.bottom() + 0.5) {
            continue;
        }
        p.setPen(QPen(kGrid, 1.0, Qt::DotLine));
        p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
        p.setPen(kText);
        const QString lab = fmtValue(v, QString());
        p.drawText(QRectF(2.0, y - 7.0, plot.left() - 6.0, 14.0),
                   Qt::AlignRight | Qt::AlignVCenter,
                   tinyFm.elidedText(lab, Qt::ElideRight, (int)plot.left() - 8));
    }
    /* 0 线加粗: "损失下降/奖励为正"这类判断都相对它 */
    if (m_lo < 0.0 && m_hi > 0.0) {
        const qreal y0 = plot.bottom() - (0.0 - m_lo) / span * plot.height();
        p.setPen(QPen(kZeroLine, 1.2));
        p.drawLine(QPointF(plot.left(), y0), QPointF(plot.right(), y0));
    }

    /* ---- 曲线 ---- */
    const int n = sampleCount();
    if (n == 0) {
        p.setFont(small);
        p.setPen(kText);
        p.drawText(plot, Qt::AlignCenter, QStringLiteral("暂无数据"));
    } else {
        for (int si = 0; si < m_series.size(); ++si) {
            const Series &s = m_series[si];
            if (s.pts.isEmpty()) {
                continue;
            }
            const int cnt = s.pts.size();
            /* 横坐标: 有多个系列时按各自的长度铺满, 短的靠右 (它们都是"最近 N 个") */
            QPolygonF poly;
            poly.reserve(cnt);
            for (int i = 0; i < cnt; ++i) {
                const qreal tx = (cnt > 1) ? (qreal)i / (qreal)(cnt - 1) : 0.5;
                const qreal x = plot.left() + tx * plot.width();
                const qreal y = plot.bottom()
                                - (s.pts[i] - m_lo) / span * plot.height();
                poly << QPointF(x, y);
            }
            /* 单点画个圆点, 否则什么都看不见 */
            if (cnt == 1) {
                p.setPen(Qt::NoPen);
                p.setBrush(s.color);
                p.drawEllipse(poly[0], 2.5, 2.5);
                continue;
            }
            QPainterPath path;
            path.moveTo(poly[0]);
            for (int i = 1; i < poly.size(); ++i) {
                path.lineTo(poly[i]);
            }
            QColor line = s.color;
            if (s.pts.size() > 200) {
                line.setAlpha(190);   /* 点很多时细一点、淡一点, 免得糊成一片 */
            }
            p.setBrush(Qt::NoBrush);
            p.setPen(QPen(line, s.pts.size() > 200 ? 1.1 : 1.5));
            p.drawPath(path);
            /* 最新点标出来 */
            p.setPen(Qt::NoPen);
            p.setBrush(s.color);
            p.drawEllipse(poly.back(), 2.2, 2.2);
        }
    }

    /* ---- 图例 + 最新值 ---- */
    p.setFont(small);
    const QFontMetrics fm(small);
    qreal lx = 8.0;
    const qreal ly = height() - 14.0;
    for (int si = 0; si < m_series.size(); ++si) {
        const Series &s = m_series[si];
        QString txt;
        if (s.valid && !s.pts.isEmpty()) {
            const double mean = s.sum / (double)s.pts.size();
            txt = QStringLiteral("%1 %2  均 %3")
                      .arg(s.name, fmtValue(s.last, m_suffix), fmtValue(mean, QString()));
        } else {
            txt = QStringLiteral("%1 -").arg(s.name);
        }
        const int tw = fm.horizontalAdvance(txt);
        if (lx + tw > width() - 6.0) {
            break;
        }
        p.setPen(Qt::NoPen);
        p.setBrush(s.color);
        p.drawRect(QRectF(lx, ly + 4.0, 9.0, 3.0));
        p.setPen(kText);
        p.drawText(QRectF(lx + 12.0, ly, tw, 12.0), Qt::AlignLeft | Qt::AlignVCenter, txt);
        lx += 14.0 + tw + 10.0;
    }
    /* 右上角: 样本数 */
    p.setPen(kText);
    p.setFont(tiny);
    const QString cntText = QStringLiteral("n=%1   双击放大").arg(n);
    p.drawText(QRectF(width() - 70.0, 3.0, 62.0, 14.0),
               Qt::AlignRight | Qt::AlignVCenter, cntText);
}

void CurveChart::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        emit doubleClicked();
        event->accept();
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

/* ============================================================
 *  CurveChartDialog
 * ============================================================ */
CurveChartDialog::CurveChartDialog(const QString &title, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(title);
    setWindowFlag(Qt::Window, true);          /* 独立窗口 (可以拖到另一个屏幕) */
    setAttribute(Qt::WA_DeleteOnClose, true); /* 关掉就析构, 不用外面管生命周期 */
    setMinimumSize(520, 320);
    resize(900, 560);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(6);

    m_chart = new CurveChart(this);
    m_chart->setTitle(title);
    m_chart->setWindow(0);   /* 放大窗口不截断: 源控件已经截过一遍了 */
    layout->addWidget(m_chart, 1);

    m_readout = new QLabel(this);
    m_readout->setWordWrap(true);
    m_readout->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(m_readout, 0);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::close);
    layout->addWidget(buttons, 0);
}

void CurveChartDialog::follow(CurveChart *source)
{
    if (m_source != nullptr) {
        disconnect(m_source, &CurveChart::dataChanged, this,
                   &CurveChartDialog::syncFromSource);
    }
    m_source = source;
    if (m_source != nullptr) {
        connect(m_source, &CurveChart::dataChanged, this,
                &CurveChartDialog::syncFromSource);
    }
    syncFromSource();
}

/*
   整体拷一份源控件的数据。放大窗口只负责"看", 不回写源控件; 条数/名字/颜色也一起
   同步 (每场对弈开始时会重命名 A/B 两条, 不跟着改名的话放大窗口会显示上一场的名字)。
*/
void CurveChartDialog::syncFromSource()
{
    if (m_source == nullptr || m_chart == nullptr) {
        return;
    }
    m_chart->clearData();
    while (m_chart->seriesCount() > 0) {
        m_chart->removeSeriesAt(m_chart->seriesCount() - 1);
    }
    for (int i = 0; i < m_source->seriesCount(); ++i) {
        const CurveChart::Series &src = m_source->series(i);
        const int idx = m_chart->addSeries(src.name, src.color);
        m_chart->setPoints(idx, src.pts);
    }
    if (m_readout != nullptr) {
        m_readout->setText(m_chart->readoutText(QStringLiteral("读数")));
    }
}