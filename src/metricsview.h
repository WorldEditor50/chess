#ifndef METRICSVIEW_H
#define METRICSVIEW_H

#include <QColor>
#include <QDialog>
#include <QString>
#include <QVector>
#include <QWidget>

class QLabel;

/*
 * CurveChart - 一个不依赖 Qt Charts 的轻量折线图控件
 * ============================================================================
 *
 * 为什么要自己画而不是用 Qt Charts:
 *   1. Qt Charts 是**额外的 Qt 模块** (需要单独装/在 CMake 里 find_package
 *      Qt6Charts), 而本工程只依赖 Widgets + Sql。为了两条曲线引入一个新依赖,
 *      编译和部署都变复杂。
 *   2. 这里要的东西很少: 若干条等间隔采样的折线 + 自动纵轴 + 图例 + 最新值。
 *      `paintEvent` 里 60 行就够了, 而 Qt Charts 的 QChartView 还要处理动画、
 *      线程与坐标轴对象的所有权。
 *
 * 数据模型: 每条曲线是一个 `QVector<double>`, 横坐标就是采样序号 (0,1,2,...),
 * 因为"第几次训练 / 第几局"本来就是离散序号。内存有上限 (setWindow), 只保留最近
 * maxPoints 个点 —— 后台训练会一直往里塞, 不设上限就是内存泄漏。
 *
 * 线程: 只允许在 GUI 线程调用 (addPoint 会 update())。后台线程的样本通过
 * ChessBoard 的信号 (队列投递) 到 GUI 线程后再进来。
 */
class CurveChart : public QWidget
{
    Q_OBJECT

public:
    explicit CurveChart(QWidget *parent = nullptr);

    struct Series {
        QString name;
        QColor color;
        QVector<double> pts;
        bool valid = false;      /* 是否已经有数据 */
        double last = 0.0;
        double mn = 0.0;
        double mx = 0.0;
        double sum = 0.0;
    };

    /* 标题 + 坐标轴文字 (纵轴只写单位/含义, 不画刻度文字) */
    void setTitle(const QString &title);
    void setValueSuffix(const QString &suffix);   /* 例如 " loss" / " 分" */

    /* 加一条曲线, 返回它的下标 (从 0 开始) */
    int addSeries(const QString &name, const QColor &color);
    /* 改第 i 条曲线的名字/颜色与数据 (放大窗口同步用) */
    void setSeriesAt(int i, const QString &name, const QColor &color,
                     const QVector<double> &values);
    /* 删掉第 i 条曲线 (放大窗口同步用) */
    void removeSeriesAt(int i);
    void addPoint(int series, double value);
    /* 一次性替换某条曲线的数据 (例如"从历史日志里重建曲线") */
    void setPoints(int series, const QVector<double> &values);

    /* 只保留最近 n 个点 (n <= 0 表示不限制) */
    void setWindow(int maxPoints);
    /* 清空所有曲线的数据 (保留曲线与颜色) */
    void clearData();

    int seriesCount() const { return m_series.size(); }
    const Series &series(int i) const { return m_series[i]; }
    int sampleCount() const;
    /* 一行"最新/均值/点数"的文字读数 (界面标签与放大窗口共用同一份格式化) */
    QString readoutText(const QString &prefix = QString()) const;

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

signals:
    /* 数据变了 (加点/清空/换序列)。放大窗口靠它保持同步。 */
    void dataChanged();
    /* 双击 (界面上用来弹出放大窗口) */
    void doubleClicked();

protected:
    void paintEvent(QPaintEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;

private:
    QRectF plotRect() const;
    void recomputeRange();
    /* 把曲线状态写进 AccessibleName/Description, 让自动化脚本能读到 (见 .cpp) */
    void updateAccessibility();

    QVector<Series> m_series;
    QString m_title;
    QString m_suffix;
    int m_maxPoints = 2000;
    double m_lo = 0.0;      /* 纵轴范围 (含留白) */
    double m_hi = 1.0;
    bool m_hasData = false;
};

/*
 * CurveChartDialog - 把某一条曲线放大到独立窗口里看
 * ============================================================================
 *
 * 为什么需要它: 右侧面板只有 ~330 px 宽, 2000 个点的曲线在那里只能看个大概
 * (纵轴刻度挤在一起、图例要省略、细节全糊)。双击就弹一个 900×560 的可缩放窗口,
 * 里面是同一个 CurveChart, 数据通过 `follow()` 与源控件保持同步 ——
 * 源控件每加一个点会 emit dataChanged, 这里整体拷一份过来。
 *
 * 非模态 (可以一边跑对弈一边看), 关闭时自动析构 (WA_DeleteOnClose)。
 */
class CurveChartDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CurveChartDialog(const QString &title, QWidget *parent = nullptr);

    CurveChart *chart() const { return m_chart; }
    /*
     * 跟随某个源控件: 立刻同步一次, 之后源控件每变一次就同步一次。
     * 传 nullptr 表示解除跟随。
     */
    void follow(CurveChart *source);

private slots:
    void syncFromSource();

private:
    CurveChart *m_chart = nullptr;
    QLabel *m_readout = nullptr;
    CurveChart *m_source = nullptr;
};

#endif // METRICSVIEW_H
