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
        /*
           `pts` = 画出来的那条曲线 (受 setWindow 限制, 默认 2000 点);
           `history` = **全部**采样点 (不受窗口限制)。
           为什么要两份 (2026-09, 用户在分析导出数据时踩到): 窗口是为了"画得动", 但导出
           是给人**分析**的 —— 用户拿导出的 CSV 当"整场 100 局"来解读, 而那个文件其实只是
           最后 2000 个点 (约十几局), 早期那个 75 的损失尖峰根本不在文件里。两份数据的
           代价很小 (一个 double 8 字节, 十万点 = 800 KB), 而"分析文件缺了一半时间线"
           这种坑会直接导致结论错。
           `toCsv()` 导出的是 **history**; 屏幕上的曲线与"均值/最小/最大"读数仍然按
           **窗口**算 (那是"最近怎么样"的口径, 语义不动)。
        */
        QVector<double> pts;
        QVector<double> history;
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
    /*
       连曲线本身一起清掉 (名字/颜色/数据全没)。

       为什么要有这个: `clearData()` **只清点、不清线**, 于是"每场对弈重建两条奖励
       曲线"的写法 (`clearData()` + `addSeries()` x2) 会每场往后**再挂两条** ——
       跑第 3 场时图上有 6 条, 其中 4 条是空的, 读数标签里就出现
       "SAC+AZ-MoE: 暂无 | Alpha-Beta: 暂无 | SAC+AZ-MoE: 最新 ... | ..." 这种
       (用户界面实录, 见 docs/agents_design.md 13.8)。要"换一批曲线"就得用这个。
    */
    void removeAllSeries();

    int seriesCount() const { return m_series.size(); }
    const Series &series(int i) const { return m_series[i]; }
    int sampleCount() const;
    /* 一行"最新/均值/点数"的文字读数 (界面标签与放大窗口共用同一份格式化) */
    QString readoutText(const QString &prefix = QString()) const;

    /*
       本图的数据导出成 CSV **文本** (2026-09)。
       为什么把"文本生成"放在控件里、把"写文件/选路径"留在窗口里:
         * 导出按钮要弹文件对话框 (自动化脚本点不动), 但格式本身是**可以测的** ——
           放进 CurveChart 之后 test_match 的 [2.9] 节能直接断言格式 (含"短的那条曲线
           后面的列留空而不是补 0"这条容易写错的规则);
         * 合并导出 (损失 + 奖励两段) 也就只需把两次 toCsv() 拼起来, 不必再抄一遍循环。
       格式:
           # <comment>
           sample,<名 1>,<名 2>,...
           1,<v>,<v>,...
       导出的样本数是 **history (全部采样点)**, 不是屏幕窗口里的点数 —— 见 Series 的说明。
    */
    QString toCsv(const QString &comment) const;
    /* 全部采样点的条数 (导出/分析用; 与 sampleCount() 的窗口口径不同) */
    int historyCount() const;

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
