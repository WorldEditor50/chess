#ifndef THINKINGINDICATOR_H
#define THINKINGINDICATOR_H

#include <QElapsedTimer>
#include <QString>
#include <QTimer>
#include <QWidget>

/*
 * ThinkingIndicator - "AI 正在思考" 的可视化控件
 *
 * 为什么需要它: AI 是在后台线程里思考的 (ChessBoard::process), 思考期间
 * mousePressEvent 会拒绝落子 —— 这是对的 (轮到 AI 走), 但界面上**没有任何反馈**,
 * 玩家看到的就是"棋子没动, 我也点不动"。加入"走子前先探索+预训练"之后单步
 * 思考时间从毫秒级涨到秒级, 这个"到底还在想还是已经卡死"的问题就变得很突出。
 *
 * 本控件同时给出三种反馈:
 *   1. 呼吸灯   —— 光晕半径/透明度按正弦缓慢起伏, 表示"活着, 还在算"
 *   2. 旋转粒子 —— 一圈粒子上的亮斑绕圈跑 (彗尾效果), 表示"正在推进"
 *   3. 沙漏     —— 上半部的沙子线性漏到下半部, 漏完一轮就复位
 * 以及右上/下方的实时耗时 (百分秒精度) 和当前阶段文字 (探索环境 / 搜索决策 / 落子)。
 *
 * 动画完全由本控件自己的 QTimer 驱动 (33 ms ≈ 30 fps), 不依赖 agent 的进度,
 * 所以即使某一步思考很久, 视觉上也一直是"在动"的, 不会看起来像死掉。
 * 每 tick 还会发 elapsedChanged(ms), 让主窗口的"AI思考时间"标签同步跳动。
 */
class ThinkingIndicator : public QWidget
{
    Q_OBJECT

public:
    explicit ThinkingIndicator(QWidget *parent = nullptr);

    /* 开始一次思考: agentName 显示在标题下方, stage 是当前阶段文字 */
    void start(const QString &agentName, const QString &stage = QString());
    /* 只更新阶段文字 (思考过程中会被调用多次) */
    void setStage(const QString &stage);
    /* 思考结束: 停表并保留最终耗时 */
    void stop();
    /* 回到"从未思考过"的初始外观 (点"开局"时用) */
    void resetToIdle();

    bool isRunning() const { return m_running; }
    /* 运行中返回已耗时, 结束后返回本次总耗时, 从未运行返回 0 */
    long long elapsedMs() const;

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

signals:
    /* 每个动画帧发一次当前耗时 (毫秒) */
    void elapsedChanged(long long ms);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    void tick();
    void drawHalo(QPainter &p, const QPointF &c) const;
    void drawParticles(QPainter &p, const QPointF &c) const;
    void drawHourglass(QPainter &p, const QPointF &c) const;

    QTimer m_timer;
    QElapsedTimer m_clock;
    bool m_running = false;
    bool m_everStarted = false;
    /* 0..1 循环相位, 三种动画都由它导出, 保证互相同步 */
    qreal m_phase = 0.0;
    long long m_finishedMs = -1;
    QString m_agent;
    QString m_stage;
};

#endif // THINKINGINDICATOR_H
