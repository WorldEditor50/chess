#ifndef BUSYDIALOG_H
#define BUSYDIALOG_H

#include <QDialog>
#include <QString>

class QLabel;
class ThinkingIndicator;

/*
 * BusyDialog - 载入/保存模型权重时的"请稍候"弹窗（里面就是那个沙漏控件）
 * ============================================================================
 *
 * 为什么需要它：载入权重不是瞬间的事 —— 启动时要读 7 组权重（老格式还是**十进制文本**，
 * DQN+MCTS 一个文件 16 MB，解析要好几秒），保存稀疏 MoE 骨干（28.7 M 参数）更要几百 MB
 * 的 IO。这段时间里界面只是"所有控件变灰 + 一行 '正在加载...'"，看起来就像卡死了。
 *
 * 做法：直接在弹窗里复用 `ThinkingIndicator`（沙漏 + 旋转粒子 + 呼吸灯 + 实时耗时）——
 * 它本来就是"AI 在想事情"的统一视觉语言，载入模型同样是在等，用户不需要学第二套符号。
 *
 * 交互上的三个决定：
 *   1. **不可关闭**：加载/保存中途被 ESC 关掉会让人以为操作取消了，其实后台线程还在跑。
 *      closeEvent/reject 都被忽略（真正的结束只有 stopBusy()）。
 *   2. **应用级模态**：加载期间不让用户去点别的 agent（那些 agent 的权重正被写）。
 *   3. **非 exec()**：加载在后台线程，`show()` 之后 GUI 事件循环继续跑，沙漏才会动；
 *      用 `exec()` 会把 GUI 线程堵在一个嵌套事件循环里（虽然也能收到队列信号，但
 *      很容量出"关窗时对话框还在"这类生命周期问题）。
 */
class BusyDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BusyDialog(QWidget *parent = nullptr);

    /* 开始等待: 标题 + 第一行说明 (内部会把沙漏跑起来) */
    void startBusy(const QString &title, const QString &message);
    /* 更新说明文字 (例如"正在载入 EVAB 权重…"), 沙漏的阶段行也跟着变 */
    void setMessage(const QString &message);
    /* 结束等待: 停沙漏并隐藏 */
    void stopBusy();

    bool isBusy() const { return m_busy; }
    QString message() const;

protected:
    /* 加载中不许关: 见头文件注释 */
    void closeEvent(QCloseEvent *event) override;
    void reject() override;

private:
    ThinkingIndicator *m_indicator = nullptr;
    QLabel *m_title = nullptr;
    QLabel *m_message = nullptr;
    bool m_busy = false;
};

#endif // BUSYDIALOG_H
