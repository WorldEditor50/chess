#include "busydialog.h"
#include "thinkingindicator.h"

#include <QCloseEvent>
#include <QLabel>
#include <QVBoxLayout>

BusyDialog::BusyDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("请稍候"));
    /*
       Qt::Dialog 是默认值, 这里显式关掉"问号"上下文帮助按钮: 加载弹窗上出现帮助按钮
       会让人以为里面有可点的东西。
    */
    setWindowFlags((windowFlags() | Qt::CustomizeWindowHint | Qt::WindowTitleHint
                    | Qt::WindowCloseButtonHint)
                   & ~Qt::WindowContextHelpButtonHint);
    setWindowModality(Qt::ApplicationModal);
    setMinimumWidth(360);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(16, 14, 16, 14);
    layout->setSpacing(8);

    m_title = new QLabel(this);
    QFont titleFont = m_title->font();
    titleFont.setPointSize(11);
    titleFont.setBold(true);
    m_title->setFont(titleFont);
    m_title->setAlignment(Qt::AlignCenter);
    layout->addWidget(m_title);

    /* 复用"AI 正在思考"的那个指示器: 沙漏 + 旋转粒子 + 呼吸灯 + 实时耗时 */
    m_indicator = new ThinkingIndicator(this);
    layout->addWidget(m_indicator, 0, Qt::AlignHCenter);

    m_message = new QLabel(this);
    m_message->setWordWrap(true);
    m_message->setAlignment(Qt::AlignCenter);
    m_message->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(m_message);

    auto *hint = new QLabel(QStringLiteral("正在读写模型权重，完成后会自动关闭。"), this);
    hint->setAlignment(Qt::AlignCenter);
    hint->setWordWrap(true);
    QFont hintFont = hint->font();
    hintFont.setPointSize(8);
    hint->setFont(hintFont);
    hint->setStyleSheet(QStringLiteral("color: #80785e;"));
    layout->addWidget(hint);
}

void BusyDialog::startBusy(const QString &title, const QString &message)
{
    m_busy = true;
    /*
       窗口标题也用 title: 这样任务栏/自动化脚本看到的就是"正在载入"/"正在保存",
       而不是一个笼统的"请稍候" (tools/verify_match_ui.ps1 就按这个名字找它)。
    */
    setWindowTitle(title);
    m_title->setText(title);
    m_message->setText(message);
    /* 沙漏的"agent 名"行用来显示当前在读写哪个模型, 阶段行显示进度说明 */
    m_indicator->start(title, message);
    show();
    raise();
    activateWindow();
}

void BusyDialog::setMessage(const QString &message)
{
    m_message->setText(message);
    if (m_busy) {
        m_indicator->setStage(message);
    }
}

void BusyDialog::stopBusy()
{
    m_busy = false;
    m_indicator->stop();
    hide();
}

QString BusyDialog::message() const
{
    return m_message->text();
}

void BusyDialog::closeEvent(QCloseEvent *event)
{
    if (m_busy) {
        /* 忽略关闭: 后台线程还在读写权重, 关掉只会让人以为操作被取消了 */
        event->ignore();
        return;
    }
    QDialog::closeEvent(event);
}

void BusyDialog::reject()
{
    if (m_busy) {
        return;   /* ESC 同理 */
    }
    QDialog::reject();
}
