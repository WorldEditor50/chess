#include "mainwindow.h"
#include "gamedb.h"

#include <QApplication>
#include <QIcon>
#include <QDebug>
#include <QMutex>
#include <QMutexLocker>
#include <QStringList>
#include <cstdio>

/*
 * ================================================================
 *  直接 printf 的诊断输出 (2026-09, 用户口径: "直接printf")
 * ================================================================
 *
 * 背景: 用户报"chess程序没有日志"。原因不是没打日志, 而是**打的通道看不见**:
 *   * 本程序是 **Console 子系统** (实测 PE Subsystem = 3), 所以 stdout/stderr 有去处;
 *   * 但代码里用的是 `qInfo()` / `qWarning()` —— MSVC 构建下 Qt 的默认消息处理器把它们
 *     送给 `OutputDebugString`, **只有挂调试器才看得到**。用户在 cmd 里跑, 于是一行都没有。
 *
 * 所以诊断路径统一改成**直连 printf** (写 stderr, 不带缓冲):
 *   * stderr 无缓冲 ⇒ 卡死/崩溃时最后几行一定已经在终端上 (stdout 是带缓冲的,
 *     重定向到文件时整块才 flush, "日志里什么都没有"正是这么来的);
 *   * 不再经过 Qt ⇒ 不依赖任何消息处理器装没装。
 *
 * ⚠ 注意: `qInfo()` 那些**业务日志** (启动权重、自检摘要) 仍然走 Qt。它们原本就设计成
 *   给"模型自检面板 + cmd 重定向"看的 (tools/verify_*.ps1 靠 stderr 抓它们),
 *   本文件不动那条路径 —— 只保证**诊断**信息一定看得见。
 */
namespace {

/* 线程安全: 日志来自 AI 线程 / 对弈线程 / 保存线程 */
QMutex g_printMutex;

void diagPrint(const QString &msg)
{
    QMutexLocker lock(&g_printMutex);
    std::fprintf(stderr, "%s\n", msg.toUtf8().constData());
    std::fflush(stderr);
}

}  // namespace

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    /*
       程序图标。两个地方各需要一份, 缺一个就会出现"窗口有图标、exe 文件没有"这种
       半吊子状态:
         * 运行时窗口/任务栏: 这里 setWindowIcon(), 路径来自 res.qrc (嵌在二进制里,
           所以不受工作目录影响); 它是**应用级**的, 所有窗口 (主窗口 / 请稍候弹窗 /
           放大曲线窗口 / 消息框) 都会自动继承。
         * exe 文件图标 (资源管理器、任务栏固定): 由 src/app.rc + src/app.ico 编进
           PE 资源, 与代码无关, 见 tools/make_app_icon.ps1。
       启动时把加载结果打出来: 图标没了是"看起来没坏但就是不对"的那种问题,
       有一行日志就不用猜 (tools/verify_app_icon.ps1 会读这一行)。
    */
    const QIcon appIcon(QStringLiteral(":/app.png"));
    if (!appIcon.isNull()) {
        a.setWindowIcon(appIcon);
        QStringList sizes;
        const QList<QSize> avail = appIcon.availableSizes();
        for (const QSize &s : avail) {
            sizes << QStringLiteral("%1x%2").arg(s.width()).arg(s.height());
        }
        diagPrint(QStringLiteral("[icon] window icon ok: %1")
                      .arg(sizes.isEmpty() ? QStringLiteral("(no size reported)")
                                           : sizes.join(QStringLiteral(", "))));
    } else {
        diagPrint(QStringLiteral("[icon] window icon FAILED to load "
                                 "(:/app.png missing from res.qrc?)"));
    }

    diagPrint(QStringLiteral("[log] 诊断输出走 stderr (直接 printf); "
                             "业务日志 (权重/自检) 仍走 Qt 的 qInfo"));

    MainWindow w;
    w.show();
    int ret = a.exec();
    GameDatabase::instance().close();
    diagPrint(QStringLiteral("[log] 退出 (return %1)").arg(ret));
    return ret;
}
