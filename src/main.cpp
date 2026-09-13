#include "mainwindow.h"
#include "gamedb.h"

#include <QApplication>
#include <QIcon>
#include <QDebug>
#include <QStringList>

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
       启动时把加载结果打进日志: 图标没了是"看起来没坏但就是不对"的那种问题,
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
        qInfo().noquote() << "[icon] window icon ok:"
                          << (sizes.isEmpty() ? QStringLiteral("(no size reported)")
                                              : sizes.join(QStringLiteral(", ")));
    } else {
        qWarning().noquote() << "[icon] window icon FAILED to load (:/app.png missing "
                                "from res.qrc?)";
    }

    MainWindow w;
    w.show();
    int ret = a.exec();
    GameDatabase::instance().close();
    return ret;
}
