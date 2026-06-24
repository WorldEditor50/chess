#include "mainwindow.h"
#include "gamedb.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    int ret = a.exec();
    GameDatabase::instance().close();
    return ret;
}
