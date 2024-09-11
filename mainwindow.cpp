#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "qssloader.hpp"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFixedSize(750, 650);
    connect(ui->resetBtn, &QPushButton::clicked,
            ui->gameWidget, &ChessBoard::reset);
    setStyleSheet(QssLoader::load(":/appstyle.qss"));
}

MainWindow::~MainWindow()
{
    delete ui;
}

