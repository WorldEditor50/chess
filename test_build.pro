SOURCES += \
    chess.cpp \
    gamedb.cpp \
    mcts.cpp \
    pos.cpp \
    stone.cpp \
    test_main.cpp

HEADERS += \
    chess.h \
    gamedb.h \
    mcts.h \
    stone.h \
    pos.h \
    test_utils.h

CONFIG += c++17 console
QT = core sql
