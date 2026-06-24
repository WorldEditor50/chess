#include "gamedb.h"
#include <QSqlQuery>
#include <QSqlError>
#include <QDebug>

/* ============================================================
 * 棋子类型名称映射
 * ============================================================ */
static const char* stoneTypeName(int type)
{
    switch (type) {
    case 0:  return "车";
    case 1:  return "马";
    case 2:  return "炮";
    case 3:  return "兵";
    case 4:  return "将";
    case 5:  return "仕";
    case 6:  return "相";
    default: return "?";
    }
}

/* ============================================================
 * 构造 / 析构
 * ============================================================ */
GameDatabase::GameDatabase()
    : m_open(false)
{
}

GameDatabase::~GameDatabase()
{
    close();
}

/* ============================================================
 * 单例
 * ============================================================ */
GameDatabase& GameDatabase::instance()
{
    static GameDatabase db;
    return db;
}

/* ============================================================
 * 打开 / 创建数据库
 * ============================================================ */
bool GameDatabase::open(const QString &path)
{
    if (m_open) {
        close();
    }

    m_db = QSqlDatabase::addDatabase("QSQLITE", "chess_connection");
    m_db.setDatabaseName(path);

    if (!m_db.open()) {
        qWarning() << "Failed to open database:" << m_db.lastError().text();
        return false;
    }

    m_open = true;
    return createTables();
}

void GameDatabase::close()
{
    if (m_open) {
        m_db.close();
        m_open = false;
    }
    if (QSqlDatabase::contains("chess_connection")) {
        QSqlDatabase::removeDatabase("chess_connection");
    }
}

/* ============================================================
 * 建表
 * ============================================================ */
bool GameDatabase::createTables()
{
    QSqlQuery query(m_db);

    /* 对局表 */
    bool ok = query.exec(
        "CREATE TABLE IF NOT EXISTS games ("
        "  id         INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  start_time TEXT    NOT NULL,"
        "  end_time   TEXT,"
        "  red_player TEXT    NOT NULL DEFAULT '玩家',"
        "  black_player TEXT  NOT NULL DEFAULT 'AI',"
        "  result     INTEGER NOT NULL DEFAULT 0,"
        "  total_moves INTEGER DEFAULT 0"
        ")"
    );
    if (!ok) {
        qWarning() << "Create games table failed:" << query.lastError().text();
        return false;
    }

    /* 走法表 */
    ok = query.exec(
        "CREATE TABLE IF NOT EXISTS moves ("
        "  id            INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  game_id       INTEGER NOT NULL,"
        "  move_number   INTEGER NOT NULL,"
        "  color         INTEGER NOT NULL,"
        "  stone_type    INTEGER NOT NULL,"
        "  from_x        INTEGER NOT NULL,"
        "  from_y        INTEGER NOT NULL,"
        "  to_x          INTEGER NOT NULL,"
        "  to_y          INTEGER NOT NULL,"
        "  captured_type INTEGER DEFAULT -1,"
        "  notation      TEXT,"
        "  FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE"
        ")"
    );
    if (!ok) {
        qWarning() << "Create moves table failed:" << query.lastError().text();
        return false;
    }

    /* 索引 */
    query.exec("CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id)");
    query.exec("CREATE INDEX IF NOT EXISTS idx_moves_move_num ON moves(game_id, move_number)");

    return true;
}

/* ============================================================
 * 开始新对局
 * ============================================================ */
int GameDatabase::startGame(const QString &redPlayer,
                            const QString &blackPlayer)
{
    QSqlQuery query(m_db);
    query.prepare(
        "INSERT INTO games (start_time, red_player, black_player) "
        "VALUES (:time, :red, :black)"
    );
    query.bindValue(":time",  QDateTime::currentDateTime().toString(Qt::ISODate));
    query.bindValue(":red",   redPlayer);
    query.bindValue(":black", blackPlayer);

    if (!query.exec()) {
        qWarning() << "startGame failed:" << query.lastError().text();
        return -1;
    }

    return query.lastInsertId().toInt();
}

/* ============================================================
 * 记录一步走法
 * ============================================================ */
void GameDatabase::recordMove(int gameId, int moveNumber, int color,
                              int stoneType,
                              int fromX, int fromY,
                              int toX, int toY,
                              int capturedType,
                              const QString &notation)
{
    QSqlQuery query(m_db);
    query.prepare(
        "INSERT INTO moves (game_id, move_number, color, stone_type, "
        "  from_x, from_y, to_x, to_y, captured_type, notation) "
        "VALUES (:gid, :mn, :col, :st, :fx, :fy, :tx, :ty, :ct, :not)"
    );
    query.bindValue(":gid", gameId);
    query.bindValue(":mn",  moveNumber);
    query.bindValue(":col", color);
    query.bindValue(":st",  stoneType);
    query.bindValue(":fx",  fromX);
    query.bindValue(":fy",  fromY);
    query.bindValue(":tx",  toX);
    query.bindValue(":ty",  toY);
    query.bindValue(":ct",  capturedType);
    query.bindValue(":not", notation);

    if (!query.exec()) {
        qWarning() << "recordMove failed:" << query.lastError().text();
    }
}

/* ============================================================
 * 结束对局
 * ============================================================ */
void GameDatabase::endGame(int gameId, int result, int totalMoves)
{
    QSqlQuery query(m_db);
    query.prepare(
        "UPDATE games SET end_time = :time, result = :res, total_moves = :mvs "
        "WHERE id = :gid"
    );
    query.bindValue(":time", QDateTime::currentDateTime().toString(Qt::ISODate));
    query.bindValue(":res",  result);
    query.bindValue(":mvs",  totalMoves);
    query.bindValue(":gid",  gameId);

    if (!query.exec()) {
        qWarning() << "endGame failed:" << query.lastError().text();
    }
}

/* ============================================================
 * 查询: 最近对局摘要
 * ============================================================ */
QStringList GameDatabase::getRecentGames(int limit)
{
    QStringList list;
    QSqlQuery query(m_db);
    query.prepare(
        "SELECT id, start_time, red_player, black_player, result, total_moves "
        "FROM games ORDER BY id DESC LIMIT :lim"
    );
    query.bindValue(":lim", limit);

    if (!query.exec()) {
        qWarning() << "getRecentGames failed:" << query.lastError().text();
        return list;
    }

    while (query.next()) {
        int id         = query.value(0).toInt();
        QString start  = query.value(1).toString();
        QString red    = query.value(2).toString();
        QString black  = query.value(3).toString();
        int result     = query.value(4).toInt();
        int moves      = query.value(5).toInt();

        QString line = QString("对局#%1 | %2 | %3(红) vs %4(黑) | %5 | %6步")
            .arg(id)
            .arg(start.mid(0, 19))
            .arg(red, black)
            .arg(resultToString(result))
            .arg(moves);
        list << line;
    }

    return list;
}

/* ============================================================
 * 查询: 某局所有走法
 * ============================================================ */
QStringList GameDatabase::getMoves(int gameId)
{
    QStringList list;
    QSqlQuery query(m_db);
    query.prepare(
        "SELECT move_number, color, stone_type, from_x, from_y, "
        "  to_x, to_y, captured_type, notation "
        "FROM moves WHERE game_id = :gid ORDER BY move_number"
    );
    query.bindValue(":gid", gameId);

    if (!query.exec()) {
        qWarning() << "getMoves failed:" << query.lastError().text();
        return list;
    }

    while (query.next()) {
        int mn   = query.value(0).toInt();
        int col  = query.value(1).toInt();
        int st   = query.value(2).toInt();
        int fx   = query.value(3).toInt();
        int fy   = query.value(4).toInt();
        int tx   = query.value(5).toInt();
        int ty   = query.value(6).toInt();
        int ct   = query.value(7).toInt();
        QString note = query.value(8).toString();

        QString colorStr = (col == 0) ? "红" : "黑";
        QString capStr;
        if (ct >= 0) {
            capStr = QString(" 吃%1").arg(stoneTypeName(ct));
        }

        QString line = QString("  %1. %2%3(%4,%5)→(%6,%7)%8")
            .arg(mn, 3)
            .arg(colorStr)
            .arg(note.isEmpty() ? stoneTypeName(st) : note)
            .arg(fx).arg(fy)
            .arg(tx).arg(ty)
            .arg(capStr);
        list << line;
    }

    return list;
}

/* ============================================================
 * 查询: 某局所有走法 (返回结构化数据)
 * ============================================================ */
QVector<DBStep> GameDatabase::getGameMoves(int gameId)
{
    QVector<DBStep> result;
    QSqlQuery query(m_db);
    query.prepare(
        "SELECT move_number, color, stone_type, from_x, from_y, "
        "  to_x, to_y, captured_type, notation "
        "FROM moves WHERE game_id = :gid ORDER BY move_number"
    );
    query.bindValue(":gid", gameId);

    if (!query.exec()) {
        qWarning() << "getGameMoves failed:" << query.lastError().text();
        return result;
    }

    while (query.next()) {
        DBStep step;
        step.moveNumber   = query.value(0).toInt();
        step.color        = query.value(1).toInt();
        step.stoneType    = query.value(2).toInt();
        step.fromX        = query.value(3).toInt();
        step.fromY        = query.value(4).toInt();
        step.toX          = query.value(5).toInt();
        step.toY          = query.value(6).toInt();
        step.capturedType = query.value(7).toInt();
        step.notation     = query.value(8).toString();
        result.push_back(step);
    }
    return result;
}

/* ============================================================
 * 删除对局
 * ============================================================ */
bool GameDatabase::deleteGame(int gameId)
{
    QSqlQuery query(m_db);

    /* 先删走法 */
    query.prepare("DELETE FROM moves WHERE game_id = :gid");
    query.bindValue(":gid", gameId);
    query.exec();

    /* 再删对局 */
    query.prepare("DELETE FROM games WHERE id = :gid");
    query.bindValue(":gid", gameId);

    if (!query.exec()) {
        qWarning() << "deleteGame failed:" << query.lastError().text();
        return false;
    }
    return true;
}

/* ============================================================
 * 辅助
 * ============================================================ */
QString GameDatabase::resultToString(int result) const
{
    switch (result) {
    case 0:  return "进行中";
    case 1:  return "红胜";
    case 2:  return "黑胜";
    case 3:  return "平局";
    default: return "未知";
    }
}

QString GameDatabase::stoneTypeToString(int type) const
{
    return QString::fromLatin1(stoneTypeName(type));
}
