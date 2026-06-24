#ifndef GAMEDB_H
#define GAMEDB_H

#include <QString>
#include <QStringList>
#include <QVector>
#include <QSqlDatabase>
#include <QDateTime>

/* ============================================================
 * 数据库走法记录结构
 * ============================================================ */
struct DBStep {
    int moveNumber;
    int color;
    int stoneType;
    int fromX, fromY;
    int toX, toY;
    int capturedType;
    QString notation;
};

/* ============================================================
 * GameDatabase - 棋局数据库管理类
 * 使用 SQLite 记录每步走法和对局结果
 * ============================================================ */
class GameDatabase
{
public:
    GameDatabase();
    ~GameDatabase();

    /* 打开数据库 (默认 chess_games.db) */
    bool open(const QString &path = "chess_games.db");
    void close();

    /* 开始新对局，返回 game_id */
    int startGame(const QString &redPlayer = "玩家",
                  const QString &blackPlayer = "AI");

    /* 记录一步走法 */
    void recordMove(int gameId, int moveNumber, int color,
                    int stoneType,
                    int fromX, int fromY,
                    int toX, int toY,
                    int capturedType,
                    const QString &notation);

    /* 结束对局 */
    void endGame(int gameId, int result, int totalMoves);

    /* 查询 */
    QStringList getRecentGames(int limit = 10);
    QStringList getMoves(int gameId);
    QVector<DBStep> getGameMoves(int gameId);
    bool deleteGame(int gameId);

    /* 单例访问 */
    static GameDatabase& instance();

    bool isOpen() const { return m_open; }

private:
    QSqlDatabase m_db;
    bool m_open;

    bool createTables();
    QString resultToString(int result) const;
    QString stoneTypeToString(int type) const;
};

#endif // GAMEDB_H
