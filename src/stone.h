#ifndef STONE_H
#define STONE_H
#include <string>
#include <vector>
#include <list>
#include <array>
#include <iostream>
#include <mutex>
#include "pos.h"

class Step
{
public:
    int id;
    int nextId;
    Pos pos;
    Pos nextPos;
    double reward;
    /*
     * true  = 由走法生成器 (getPossibleSteps / 飞将分支) 产生, 可以拿去落子;
     * false = 默认构造的"无走法"占位对象。
     *
     * 这个字段取代了以前那种 "id == Stone::ID_NONE" / "pos == (0,0)" 的哨兵判断:
     * 前者永远不成立 (默认 Step 的 id 是 0, 而 ID_NONE 是 33), 后者会被黑方
     * 底线起点的合法走法误命中。
     */
    bool valid;
public:
    Step():id(0), nextId(0), reward(0.0), valid(false){}
    Step(int fromID_, const Pos &fromPos_, int toID_, const Pos &toPos_, double r):
        id(fromID_), nextId(toID_), pos(fromPos_), nextPos(toPos_), reward(r), valid(true){}
    Step(const Step &s):id(s.id),nextId(s.nextId),pos(s.pos),nextPos(s.nextPos),reward(s.reward),valid(s.valid){}
    Step& operator=(const Step& s)
    {
        if (this == &s) {
            return *this;
        }
        id = s.id;
        nextId = s.nextId;
        pos = s.pos;
        nextPos = s.nextPos;
        /* reward / valid 以前漏掉了: 拷贝构造会复制而赋值不会, 行为不一致 */
        reward = s.reward;
        valid = s.valid;
        return *this;
    }
};
class Steps
{
private:
    /*
       空闲链改为 thread_local —— 每条线程一份, get()/put() 不需要任何同步。

       历史: 这里先是完全无锁的全局 std::list (并发使用会破坏链表指针), 我给它
       加了一把 std::mutex; 但走法生成是搜索的热路径, 每个候选走法都要 get() 一次,
       一次 sample() 就是几十次加解锁, 于是又改成按线程分开。

       Step 只在申请的线程里归还 (每个搜索/训练线程自成一体, 不会把 Step 交给别的
       线程), 所以不存在跨线程访问。线程退出时它的空闲链就地销毁 —— Step 是只有
       基本成员的 POD, 泄漏量等于该线程的峰值同时使用量 (几十~几百个), 有界。
    */
    static std::vector<Step*> &freeList()
    {
        static thread_local std::vector<Step*> list;
        return list;
    }
public:
    Steps(){}
    ~Steps() = default;
    inline static Steps& instance()
    {
        static Steps pool;
        return pool;
    }
    Step* get()
    {
        std::vector<Step*> &list = freeList();
        if (list.empty()) {
            return new Step;
        }
        Step *ptr = list.back();
        list.pop_back();
        return ptr;
    }
    void put(const std::vector<Step*> &steps)
    {
        std::vector<Step*> &list = freeList();
        for (size_t i = 0; i < steps.size(); i++) {
            Step *ptr = steps.at(i);
            if (ptr != nullptr) {
                list.push_back(ptr);
            }
        }
        return;
    }
    void put(Step *step)
    {
        if (step != nullptr) {
            freeList().push_back(step);
        }
    }
};

class Stone;

template<typename T>
class StoneMap
{
public:
    constexpr static int row = 10;
    constexpr static int col = 9;
    T* data[10][9];
public:
    /*
       data 以前完全没有初始化, 而 Chess 的 32 个棋子构造函数只写自己所在的
       32 个格子, 剩下 58 个格子保持不确定值。只要在 reset() 之前访问空交叉点
       (chessboard.cpp 的 selectStone / moveStone, 以及 isAttacked) 就会解引用
       垃圾指针 —— MSVC Debug 下 0xCDCDCDCD 必崩。
    */
    StoneMap(){ clear(); }
    inline T* &operator[](const Pos &pos) {return data[pos.x][pos.y];}
    inline bool isInner(const Pos &pos) const
    {
        if ((pos.x >= 0 && pos.x < row) &&
                pos.y >= 0 && pos.y < col) {
            return true;
        }
        return false;
    }
    void clear()
    {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                data[i][j] = nullptr;
            }
        }
        return;
    }
    int countStoneOnLine(const Pos &p1, const Pos &p2)
    {
        /* 点 */
        if (p1 == p2) {
            return -1;
        }
        /* 斜线 */
        if (p1.x != p2.x && p1.y != p2.y) {
            return -1;
        }
        int count = 0;
        if (p1.x == p2.x) {
            int x = p1.x;
            int y0 = std::min(p1.y, p2.y) + 1;
            int yn = std::max(p1.y, p2.y);
            for (int i = y0; i < yn; i++) {
                if (data[x][i] != nullptr) {
                    count++;
                }
            }
        }
        if (p1.y == p2.y) {
            int y = p1.y;
            int x0 = std::min(p1.x, p2.x) + 1;
            int xn = std::max(p1.x, p2.x);
            for (int i = x0; i < xn; i++) {
                if (data[i][y] != nullptr) {
                    count++;
                }
            }
        }
        return count;
    }
    T* get(int id)
    {
        T *ptr = nullptr;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (data[i][j] == nullptr) {
                    continue;
                }
                if (data[i][j]->id == id) {
                    ptr = data[i][j];
                    break;
                }
            }
        }
        return ptr;
    }
    void show()
    {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (data[i][j] == nullptr) {
                    std::cout<<" ";
                } else {
                    std::cout<<data[i][j]->name;
                }
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
        return;
    }
};

class Stone
{
public:
    enum ID {
        ID_RED = 0,
        ID_RED_CHE1 = 0,
        ID_RED_MA1,
        ID_RED_XIANG1,
        ID_RED_SHI1,
        ID_RED_JIANG,
        ID_RED_SHI2,
        ID_RED_XIANG2,
        ID_RED_MA2,
        ID_RED_CHE2,
        ID_RED_PAO1,
        ID_RED_PAO2,
        ID_RED_BING1,
        ID_RED_BING2,
        ID_RED_BING3,
        ID_RED_BING4,
        ID_RED_BING5,
        ID_RED_END = 16,
        ID_BLACK = 16,
        ID_BLACK_CHE1 = 16,
        ID_BLACK_MA1,
        ID_BLACK_XIANG1,
        ID_BLACK_SHI1,
        ID_BLACK_JIANG,
        ID_BLACK_SHI2,
        ID_BLACK_XIANG2,
        ID_BLACK_MA2,
        ID_BLACK_CHE2,
        ID_BLACK_PAO1,
        ID_BLACK_PAO2,
        ID_BLACK_BING1,
        ID_BLACK_BING2,
        ID_BLACK_BING3,
        ID_BLACK_BING4,
        ID_BLACK_BING5,
        ID_BLACK_END,
        ID_NONE
    };
    enum Color {
        COLOR_RED = 0,
        COLOR_BLACK,
        COLOR_NONE
    };
    enum Type {
        TYPE_CHE = 0,
        TYPE_MA,
        TYPE_PAO,
        TYPE_BING,
        TYPE_JIANG,
        TYPE_SHI,
        TYPE_XIANG,
    };

    constexpr static double value_che = 0.5;
    constexpr static double value_ma = 0.3;
    constexpr static double value_pao = 0.3;
    constexpr static double value_bing = 0.1;
    constexpr static double value_jiang = 1000;
    constexpr static double value_shi = 0.2;
    constexpr static double value_xiang = 0.2;
    constexpr static double value_infi = 100000.0;
    using Offset = Pos;

    StoneMap<Stone> *m_map;
    std::array<Stone*, 32> *m_children;
public:
    int id;
    int type;
    int color;
    int alive;
    double value;
    Pos pos;
    std::string name;
public:
    Stone():
        m_map(nullptr), m_children(nullptr) {}
    explicit Stone(int id_, int type_, int color_, int alive_, double value_, const Pos &pos_,
                   StoneMap<Stone> *mapPtr = nullptr,
                   std::array<Stone*, 32> *childrenPtr = nullptr):
        m_map(mapPtr), m_children(childrenPtr),
        id(id_), type(type_), color(color_), alive(alive_),
        value(value_), pos(pos_){}
    Stone(const Stone& s):
        m_map(s.m_map), m_children(s.m_children),
        id(s.id), type(s.type), color(s.color), alive(s.alive),
        value(s.value),pos(s.pos),name(s.name){}
    Stone& operator = (const Stone &s)
    {
        if (this == &s) {
            return *this;
        }
        id = s.id;
        type = s.type;
        color = s.color;
        alive = s.alive;
        value = s.value;
        pos = s.pos;
        name = s.name;
        return *this;
    }
    virtual bool tryMoveTo(const Pos &pos){return true;}
    bool moveTo(const Pos &pos_)
    {
        if (tryMoveTo(pos_) == false) {
            return false;
        }
        Stone *dst = (*m_map)[pos_];
        if (dst != nullptr) {
            if (dst->color == color) {
                return false;
            }
            dst->alive = false;
        }
        (*m_map)[pos] = nullptr;
        (*m_map)[pos_] = this;
        pos = pos_;
        return true;
    }
    virtual void getPossibleSteps(std::vector<Step*> &steps){}
    void getAllPossibleSteps(const std::vector<Offset>& offsets, std::vector<Step*> &steps)
    {
        for (std::size_t i = 0; i < offsets.size(); i++) {
            int x = pos.x + offsets[i].x;
            int y = pos.y + offsets[i].y;
            Pos dstPos(x, y);
            if (m_map->isInner(dstPos) == false) {
                continue;
            }
            if (tryMoveTo(dstPos) == false) {
                continue;
            }
            Stone *stone = (*m_map)[dstPos];
            if (stone != nullptr) {
                if (stone->color == color) {
                    continue;
                }
            }
            Step *step = Steps::instance().get();
            step->id = id;
            step->pos = pos;
            if (stone == nullptr) {
                step->nextId = ID_NONE;
            } else {
                step->nextId = stone->id;
            }
            step->nextPos = dstPos;
            step->reward = 0;
            step->valid = true;
            steps.push_back(step);
        }
        return;
    }
    void show()
    {
        std::cout<<"id:"<<id<<",type:"<<type<<",color:"<<color<<",alive:"<<alive<<",value:"<<value<<",x:"<<pos.x<<",y:"<<pos.y<<",name:"<<name<<std::endl;
        return;
    }
};

class Che : public Stone
{
public:
    Che(){}
    explicit Che(int id_, int color_, const Pos &pos_,
                 StoneMap<Stone> *mapPtr, std::array<Stone*, 32> *childrenPtr):
        Stone(id_, TYPE_CHE, color_, 1, value_che, pos_, mapPtr, childrenPtr)
    {
        name = "车";
        (*m_children)[id_] = this;
        (*m_map)[pos_] = this;
    }
    Che(const Che &r):Stone(r){}
    Che& operator=(const Che& r)
    {
        if (this == &r) {
            return *this;
        }
        Stone::operator=(r);
        return *this;
    }
    bool tryMoveTo(const Pos &pos_)override
    {
        if (m_map->countStoneOnLine(pos, pos_) == 0) {
            return true;
        }
        return false;
    };
    void getPossibleSteps(std::vector<Step*> &steps) override
    {
        std::vector<Offset> offsets;
        //纵向搜索
        for (int i = 0; i < m_map->row; i++) {
            if (i == pos.x) {
                continue;
            }
            offsets.push_back(Offset(i - pos.x, 0));
        }
        //横向搜索
        for (int i = 0; i < m_map->col; i++) {
            if (i == pos.y) {
                continue;
            }
            offsets.push_back(Offset(0, i - pos.y));
        }
        Stone::getAllPossibleSteps(offsets, steps);
        return;
    }
};

class Ma : public Stone
{
public:
    Ma(){}
    explicit Ma(int id_, int color_, const Pos &pos_,
                StoneMap<Stone> *mapPtr, std::array<Stone*, 32> *childrenPtr):
        Stone(id_, TYPE_MA, color_, 1, value_ma, pos_, mapPtr, childrenPtr)
    {
        name = "马";
        (*m_children)[id_] = this;
        (*m_map)[pos_] = this;
    }
    Ma(const Ma &r):Stone(r){}
    Ma& operator=(const Ma& r)
    {
        if (this == &r) {
            return *this;
        }
        Stone::operator=(r);
        return *this;
    }
    bool tryMoveTo(const Pos &pos_)override
    {
        int delta = std::abs(pos_.x - pos.x)*10 + std::abs(pos_.y - pos.y);
        if (delta != 12 && delta != 21) {
            return false;
        }
        int midx = 0;
        int midy = 0;
        if (delta == 12) {
            midy = (pos.y + pos_.y)/2;
            midx = pos.x;
        } else {
            midx = (pos.x + pos_.x)/2;
            midy = pos.y;
        }
        if ((*m_map)[Pos(midx, midy)] != nullptr) {
            return false;
        }
        return true;
    };
    void getPossibleSteps(std::vector<Step*> &steps) override
    {
        std::vector<Offset> offsets ={{2,1},{-2,1},{2,-1},{-2,-1},
                                      {1,2},{1,-2},{-1,2},{-1,-2}};
        Stone::getAllPossibleSteps(offsets, steps);
        return;
    }
};
class Xiang : public Stone
{
public:
    Xiang(){}
    explicit Xiang(int id_, int color_, const Pos &pos_,
                   StoneMap<Stone> *mapPtr, std::array<Stone*, 32> *childrenPtr):
        Stone(id_, TYPE_XIANG, color_, 1, value_xiang, pos_, mapPtr, childrenPtr)
    {
        if (color_ == COLOR_RED) {
            name = "相";
        } else {
            name = "象";
        }
        (*m_children)[id_] = this;
        (*m_map)[pos_] = this;
    }
    Xiang(const Xiang &r):Stone(r){}
    Xiang& operator=(const Xiang& r)
    {
        if (this == &r) {
            return *this;
        }
        Stone::operator=(r);
        return *this;
    }
    bool tryMoveTo(const Pos &pos_)override
    {
        /* 相/象不能过河: 红方限 x>=5, 黑方限 x<=4 (原来两边各放宽了一格) */
        if (color == Stone::COLOR_RED) {
            if (pos_.x < 5) {
                return false;
            }
        } else {
            if (pos_.x > 4) {
                return false;
            }
        }
        int midx = (pos.x + pos_.x)/2;
        int midy = (pos.y + pos_.y)/2;
        if ((*m_map)[Pos(midx, midy)] != nullptr) {
            return false;
        }
        int delta = std::abs(pos_.x - pos.x)*10 + std::abs(pos_.y - pos.y);
        if (delta == 22) {
            return true;
        }
        return false;
    };
    void getPossibleSteps(std::vector<Step*> &steps) override
    {
        std::vector<Offset> offsets = {{2,2}, {-2,2}, {2,-2}, {-2,-2}};
        Stone::getAllPossibleSteps(offsets, steps);
        return;
    }
};
class Shi : public Stone
{
public:
    Shi(){}
    explicit Shi(int id_, int color_, const Pos &pos_,
                 StoneMap<Stone> *mapPtr, std::array<Stone*, 32> *childrenPtr):
        Stone(id_, TYPE_SHI, color_, 1, value_shi, pos_, mapPtr, childrenPtr)
    {
        name = "仕";
        (*m_children)[id_] = this;
        (*m_map)[pos_] = this;
    }
    Shi(const Shi &r):Stone(r){}
    Shi& operator=(const Shi& r)
    {
        if (this == &r) {
            return *this;
        }
        Stone::operator=(r);
        return *this;
    }
    bool tryMoveTo(const Pos &pos_) override
    {
        if (pos_.y < 3 || pos_.y > 5) {
            return false;
        }
        if (color == Stone::COLOR_RED) {
            if(pos_.x < 7 || pos_.x > 9) {
               return false;
            }
        } else {
            if(pos_.x < 0 || pos_.x > 2) {
                return false;
            }
        }
        int delta = std::abs(pos_.x - pos.x)*10 + std::abs(pos_.y - pos.y);
        if (delta == 11) {
            return true;
        }
        return false;
    };
    void getPossibleSteps(std::vector<Step*> &steps) override
    {
        std::vector<Offset> offsets = {{1,1}, {-1,1}, {1,-1}, {-1,-1}};
        Stone::getAllPossibleSteps(offsets, steps);
        return;
    }
};
class Jiang : public Stone
{
public:
    Jiang(){}
    explicit Jiang(int id_, int color_, const Pos &pos_,
                   StoneMap<Stone> *mapPtr, std::array<Stone*, 32> *childrenPtr):
        Stone(id_, TYPE_JIANG, color_, 1, value_jiang, pos_, mapPtr, childrenPtr)
    {
        if (color_ == COLOR_RED) {
            name = "帅";
        } else {
            name = "将";
        }
        (*m_children)[id_] = this;
        (*m_map)[pos_] = this;
    }
    Jiang(const Jiang &r):Stone(r){}
    Jiang& operator=(const Jiang& r)
    {
        if (this == &r) {
            return *this;
        }
        Stone::operator=(r);
        return *this;
    }
    bool tryMoveTo(const Pos &pos_) override
    {
        Stone *dst = (*m_map)[pos_];
        if (dst != nullptr) {
            if (dst->type == Stone::TYPE_JIANG) {
                return m_map->countStoneOnLine(pos, pos_) == 0;
            }
        }
        if (pos_.y < 3 || pos_.y > 5) {
            return false;
        }
        if (color == Stone::COLOR_RED) {
            if(pos_.x < 7 || pos_.x > 9) {
               return false;
            }
        } else {
            if(pos_.x < 0 || pos_.x > 2) {
                return false;
            }
        }
        int delta = std::abs(pos_.x - pos.x)*10 + std::abs(pos_.y - pos.y);
        if (delta == 1 || delta == 10) {
            return true;
        }
        return false;
    };
    void getPossibleSteps(std::vector<Step*> &steps) override
    {
        std::vector<Offset> offsets = {{0,1}, {1,0}, {0,-1}, {-1,0}};
        Stone::getAllPossibleSteps(offsets, steps);
        /* 飞将 */
        int dstID = 0;
        if (id == ID_RED_JIANG) {
            dstID = ID_BLACK_JIANG;
        } else {
            dstID = ID_RED_JIANG;
        }
        Stone *stone = (*m_children)[dstID];
        if (stone == nullptr) {
            return;
        }
        if (stone->alive == false) {
            /* 对方将/帅已被吃, 它留在 pos 上的坐标已经失效 */
            return;
        }
        if (tryMoveTo(stone->pos) == false) {
            return;
        }
        Step *step = Steps::instance().get();
        step->id = id;
        step->pos = pos;
        step->nextId = stone->id;
        step->nextPos = stone->pos;
        step->reward = 0;
        step->valid = true;
        steps.push_back(step);
        return;
    }
};
class Pao : public Stone
{
public:
    Pao(){}
    explicit Pao(int id_, int color_, const Pos &pos_,
                 StoneMap<Stone> *mapPtr, std::array<Stone*, 32> *childrenPtr):
        Stone(id_, TYPE_PAO, color_, 1, value_pao, pos_, mapPtr, childrenPtr)
    {
        name = "炮";
        (*m_children)[id_] = this;
        (*m_map)[pos_] = this;
    }
    Pao(const Pao &r):Stone(r){}
    Pao& operator=(const Pao& r)
    {
        if (this == &r) {
            return *this;
        }
        Stone::operator=(r);
        return *this;
    }
    bool tryMoveTo(const Pos &pos_) override
    {
        int count = m_map->countStoneOnLine(pos, pos_);
        Stone *dst = (*m_map)[pos_];
        if (dst != nullptr) {
            if (count == 1) {
                return true;
            }
        } else {
            if (count == 0) {
                return true;
            }
        }
        return false;
    }
    void getPossibleSteps(std::vector<Step*> &steps) override
    {
        std::vector<Offset> offsets;
        //纵向搜索
        for (int i = 0; i < m_map->row; i++) {
            if (i == pos.x) {
                continue;
            }
            offsets.push_back(Offset(i - pos.x, 0));
        }
        //横向搜索
        for (int i = 0; i < m_map->col; i++) {
            if (i == pos.y) {
                continue;
            }
            offsets.push_back(Offset(0, i - pos.y));
        }
        Stone::getAllPossibleSteps(offsets, steps);
        return;
    }
};
class Bing : public Stone
{
public:
    Bing(){}
    explicit Bing(int id_, int color_, const Pos &pos_,
                  StoneMap<Stone> *mapPtr, std::array<Stone*, 32> *childrenPtr):
        Stone(id_, TYPE_BING, color_, 1, value_bing, pos_, mapPtr, childrenPtr)
    {
        if (color_ == COLOR_RED) {
            name = "兵";
        } else {
            name = "卒";
        }
        (*m_children)[id_] = this;
        (*m_map)[pos_] = this;
    }
    Bing(const Bing &r):Stone(r){}
    Bing& operator=(const Bing& r)
    {
        if (this == &r) {
            return *this;
        }
        Stone::operator=(r);
        return *this;
    }
    bool tryMoveTo(const Pos &pos_)override
    {
        /* 不能后退 */
        if (color == Stone::COLOR_RED) {
            if (pos_.x - pos.x > 0) {
               return false;
            }
            /* 未过河时不能左右走 */
            if (pos.x >= 5 && pos.x == pos_.x) {
                return false;
            }
        } else {
            if (pos_.x - pos.x < 0) {
                return false;
            }
            /* 未过河时不能左右走 */
            if (pos.x <= 4 && pos.x == pos_.x) {
                return false;
            }
        }
        /* 过河后可以水平移动，前进时x方向必须变化 */
        int delta = std::abs(pos_.x - pos.x) + std::abs(pos_.y - pos.y);
        if (delta == 1) {
            return true;
        }
        return false;
    };
    void getPossibleSteps(std::vector<Step*> &steps) override
    {
        std::vector<Offset> offsets = {{0,1}, {1,0}, {0,-1}, {-1,0}};
        Stone::getAllPossibleSteps(offsets, steps);
        return;
    }
};

/* ====================================================================
 *  奖励的"视角"换算 —— 老一代 agent 必须显式做这一步 (2026-09)
 * ====================================================================
 *
 *  本工程里同时存在两套**价值口径**, 用错不会报任何错, 只会让一半样本的训练目标
 *  整体反号 (症状是"学不动", 不是崩溃), 所以把换算写成一个有名字的函数:
 *
 *    * 老一代 PG / DQN / DQN+MCTS: encodeState 是"棋盘绝对坐标 + 黑子为正"
 *      (红子取负), 网络表达的是**对黑方的价值** -> 奖励必须是**黑方视角**。
 *    * 新一代 PPOMCTS / SACAZ / EVAB: encodeState 是**规范视角** (轮到黑方时整盘
 *      镜像), 网络表达的是**走子方**的价值 -> 奖励直接就是**走子方视角**,
 *      **不要**调用下面这个函数。
 *
 *  而 Chess::moveForward 的 totalReward、computeReward、rolloutFromCurrent 产出的
 *  都是走子方视角 (吃子者为正)。于是老一代 agent 在红方走子时要把符号翻过来。
 *  审查结果与每一处的判定见 docs/agents_design.md 的 §17。
 */
inline float moverRewardToBlackFrame(float moverReward, int moverColor)
{
    return (moverColor == Stone::COLOR_RED) ? -moverReward : moverReward;
}

/*
 *  同上, 但走子方是从 Step::id 推的 —— 给 rolloutFromCurrent 的回调用:
 *  那个回调拿得到 Step, 拿不到"当前走子方"这个变量 (rollout 内部会换手)。
 *  红方 id 是 0..15, 黑方是 16..31 (见上面 ID_* 枚举)。
 */
inline float moverRewardToBlackFrame(float moverReward, const Step &s)
{
    return (s.id < Stone::ID_BLACK) ? -moverReward : moverReward;
}

#endif // STONE_H
