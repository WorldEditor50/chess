#ifndef STONE_H
#define STONE_H
#include <string>
#include <vector>
#include <list>
#include <array>
#include <iostream>
#include "pos.h"

class Step
{
public:
    int id;
    int nextId;
    Pos pos;
    Pos nextPos;
    double reward;
public:
    Step():id(0), nextId(0){}
    Step(int fromID_, const Pos &fromPos_, int toID_, const Pos &toPos_, double r):
        id(fromID_), nextId(toID_), pos(fromPos_), nextPos(toPos_), reward(r){}
    Step(const Step &s):id(s.id),nextId(s.nextId),pos(s.pos),nextPos(s.nextPos),reward(s.reward){}
    Step& operator=(const Step& s)
    {
        if (this == &s) {
            return *this;
        }
        id = s.id;
        nextId = s.nextId;
        pos = s.pos;
        nextPos = s.nextPos;
        return *this;
    }
};
class Steps
{
private:
    std::list<Step*> stepList;
public:
    Steps(){}
    ~Steps()
    {
        for (auto it = stepList.begin(); it != stepList.end(); it++) {
            Step *p = *it;
            delete p;
        }
    }
    inline static Steps& instance()
    {
        static Steps pool;
        return pool;
    }
    Step* get()
    {
        Step *ptr = nullptr;
        if (stepList.empty()) {
            ptr = new Step;
        } else {
            ptr = stepList.back();
            stepList.pop_back();
        }
        return ptr;
    }
    void put(const std::vector<Step*> &steps)
    {
        for (size_t i = 0; i < steps.size(); i++) {
            Step *ptr = steps.at(i);
            if (ptr != nullptr) {
                stepList.push_back(ptr);
            }
        }
        return;
    }
};

template<typename T>
class StoneMap
{
public:
    constexpr static int row = 10;
    constexpr static int col = 9;
    T* data[10][9];
public:
    StoneMap(){}
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
    static StoneMap<Stone> map;

    static std::array<Stone*, 32> children;
public:
    int id;
    int type;
    int color;
    int alive;
    double value;
    Pos pos;
    std::string name;
public:
    Stone(){}
    explicit Stone(int id_, int type_, int color_, int alive_, double value_, const Pos &pos_):
        id(id_), type(type_), color(color_), alive(alive_),
        value(value_), pos(pos_){}
    Stone(const Stone& s):
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
        Stone *dst = Stone::map[pos_];
        if (dst != nullptr) {
            if (dst->color == color) {
                return false;
            }
            dst->alive = false;
        }
        Stone::map[pos] = nullptr;
        Stone::map[pos_] = this;
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
            if (Stone::map.isInner(dstPos) == false) {
                continue;
            }
            if (tryMoveTo(dstPos) == false) {
                continue;
            }
            Stone *stone = Stone::map[dstPos];
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
    explicit Che(int id_, int color_, const Pos &pos_):
        Stone(id_, TYPE_CHE, color_, 1, value_che, pos_)
    {
        name = "车";
        Stone::children[id_] = this;
        Stone::map[pos_] = this;
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
        if (Stone::map.countStoneOnLine(pos, pos_) == 0) {
            return true;
        }
        return false;
    };
    void getPossibleSteps(std::vector<Step*> &steps) override
    {
        std::vector<Offset> offsets;
        //纵向搜索
        for (int i = 0; i < Stone::map.row; i++) {
            if (i == pos.x) {
                continue;
            }
            offsets.push_back(Offset(i - pos.x, 0));
        }
        //横向搜索
        for (int i = 0; i < Stone::map.col; i++) {
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
    explicit Ma(int id_, int color_, const Pos &pos_):
        Stone(id_, TYPE_MA, color_, 1, value_ma, pos_)
    {
        name = "马";
        Stone::children[id_] = this;
        Stone::map[pos_] = this;
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
        if (Stone::map[Pos(midx, midy)] != nullptr) {
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
    explicit Xiang(int id_, int color_, const Pos &pos_):
        Stone(id_, TYPE_XIANG, color_, 1, value_xiang, pos_)
    {
        if (color_ == COLOR_RED) {
            name = "相";
        } else {
            name = "象";
        }
        Stone::children[id_] = this;
        Stone::map[pos_] = this;
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
        if (color == Stone::COLOR_RED) {
            if (pos_.x < 4) {
                return false;
            }
        } else {
            if (pos_.x > 5) {
                return false;
            }
        }
        int midx = (pos.x + pos_.x)/2;
        int midy = (pos.y + pos_.y)/2;
        if (Stone::map[Pos(midx, midy)] != nullptr) {
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
    explicit Shi(int id_, int color_, const Pos &pos_):
        Stone(id_, TYPE_SHI, color_, 1, value_shi, pos_)
    {
        name = "仕";
        Stone::children[id_] = this;
        Stone::map[pos_] = this;
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
    explicit Jiang(int id_, int color_, const Pos &pos_):
        Stone(id_, TYPE_JIANG, color_, 1, value_jiang, pos_)
    {
        if (color_ == COLOR_RED) {
            name = "帅";
        } else {
            name = "将";
        }
        Stone::children[id_] = this;
        Stone::map[pos_] = this;
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
        Stone *dst = Stone::map[pos_];
        if (dst != nullptr) {
            if (dst->type == Stone::TYPE_JIANG) {
                return Stone::map.countStoneOnLine(pos, pos_) == 0;
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
        Stone *stone = Stone::children[dstID];
        if (stone == nullptr) {
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
        steps.push_back(step);
        return;
    }
};
class Pao : public Stone
{
public:
    Pao(){}
    explicit Pao(int id_, int color_, const Pos &pos_):
        Stone(id_, TYPE_PAO, color_, 1, value_pao, pos_)
    {
        name = "炮";
        Stone::children[id_] = this;
        Stone::map[pos_] = this;
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
        int count = Stone::map.countStoneOnLine(pos, pos_);
        Stone *dst = Stone::map[pos_];
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
        for (int i = 0; i < Stone::map.row; i++) {
            if (i == pos.x) {
                continue;
            }
            offsets.push_back(Offset(i - pos.x, 0));
        }
        //横向搜索
        for (int i = 0; i < Stone::map.col; i++) {
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
    explicit Bing(int id_, int color_, const Pos &pos_):
        Stone(id_, TYPE_BING, color_, 1, value_bing, pos_)
    {
        if (color_ == COLOR_RED) {
            name = "兵";
        } else {
            name = "卒";
        }
        Stone::children[id_] = this;
        Stone::map[pos_] = this;
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
            if (pos.x >= 5 && pos.x == pos_.x) {
                return false;
            }
        } else {
            if (pos_.x - pos.x < 0) {
                return false;
            }
            if (pos.x <= 4 && pos.x == pos_.x) {
                return false;
            }
        }
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

#endif // STONE_H
