#ifndef POS_H
#define POS_H


class Pos
{
public:
    int x;
    int y;
public:
    Pos():x(0), y(0){}
    Pos(int x_, int y_):x(x_), y(y_){}
    Pos(const Pos &pos):x(pos.x), y(pos.y){}
    Pos& operator = (const Pos &pos);
    bool operator == (const Pos &pos) const;
    Pos& operator += (const Pos &pos);
    Pos& operator -= (const Pos &pos);
    Pos& operator += (int scale);
    Pos& operator -= (int scale);
    Pos& operator *= (int scale);
    Pos& operator /= (int scale);
    Pos operator + (const Pos &p);
    Pos operator - (const Pos &p);
    Pos operator + (int scale);
    Pos operator - (int scale);
    Pos operator * (int scale);
    Pos operator / (int scale);
};
#endif // POS_H
