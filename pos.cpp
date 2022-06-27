#include "pos.h"

Pos& Pos::operator = (const Pos &pos)
{
    if (this == &pos) {
        return *this;
    }
    x = pos.x;
    y = pos.y;
    return *this;
}
bool Pos::operator == (const Pos &pos) const
{
    return  x == pos.x && y == pos.y;
}
Pos& Pos::operator += (const Pos &pos)
{
    if (this == &pos) {
        return *this;
    }
    x += pos.x;
    y += pos.y;
    return *this;
}
Pos& Pos::operator -= (const Pos &pos)
{
    if (this == &pos) {
        return *this;
    }
    x -= pos.x;
    y -= pos.y;
    return *this;
}
Pos& Pos::operator += (int scale)
{
    x += scale;
    y += scale;
    return *this;
}
Pos& Pos::operator -= (int scale)
{
    x -= scale;
    y -= scale;
    return *this;
}
Pos& Pos::operator *= (int scale)
{
    x *= scale;
    y *= scale;
    return *this;
}
Pos& Pos::operator /= (int scale)
{
    x /= scale;
    y /= scale;
    return *this;
}
Pos Pos::operator + (const Pos &p)
{
    Pos pos;
    pos.x = x + p.x;
    pos.y = y + p.y;
    return pos;
}
Pos Pos::operator - (const Pos &p)
{
    Pos pos;
    pos.x = x - p.x;
    pos.y = y - p.y;
    return pos;
}
Pos Pos::operator + (int scale)
{
    Pos pos;
    pos.x = x + scale;
    pos.y = y + scale;
    return pos;
}
Pos Pos::operator - (int scale)
{
    Pos pos;
    pos.x = x - scale;
    pos.y = y - scale;
    return pos;
}
Pos Pos::operator * (int scale)
{
    Pos pos;
    pos.x = x * scale;
    pos.y = y * scale;
    return pos;
}
Pos Pos::operator / (int scale)
{
    Pos pos;
    pos.x = x / scale;
    pos.y = y / scale;
    return pos;
}
