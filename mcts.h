#ifndef MCTS_H
#define MCTS_H

#include <vector>


class State
{
public:
    int id;
    int value;
    std::vector<int> cumulateChoice;
public:
    bool isTerminate();
    int computeReward();
    void setValue(int value_);
};

class Node
{
public:
    int visitTimes;
    int quality;
    int parentID;
    State state;
    std::vector<int> childID;
public:
    bool isAllExpanded();
    void setParent(int parentID_);
    void addChirld(int childID_);
};

class MCTS
{
public:
    std::vector<int> indexes;
    std::vector<Node> nodes;
public:
    MCTS();
    void impl();
    void expand();
    void treePolicy();
    void defaultPolicy();
    void bestChild();
    void backup();
};

#endif // MCTS_H
