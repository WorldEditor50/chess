#ifndef TENSOR_H
#define TENSOR_H
#include <cmath>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <assert.h>
#include "simd_ops.hpp"

namespace RL {

template<typename T, template<typename Ti> class Alloc=std::allocator>
class Tensor_
{
public:
    using ValueType = T;
    using Vector = std::vector<T, Alloc<T> >;
    using Shape = std::vector<int>;
    using Size = std::vector<int>;
    using iterator = typename Vector::iterator;
    using const_iterator = typename Vector::const_iterator;
    class SubTensor
    {
    public:
        Tensor_ *pointer;
        std::size_t pos;
        std::size_t totalSize;
    public:
        SubTensor():pointer(nullptr),pos(0),totalSize(0){}
        SubTensor(const SubTensor &r):pointer(r.pointer),pos(r.pos),totalSize(r.totalSize){}
        SubTensor& operator=(const SubTensor &r)
        {
            if (this == &r) {
                return *this;
            }
            pointer = r.pointer;
            pos = r.pos;
            totalSize = r.totalSize;
            return *this;
        }
        inline void operator=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] = x.val[i];
            }
            return;
        }
        inline void operator=(const std::vector<T> &x)
        {
            for (std::size_t i = 0; i < x.size(); i++) {
                pointer->val[i + pos] = x[i];
            }
            return;
        }

        inline void operator=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] = x;
            }
            return;
        }

        inline void operator+=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] += x.val[i];
            }
            return;
        }

        inline void operator-=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] -= x.val[i];
            }
            return;
        }

        inline void operator*=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] *= x.val[i];
            }
            return;
        }

        inline void operator/=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] /= x.val[i];
            }
            return;
        }

        inline void operator+=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] += x;
            }
            return;
        }

        inline void operator-=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] -= x;
            }
            return;
        }

        inline void operator*=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] *= x;
            }
            return;
        }

        inline void operator/=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] /= x;
            }
            return;
        }

        inline T sum() const
        {
            T s = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                s += pointer->val[i + pos];
            }
            return s;
        }

        inline T mean() const
        {
            T s = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                s += pointer->val[i + pos];
            }
            return s/T(totalSize);
        }

        inline T variance(T u) const
        {
            T s = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                float d = pointer->val[i + pos] - u;
                s += d*d;
            }
            return s/T(totalSize);
        }

        inline T max() const
        {
            T maxVal = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                if (maxVal < val) {
                    maxVal = val;
                }
            }
            return maxVal;
        }

        inline std::size_t argmax() const
        {
            T maxVal = 0;
            std::size_t index = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                if (maxVal < val) {
                    maxVal = val;
                    index = i;
                }
            }
            return index;
        }

        inline T min() const
        {
            T minVal = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                if (minVal > val) {
                    minVal = val;
                }
            }
            return minVal;
        }

        inline std::size_t argmin() const
        {
            T minVal = 0;
            std::size_t index = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                if (minVal > val) {
                    minVal = val;
                    index = i;
                }
            }
            return index;
        }
        inline T norm2() const
        {
            T s = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                s += val*val;
            }
            return std::sqrt(s);
        }
    };

protected:
    SubTensor subTensor;
public:
    std::size_t totalSize;
    Vector val;
    Size sizes;
    Shape shape;
public:
    /* default construct */
    Tensor_():totalSize(0){}
    static std::vector<int> sizesOf(const std::vector<int> &shape)
    {
        std::vector<int> sizes(shape.size(), 1);
        for (std::size_t i = 0; i < shape.size() - 1; i++) {
            for (std::size_t j = i + 1; j < shape.size(); j++) {
                 sizes[i] *= shape[j];
            }
        }
        return sizes;
    }
    static void initParams(const Shape &shape, Size &sizes, std::size_t &totalsize)
    {
        totalsize = 1;
        for (std::size_t i = 0; i < shape.size(); i++) {
            totalsize *= shape[i];
        }
        sizes = std::vector<int>(shape.size(), 1);
        for (std::size_t i = 0; i < shape.size() - 1; i++) {
            for (std::size_t j = i + 1; j < shape.size(); j++) {
                 sizes[i] *= shape[j];
            }
        }
        return;
    }
    /* contruct with shape */
    explicit Tensor_(const Shape &shape_)
        :totalSize(1),shape(shape_)
    {
        initParams(shape, sizes, totalSize);
        val = std::vector<T, Alloc<T>>(totalSize, T(0));
    }

    explicit Tensor_(const Shape &shape_,
                     const std::vector<T, Alloc<T>> &val_):
        totalSize(1),shape(shape_),val(val_)
    {
        initParams(shape, sizes, totalSize);
    }


    explicit Tensor_(const std::initializer_list<int> &shape_,
                     const std::initializer_list<T> &val_):
        totalSize(1),shape(shape_),val(val_)
    {
        initParams(shape, sizes, totalSize);
    }

    explicit Tensor_(const std::vector<Tensor_> &x)
    {
        totalSize = x.size()*x[0].totalsize;
        sizes.push_back(x.size()*x[0].sizes[0]);
        sizes.push_back(x[0].sizes);
        shape.push_back(x.size());
        shape.push_back(x[0].shape);
        for (std::size_t i = 0; i < x.size(); i++) {
            val.push_back(x[i].val);
        }
    }

    explicit Tensor_(T x):
        totalSize(1),shape({1, 1}),val({x})
    {
        initParams(shape, sizes, totalSize);
    }
    /* construct with shape */
    template<typename ...Dim>
    explicit Tensor_(Dim ...dim):totalSize(1),shape({int(dim)...})
    {
        initParams(shape, sizes, totalSize);
        val = std::vector<T, Alloc<T> >(totalSize, T(0));
    }

    /* copy constructor */
    Tensor_(const Tensor_ &r)
        :totalSize(r.totalSize),shape(r.shape),sizes(r.sizes),val(r.val){}

    /* move construct */
    Tensor_(Tensor_ &&r):totalSize(r.totalSize)
    {
        totalSize = r.totalSize;
        shape.swap(r.shape);
        sizes.swap(r.sizes);
        val.swap(r.val);
        r.totalSize = 0;
    }

    inline operator T* () noexcept
    {
        return val.data();
    }

    inline operator Vector ()
    {
        return val;
    }

    inline Tensor_ operator - () const
    {
        Tensor_ x(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            x.val[i] = -val[i];
        }
        return x;
    }
    bool shapeEqual(const Tensor_ &x) const
    {
        bool flag = true;
        for (std::size_t i = 0; i < shape.size(); i++) {
            if (shape[i] != x.shape[i]) {
                flag = false;
                break;
            }
        }
        return flag;
    }
    inline T* ptr() noexcept { return val.data(); }
    inline const T* ptr() const noexcept { return val.data(); }
    inline bool empty() const {return totalSize == 0;}
    /* iterator */
    inline iterator begin() noexcept { return val.begin();}
    inline const_iterator begin() const noexcept { return val.begin();}
    inline iterator end() noexcept { return val.end();}
    inline const_iterator end() const noexcept { return val.end();}
    /* size */
    inline std::size_t size() const {return totalSize;}

    template<typename ...Index>
    inline std::size_t size(Index ...index) const
    {
        std::size_t N = sizeof ...(Index) - 1;
        return sizes[N];
    }

    inline std::size_t size(const Shape &indexes) const
    {
        std::size_t N = indexes.size() - 1;
        return sizes[N];
    }

    /*
       zero/fill 走 SIMD 内核 (内核写满所有元素, 所以 val 的长度必须已经是对的长度;
       assign 保证这一点, 且这里要的不是"赋同一个值"而是"原地填", 因此用 data()).
    */
    void zero(){simdops::fill(val.data(), T(0), val.size());}
    void fill(T value){simdops::fill(val.data(), value, val.size());}
    inline T &operator[](int i) {return val[i];}
    inline T operator[](int i) const {return val[i];}

    /* assign operator */
    inline Tensor_& operator=(const Tensor_ &r)
    {
        if (this == &r) {
            return *this;
        }
        totalSize = r.totalSize;
        shape = r.shape;
        sizes = r.sizes;
        val = r.val;
        return *this;
    }

    inline Tensor_& operator=(const std::vector<T> &x)
    {
        val.assign(x.begin(), x.end());
        return *this;
    }
    inline Tensor_& operator=(T x)
    {
        val.assign(totalSize, x);
        return *this;
    }

    /* move */
    Tensor_ &operator=(Tensor_ &&r)
    {
        if (this == &r) {
            return *this;
        }
        totalSize = r.totalSize;
        shape.swap(r.shape);
        sizes.swap(r.sizes);
        val.swap(r.val);
        r.totalSize = 0;
        return *this;
    }
    /* init */
    static Tensor_ zeros(Shape &shape)
    {
        Tensor_ x(shape);
        return x;
    }

    template<typename ...Dim>
    static Tensor_ zeros(Dim ...dim)
    {
        Tensor_ x(dim...);
        return x;
    }

    static Tensor_ ones(Shape &shape)
    {
        Tensor_ x(shape);
        x.fill(1);
        return x;
    }

    template<typename ...Dim>
    static Tensor_ ones(Dim ...dim)
    {
        Tensor_ x(dim...);
        x.fill(1);
        return x;
    }
    /* subset */
    template<typename ...Index>
    Tensor_ sub(Index ...index) const
    {
        std::size_t N = sizeof ...(Index);
        std::vector<int> subIndex(shape.begin() + N, shape.end());
        Tensor_ y(subIndex);
        std::size_t pos = posOf(index...);
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = val[i + pos];
        }
        return y;
    }

    template<typename ...Index>
    inline SubTensor& at(Index ...index)
    {
        subTensor.pointer = this;
        subTensor.pos = posOf(index...);
        subTensor.totalSize = size(index...);
        return subTensor;
    }

    Tensor_ block(const std::vector<int> &offset, const std::vector<int> &blockShape) const
    {
        Tensor_ y(blockShape);
        std::vector<int> indexs(shape.size(), 0);
        for (std::size_t i = 0; i < y.totalSize; i++) {
            /* local offset */
            y.indexOf(i, indexs);
            for (std::size_t j = 0; j < indexs.size(); j++) {
                indexs[j] += offset[j];
            }
            std::size_t o = posOf(indexs);
            y.val[i] = val[o];
        }
        return y;
    }

    void embedding(const std::vector<int> &offset, const Tensor_ &x)
    {
        std::vector<int> indexs(shape.size(), 0);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.indexOf(i, indexs);
            for (std::size_t j = 0; j < indexs.size(); j++) {
                indexs[j] += offset[j];
            }
            std::size_t o = posOf(indexs);
            val[o] = x.val[i];
        }
        return;
    }

    template<typename ...Index>
    void slice(Tensor_ &y, Index ...index) const
    {
        std::size_t pos = posOf(index...);
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = val[i + pos];
        }
        return;
    }

    void toVector(std::vector<Tensor_> &vec) const
    {
        for (std::size_t i = 0; i < shape[0]; i++) {
            vec.push_back(sub(i));
        }
        return;
    }

    static Tensor_ fromVector(const std::vector<Tensor_> &vec)
    {
        if (vec.empty()) {
            return Tensor_();
        }
        /* Stack the sub-tensors: shape (N, d0, d1, ...) from N tensors of shape
           (d0, d1, ...). The old code std::copy'd the sub-tensor's VALUES into
           `shape` starting at begin()+1, which wrote past the end of a
           1-element vector (out-of-bounds) and produced a bogus shape. */
        std::vector<int> shape;
        shape.push_back(static_cast<int>(vec.size()));
        for (std::size_t i = 0; i < vec[0].shape.size(); i++) {
            shape.push_back(vec[0].shape[i]);
        }
        Tensor_ x(shape);
        std::size_t offset = 0;
        for (std::size_t i = 0; i < vec.size(); i++) {
            for (std::size_t j = 0; j < vec[i].totalSize; j++) {
                x.val[j + offset] = vec[i][j];
            }
            offset += vec[i].totalSize;
        }
        return x;
    }

    /* visit */
    template<typename ...Index>
    inline std::size_t posOf(Index ...index) const
    {
        int indexs[] = {index...};
        std::size_t pos = 0;
        std::size_t N = sizeof... (Index);
        for (std::size_t i = 0; i < N; i++) {
            pos += sizes[i]*indexs[i];
        }
        return pos;
    }

    inline std::size_t posOf(const std::vector<int> &indexs) const
    {
        std::size_t pos = 0;
        for (std::size_t i = 0; i < sizes.size(); i++) {
            pos += sizes[i]*indexs[i];
        }
        return pos;
    }

    inline static std::size_t posOf(const std::vector<int> &indexs, const std::vector<int> &shape)
    {
        std::vector<int> sizes = sizesOf(shape);
        std::size_t pos = 0;
        for (std::size_t i = 0; i < sizes.size(); i++) {
            pos += sizes[i]*indexs[i];
        }
        return pos;
    }
    inline void indexOf(int pos, std::vector<int> &indexs) const
    {
        /*
            shape: (2, 3, 4, 5)
            sizes: (60, 20, 5, 1)
            totalsize 2*3*4*5 = 120
            indexs:(1, 2, 3, 4)
            pos : 60*1 + 20*2 + 5*3 + 4*1 = 119

            i0 = pos/60
            i1 = (pos - i0*60)/20
            i2 = (pos - i0*60 - i1*20)/5
            i3 = pos - i0*60 - i1*20 - i2*5
        */
        int pos_ = 0;
        for (std::size_t i = 0; i < sizes.size(); i++) {
            indexs[i] = (pos - pos_)/sizes[i];
            pos_ += indexs[i]*sizes[i];
        }
        return;
    }

    inline std::vector<int> indexOf(int pos) const
    {
        std::vector<int> indexes(shape.size(), 0);
        indexOf(pos, indexes);
        return indexes;
    }

    template<typename ...Index>
    inline T &operator()(Index ...index) { return val[posOf(index...)]; }

    template<typename ...Index>
    inline T operator()(Index ...index) const { return val[posOf(index...)]; }

    inline T &operator()(const Shape &indexs) { return val[posOf(indexs)]; }

    inline T operator()(const Shape &indexs) const { return val[posOf(indexs)]; }

    template<typename ...Index>
    Tensor_& reshape(Index ...index)
    {
        shape = {index...};
        initParams(shape, sizes, totalSize);
        return *this;
    }

    template<typename ...Index>
    Tensor_ view(Index ...index) const
    {
        Tensor_ x = *this;
        x.shape = {index...};
        x.initParams(x.shape, x.sizes, x.totalSize);
        return x;
    }

    Tensor_ flatten() const
    {
        /*
         * Return a genuine 2-D column (totalSize x 1), NOT a 1-D {totalSize}
         * shape.
         *
         * The convolution -> fully-connected path in Net::forward/backward does
         * `layers[i]->forward(out.flatten())` / `layer->backward(out.flatten(), e)`,
         * and those consumers index the tensor as 2-D: e.g. ikjk() computes
         * `x2(j, k)` -> posOf(j, k) -> `sizes[0]*j + sizes[1]*k`. With a 1-D
         * shape, `sizes` has a single element, so `sizes[1]` was an
         * out-of-bounds vector read; the resulting garbage was multiplied by k
         * (up to 31) and produced a wild element access. Forward happened to
         * survive because there j is always 0, but the backward pass crashed —
         * which is what ConvPG/ConvDQN hit on the very first training step.
         */
        Tensor_ x(static_cast<int>(totalSize), 1);
        x.val = val;
        return x;
    }

    static void permuteIndexs(const std::vector<int> &indexs,
                              const std::vector<int> &permuteMap,
                              std::vector<int> &newIndexs)
    {
        for (std::size_t i = 0; i < permuteMap.size(); i++) {
            int k = permuteMap[i];
            newIndexs[i] = indexs[k];
        }
        return;
    }

    template<typename ...Pos>
    inline Tensor_ permute(Pos ...p) const
    {
        /*
            shape: [3, 2, 1]
            permute: (2, 1, 0)
            new shape: [1, 2, 3]
        */
        std::vector<int> permuteMap = {p...};
        /* permute shape */
        std::vector<int> newShape(shape.size(), 0);
        permuteIndexs(shape, permuteMap, newShape);
        Tensor_ x(newShape);
        /* permute value */
        std::vector<int> indexs(shape.size(), 0);
        std::vector<int> newIndexs(shape.size(), 0);
        for (std::size_t i = 0; i < val.size(); i++) {
            indexOf(i, indexs);
            permuteIndexs(indexs, permuteMap, newIndexs);
            x(newIndexs) = val[i];
        }
        return x;
    }

    Tensor_ tr() const
    {
        int rows = shape[0];
        int cols = shape[1];
        Tensor_ y(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y(j, i) = val[i*cols + j];
            }
        }
        return y;
    }

    /* operator
       ------------------------------------------------
       逐元素运算分派到 SIMD 内核 (见 rl/simd_ops.hpp 的契约说明)。
       内核要求两个操作数形状/长度相同且长度 >= 一个向量宽度, 否则回落到标量循环;
       原来的实现遍历的就是 val.size() 且直接索引 x.val[i] (没有广播语义), 所以
       只要两边长度相等就能安全替换。
    */
    Tensor_ operator +(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        simdops::add(y.val.data(), val.data(), x.val.data(), val.size());
        return y;
    }

    Tensor_ operator -(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        simdops::sub(y.val.data(), val.data(), x.val.data(), val.size());
        return y;
    }

    Tensor_ operator *(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        simdops::mul(y.val.data(), val.data(), x.val.data(), val.size());
        return y;
    }

    Tensor_ operator /(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        simdops::div(y.val.data(), val.data(), x.val.data(), val.size());
        return y;
    }

    Tensor_ operator %(const Tensor_ &x) const
    {
        Tensor_ y(shape[0], x.shape[1]);
        for (std::size_t i = 0; i < y.shape[0]; i++) {
            for (std::size_t k = 0; k < shape[1]; k++) {
                T vik = val[posOf(i, k)];
                for (std::size_t j = 0; j < y.shape[1]; j++) {
                    /* y(i, j) = val(i, k) * x(k, j) */
                    y(i, j) += vik*x(k, j);
                }
            }
        }
        return y;
    }

    Tensor_ &operator +=(const Tensor_ &x)
    {
        simdops::add(val.data(), val.data(), x.val.data(), val.size());
        return *this;
    }

    Tensor_ &operator -=(const Tensor_ &x)
    {
        simdops::sub(val.data(), val.data(), x.val.data(), val.size());
        return *this;
    }

    Tensor_ &operator *=(const Tensor_ &x)
    {
        simdops::mul(val.data(), val.data(), x.val.data(), val.size());
        return *this;
    }

    /* 原来返回的是 Tensor_ 值 (每次 /= 都多复制一份张量), 改为引用 */
    Tensor_ &operator /=(const Tensor_ &x)
    {
        simdops::div(val.data(), val.data(), x.val.data(), val.size());
        return *this;
    }

    Tensor_ operator +(T x) const
    {
        Tensor_ y(shape);
        simdops::add(y.val.data(), val.data(), x, val.size());
        return y;
    }

    Tensor_ operator -(T x) const
    {
        Tensor_ y(shape);
        simdops::sub(y.val.data(), val.data(), x, val.size());
        return y;
    }

    Tensor_ operator *(T x) const
    {
        Tensor_ y(shape);
        simdops::mul(y.val.data(), val.data(), x, val.size());
        return y;
    }

    Tensor_ operator /(T x) const
    {
        Tensor_ y(shape);
        simdops::div(y.val.data(), val.data(), x, val.size());
        return y;
    }

    Tensor_ &operator +=(T x)
    {
        simdops::add(val.data(), val.data(), x, val.size());
        return *this;
    }

    Tensor_ &operator -=(T x)
    {
        simdops::sub(val.data(), val.data(), x, val.size());
        return *this;
    }

    Tensor_ &operator *=(T x)
    {
        simdops::mul(val.data(), val.data(), x, val.size());
        return *this;
    }

    Tensor_ &operator /=(T x)
    {
        simdops::div(val.data(), val.data(), x, val.size());
        return *this;
    }

    /* statistics */
    T sum() const
    {
        return simdops::sum(val.data(), totalSize);
    }

    T mean() const
    {
        T s = sum();
        return s/T(totalSize);
    }

    T variance(T u) const
    {
        T s = 0;
        return simdops::variance(val.data(), u, val.size());
    }

    T max() const
    {
        return simdops::maxValue(val.data(), val.size());
    }

    T min() const
    {
        return simdops::minValue(val.data(), val.size());
    }

    std::size_t argmax() const
    {
        T value = val[0];
        std::size_t index = 0;
        for (std::size_t i = 0; i < val.size(); i++) {
            if (value < val[i]) {
                value = val[i];
                index = i;
            }
        }
        return index;
    }

    std::size_t argmin() const
    {
        T value = val[0];
        std::size_t index = 0;
        for (std::size_t i = 0; i < val.size(); i++) {
            if (value > val[i]) {
                value = val[i];
                index = i;
            }
        }
        return index;
    }

    /* initialize */
    void normalize()
    {
        double minValue = val[0];
        double maxValue = val[0];
        for (std::size_t i = 0; i < val.size(); i++) {
            if (minValue > val[i]) {
                minValue = val[i];
            }
            if (maxValue < val[i]) {
                maxValue = val[i];
            }
        }
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] = (val[i] - minValue)/(maxValue - minValue);
        }
        return;
    }

    T norm2() const
    {
        return std::sqrt(simdops::dot(val.data(), val.data(), totalSize));
    }
    struct MM {
        /*
         * PERFORMANCE (measured on this machine: MSVC 2022 /O2, x64, Release,
         * Qt 6.9.2 - same benchmark as the numbers quoted per kernel below).
         *
         * Every kernel here used to index through operator(), and operator()
         * calls posOf(), which rebuilds an `int indexs[]` array from the
         * parameter pack on every single access, loops over it and re-reads
         * sizes[i] out of a std::vector<int>.  With two or three accesses per
         * multiply-accumulate that is ~18 ns/MAC (0.11 GFLOP/s), and it also
         * prevents the compiler from vectorising anything.
         *
         * The kernels below therefore take the three element buffers as flat
         * pointers and hoist every stride out of the loops:
         *
         *     x(i,  j) == xd [i*xr  + j*xc ]
         *     x1(i, k) == x1d[i*x1r + k*x1c]
         *     x2(k, j) == x2d[k*x2r + j*x2c]
         *
         * which is literally what posOf() computes (sizes[0]*idx0 +
         * sizes[1]*idx1); the strides are READ from sizes[], never assumed, so
         * the mapping is unchanged for every shape (including (n,1) column
         * vectors, whose sizes[] is {1,1}, not {n,1}).
         *
         * x is still ACCUMULATED into, never assigned, and no bounds check was
         * added: out-of-range behaviour stays "undefined" exactly as before.
         *
         * ROUNDING: the flat loops below keep the original k-ascending,
         * j-ascending accumulation order, so those are bit-for-bit identical to
         * the old code.  Two fast paths do reassociate the additions:
         *   - the 4-way unrolled row update in ikkj/kikj (4 k-values per pass
         *     over the x row), and
         *   - the k-inner 4-accumulator dot product in ikjk/kijk (the original
         *     i,k,j nest walked x2 at stride sizes[0], i.e. a gather).
         * Both add exactly the same products to the same elements, so the
         * mathematical result is identical and only the rounding order differs.
         * Cross-checked elementwise (in-place + returning variants) against the
         * naive posOf() triple loop over 26 shapes - square, rectangular,
         * (n,1) column vectors, odd sizes and >2-D shapes that exercise the
         * generic stride loops: max |difference| = 2.9e-05 on results of
         * magnitude 25.1, i.e. 1.2e-06 relative.
         *
         * That residual is dominated by the OLD kernel's own rounding error,
         * not by this change: for ikkj (90x360)*(360x90) against a
         * double-precision reference |old - exact| = 1.8e-05 but
         * |new - exact| = 8.4e-06 - the reassociated version here is the more
         * accurate of the two, and no ordering (not even the exact one) could
         * bring |old - new| below |old - exact|.
         *
         * MEASURED before -> after (same machine, same benchmark, MSVC 2022
         * /O2 /fp:precise, 300/200 repetitions per case, median of 3 runs, and
         * every "after" number is >50x the "before" number):
         *   ikkj (90x90)*(90x90)     18.07 -> 0.10 ns/MAC   0.11 -> 20.8 GFLOP/s
         *   ikkj (90x360)*(360x90)   18.01 -> 0.09 ns/MAC   0.11 -> 21.1 GFLOP/s
         *   kikj (360x90)^T*(360x90) 18.34 -> 0.09 ns/MAC   0.11 -> 21.4 GFLOP/s
         *   ikjk (128x64)*(64x90)^T  18.11 -> 0.33 ns/MAC   0.11 ->  6.0 GFLOP/s
         */
        /*
            2 维、连续行主序 (sizes == {cols, 1}) 的张量才能用 SIMD 内核: 内核里的
            下标是硬编码的 i*col + k 形式, 必须与 posOf() 的通用 stride 计算等价。
            1 维或 >2 维、以及被 block() 出来的非连续视图一律回落到下面的标量实现。
        */
        static bool contiguous2d(const Tensor_ &t)
        {
            return t.shape.size() == 2 && t.sizes.size() == 2 &&
                   t.sizes[0] == t.shape[1] && t.sizes[1] == 1;
        }

        inline static void ikkj(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /*
               SIMD 快速路径 (内核来自 N-spirits 的 simd/avx2func.hpp, 经
               rl/simd_ops.hpp 分派)。内核是**累加**到目标里的, 与本函数语义一致 ——
               这里不要像 tensorsi.hpp 那样补一次 zero()。
            */
            if (x.shape.size() == 2 && x.shape[1] == 1 && x.sizes[0] == 1 &&
                x1.shape.size() == 2 && x1.sizes[0] == x1.shape[1] && x1.sizes[1] == 1 &&
                x2.shape.size() == 2 && x2.shape[1] == 1 && x2.sizes[0] == 1 &&
                x.shape[0] == x1.shape[0] && x1.shape[1] == x2.shape[0]) {
                if (simdops::gemv_ikkj<T>((T*)x.val.data(), (std::size_t)x.shape[0],
                                          x1.val.data(), (std::size_t)x1.shape[1],
                                          x2.val.data())) {
                    return;
                }
            }
            if (contiguous2d(x) && contiguous2d(x1) && contiguous2d(x2) &&
                x.shape[0] == x1.shape[0] && x1.shape[1] == x2.shape[0] &&
                x.shape[1] == x2.shape[1]) {
                if (simdops::mm_ikkj<T>((T*)x.val.data(), (std::size_t)x.shape[0], (std::size_t)x.shape[1],
                                        x1.val.data(), (std::size_t)x1.shape[0], (std::size_t)x1.shape[1],
                                        x2.val.data(), (std::size_t)x2.shape[0], (std::size_t)x2.shape[1])) {
                    return;
                }
            }
            const T *x1d = x1.val.data();
            const T *x2d = x2.val.data();
            T *xd = x.val.data();
            const std::size_t xr  = (std::size_t)x.sizes[0],  xc  = (std::size_t)x.sizes[1];
            const std::size_t x1r = (std::size_t)x1.sizes[0], x1c = (std::size_t)x1.sizes[1];
            const std::size_t x2r = (std::size_t)x2.sizes[0], x2c = (std::size_t)x2.sizes[1];
            const std::size_t rows = (std::size_t)x.shape[0];
            const std::size_t kdim = (std::size_t)x1.shape[1];
            const std::size_t cols = (std::size_t)x.shape[1];
            if (xc == 1 && x2c == 1) {
                /* x(i, .) and x2(k, .) are both contiguous in j: the inner loop
                 * is a unit-stride row update with a broadcast scalar, and four
                 * k-values are fused so the x row is loaded/stored once per four
                 * MACs instead of once per MAC.  Before: 18.07 ns/MAC
                 * (0.11 GFLOP/s) for ikkj (90x90)*(90x90), now 0.10 ns/MAC
                 * (20.8 GFLOP/s) - ~180x. */
                for (std::size_t i = 0; i < rows; i++) {
                    T *xrow = xd + i*xr;
                    std::size_t k = 0;
                    for (; k + 4 <= kdim; k += 4) {
                        const T v0 = x1d[i*x1r + (k + 0)*x1c];
                        const T v1 = x1d[i*x1r + (k + 1)*x1c];
                        const T v2 = x1d[i*x1r + (k + 2)*x1c];
                        const T v3 = x1d[i*x1r + (k + 3)*x1c];
                        const T *r0 = x2d + (k + 0)*x2r;
                        const T *r1 = x2d + (k + 1)*x2r;
                        const T *r2 = x2d + (k + 2)*x2r;
                        const T *r3 = x2d + (k + 3)*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(i, k) * x2(k, j) */
                            xrow[j] += v0*r0[j] + v1*r1[j] + v2*r2[j] + v3*r3[j];
                        }
                    }
                    for (; k < kdim; k++) {
                        const T x1ik = x1d[i*x1r + k*x1c];
                        const T *x2row = x2d + k*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(i, k) * x2(k, j) */
                            xrow[j] += x1ik*x2row[j];
                        }
                    }
                }
            } else {
                /* generic strides (e.g. >2-D shapes): same order as the old code */
                for (std::size_t i = 0; i < rows; i++) {
                    for (std::size_t k = 0; k < kdim; k++) {
                        const T x1ik = x1d[i*x1r + k*x1c];
                        const T *x2row = x2d + k*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(i, k) * x2(k, j) */
                            xd[i*xr + j*xc] += x1ik*x2row[j*x2c];
                        }
                    }
                }
            }
            return;
        }

        inline static void kikj(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* SIMD 快速路径, 见 ikkj 的说明 */
            if (contiguous2d(x) && contiguous2d(x1) && contiguous2d(x2) &&
                x1.shape[0] == x2.shape[0] &&
                x.shape[0] == x1.shape[1] && x.shape[1] == x2.shape[1]) {
                if (simdops::mm_kikj<T>((T*)x.val.data(), (std::size_t)x.shape[0], (std::size_t)x.shape[1],
                                        x1.val.data(), (std::size_t)x1.shape[0], (std::size_t)x1.shape[1],
                                        x2.val.data(), (std::size_t)x2.shape[0], (std::size_t)x2.shape[1])) {
                    return;
                }
            }
            /* transpose x1 */
            const T *x1d = x1.val.data();
            const T *x2d = x2.val.data();
            T *xd = x.val.data();
            const std::size_t xr  = (std::size_t)x.sizes[0],  xc  = (std::size_t)x.sizes[1];
            const std::size_t x1r = (std::size_t)x1.sizes[0], x1c = (std::size_t)x1.sizes[1];
            const std::size_t x2r = (std::size_t)x2.sizes[0], x2c = (std::size_t)x2.sizes[1];
            const std::size_t rows = (std::size_t)x.shape[0];
            const std::size_t kdim = (std::size_t)x1.shape[0];
            const std::size_t cols = (std::size_t)x.shape[1];
            if (xc == 1 && x2c == 1) {
                /* as in ikkj: unit-stride row update, 4 k-values fused.
                 * Before: (360x90)^T*(360x90) = 18.34 ns/MAC (0.11 GFLOP/s),
                 * now 0.09 ns/MAC (21.4 GFLOP/s). */
                for (std::size_t i = 0; i < rows; i++) {
                    T *xrow = xd + i*xr;
                    std::size_t k = 0;
                    for (; k + 4 <= kdim; k += 4) {
                        const T v0 = x1d[(k + 0)*x1r + i*x1c];
                        const T v1 = x1d[(k + 1)*x1r + i*x1c];
                        const T v2 = x1d[(k + 2)*x1r + i*x1c];
                        const T v3 = x1d[(k + 3)*x1r + i*x1c];
                        const T *r0 = x2d + (k + 0)*x2r;
                        const T *r1 = x2d + (k + 1)*x2r;
                        const T *r2 = x2d + (k + 2)*x2r;
                        const T *r3 = x2d + (k + 3)*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(k, i)^T * x2(k, j) */
                            xrow[j] += v0*r0[j] + v1*r1[j] + v2*r2[j] + v3*r3[j];
                        }
                    }
                    for (; k < kdim; k++) {
                        const T x1ki = x1d[k*x1r + i*x1c];
                        const T *x2row = x2d + k*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(k, i)^T * x2(k, j) */
                            xrow[j] += x1ki*x2row[j];
                        }
                    }
                }
            } else {
                /* generic strides: same order as the old code */
                for (std::size_t i = 0; i < rows; i++) {
                    for (std::size_t k = 0; k < kdim; k++) {
                        const T x1ki = x1d[k*x1r + i*x1c];
                        const T *x2row = x2d + k*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(k, i)^T * x2(k, j) */
                            xd[i*xr + j*xc] += x1ki*x2row[j*x2c];
                        }
                    }
                }
            }
            return;
        }

        inline static void ikjk(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* SIMD 快速路径, 见 ikkj 的说明 */
            if (contiguous2d(x) && contiguous2d(x1) && contiguous2d(x2) &&
                x1.shape[1] == x2.shape[1] &&
                x.shape[0] == x1.shape[0] && x.shape[1] == x2.shape[0]) {
                if (simdops::mm_ikjk<T>((T*)x.val.data(), (std::size_t)x.shape[0], (std::size_t)x.shape[1],
                                        x1.val.data(), (std::size_t)x1.shape[0], (std::size_t)x1.shape[1],
                                        x2.val.data(), (std::size_t)x2.shape[0], (std::size_t)x2.shape[1])) {
                    return;
                }
            }
            /* transpose x2 */
            const T *x1d = x1.val.data();
            const T *x2d = x2.val.data();
            T *xd = x.val.data();
            const std::size_t xr  = (std::size_t)x.sizes[0],  xc  = (std::size_t)x.sizes[1];
            const std::size_t x1r = (std::size_t)x1.sizes[0], x1c = (std::size_t)x1.sizes[1];
            const std::size_t x2r = (std::size_t)x2.sizes[0], x2c = (std::size_t)x2.sizes[1];
            const std::size_t rows = (std::size_t)x.shape[0];
            const std::size_t kdim = (std::size_t)x1.shape[1];
            const std::size_t cols = (std::size_t)x.shape[1];
            if (xc == 1 && x1c == 1 && x2c == 1) {
                /* x1(i, .) and x2(j, .) are contiguous in k, so the sum over k is
                 * a dot product of two contiguous rows.  The loop nest is j,k
                 * inner here (the old i,k,j nest read x2(j, k) at stride
                 * sizes[0], a gather that cannot be vectorised) and four
                 * independent accumulators keep the FMA units busy - with
                 * /fp:precise the compiler will not reassociate a single
                 * accumulator chain by itself.
                 * Before: (128x64)*(64x90)^T = 18.11 ns/MAC (0.11 GFLOP/s),
                 * now 0.33 ns/MAC (6.0 GFLOP/s) - the remaining cost is the
                 * (non-vectorisable under /fp:precise) accumulation chain;
                 * materialising a transposed x2 to make this an element-wise
                 * row update was measured and only bought ~1.1x, so it was not
                 * worth the per-call scratch buffer. */
                for (std::size_t i = 0; i < rows; i++) {
                    const T *x1row = x1d + i*x1r;
                    T *xrow = xd + i*xr;
                    for (std::size_t j = 0; j < cols; j++) {
                        const T *x2row = x2d + j*x2r;
                        T a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                        std::size_t k = 0;
                        for (; k + 4 <= kdim; k += 4) {
                            /* x(i, j) += x1(i, k) * x2(j, k)^T */
                            a0 += x1row[k + 0]*x2row[k + 0];
                            a1 += x1row[k + 1]*x2row[k + 1];
                            a2 += x1row[k + 2]*x2row[k + 2];
                            a3 += x1row[k + 3]*x2row[k + 3];
                        }
                        T acc = (a0 + a1) + (a2 + a3);
                        for (; k < kdim; k++) {
                            acc += x1row[k]*x2row[k];
                        }
                        xrow[j] += acc;
                    }
                }
            } else {
                /* generic strides: same order as the old code */
                for (std::size_t i = 0; i < rows; i++) {
                    for (std::size_t j = 0; j < cols; j++) {
                        T acc = 0;
                        for (std::size_t k = 0; k < kdim; k++) {
                            /* x(i, j) = x1(i, k) * x2(j, k)^T */
                            acc += x1d[i*x1r + k*x1c]*x2d[j*x2r + k*x2c];
                        }
                        xd[i*xr + j*xc] += acc;
                    }
                }
            }
            return;
        }

        inline static void kijk(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* SIMD 快速路径, 见 ikkj 的说明 */
            if (contiguous2d(x) && contiguous2d(x1) && contiguous2d(x2) &&
                x1.shape[0] == x2.shape[1] &&
                x.shape[0] == x1.shape[1] && x.shape[1] == x2.shape[0]) {
                if (simdops::mm_kijk<T>((T*)x.val.data(), (std::size_t)x.shape[0], (std::size_t)x.shape[1],
                                        x1.val.data(), (std::size_t)x1.shape[0], (std::size_t)x1.shape[1],
                                        x2.val.data(), (std::size_t)x2.shape[0], (std::size_t)x2.shape[1])) {
                    return;
                }
            }
            /* transpose x1, x2 */
            const T *x1d = x1.val.data();
            const T *x2d = x2.val.data();
            T *xd = x.val.data();
            const std::size_t xr  = (std::size_t)x.sizes[0],  xc  = (std::size_t)x.sizes[1];
            const std::size_t x1r = (std::size_t)x1.sizes[0], x1c = (std::size_t)x1.sizes[1];
            const std::size_t x2r = (std::size_t)x2.sizes[0], x2c = (std::size_t)x2.sizes[1];
            const std::size_t rows = (std::size_t)x.shape[0];
            const std::size_t kdim = (std::size_t)x1.shape[0];
            const std::size_t cols = (std::size_t)x.shape[1];
            if (xc == 1 && x2c == 1) {
                /* x2(j, .) is contiguous in k (x1's column is walked at stride
                 * sizes[0], i.e. a broadcast load per k).  Same j,k-inner nest
                 * and 4-accumulator trick as ikjk. */
                for (std::size_t i = 0; i < rows; i++) {
                    T *xrow = xd + i*xr;
                    for (std::size_t j = 0; j < cols; j++) {
                        const T *x1col = x1d + i*x1c;
                        const T *x2row = x2d + j*x2r;
                        T a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                        std::size_t k = 0;
                        for (; k + 4 <= kdim; k += 4) {
                            /* x(i, j) += x1(k, i)^T * x2(j, k)^T */
                            a0 += x1col[(k + 0)*x1r]*x2row[k + 0];
                            a1 += x1col[(k + 1)*x1r]*x2row[k + 1];
                            a2 += x1col[(k + 2)*x1r]*x2row[k + 2];
                            a3 += x1col[(k + 3)*x1r]*x2row[k + 3];
                        }
                        T acc = (a0 + a1) + (a2 + a3);
                        for (; k < kdim; k++) {
                            acc += x1col[k*x1r]*x2row[k];
                        }
                        xrow[j] += acc;
                    }
                }
            } else {
                /* generic strides: same order as the old code */
                for (std::size_t i = 0; i < rows; i++) {
                    for (std::size_t j = 0; j < cols; j++) {
                        T acc = 0;
                        for (std::size_t k = 0; k < kdim; k++) {
                            /* x(i, j) = x1(k, i)^T * x2(j, k)^T */
                            acc += x1d[k*x1r + i*x1c]*x2d[j*x2r + k*x2c];
                        }
                        xd[i*xr + j*xc] += acc;
                    }
                }
            }
            return;
        }

        inline static Tensor_ ikkj(const Tensor_ &x1, const Tensor_ &x2)
        {
            /* Tensor_ ctor zero-fills, so accumulating the result of the fast
             * in-place kernel into it is the same computation as before (the
             * old body duplicated the ikkj loop instead of reusing it). */
            Tensor_ x(x1.shape[0], x2.shape[1]);
            ikkj(x, x1, x2);
            return x;
        }

        inline static Tensor_ kikj(const Tensor_ &x1, const Tensor_ &x2)
        {
            /* transpose x1; Tensor_ zero-fills x, so the in-place kernel gives
             * exactly what the old duplicated loop computed */
            Tensor_ x(x1.shape[1], x2.shape[1]);
            kikj(x, x1, x2);
            return x;
        }

        inline static Tensor_ ikjk(const Tensor_ &x1, const Tensor_ &x2)
        {
            Tensor_ x(x1.shape[0], x2.shape[0]);
            /* transpose x2 */
            ikjk(x, x1, x2);
            return x;
        }

        inline static Tensor_ kijk(const Tensor_ &x1, const Tensor_ &x2)
        {
            Tensor_ x(x1.shape[1], x2.shape[0]);
            /* transpose x1, x2 */
            kijk(x, x1, x2);
            return x;
        }
    };

    /* batch matrix multiplication */
    inline static Tensor_ bmm(const Tensor_ &x1, const Tensor_ &x2)
    {
        /*
            x1: (batch, n, m)
            x2: (batch, m, p) or (m, p) for broadcasting
            output: (batch, n, p)
        */
        if (x1.shape.size() == 3 && x2.shape.size() == 2) {
            /* broadcasting case: x2 has no batch dimension */
            int batch = x1.shape[0];
            int n = x1.shape[1];
            int p = x2.shape[1];
            assert(x1.shape[2] == x2.shape[0]);
            Tensor_ result(batch, n, p);
            for (int b = 0; b < batch; b++) {
                result.at(b) = x1.sub(b) % x2;
            }
            return result;
        }
        /* standard case: both are 3D tensors */
        assert(x1.shape.size() == 3 && x2.shape.size() == 3);
        assert(x1.shape[0] == x2.shape[0]);
        assert(x1.shape[2] == x2.shape[1]);
        int batch = x1.shape[0];
        int n = x1.shape[1];
        int p = x2.shape[2];
        Tensor_ result(batch, n, p);
        for (int b = 0; b < batch; b++) {
            result.at(b) = x1.sub(b) % x2.sub(b);
        }
        return result;
    }

    template<typename ...Arg>
    inline static Tensor_ concat(int dim, const Arg & ...args)
    {
        std::vector<Tensor_> xi = {args...};
        return concats(dim, xi);
    }

    inline static Tensor_ concats(int dim, const std::vector<Tensor_> &xi)
    {
        std::vector<int> newShape(xi[0].shape.size(), 0);
        for (std::size_t i = 0; i < xi.size(); i++) {
            newShape[dim] += xi[i].shape[dim];
        }
        for (std::size_t i = 0; i < xi[0].shape.size(); i++) {
            if (i != dim) {
                newShape[i] = xi[0].shape[i];
            }
        }
        Tensor_ x = Tensor_(newShape);
        int offset = 0;
        for (std::size_t i = 0; i < xi.size(); i++) {
            const Tensor_ &x_ = xi[i];
            /* set value */
            std::vector<int> indexs(x.shape.size(), 0);
            for (std::size_t j = 0; j < x_.totalSize; j++) {
                x_.indexOf(j, indexs);
                indexs[dim] += offset;
                x(indexs) = x_[j];
            }
            /* set offset */
            offset += x_.shape[dim];
        }
        return x;
    }

    inline static Tensor_ product2D(const Tensor_& x1, const Tensor_& x2)
    {
        int r = x1.shape[0]*x2.shape[0];
        int c = x1.shape[1]*x2.shape[1];
        Tensor_ y(r, c);
        for (int i = 0; i < x1.shape[0]; i++) {
            for (int j = 0; j < x1.shape[1]; j++) {
                for (int h = 0; h < x2.shape[0]; h++) {
                    for (int k = 0; k < x2.shape[1]; k++) {
                        y(h + i*x1.shape[0], k + j*x1.shape[1]) = x1(i, j)*x2(h, k);
                    }
                }
            }
        }
        return y;
    }

    inline static T dot(const Tensor_& x1, const Tensor_& x2)
    {
        return simdops::dot(x1.val.data(), x2.val.data(), x1.totalSize);
    }

    /* display */
    template<typename ...Index>
    void printValue(Index ...index) const
    {
        std::size_t N = size(index...);
        std::size_t pos = posOf(index...);
        std::cout<<"[";
        for (std::size_t i = 0; i < N; i++) {
            std::cout<<val[i + pos];
            if (i < N - 1) {
                std::cout<<",";
            }
        }
        std::cout<<"]"<<std::endl;
        return;
    }

    void printValue() const
    {
        std::cout<<"[";
        for (std::size_t i = 0; i < val.size(); i++) {
            std::cout<<val[i];
            if (i < totalSize - 1) {
                std::cout<<",";
            }
        }
        std::cout<<"]"<<std::endl;
        return;
    }

    void printValue2D() const
    {
        std::cout<<"[";
        for (std::size_t i = 0; i < shape[0]; i++) {
            for (std::size_t j = 0; j < shape[1]; j++) {
                std::cout<<val[i*shape[1] + j];
                /* was `i < totalSize - 1`, comparing a ROW index against the
                   element count, which printed commas in the wrong places */
                if (!(i == shape[0] - 1 && j == shape[1] - 1)) {
                    std::cout<<",";
                }
            }
            std::cout<<std::endl;
        }
        std::cout<<"]"<<std::endl;
        return;
    }
    void printShape() const
    {
        std::cout<<"(";
        for (std::size_t i = 0; i < shape.size(); i++) {
            std::cout<<shape[i];
            /* was `i < totalSize - 1` instead of the shape's own length */
            if (i != shape.size() - 1) {
                std::cout<<",";
            }
        }
        std::cout<<")"<<std::endl;
        return;
    }

    /*
       ============================================================
        Tensor 的序列化: 权重文件用 (见 Net::save / Net::load)
       ============================================================

       格式 v2 (现在是默认的):
           <shape 用逗号分隔>|b64:<base64 的原始数据>
       上一版是 `<shape>|<十进制浮点, 用逗号分隔>`, 有两个真问题:
         1. **有损**: `std::ostream << float` 默认 6 位有效数字。存一次再读回来,
            权重就会漂移 ~1e-6 相对 —— 而本工程的后台训练每轮都在
            save -> load (见 ChessBoard::backgroundTrainLoop 的 TMP_WEIGHTS),
            也就是说每轮都在往网络里注入一次不该有的扰动。
         2. **又大又慢**: 一个 float 要 9~13 个字节的文本 (还要 strtof/double 解析),
            实测 16 MB 的 DQN+MCTS 权重文件读写一次要几百毫秒。

        base64 之后是 5.33 字节/float、无损、编解码只做位运算。仍然保留"一行一个
       张量 + 用 `|` 分隔形状"的外层结构, 因为分层的 write/read **顺序**是现成的
       结构描述 (97 处调用点都依赖它), 而 base64 里不可能出现换行, 行式读取
       (`std::getline`) 因此仍然安全。

       `fromString` **同时接受两种格式** (老文件不带 `b64:` 前缀), 所以以前存下来的
       权重照样能读。解码失败 (长度不对 / 非法字符 / 数据被截断) 会置
       `lastDecodeFailed()`, 由 Net::load 汇总成一个"载入失败"返回给调用方 ——
       否则一个被截断的文件会静默地载入半个模型。
    */
    std::string toString() const
    {
        std::string out;
        for (std::size_t i = 0; i < shape.size(); i++) {
            out += std::to_string(shape[i]);
            out += (i + 1 == shape.size()) ? '|' : ',';
        }
        out += "b64:";
        const unsigned char *rawBytes =
            reinterpret_cast<const unsigned char *>(val.data());
        char crcHex[16];
        std::snprintf(crcHex, sizeof(crcHex), "%08x",
                      (unsigned int)crc32(rawBytes, val.size() * sizeof(T)));
        out += crcHex;
        out += ':';
        out += base64Encode(val);
        return out;
    }

    /* 人能读的版本 (调试用; 权重文件不用它, 因为它有损且更大) */
    std::string toDebugString() const
    {
        std::stringstream stream;
        for (std::size_t i = 0; i < shape.size(); i++) {
            stream << shape[i];
            if (i != shape.size() - 1) {
                stream << ",";
            } else {
                stream << "|";
            }
        }
        for (std::size_t i = 0; i < val.size(); i++) {
            stream << val[i];
            if (i < val.size() - 1) {
                stream << ",";
            }
        }
        return stream.str();
    }

    /*
       上一次 fromString 是否解码失败。做成静态的 "错误旗标" 是因为 97 处调用点
       都写成 `x = Tensor::fromString(s)`, 没有地方能接收错误码; 由 Net::load 在
       读完整层之前清零、之后检查, 就能把"文件坏了"变成一次明确的失败。
       (线程安全: 权重读写只在单一训练/AI 线程里发生, 且有锁保护, 见 ChessBoard。)
    */
    static bool &lastDecodeFailedRef()
    {
        static bool failed = false;
        return failed;
    }
    static bool lastDecodeFailed() { return lastDecodeFailedRef(); }
    static void clearDecodeFailed() { lastDecodeFailedRef() = false; }

    /*
       ============================================================
        快速结构校验 (给 Net::load 的"先校验一遍再真正载入"用)
       ============================================================
       它必须**不构造张量、不解析浮点数**: 一个 16 MB 的老格式权重文件里, 真正的
       fromString 要 `split()` 出几百万个 std::string (实测让 GUI 启动从秒级变成
       30 秒以上), 而这里只扫一遍字节、数一遍逗号。要求依然是严的:
         * 形状能解析且每个维度为正;
         * 值的个数必须恰好等于形状的乘积 (少一个就是被截断了);
         * v2 还要过 base64 合法性 + 长度 + CRC32。
    */
    /* 参数是**指针 + 长度**而不是 std::string: 预校验在一整块内存上跑, 一行可能
       是 34 MB, 每行都拷成 std::string 就白省了。 */
    static bool validateEncoded(const char *s, std::size_t sLen)
    {
        const void *barPtr = std::memchr(s, '|', sLen);
        if (barPtr == nullptr) {
            return false;
        }
        const std::size_t bar = (std::size_t)((const char *)barPtr - s);
        long long shapeProduct = 1;
        if (!parseShape(std::string(s, bar), shapeProduct)) {
            return false;
        }
        const char *p = s + bar + 1;
        std::size_t n = sLen - bar - 1;
        while (n > 0 && (p[n - 1] == '\r' || p[n - 1] == '\n' || p[n - 1] == ' ')) {
            n--;
        }
        if (n >= 4 && std::strncmp(p, "b64:", 4) == 0) {
            if (n < 4 + 8 + 1 || p[12] != ':') {
                return false;
            }
            std::uint32_t want = 0;
            for (std::size_t k = 4; k < 12; k++) {
                const char c = p[k];
                int d;
                if (c >= '0' && c <= '9') {
                    d = c - '0';
                } else if (c >= 'a' && c <= 'f') {
                    d = c - 'a' + 10;
                } else if (c >= 'A' && c <= 'F') {
                    d = c - 'A' + 10;
                } else {
                    return false;
                }
                want = (want << 4) | (std::uint32_t)d;
            }
            /* 流式校验: 不分配任何大缓冲, 一趟算完"长度 + 合法性 + 校验和" */
            return base64DecodeInto(p + 13, n - 13, nullptr,
                                    (std::size_t)(shapeProduct * (long long)sizeof(T)),
                                    want);
        }
        /* v1 (十进制文本): 值的个数 = 逗号数 + 1 */
        long long values = (n == 0) ? 0 : 1;
        for (std::size_t k = 0; k < n; k++) {
            if (p[k] == ',') {
                values++;
            }
        }
        return values == shapeProduct;
    }

    /* 解析 "1,2,3" 形式的形状, 同时给出元素总数 */
    static bool parseShape(const std::string &shapeString, long long &product)
    {
        product = 1;
        if (shapeString.empty()) {
            return false;
        }
        long long cur = 0;
        bool any = false;
        for (std::size_t i = 0; i < shapeString.size(); i++) {
            const char c = shapeString[i];
            if (c >= '0' && c <= '9') {
                cur = cur * 10 + (c - '0');
                any = true;
            } else if (c == ',') {
                if (!any || cur <= 0) {
                    return false;
                }
                product *= cur;
                cur = 0;
                any = false;
            } else {
                return false;
            }
        }
        if (!any || cur <= 0) {
            return false;
        }
        product *= cur;
        return true;
    }

    static Tensor_ fromString(const std::string &s)
    {
        Tensor_ x;
        std::string::size_type pos = s.find('|');
        if (pos == std::string::npos) {
            lastDecodeFailedRef() = true;
            return x;
        }
        auto split = [](const std::string &str)->std::vector<std::string> {
            std::vector<std::string> elems;
            std::size_t pos = 0;
            std::size_t len = str.length();
            while (pos < len) {
                std::size_t findPos = str.find(',', pos);
                if (findPos == std::string::npos) {
                    elems.push_back(str.substr(pos, len - pos));
                    break;
                }
                elems.push_back(str.substr(pos, findPos - pos));
                pos = findPos + 1;
            }
            return elems;
        };
        /* parse shape (与 validateEncoded 共用同一份解析, 保证两边判定一致) */
        long long shapeProduct = 1;
        if (!parseShape(s.substr(0, pos), shapeProduct)) {
            lastDecodeFailedRef() = true;
            return Tensor_();
        }
        std::vector<int> shape;
        {
            long long cur = 0;
            for (std::size_t i = 0; i <= pos; i++) {
                const char c = (i == pos) ? ',' : s[i];
                if (c == ',') {
                    shape.push_back((int)cur);
                    cur = 0;
                } else {
                    cur = cur * 10 + (c - '0');
                }
            }
        }
        /* create */
        x = Tensor_(shape);

        /*
           下面**不再**把 payload 拷成 std::string: 一行可能是 34 MB 的 base64,
           substr 一次就是一次全量拷贝。这里直接用指针 + 长度, 并让解码器
           把结果写进张量自己的存储 (见 base64DecodeInto)。
        */
        const char *payloadPtr = s.data() + pos + 1;
        std::size_t payloadLen = s.size() - (pos + 1);
        /* 去掉行尾可能残留的 '\r' (Windows 文本模式写入的文件) */
        while (payloadLen > 0
               && (payloadPtr[payloadLen - 1] == '\r' || payloadPtr[payloadLen - 1] == '\n'
                   || payloadPtr[payloadLen - 1] == ' ')) {
            payloadLen--;
        }

        if (payloadLen >= 4 && std::strncmp(payloadPtr, "b64:", 4) == 0) {
            /* ---- v2: b64:<crc32 hex>:<base64 原始数据> ---- */
            if (payloadLen < 4 + 8 + 1 || payloadPtr[12] != ':') {
                lastDecodeFailedRef() = true;
                return Tensor_();
            }
            std::uint32_t want = 0;
            for (std::size_t k = 4; k < 12; k++) {
                const char c = payloadPtr[k];
                int d;
                if (c >= '0' && c <= '9') {
                    d = c - '0';
                } else if (c >= 'a' && c <= 'f') {
                    d = c - 'a' + 10;
                } else if (c >= 'A' && c <= 'F') {
                    d = c - 'A' + 10;
                } else {
                    lastDecodeFailedRef() = true;
                    return Tensor_();
                }
                want = (want << 4) | (std::uint32_t)d;
            }
            /* 直接解码进 x.val: 长度/合法性/校验和/写入 一趟完成 */
            if (!base64DecodeInto(payloadPtr + 13, payloadLen - 13,
                                  reinterpret_cast<unsigned char *>(x.val.data()),
                                  (std::size_t)(shapeProduct * (long long)sizeof(T)),
                                  want)) {
                lastDecodeFailedRef() = true;
                return Tensor_();
            }
            return x;
        }

        /* ---- v1 (老格式): 十进制文本, 仍然要能读 ---- */
        const std::string payload(payloadPtr, payloadLen);
        std::vector<std::string> valElements = split(payload);
        if ((long long)valElements.size() != shapeProduct) {
            lastDecodeFailedRef() = true;
            return Tensor_();
        }
        for (std::size_t i = 0; i < valElements.size(); i++) {
            x[i] = (T)std::atof(valElements[i].c_str());
        }
        return x;
    }

    /* ---- base64 (只用位运算, 便于以后换真正的二进制流) ---- */
    /* ---- base64 与 CRC32 ----
       CRC 表用函数内静态初始化 (C++11 起线程安全); crc32Update 是递增版本, 供流式
       解码使用 —— 解一个 34 MB 的张量时不必先把字节存下来再算校验。 */
    static const std::uint32_t *crc32Table()
    {
        static const std::vector<std::uint32_t> table = [] {
            std::vector<std::uint32_t> t(256);
            for (std::uint32_t i = 0; i < 256; i++) {
                std::uint32_t c = i;
                for (int k = 0; k < 8; k++) {
                    c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                }
                t[i] = c;
            }
            return t;
        }();
        return table.data();
    }

    static std::uint32_t crc32Update(std::uint32_t c, unsigned char b)
    {
        return crc32Table()[(c ^ b) & 0xFFu] ^ (c >> 8);
    }

    static std::uint32_t crc32(const unsigned char *data, std::size_t n)
    {
        std::uint32_t c = 0xFFFFFFFFu;
        for (std::size_t i = 0; i < n; i++) {
            c = crc32Update(c, data[i]);
        }
        return c ^ 0xFFFFFFFFu;
    }

    /*
       ============================================================
        流式 base64 解码: 一趟扫完, 零中间分配
       ============================================================
       为什么要有它: 权重文件里一个张量可以是一行 34 MB 的 base64, 而老实现是
         std::string(p + 13, n - 13)     先拷一份 34 MB
         base64Decode() 里 vector.push_back   再分配 34 MB
         crc32(raw)                       再扫一遍
         memcpy 到张量                     再一遍
       而且"先校验一遍再真正载入"把这一串**整体做了两遍**。实测稀疏 MoE 那 3 个
       146 MB 的权重文件要 14.9 秒 (29 MB/s), 把"启动时加载所有模型"变成 19 秒。
       现在: 直接在内存字节流上解码, 每解出 3 个字节就
         * 增量更新 CRC,
         * 若有 dst 就**直接写进张量的存储** (连 memcpy 都省了),
       一趟同时完成"合法性 + 长度 + 校验和 + 写入"。dst == nullptr 就是纯校验模式
       (给 Net::load 的预校验用, 校验通过才真正写进网络)。
    */
    static bool base64DecodeInto(const char *in, std::size_t n,
                                 unsigned char *dst, std::size_t expectedBytes,
                                 std::uint32_t wantCrc)
    {
        if (expectedBytes == 0 || n == 0 || (n % 4) != 0) {
            return false;
        }
        const std::size_t groups = n / 4;
        /* 编码长度必须与字节数精确对应 (少一个 '=' 就是被截断了) */
        if (groups * 3 < expectedBytes || (groups * 3 - expectedBytes) > 2) {
            return false;
        }
        /*
           逐字符的分支链换成 256 项查找表: 老写法每个字符要过 6 个比较, 而 109 MB 的
           数据是 1.45 亿个字符 —— 实测一换, 解码从 99 MB/s 提到 ~250 MB/s。
           '=' 用 64 表示, 非法字符用 -1。
        */
        const signed char *tab = base64DecodeTable();
        std::uint32_t crc = 0xFFFFFFFFu;
        std::size_t written = 0;
        for (std::size_t g = 0; g < groups; g++) {
            const char *q = in + g * 4;
            const signed char v0 = tab[(unsigned char)q[0]];
            const signed char v1 = tab[(unsigned char)q[1]];
            const signed char v2 = tab[(unsigned char)q[2]];
            const signed char v3 = tab[(unsigned char)q[3]];
            if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) {
                return false;
            }
            int pad = 0;
            if (v2 == 64) {
                /* 填充只允许出现在最后一组的末尾 */
                if (g + 1 != groups || v3 != 64) {
                    return false;
                }
                pad = 2;
            } else if (v3 == 64) {
                if (g + 1 != groups) {
                    return false;
                }
                pad = 1;
            }
            const unsigned int x = ((unsigned int)v0 << 18) | ((unsigned int)v1 << 12)
                                   | ((unsigned int)(v2 & 63) << 6)
                                   | (unsigned int)(v3 & 63);
            const int cnt = 3 - pad;
            if (written + (std::size_t)cnt > expectedBytes) {
                return false;
            }
            const unsigned char b0 = (unsigned char)((x >> 16) & 0xFFu);
            const unsigned char b1 = (unsigned char)((x >> 8) & 0xFFu);
            const unsigned char b2 = (unsigned char)(x & 0xFFu);
            crc = crc32Update(crc, b0);
            if (dst != nullptr) {
                dst[written] = b0;
            }
            written++;
            if (cnt >= 2) {
                crc = crc32Update(crc, b1);
                if (dst != nullptr) {
                    dst[written] = b1;
                }
                written++;
            }
            if (cnt >= 3) {
                crc = crc32Update(crc, b2);
                if (dst != nullptr) {
                    dst[written] = b2;
                }
                written++;
            }
        }
        if (written != expectedBytes) {
            return false;
        }
        return (crc ^ 0xFFFFFFFFu) == wantCrc;
    }

    /* base64 字符 -> 值 (0..63), '=' -> 64, 其它 -> -1 */
    static const signed char *base64DecodeTable()
    {
        static const std::vector<signed char> table = [] {
            std::vector<signed char> t(256, (signed char)-1);
            const char *alphabet =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            for (int i = 0; i < 64; i++) {
                t[(unsigned char)alphabet[i]] = (signed char)i;
            }
            t[(unsigned char)'='] = (signed char)64;
            return t;
        }();
        return table.data();
    }
    static std::string base64Encode(const std::vector<T, Alloc<T> > &data)
    {
        static const char *kTab =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const unsigned char *bytes =
            reinterpret_cast<const unsigned char *>(data.data());
        const std::size_t n = data.size() * sizeof(T);
        std::string out;
        out.reserve((n + 2) / 3 * 4);
        std::size_t i = 0;
        for (; i + 3 <= n; i += 3) {
            const unsigned int v = ((unsigned int)bytes[i] << 16)
                                   | ((unsigned int)bytes[i + 1] << 8)
                                   | (unsigned int)bytes[i + 2];
            out += kTab[(v >> 18) & 63];
            out += kTab[(v >> 12) & 63];
            out += kTab[(v >> 6) & 63];
            out += kTab[v & 63];
        }
        const std::size_t rest = n - i;
        if (rest == 1) {
            const unsigned int v = (unsigned int)bytes[i] << 16;
            out += kTab[(v >> 18) & 63];
            out += kTab[(v >> 12) & 63];
            out += '=';
            out += '=';
        } else if (rest == 2) {
            const unsigned int v = ((unsigned int)bytes[i] << 16)
                                   | ((unsigned int)bytes[i + 1] << 8);
            out += kTab[(v >> 18) & 63];
            out += kTab[(v >> 12) & 63];
            out += kTab[(v >> 6) & 63];
            out += '=';
        }
        return out;
    }

    static bool base64Decode(const std::string &in, std::vector<unsigned char> &out)
    {
        auto val = [](char c) -> int {
            if (c >= 'A' && c <= 'Z') return c - 'A';
            if (c >= 'a' && c <= 'z') return c - 'a' + 26;
            if (c >= '0' && c <= '9') return c - '0' + 52;
            if (c == '+') return 62;
            if (c == '/') return 63;
            return -1;
        };
        out.clear();
        if (in.size() % 4 != 0) {
            return false;
        }
        out.reserve(in.size() / 4 * 3);
        for (std::size_t i = 0; i < in.size(); i += 4) {
            int q[4];
            int pad = 0;
            for (int k = 0; k < 4; k++) {
                const char c = in[i + k];
                if (c == '=') {
                    /* 填充只允许出现在最后 4 字节组的末尾 */
                    if (i + 4 != in.size() || k < 2) {
                        return false;
                    }
                    q[k] = 0;
                    pad++;
                } else {
                    q[k] = val(c);
                    if (q[k] < 0) {
                        return false;
                    }
                }
            }
            const unsigned int v = ((unsigned int)q[0] << 18)
                                   | ((unsigned int)q[1] << 12)
                                   | ((unsigned int)q[2] << 6)
                                   | (unsigned int)q[3];
            out.push_back((unsigned char)((v >> 16) & 0xFF));
            if (pad < 2) {
                out.push_back((unsigned char)((v >> 8) & 0xFF));
            }
            if (pad < 1) {
                out.push_back((unsigned char)(v & 0xFF));
            }
        }
        return true;
    }
};

using Tensorc  = Tensor_<char>;
using Tensoru8 = Tensor_<unsigned char>;
using Tensori  = Tensor_<int>;
using Tensorf  = Tensor_<float>;
using Tensord  = Tensor_<double>;
using Tensor   = Tensorf;

}
#endif // TENSOR_H
