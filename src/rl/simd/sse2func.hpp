#ifndef SSE2FUNC_HPP
#define SSE2FUNC_HPP
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <iostream>
/* Chess: basic_def.h is not part of this project - the two macros it supplied
   are defined here instead (and only if nothing else defined them already).
   The global pi it also declared is deliberately NOT carried over: RL already
   has its own RL::pi in util.hpp. */
#ifndef FORCE_INLINE
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
        #define VECTORCALL   __vectorcall
    #else
        #define FORCE_INLINE __attribute__((always_inline)) inline
        #define VECTORCALL
        #ifndef __vectorcall
            #define __vectorcall
        #endif
    #endif
#endif

namespace simd {

/*
    simd::Step<T> reports the vector width of the active instruction set.
    avx2func.hpp defines the same specialisations behind __AVX2__, so they are
    only emitted here when AVX2 is off - otherwise the two headers would
    redefine the template and the specialisations.
*/
#if !defined(__AVX2__)
template<typename T>
struct Step {
    constexpr static std::size_t value = 0;
};
template<>
struct Step<double> {
    constexpr static std::size_t value = sizeof (__m128d)/sizeof (double);
};
template<>
struct Step<float> {
    constexpr static std::size_t value = sizeof (__m128)/sizeof (float);
};
#endif
struct SSE2 {

    /*
        SSE2 has no fused multiply add: _mm_fmadd_* is encoded as FMA3 and
        would fault on a cpu without it, even though this header is compiled
        for plain SSE2.  Emulate it with mul + add.
    */
    inline static __m128 fmadd(__m128 a, __m128 b, __m128 c)
    {
        return _mm_add_ps(_mm_mul_ps(a, b), c);
    }
    inline static __m128d fmadd(__m128d a, __m128d b, __m128d c)
    {
        return _mm_add_pd(_mm_mul_pd(a, b), c);
    }

    inline static float reduce(__m128& ymm)
    {
        /*
            horizontal sum of the four lanes.
            [a0,a1,a2,a3] + [a2,a3,a0,a1] = [a0+a2, a1+a3, ...]
                                     + [b1,b0, ...] = a0+a1+a2+a3
        */
        __m128 shuf = _mm_shuffle_ps(ymm, ymm, _MM_SHUFFLE(1, 0, 3, 2));
        ymm = _mm_add_ps(ymm, shuf);
        shuf = _mm_shuffle_ps(ymm, ymm, _MM_SHUFFLE(2, 3, 0, 1));
        ymm = _mm_add_ps(ymm, shuf);
        return _mm_cvtss_f32(ymm);
    }

    inline static double reduce(__m128d& ymm)
    {
        /* horizontal sum of the two lanes */
        __m128d shuf = _mm_shuffle_pd(ymm, ymm, 1);
        ymm = _mm_add_pd(ymm, shuf);
        return _mm_cvtsd_f64(ymm);
    }

    inline static void fill(double* __restrict x, double x0, std::size_t N)
    {
        double* px = x;
        std::size_t r = N%2;
        __m128d vecx0 = _mm_setr_pd(x0, x0);
        for (std::size_t i = 0; i < N - r; i+=2) {
            _mm_storeu_pd(px + i, vecx0);
        }
        for (std::size_t i = N - r; i < N; i++) {
            x[i] = x0;
        }
        return;
    }

    inline static void fill(float* __restrict x, float x0, std::size_t N)
    {
        float* px = x;
        std::size_t r = N%4;
        __m128 vecx0 = _mm_setr_ps(x0, x0, x0, x0);
        for (std::size_t i = 0; i < N - r; i+=4) {
            _mm_storeu_ps(px + i, vecx0);
        }
        for (std::size_t i = N - r; i < N; i++) {
            x[i] = x0;
        }
        return;
    }

    inline static void add(double* z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: sse double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* step */
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 2 double into __m128d */
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_loadu_pd(py + i);
            /* add */
            vecz = _mm_add_pd(vecx, vecy);
            /* store result */
            _mm_storeu_pd(pz + i, vecz);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] + x[i];
        }
        return;
    }

    inline static void sub(double* z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: sse double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* offset */
        /* step */
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 2 double into __m128d */
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_loadu_pd(py + i);
            /* add */
            vecz = _mm_sub_pd(vecy, vecx);      /* z = y - x */
            /* store result */
            _mm_storeu_pd(pz + i, vecz);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] - x[i];
        }
        return;
    }

    inline static void mul(double* z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: sse double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* step */
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 2 double into __m256d */
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_loadu_pd(py + i);
            /* add */
            vecz = _mm_mul_pd(vecx, vecy);
            /* store result */
            _mm_storeu_pd(pz + i, vecz);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] * x[i];
        }
        return;
    }

    inline static void div(double* z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: sse double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* step */
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_loadu_pd(py + i);
            /* add */
            vecz = _mm_div_pd(vecy, vecx);      /* z = y / x */
            /* store result */
            _mm_storeu_pd(pz + i, vecz);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] / x[i];
        }
        return;
    }

    inline static void add(float* z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: sse2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* step */
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_loadu_ps(py + i);
            /* add */
            vecz = _mm_add_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz + i, vecz);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] + x[i];
        }
        return;
    }

    inline static void sub(float* z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: sse2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* step */
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_loadu_ps(py + i);
            /* sub */
            vecz = _mm_sub_ps(vecy, vecx);      /* z = y - x */
            /* store result */
            _mm_storeu_ps(pz + i, vecz);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] - x[i];
        }
        return;
    }

    inline static void mul(float* z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: sse2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* step */
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_loadu_ps(py + i);
            /* add */
            vecz = _mm_mul_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz + i, vecz);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] * x[i];
        }
        return;
    }

    inline static void div(float* z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: sse2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* step */
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_loadu_ps(py + i);
            /* add */
            vecz = _mm_div_ps(vecy, vecx);      /* z = y / x */
            /* store result */
            _mm_storeu_ps(pz + i, vecz);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] / x[i];
        }
        return;
    }

    inline static void add(double* z, const double* __restrict y, double x, std::size_t N)
    {
        const double *py = y;
        double *pz = z;
        __m128d vecx = _mm_setr_pd(x, x);
        __m128d vecy;
        __m128d vecz;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_pd(py + i);
            vecz = _mm_add_pd(vecy, vecx);
            _mm_storeu_pd(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] + x;
        }
        return;
    }

    inline static void sub(double* z, const double* __restrict y, double x, std::size_t N)
    {
        const double *py = y;
        double *pz = z;
        __m128d vecx = _mm_setr_pd(x, x);
        __m128d vecy;
        __m128d vecz;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_pd(py + i);
            vecz = _mm_sub_pd(vecy, vecx);
            _mm_storeu_pd(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] - x;
        }
        return;
    }

    inline static void mul(double* z, const double* __restrict y, double x, std::size_t N)
    {
        const double *py = y;
        double *pz = z;
        __m128d vecx = _mm_setr_pd(x, x);
        __m128d vecy;
        __m128d vecz;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_pd(py + i);
            vecz = _mm_mul_pd(vecy, vecx);
            _mm_storeu_pd(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] * x;
        }
        return;
    }

    inline static void div(double* z, const double* __restrict y, double x, std::size_t N)
    {
        const double *py = y;
        double *pz = z;
        __m128d vecx = _mm_setr_pd(x, x);
        __m128d vecy;
        __m128d vecz;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_pd(py + i);
            vecz = _mm_div_pd(vecy, vecx);
            _mm_storeu_pd(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] / x;
        }
        return;
    }

    inline static void add(float* z, const float* __restrict y, float x, std::size_t N)
    {
        const float *py = y;
        float *pz = z;
        __m128 vecx = _mm_setr_ps(x, x, x, x);
        __m128 vecy;
        __m128 vecz;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_ps(py + i);
            vecz = _mm_add_ps(vecy, vecx);
            _mm_storeu_ps(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] + x;
        }
        return;
    }

    inline static void sub(float* z, const float* __restrict y, float x, std::size_t N)
    {
        const float *py = y;
        float *pz = z;
        __m128 vecx = _mm_setr_ps(x, x, x, x);
        __m128 vecy;
        __m128 vecz;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_ps(py + i);
            vecz = _mm_sub_ps(vecy, vecx);
            _mm_storeu_ps(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] - x;
        }
        return;
    }

    inline static void mul(float* z, const float* __restrict y, float x, std::size_t N)
    {
        const float *py = y;
        float *pz = z;
        __m128 vecx = _mm_setr_ps(x, x, x, x);
        __m128 vecy;
        __m128 vecz;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_ps(py + i);
            vecz = _mm_mul_ps(vecy, vecx);
            _mm_storeu_ps(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] * x;
        }
        return;
    }

    inline static void div(float* z, const float* __restrict y, float x, std::size_t N)
    {
        const float *py = y;
        float *pz = z;
        __m128 vecx = _mm_setr_ps(x, x, x, x);
        __m128 vecy;
        __m128 vecz;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_ps(py + i);
            vecz = _mm_div_ps(vecy, vecx);
            _mm_storeu_ps(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] / x;
        }
        return;
    }


    inline static double max(const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        /* align */
        std::size_t r = N%step;
        /* find max value per 2 element */
        __m128d maxVec = _mm_loadu_pd(px);
        for (std::size_t i = step; i < N - r; i+=step) {
            __m128d vecx = _mm_loadu_pd(px + i);
            maxVec = _mm_max_pd(vecx, maxVec);
        }
        /* find max value in result */
        double result[2];
        _mm_storeu_pd(result, maxVec);
        double maxValue = result[0];
        for (std::size_t i = 1; i < step; i++) {
            maxValue = result[i] > maxValue ? result[i] : maxValue;
        }
        /* find max value in the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            maxValue = px[i] > maxValue ? px[i] : maxValue;
        }
        return maxValue;
    }

    inline static float max(const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        std::size_t step = sizeof (__m128)/sizeof (float);
        /* align */
        std::size_t r = N%step;
        /* find max value per 4 element */
        __m128 maxVec = _mm_loadu_ps(px);
        for (std::size_t i = step; i < N - r; i+=step) {
            __m128 vecx = _mm_loadu_ps(px + i);
            maxVec = _mm_max_ps(vecx, maxVec);
        }
        /* find max value in result */
        float result[4];
        _mm_storeu_ps(result, maxVec);
        float maxValue = result[0];
        for (std::size_t i = 1; i < step; i++) {
            maxValue = result[i] > maxValue ? result[i] : maxValue;
        }
        /* find max value in the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            maxValue = px[i] > maxValue ? px[i] : maxValue;
        }
        return maxValue;
    }

    inline static double min(const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        /* align */
        std::size_t r = N%step;
        /* find min value per 2 element */
        __m128d minVec = _mm_loadu_pd(px);
        for (std::size_t i = step; i < N - r; i+=step) {
            __m128d vecx = _mm_loadu_pd(px + i);
            minVec = _mm_min_pd(vecx, minVec);
        }
        /* find min value in result */
        double result[2];
        _mm_storeu_pd(result, minVec);
        double minValue = result[0];
        for (std::size_t i = 1; i < step; i++) {
            minValue = result[i] < minValue ? result[i] : minValue;
        }
        /* find min value in the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            minValue = px[i] < minValue ? px[i] : minValue;
        }
        return minValue;
    }

    inline static float min(const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        std::size_t step = sizeof (__m128)/sizeof (float);
        /* align */
        std::size_t r = N%step;
        /* find min value per 4 element */
        __m128 minVec = _mm_loadu_ps(px);
        for (std::size_t i = step; i < N - r; i+=step) {
            __m128 vecx = _mm_loadu_ps(px + i);
            minVec = _mm_min_ps(vecx, minVec);
        }
        /* find min value in result */
        float result[4];
        _mm_storeu_ps(result, minVec);
        float minValue = result[0];
        for (std::size_t i = 1; i < step; i++) {
            minValue = result[i] < minValue ? result[i] : minValue;
        }
        /* find min value in the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            minValue = px[i] < minValue ? px[i] : minValue;
        }
        return minValue;
    }

    inline static double sum(const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        /* init */
        std::size_t r = N%step;
        double result[2] = {0};
        __m128d vecs = _mm_loadu_pd(result);
        for (std::size_t i = 0; i < N - r; i+=step) {
            __m128d vecx = _mm_loadu_pd(px + i);
            vecs = _mm_add_pd(vecs, vecx);
        }
        /* sum up result */
        _mm_storeu_pd(result, vecs);
        double s = 0;
        for (std::size_t i = 0; i < step; i++) {
            s += result[i];
        }
        /* sum up the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px[i];
        }
        return s;
    }

    inline static float sum(const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        std::size_t step = sizeof (__m128)/sizeof (float);
        /* init */
        std::size_t r = N%step;
        __m128 vecs = _mm_setzero_ps();
        for (std::size_t i = 0; i < N - r; i+=step) {
            __m128 vecx = _mm_loadu_ps(px + i);
            vecs = _mm_add_ps(vecs, vecx);
        }
        /* sum up result */
        float s = reduce(vecs);
        /* sum up the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px[i];
        }
        return s;
    }
    inline static double product(const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        /* init */
        std::size_t r = N%step;
        double result[2] = {0};
        __m128d vecx;
        __m128d vecs = _mm_setr_pd(1, 1);
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx = _mm_loadu_pd(px + i);
            vecs = _mm_mul_pd(vecs, vecx);
        }
        float s = 1;
        _mm_storeu_pd(result, vecs);
        for (std::size_t i = 0; i < step; i++) {
            s *= result[i];
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s *= px[i];
        }
        return s;
    }

    inline static float product(const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        std::size_t step = sizeof (__m128)/sizeof (float);
        /* init */
        std::size_t r = N%step;
        float result[4] = {0};
        __m128 vecs = _mm_setr_ps(1, 1, 1, 1);
        for (std::size_t i = 0; i < N - r; i+=step) {
            __m128 vecx = _mm_loadu_ps(px + i);
            vecs = _mm_mul_ps(vecs, vecx);
        }
        float s = 1;
        _mm_storeu_ps(result, vecs);
        for (std::size_t i = 0; i < step; i++) {
            s *= result[i];
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s *= px[i];
        }
        return s;
    }

    inline static void sqrt(double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        double *py = y;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        __m128d vecx;
        __m128d vecy;
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_sqrt_pd(vecx);
            _mm_storeu_pd(py + i, vecy);
        }
        /* sqrt the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] = std::sqrt(px[i]);
        }
        return;
    }

    inline static void sqrt(float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        float *py = y;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        __m128 vecx;
        __m128 vecy;
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_sqrt_ps(vecx);
            _mm_storeu_ps(py + i, vecy);
        }
        /* sqrt the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] = std::sqrt(px[i]);
        }
        return;
    }

    inline static double dot(const double* __restrict x1, const double* x2, std::size_t N)
    {
        const double *px1 = x1;
        const double *px2 = x2;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        __m128d vecx1;
        __m128d vecx2;
        __m128d vecy = _mm_setzero_pd();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_pd(px1 + i);
            vecx2 = _mm_loadu_pd(px2 + i);
            vecy = fmadd(vecx1, vecx2, vecy);
        }
        /* sum up result */
        double s = reduce(vecy);
        /* dot the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px1[i]*px2[i];
        }
        return s;
    }

    /*
        element wise absolute value: z = |x|.
        The exponent/sign split makes this exact for every finite value and for
        infinities; NaN keeps its payload.  z may alias x.
    */
    inline static void abs(double* z, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        double *pz = z;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        const __m128d signMask = _mm_castsi128_pd(_mm_set1_epi64x(0x7fffffffffffffffLL));
        for (std::size_t i = 0; i < N - r; i+= step) {
            _mm_storeu_pd(pz + i, _mm_and_pd(_mm_loadu_pd(px + i), signMask));
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = std::abs(x[i]);
        }
        return;
    }

    inline static void abs(float* z, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        float *pz = z;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        const __m128 signMask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
        for (std::size_t i = 0; i < N - r; i+= step) {
            _mm_storeu_ps(pz + i, _mm_and_ps(_mm_loadu_ps(px + i), signMask));
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = std::abs(x[i]);
        }
        return;
    }

    /*
        element wise clamp: z = x < lo ? lo : (x > hi ? hi : x).
        Explicit comparisons (instead of min/max) so that NaN and lo > hi
        behave exactly like the scalar Tensor_::clip.  z may alias x.
    */
    inline static void clip(double* z, const double* __restrict x, double lo, double hi, std::size_t N)
    {
        const double *px = x;
        double *pz = z;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        const __m128d vlo = _mm_set1_pd(lo);
        const __m128d vhi = _mm_set1_pd(hi);
        for (std::size_t i = 0; i < N - r; i+= step) {
            __m128d v = _mm_loadu_pd(px + i);
            __m128d lt = _mm_cmplt_pd(v, vlo);
            v = _mm_or_pd(_mm_and_pd(lt, vlo), _mm_andnot_pd(lt, v));
            __m128d gt = _mm_cmpgt_pd(v, vhi);
            v = _mm_or_pd(_mm_and_pd(gt, vhi), _mm_andnot_pd(gt, v));
            _mm_storeu_pd(pz + i, v);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]);
        }
        return;
    }

    inline static void clip(float* z, const float* __restrict x, float lo, float hi, std::size_t N)
    {
        const float *px = x;
        float *pz = z;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        const __m128 vlo = _mm_set1_ps(lo);
        const __m128 vhi = _mm_set1_ps(hi);
        for (std::size_t i = 0; i < N - r; i+= step) {
            __m128 v = _mm_loadu_ps(px + i);
            __m128 lt = _mm_cmplt_ps(v, vlo);
            v = _mm_or_ps(_mm_and_ps(lt, vlo), _mm_andnot_ps(lt, v));
            __m128 gt = _mm_cmpgt_ps(v, vhi);
            v = _mm_or_ps(_mm_and_ps(gt, vhi), _mm_andnot_ps(gt, v));
            _mm_storeu_ps(pz + i, v);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]);
        }
        return;
    }

    inline static float dot(const float* __restrict x1,
                           const float* x2,
                           std::size_t N)
    {
        const float *px1 = x1;
        const float *px2 = x2;
        /* 4 floats per __m128, not 2 */
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        __m128 vecx1;
        __m128 vecx2;
        __m128 vecy = _mm_setzero_ps();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_ps(px1 + i);
            vecx2 = _mm_loadu_ps(px2 + i);
            vecy = fmadd(vecx1, vecx2, vecy);
        }
        /* sum up result */
        float s = reduce(vecy);
        /* dot the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px1[i]*px2[i];
        }
        return s;
    }

    inline static double norm2s(const double* __restrict x1, const double* x2, std::size_t N)
    {
        const double *px1 = x1;
        const double *px2 = x2;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        __m128d vecx1;
        __m128d vecx2;
        __m128d vecx;
        __m128d vecy = _mm_setzero_pd();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_pd(px1 + i);
            vecx2 = _mm_loadu_pd(px2 + i);
            vecx = _mm_sub_pd(vecx1, vecx2);
            vecy = fmadd(vecx, vecx, vecy);
        }
        /* sum up result */
        double s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px1[i] - px2[i])*(px1[i] - px2[i]);
        }
        return s;
    }

    inline static float norm2s(const float* __restrict x1, const float* x2, std::size_t N)
    {
        const float *px1 = x1;
        const float *px2 = x2;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        __m128 vecx1;
        __m128 vecx2;
        __m128 vecx;
        __m128 vecy = _mm_setzero_ps();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_ps(px1 + i);
            vecx2 = _mm_loadu_ps(px2 + i);
            vecx = _mm_sub_ps(vecx1, vecx2);
            vecy = fmadd(vecx, vecx, vecy);
        }
        /* sum up result */
        float s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px1[i] - px2[i])*(px1[i] - px2[i]);
        }
        return s;
    }

    inline static double variance(const double* __restrict x,  double  u, std::size_t N)
    {
        const double *px = x;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        __m128d vecx1;
        __m128d vecu = _mm_set1_pd(u);
        __m128d vecx;
        __m128d vecy = _mm_setzero_pd();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_pd(px + i);
            vecx = _mm_sub_pd(vecx1, vecu);
            vecy = fmadd(vecx, vecx, vecy);
        }
        /* sum up result */
        double s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px[i] - u)*(px[i] - u);
        }
        return s/double(N);
    }

    inline static float variance(const float* __restrict x,  float  u, std::size_t N)
    {
        const float *px = x;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        __m128 vecx1;
        __m128 vecu = _mm_set1_ps(u);
        __m128 vecx;
        __m128 vecy = _mm_setzero_ps();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_ps(px + i);
            vecx = _mm_sub_ps(vecx1, vecu);
            vecy = fmadd(vecx, vecx, vecy);
        }
        /* sum up result */
        float s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px[i] - u)*(px[i] - u);
        }
        return s/float(N);
    }

    inline static void matMul(double* __restrict z, std::size_t zRow, std::size_t zCol,
                              const double* __restrict x, std::size_t xRow, std::size_t xCol,
                              const double* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const double *x_ = x;
        const double *y_ = y;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        double *z_ = z;
        std::size_t r = zCol%step;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                double xik = x_[i*xCol + k];
                __m128d vecx = _mm_set1_pd(xik);
                for (std::size_t j = 0; j < zCol - r; j+=step) {
                    __m128d vecy = _mm_loadu_pd(y_ + k*yCol + j);
                    __m128d vecz = _mm_loadu_pd(z_ + i*zCol + j);
                    /* fmadd(a, b, c): a*b + c */
                    vecz = fmadd(vecx, vecy, vecz);
                    /* store result */
                    _mm_storeu_pd(z_ + i*zCol + j, vecz);
                }
                for (std::size_t j = zCol - r; j < zCol; j++) {
                    z_[i*zCol + j] += xik * y_[k*yCol + j];
                }
            }
        }
        return;
    }

    inline static void matMul8(double* __restrict z, std::size_t zRow, std::size_t zCol,
                               const double* __restrict x, std::size_t xRow, std::size_t xCol,
                               const double* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const double *x_ = x;
        const double *y_ = y;
        std::size_t step = 4;
        double *z_ = z;
        std::size_t r1 = xCol%step;
        std::size_t r2 = zCol%step;
        double xik[4];
        __m128d vecx[4];
        __m128d vecy[4];
        __m128d vecz;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol - r1; k+=step) {
                xik[0] = x_[i*xCol + k];
                xik[1] = x_[i*xCol + k + 1];
                xik[2] = x_[i*xCol + k + 2];
                xik[3] = x_[i*xCol + k + 3];
                vecx[0] = _mm_set1_pd(xik[0]);
                vecx[1] = _mm_set1_pd(xik[1]);
                vecx[2] = _mm_set1_pd(xik[2]);
                vecx[3] = _mm_set1_pd(xik[3]);
                for (std::size_t j = 0; j < zCol - r2; j+=step) {
                    /* put 2 double into __m128d */
                    vecy[0] = _mm_loadu_pd(y_ + k*yCol + j);
                    vecy[1] = _mm_loadu_pd(y_ + (k + 1)*yCol + j);
                    vecy[2] = _mm_loadu_pd(y_ + (k + 2)*yCol + j);
                    vecy[3] = _mm_loadu_pd(y_ + (k + 3)*yCol + j);
                    vecz  = _mm_loadu_pd(z_ + i*zCol + j);
                    /* fmadd(a, b, c): a*b + c */
                    vecz = fmadd(vecx[0], vecy[0], vecz);
                    vecz = fmadd(vecx[1], vecy[1], vecz);
                    vecz = fmadd(vecx[2], vecy[2], vecz);
                    vecz = fmadd(vecx[3], vecy[3], vecz);
                    /* store result */
                    _mm_storeu_pd(z_ + i*zCol + j, vecz);
                }
                for (std::size_t j = zCol - r2; j < zCol; j++) {
                    z_[i*zCol + j] += xik[0] * y_[k*yCol + j];
                    z_[i*zCol + j] += xik[1] * y_[(k + 1)*yCol + j];
                    z_[i*zCol + j] += xik[2] * y_[(k + 2)*yCol + j];
                    z_[i*zCol + j] += xik[3] * y_[(k + 3)*yCol + j];
                }
            }
        }
        return;
    }

    inline static void matMul(float* __restrict z, std::size_t zRow, std::size_t zCol,
                              const float* __restrict x, std::size_t xRow, std::size_t xCol,
                              const float* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        /*
            origin:
                https://blog.csdn.net/StandCrow/article/details/120206063
        */
        const float *x_ = x;
        const float *y_ = y;
        float *z_ = z;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = zCol%step;
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                float xik = x_[i*xCol + k];
                vecx = _mm_set1_ps(xik);
                for (std::size_t j = 0; j < zCol - r; j+=step) {
                    /* put 4 float into __m128 */
                    vecy = _mm_loadu_ps(y_ + k*yCol + j);
                    vecz = _mm_loadu_ps(z_ + i*zCol + j);
                    /* fmadd(a, b, c): a*b + c */
                    vecz = fmadd(vecx, vecy, vecz);
                    /* store result */
                    _mm_storeu_ps(z_ + i*zCol + j, vecz);
                }
                /* the rest column */
                for (std::size_t j = zCol - r; j < zCol; j++) {
                    z_[i*zCol + j] += xik * y_[k*yCol + j];
                }
            }
        }
        return;
    }

    inline static void matMul32(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const float *x_ = x;
        const float *y_ = y;
        float *z_ = z;
        std::size_t step = 8;                                  /* rows of x per block */
        std::size_t jStep = sizeof (__m128)/sizeof (float);    /* columns per vector */
        std::size_t r1 = xCol%step;
        std::size_t r2 = zCol%jStep;
        float xik[8];
        __m128 vecx[8];
        __m128 vecy[8];
        __m128 vecz;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol - r1; k+=step) {
                xik[0] = x_[i*xCol + k];
                xik[1] = x_[i*xCol + k + 1];
                xik[2] = x_[i*xCol + k + 2];
                xik[3] = x_[i*xCol + k + 3];
                xik[4] = x_[i*xCol + k + 4];
                xik[5] = x_[i*xCol + k + 5];
                xik[6] = x_[i*xCol + k + 6];
                xik[7] = x_[i*xCol + k + 7];
                vecx[0] = _mm_set1_ps(xik[0]);
                vecx[1] = _mm_set1_ps(xik[1]);
                vecx[2] = _mm_set1_ps(xik[2]);
                vecx[3] = _mm_set1_ps(xik[3]);
                vecx[4] = _mm_set1_ps(xik[4]);
                vecx[5] = _mm_set1_ps(xik[5]);
                vecx[6] = _mm_set1_ps(xik[6]);
                vecx[7] = _mm_set1_ps(xik[7]);
                for (std::size_t j = 0; j < zCol - r2; j+=jStep) {
                    /* put 4 float into __m128 */
                    vecy[0] = _mm_loadu_ps(y_ + k*yCol + j);
                    vecy[1] = _mm_loadu_ps(y_ + (k + 1)*yCol + j);
                    vecy[2] = _mm_loadu_ps(y_ + (k + 2)*yCol + j);
                    vecy[3] = _mm_loadu_ps(y_ + (k + 3)*yCol + j);
                    vecy[4] = _mm_loadu_ps(y_ + (k + 4)*yCol + j);
                    vecy[5] = _mm_loadu_ps(y_ + (k + 5)*yCol + j);
                    vecy[6] = _mm_loadu_ps(y_ + (k + 6)*yCol + j);
                    vecy[7] = _mm_loadu_ps(y_ + (k + 7)*yCol + j);
                    vecz  = _mm_loadu_ps(z_ + i*zCol + j);
                    /* fmadd(a, b, c): a*b + c */
                    vecz = fmadd(vecx[0], vecy[0], vecz);
                    vecz = fmadd(vecx[1], vecy[1], vecz);
                    vecz = fmadd(vecx[2], vecy[2], vecz);
                    vecz = fmadd(vecx[3], vecy[3], vecz);
                    vecz = fmadd(vecx[4], vecy[4], vecz);
                    vecz = fmadd(vecx[5], vecy[5], vecz);
                    vecz = fmadd(vecx[6], vecy[6], vecz);
                    vecz = fmadd(vecx[7], vecy[7], vecz);
                    /* store result */
                    _mm_storeu_ps(z_ + i*zCol + j, vecz);
                }

                for (std::size_t j = zCol - r2; j < zCol; j++) {
                    z_[i*zCol + j] += xik[0] * y_[k*yCol + j];
                    z_[i*zCol + j] += xik[1] * y_[(k + 1)*yCol + j];
                    z_[i*zCol + j] += xik[2] * y_[(k + 2)*yCol + j];
                    z_[i*zCol + j] += xik[3] * y_[(k + 3)*yCol + j];
                    z_[i*zCol + j] += xik[4] * y_[(k + 4)*yCol + j];
                    z_[i*zCol + j] += xik[5] * y_[(k + 5)*yCol + j];
                    z_[i*zCol + j] += xik[6] * y_[(k + 6)*yCol + j];
                    z_[i*zCol + j] += xik[7] * y_[(k + 7)*yCol + j];
                }
            }
            /* the remaining rows of x: xCol % step of them are not covered by
               the blocked loop above */
            for (std::size_t k = xCol - r1; k < xCol; k++) {
                const float xik0 = x_[i*xCol + k];
                for (std::size_t j = 0; j < zCol; j++) {
                    z_[i*zCol + j] += xik0 * y_[k*yCol + j];
                }
            }
        }
        return;
    }

    struct MatMul {

        inline static void ikkj(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            const float *x_ = x;
            const float *y_ = y;
            float *z_ = z;
            std::size_t step = sizeof (__m128)/sizeof (float);
            std::size_t r = zCol%step;
            __m128 vecx;
            __m128 vecy;
            __m128 vecz;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xCol; k++) {
                    float xik = x_[i*xCol + k];
                    vecx = _mm_set1_ps(xik);
                    for (std::size_t j = 0; j < zCol - r; j+=step) {
                        /* put 4 float into __m128 */
                        vecy = _mm_loadu_ps(y_ + k*yCol + j);
                        vecz = _mm_loadu_ps(z_ + i*zCol + j);
                        /* fmadd(a, b, c): a*b + c */
                        vecz = fmadd(vecx, vecy, vecz);
                        /* store result */
                        _mm_storeu_ps(z_ + i*zCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*zCol + j] += xik * y_[k*yCol + j];
                    }
                }
            }
            return;
        }

        inline static void kikj(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            /* z = x^T * y */
            const float *x_ = x;
            const float *y_ = y;
            float *z_ = z;
            std::size_t step = sizeof (__m128)/sizeof (float);
            std::size_t r = zCol%step;
            __m128 vecx;
            __m128 vecy;
            __m128 vecz;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xRow; k++) {
                    float xki = x_[k*xCol + i];
                    vecx = _mm_set1_ps(xki);
                    for (std::size_t j = 0; j < zCol - r; j+=step) {
                        /* put 4 float into __m128 */
                        vecy = _mm_loadu_ps(y_ + k*yCol + j);
                        vecz = _mm_loadu_ps(z_ + i*zCol + j);
                        /* fmadd(a, b, c): a*b + c */
                        vecz = fmadd(vecx, vecy, vecz);
                        /* store result */
                        _mm_storeu_ps(z_ + i*zCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*zCol + j] += xki * y_[k*yCol + j];
                    }
                }
            }
            return;
        }
        inline static void ikjk(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            /*
                z = x * y^T : z(i, j) = sum_k x(i, k) * y(j, k), k in [0, xCol)

                x2(k, j) is not contiguous in j, so a vector over j cannot be
                formed from y.  Both operands are however row contiguous in k,
                therefore every element is a vectorized dot product of two rows.
            */
            (void)xRow;
            (void)yRow;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t j = 0; j < zCol; j++) {
                    z[i*zCol + j] = dot(x + i*xCol, y + j*yCol, xCol);
                }
            }
            return;
        }
        inline static void kijk(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            /*
                z = x^T * y^T : z(i, j) = sum_k x(k, i) * y(j, k), k in [0, xRow)

                y row j is contiguous, x is not: column i is gathered once and
                reused by every j.
            */
            (void)yRow;
            std::vector<float> col(xRow, 0.0f);
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xRow; k++) {
                    col[k] = x[k*xCol + i];
                }
                for (std::size_t j = 0; j < zCol; j++) {
                    z[i*zCol + j] = dot(col.data(), y + j*yCol, xRow);
                }
            }
            return;
        }

    };
};

}

#endif // SSE2FUNC_HPP
