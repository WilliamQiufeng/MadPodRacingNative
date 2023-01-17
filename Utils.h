//
// Created by mac on 2022/12/25.
//

#ifndef MADPODRACING_UTILS_H
#define MADPODRACING_UTILS_H

#include <cmath>
#include <iostream>

const double eps = 1e-5;

template<typename T>
inline bool FpEqual(T a, T b) {
    return abs(a - b) < eps;
}

template<typename T>
inline T Bound(T v, T lb, T ub) {
    if (v < lb) return lb;
    if (v > ub) return ub;
    return v;
}
// Given lb < 0, ub > 0,
// OutSignBound(v) =    lb  if v > lb and v < 0
//                      ub  if v < ub and v > 0
//                      v   otherwise
template<typename T>
inline T OutSignBound(T v, T lb, T ub) {
    if (v > lb && v < 0) return lb;
    if (v < ub && v > 0) return ub;
    return v;
}

template<typename T>
inline T MapVal(T v, T a, T b, T c, T d) {
    return (T) (c + ((double) (d - c)) * (v - a) / (b - a));
}

template<typename T>
inline T Normalize(T v, T a, T b) {
    return MapVal<T>(v, a, b, 0, 1);
}

template<typename T>
inline T Denormalize(T v, T a, T b) {
    return MapVal<T>(v, 0, 1, a, b);
}

template<typename T>
inline T Lerp(T a, T b, double t) {
    return (T) (a + (b - a) * t);
}

inline double DegToRad(double deg) { return deg * (M_PI / 180); }

inline double RadToDeg(double rad) { return rad * 180 / M_PI; }

inline double SinN(double x) { return sin(x * M_PI / 2); }  // [0..1] maps to [0..1]
inline size_t Cycle(int i, size_t n) { return (size_t) (i < 0 ? i + n : i % n); }
inline double ClampRadian(double rad) {
    while (rad < -M_PI) rad += M_PI * 2;
    while (rad > M_PI) rad -= M_PI * 2;
    return rad;
}
//inline double BestRotation(double rad) {
//    rad = ClampRadian(rad);
//
//}
struct Vec {
    double x, y;

    double Abs() const;

    double Angle() const;

    // double angle(Vec base) { return atan2(Det(base), dot(base)); }
    double Angle(Vec base) const;

    double Dot(Vec b) const;

    double Det(Vec b) const;

    double SqDist() const;

    double SqDist(Vec b) const;;

    Vec Unit() const;

    Vec Rotate(double rad) const;

    Vec Floor() const;

    void OutputReal(std::ostream& os);

    friend std::ostream& operator<<(std::ostream& os, const Vec& p);

    friend std::istream& operator>>(std::istream& os, Vec& p);

    Vec operator+(Vec b) const;

    Vec operator-(Vec b) const;

    Vec operator-() const;

    Vec operator*(double s) const;

    Vec operator/(double s) const;

    Vec& operator+=(const Vec b);

    Vec& operator*=(const double scale);

    bool operator==(Vec b) const;

    bool operator!=(Vec b) const;
};

Vec operator*(double a, Vec b);
Vec MinImpulse(double minI, double m, Vec v);

const inline Vec UnitRight = {1, 0};

inline std::pair<Vec, Vec> ElasticCollision(double m1, double m2, Vec u1, Vec u2) {
    double factor = 1 / (m1 + m2);
    return {
            factor * (m1 * u1 - m2 * u1 + 2 * m2 * u2),
            factor * (2 * m1 * u1 + m2 * u2 - m1 * u2)
    };
}
inline std::pair<Vec, Vec> ElasticCollision(double m1, double m2, Vec u1, Vec u2, Vec p1, Vec p2, double minImpulse = 120) {
    Vec j = (p1 - p2).Unit();
    double Jmod = OutSignBound(2 * m1 * m2 / (m1 + m2) * j.Dot(u2 - u1), -minImpulse, minImpulse);
    Vec J = Jmod * j;
    return {
        u1 + 1 / m1 * J,
        u2 - 1 / m2 * J
    };
}
inline double CollisionTime(Vec v1, Vec v2, Vec p1, Vec p2, double r1, double r2) {
    Vec dv = v1 - v2;
    Vec dp = p1 - p2;
    double pv = dp.Dot(dv);
    double vv = dv.SqDist();
    double pp = dp.SqDist();
    double delta = pv * pv - vv * (pp - (r1 + r2) * (r1 + r2));
    if (delta < 0) return -1;
    double sqrtDelta = std::sqrt(delta);
    double t1 = (- pv - sqrtDelta) / vv;
    double t2 = (- pv + sqrtDelta) / vv;
    if (t1 >= 0 && t1 <= 1) return t1;
    else if (t2 >= 0 && t2 <= 1) return t2;
    else return -1;
}

#endif //MADPODRACING_UTILS_H
