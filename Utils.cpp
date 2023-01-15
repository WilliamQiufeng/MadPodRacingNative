//
// Created by mac on 2022/12/25.
//

#include "Utils.h"

double Vec::Abs() const { return sqrt(SqDist()); }

double Vec::Angle() const { return atan2(y, x); }

double Vec::Angle(Vec base) const { return Angle() - base.Angle(); }

double Vec::Dot(Vec b) const { return x * b.x + y * b.y; }

double Vec::Det(Vec b) const { return x * b.y - y * b.x; }

double Vec::SqDist() const { return x * x + y * y; }

double Vec::SqDist(Vec b) const { return operator-(b).SqDist(); }

Vec Vec::Unit() const { return operator/(Abs()); }

Vec Vec::Rotate(double rad) const {
    auto s = sin(rad), c = cos(rad);
    return {x * c - y * s, x * s + y * c};
}

void Vec::OutputReal(std::ostream& os) { os << "(" << x << ", " << y << ")"; }

std::ostream& operator<<(std::ostream& os, const Vec& p) {
    os << (int) p.x << ' ' << (int) p.y;
    return os;
}

std::istream& operator>>(std::istream& os, Vec& p) {
    os >> p.x >> p.y;
    return os;
}

Vec operator*(double a, Vec b) {
    return b * a;
}

Vec MinImpulse(double minI, double m, Vec v) {
    if (m * v.Abs() < minI) return v.Unit() * (minI / m);
    return v;
}

bool Vec::operator!=(Vec b) const { return !operator==(b); }

bool Vec::operator==(Vec b) const { return FpEqual(x, b.x) && FpEqual(y, b.y); }

Vec Vec::operator/(double s) const { return {x / s, y / s}; }

Vec Vec::operator*(double s) const { return {x * s, y * s}; }

Vec Vec::operator-() const { return {-x, -y}; }

Vec Vec::operator-(Vec b) const { return {x - b.x, y - b.y}; }

Vec Vec::operator+(Vec b) const { return {x + b.x, y + b.y}; }

Vec& Vec::operator+=(const Vec b) {
    x += b.x;
    y += b.y;
    return *this;
}

Vec& Vec::operator*=(const double scale) {
    x *= scale;
    y *= scale;
    return *this;
}

Vec Vec::Floor() const {
    return {std::floor(x), std::floor(y)};
}
