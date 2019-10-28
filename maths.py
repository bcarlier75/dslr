import math
from typing import List

Vector = List[float]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w)

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def total_sum_of_squares(y: Vector) -> float:
    return sum(v ** 2 for v in de_mean(y))


def sum_of_squares(v: Vector) -> float:
    return dot(v, v)


def predict(theta_0: float, theta_1: float, x_i: float) -> float:
    return theta_1 * x_i + theta_0


def error(theta_0: float, theta_1: float, x_i: float, y_i: float) -> float:
    return predict(theta_0, theta_1, x_i) - y_i


def sum_of_sqerrors(theta_0: float, theta_1: float, x: Vector, y: Vector) -> float:
    return sum(error(theta_0, theta_1, x_i, y_i) ** 2
               for x_i, y_i in list(zip(x, y)))


def r_squared(theta_0: float, theta_1: float, x: Vector, y: Vector) -> float:
    return 1.0 - (sum_of_sqerrors(theta_0, theta_1, x, y) /
                  total_sum_of_squares(y))


def quantile(xs: List[float], p: float) -> float:
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


def variance(xs: List[float]) -> float:
    assert len(xs) >= 2

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))


def count(xs: List[float]) -> float:
    return float(len(xs))


def min_(xs: List[float]) -> float:
    my_min = xs[0]
    for elem in xs:
        if elem < my_min:
            my_min = elem
    return my_min


def max_(xs: List[float]) -> float:
    my_min = xs[0]
    for elem in xs:
        if elem > my_min:
            my_min = elem
    return my_min
