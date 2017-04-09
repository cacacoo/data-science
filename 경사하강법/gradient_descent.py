# coding=utf-8
import random
import math
import matplotlib.pyplot as plt
from ..선형대수 import vector


def sum_of_squares(v):
    """V에 속해 있는 항목들의 제곱합을 계산한다"""
    return sum(v_i ** 2 for v_i in v)


print(sum_of_squares([2, 3, 4, 5]))


# 미분값은 함수 변화율의 극한값이다
# f는 x에 따라 값이 변화하는 함수, x는 점의 위치, h는 x의 변화량
def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h


# 즉 기울기, x가 x + h까지 변화할 때, f(x)가 f(x+h)만큼 변화량을 h로 나눈다면 이를 "평균 변화율"이라 한다.
# h가 충분히 작아 0에 수렴할 때, 이를 순간 변화율이라 한다.
# 그래서 미분은 함수 변화율(평균 변화율)의 0에 수렴하는 극한값이라 한다.

# 도함수 :함수 f(x) 의 특정 구간을 정의역으로 하고 미분계수를 치역으로 하는 함수 f'(x)를 f(x)에 대한 도함수라고 한다
def square(x):
    return x * x


def derivative_square(x):
    return 2 * x


derivative_estimate = lambda x: difference_quotient(square, x, h=0.1)

x = range(-10, 10)

plt.title("Actual Derivatives vs. Estimates")
plt.plot(x, map(derivative_square, x), 'rx', label='Actual')
plt.plot(x, map(derivative_estimate, x), 'b+', label='Estimate')
plt.legend(loc=9)
plt.show()


# f가 다변수 함수면 여러 개의 입력 변수 중 하나에 작은 변화가 있을 때, f(x)의 변화량을 알려주는 편도함수(partial derivative) 역시 여러개 존재한다.
# i 번째 편도함수는, i 번째 변수를 제외한 다른 모든 입력변수를 고정시켜서 계산한다
def partial_difference_quotient(f, v, i, h):
    """함수 f의 i번째 편도함수가 v에서 가지는 값"""
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]


# sum_of_squares는 v가 0 백터일 때 가장 작은 값을 가진다. 그런데 이 사실을 모를 때, 최솟값을 어떻게 구할까??
# 경사하강법을 이용해 최솟값을 구해보자
# 임의의 시작점을 잡고 gradient가 아주 작아질 때 까지 경사의 반대 방향으로 무한히 이동시킨다.
def step(v, direction, step_size):
    """v에서 step_size만큼 이동하기"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]


def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


# 임의의 시작점을 선택
v = [random.randint(-10, 10) for i in range(3)]

tolerance = 0.0000001

while True:
    gradient = sum_of_squares_gradient(v)  # v의 경사도 계산
    next_v = step(v, gradient, -0.01)  # 경사도의 음수만큼 이동
    if vector.distance(next_v, v) < tolerance:
        break
    v = next_v

# 하지만 적절한 이동거리(step_size)를 정하는 것은 쉽지 않다. 너무 작게 잡으면 너무 많은 cost가 들지만 정확하다. 그 반대도 있고
# 몇가지 방법이 있는데, 이동거리를 고정, 시간혹은 횟수에 따라 이동거리를 줄임, 이동할 때마다 목적 함수를 최소화하는 이동거리로 정함
# 세번째 방법이 자주 쓰이며 이때 몇몇 정해진 이동 거리를 시도해 보고 그중에서 목적함수를 최소화하는 값을 고르는 방법이 있다.

step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]


# 하지만 step_size를 잘못 넣어 오류가 날 수 있으니, 오류값이 들어가면 그냥 제외되도록 무한대를 반환하게하는 함수를 만들자
def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')

    return safe_f


def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """목적 함수를 최소화시키는 theta를 경사 하강법을 사용해서 찾아주는 함수"""
    theta = theta_0  # theta를 시작점으로 설정
    target_fn = safe(target_fn)  # 오류를 처리할 수 있는 target_fn 으로 변환
    value = target_fn(theta)  # 최소화시키려는 값

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        # 함수를 최소화시키는 theta 선택
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        # tolerance만큼 수렴하면 멈춤
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value


# 함수를 최대화해야 할 떄도 있는데 그때는 목적함수의 음수값을 최소화하면 된다
def negate(f):
    """x를 입력하면 -f(x)를 반환해 주는 함수 생성"""
    return lambda *args, **kwargs: -f(*args, **kwargs)


def negate_all(f):
    """f가 여러 숫자를 반환할 때 모든 숫자를 음수로 반환"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]


def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)


