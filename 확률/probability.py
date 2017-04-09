# coding=utf-8
import random
import matplotlib.pyplot as plt
import math


# 조건부 확률
# P(E,F) = P(E)P(F)
# P(E|F) = P(E,F) / P(F)
# P(E,F) = P(E|F)P(F)
def random_kid():
    return random.choice(["boy", "girl"])


both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)
for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == "girl":
        older_girl += 1
    if older == "girl" and younger == "girl":
        both_girls += 1
    if older == "girl" or younger == "girl":
        either_girl += 1

print "P(both | older):", both_girls / older_girl
print "P(both | either):", both_girls / either_girl


# 베이즈 정리
# P(E|F) = P(E,F)/P(F) = P(F|E)P(E)/P(F)
# P(F) = P(F,E) + P(F,^E)  -> ^E는 E가 일어나지 않은 확률
# P(E|F) = P(F|E)P(E) / P(F,E) + P(F,^E) = P(F|E)P(E) / [P(F|E)P(E) + P(F|^E)P(^E)]

# 연속 분포
# 이산형 분포 : 각각의 결과에 확률을 계산 가능한 분포(동전 던지기)
# 균등 분포 : 0~1사이에 모든 값에 동등한 비중을 준 분포
# 0~1사이에는 무한히 많은 숫자가 존재 > 따라서 특정 구간을 적분한 값으로 확률을 나타내는 확률밀도함수(probability density function, pdf)로 연속 분포를 표현
def uniform_pdf(x):
    """균등분포의 확률밀도함수균등분포의 확률밀도함수"""
    return 1 if 0 <= x < 1 else 0


# 누적 분포 함수(cumulative distribution function, cdf) : 확률변수의값이 특정 값보다 작거나 클 확률
def uniform_cdf(x):
    """균등분포를 따르는 확률변수의 값이 x보다 작거나 같을 확률을 반화"""
    if x < 0:
        return 0  # 균등분포의 확률은 절대로 0보다 작을 수 없음
    elif x < 1:
        return x  # 예시 : P(X <= 0.4) = 0.4
    else:
        return 1  # 균등분포의 확률은 항상 1보다 작음


# 정규분포(normal distribution)
# 종모양, 뮤(평균), 시그마(표준편차)의 두 파라미터로 정의
def normal_pdf(x, mu=0, sigma=1):  # x는 정규분포내 값, mu는 평균, sigma는 표준편차 1 => default : 표준정규분포
    """정규분포의 밀도함수"""
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2)) / (sqrt_two_pi * sigma)


xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')


# 표준정규분포(standard normal distribution) : mu가 0, sigma가 1인 정규분포
# Z를 표준정규분포의 확률변수로로 나타내면, X = sigma*Z + mu
# Z = X - mu / sigma
def normal_cdf(x, mu=0, sigma=1):
    """정규분포의 누적분포함수"""
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """이진 검색을 사용해서 역함수를 근사"""
    # 표준정규분포가 아니라면 표준정규분포로 변환
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0
    hi_z, hi_p = 10.0, 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2  # 중간 값
        mid_p = normal_cdf(mid_z)  # 중간 값의 누적분포
        if mid_p < p:
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            hi_z, hi_p = mid_z, mid_p
        else:
            break

    return mid_z

# 중심극한정리(central limit theorem)
# 동일한 분포에 대한 독립적인 확률변수의 평균을 나타내는 확률변수가 대략적으로 정규분포를 따른다는 정리
# 예로써, x_1, x_2 ... x_n 을 평균 mu, 표준편차 sigma를 갖는 확률변수라고 할때, n이 적당히 크다면
# (x_1 + x_2 + x_3 ... + x_n) / n 또한 평균이 mu, 표준편차가  sigma / n 인 정규분포와 비슷해 질 것이다

# 만약, 동전을 100번 던져서 앞면이 60번 이상 나올 확률을 알고 싶다.했을 때
# 중심극한정리를 사용하여, 평균 50, 표준편차가 5인 정규분포에서 확률변수가 60이상일 확률로 근사할 수 있다는 점에서 그 가치가 있다
# 이 방법은 이항 함수의 밀도함수에서 확률을 직접 계산하는 것보다 훨씬 쉽기때문.
