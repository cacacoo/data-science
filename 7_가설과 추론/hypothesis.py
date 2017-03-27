# coding=utf-8
import math


# 가설이란 '이 동전은 앞뒤가 나올 확률이 공평한 동전이다' 등과 같은 주장을 의미한다
# 이러한 주장들은 통계치들을 통해 얼마나 타당한지 알 수 있게 한다

# 고전적인 가설검정
# 귀무가설(H0, null hypothesis) : 기본적인 가설, 동전이 앞면이 나올 확률은 0.5이다.
# 대립가설(H1, alternative hypothesis) : 비교대상이 되는 대립하는 가설, 동전이 앞면이 나올 확률은 0.5가 아니다.
# 통계를 통해서 H0를 기각할지 말지를 결정한다.
def normal_approximation_to_binomial(n, p):
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma


def normal_pdf(x, mu=0, sigma=1):  # x는 정규분포내 값, mu는 평균, sigma는 표준편차 1 => default : 표준정규분포
    """정규분포의 밀도함수"""
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2)) / (sqrt_two_pi * sigma)


# print normal_pdf(0)
# print normal_pdf(1)


def normal_cdf(x, mu=0, sigma=1):
    """정규분포의 누적분포함수"""
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


# print normal_cdf(0)
# print normal_cdf(1)

# 실제 동전 던지기로부터 얻은 값이 구간 안(혹은 밖)에 존재할 확률 구하기
# 누적분포함수는 확률변수가 특정 값보다 작을 확률을 나타낸다
normal_probability_below = normal_cdf


# 만약 확률변수가 특정 값보다 작지 않다면 특정 값보다 크다는 것을 의미
def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)


# 확률변수가 hi, lo 사이에 존재하는 경우
def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)


# 확률변수가 범위 밖에 존재하는 경우
def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)


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


# 분포의 60%를 차지하는 평균 중심 구간 구하기 = 양쪽 꼬리 20% 차지하는 지점 구하기
def normal_upper_bound(probability, mu=0, sigma=1):
    """P(Z <= z) = probability 인 z 값을 반환"""
    return inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability, mu=0, sigma=1):
    """P(Z >= z) = probability 인 z 값을 반환"""
    return inverse_normal_cdf(1 - probability, mu, sigma)


def normal_two_side_bound(probability, mu=0, sigma=1):
    tail_probability = (1 - probability) / 2
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound


mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
print mu_0, sigma_0

# 유의수준(significance) : 제 1종 오류를 얼마나 허용해 줄 것인지 의미
# 1종 오류 : H0라 참이지만 H0를 기각하는 'false positive(가양성)'오류를 의미
# 보통 5%나 1%로 설정하는 경우가 많다.

# 5%를 유의수준으로 잡으면 469와 531을 벗어나는 결과가 나올 확률이 20번 중에 1번이라는 뜻이 된다
print normal_two_side_bound(0.95, mu_0, sigma_0)

# 검정력(power) : 2종 오류인 'H0가 거짓이지만 H0를 기각하지 않은 오류를 범하지 않을 확률'을 의미
# 2종 오류 검증을 위해서는 먼저 H0가 거짓이라는 것이 무엇을 의

# p가 0.5라고 가정할 때 유의수준이 5%인 구간
lo, hi = normal_two_side_bound(0.95, mu_0, sigma_0)

# p = 0.55인 경우의 실제 평균가 표준편자
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# 제 2종 오류란 귀무가설(H0)를 기각하지 못한다는 의미
# 즉, X가 주어진 구간 안에 존재할 경우를 의미
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability
print power

hi = normal_upper_bound(0.95, mu_0, sigma_0)  # 결과값은 526, (< 531, 분포 상위 부분에 더 높은 확률을 주기 위해서)
type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability
print power


# p-value : H0가 참이라고 가정하고 실제로 관측된 값보다 더 극단적인 값이 나올 확률을 구하는 것
def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)


print two_sided_p_value(529.5, mu_0, sigma_0)

# 사건에 대한 분포를 모를 때 관측된 값에 대한 신뢰구간을 사용하여 가설을 검증할 수 있다
# 동전 1000번 던졌을 때, 앞면이 525번 나왔다면, p는 0.525 로 추정할 수 있다.
# 하지만 이 p값을 얼마나 신뢰할 수 있을까?? 이를 증명하는 것이 중요!

# 정확한 p값을 알고 있다면, 중심극한정리를 사용하여 베르누이 확룰변수들의
# 평균은 대략 평균이 p고 표준편차는 math.sqrt(p * (1 - p) / 1000) 인 정규분포로 추정가능하다.

# 정확한 p값을 모른다면, 추정치를 사용할 수 있다.
p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)

normal_two_side_bound(0.95, mu, sigma)  # [0.4940, 0.5560]


# 이 경우, 0.5는 신뢰구간 안에 있기때문에, 동전은 공평하지 않다고 결론을 내릴수 없다. 그런데 540번 나왔다면 95%에 대해 통과못하게된다


# p-value 해킹
# 귀무가설을 잘못 기각하는 경우가 5%인 가설검정은 정의에서 알 수 있듯이 모든 경우의 5%에서 귀무가설을 잘못 기각한다.??


# A/B 테스트 해보기
# P(a) : A광고를 클릭할 확률, 표준편차는 math.sqrt(P(a) * (1 - P(a)) / n)
# P(b) : B광고를 클릭할 확률, 표준편차는 math.sqrt(P(b) * (1 - P(b)) / n)
def estimated_parameters(N, n):
    p = n / N
    s = math.sqrt(p * (1 - p) / n)
    return p, s


# 두 정규분포가 독립이라면 두 정규분포의 차이는 평균이 P(b) - P(a), 표준편차는 math.sqrt(sigma_a^2 + sigma_b^2) 인 정규분포를 따른다.
# 따라서 A와 B가 서로 차이가 없다면, 귀무가설로서 H0는 P(b) - P(a) = 0 이 되어야 한다.
def a_b_test_statistics(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)


# 1000명이 A를 200번, B를 180번 누른 경우, 평균이 같다고 가정했을 때 이러한 차이가 발생할 확률은 0.254 (25.4%)
z = a_b_test_statistics(1000, 200, 1000, 180)
print two_sided_p_value(z)

# 1000명이 A를 200번, B를 150번 누른 경우, 평균이 같다고 가정했을 때 이러한 차이가 발생할 확률은 0.003 (3%)
z = a_b_test_statistics(1000, 200, 1000, 150)
print two_sided_p_value(z)


# 베이지안 추론
# 위의 경우와는 다른 방식의 추론은 없을까? 음...