---
{"dg-publish":true,"permalink":"/ygy/causal-inference-for-the-brave-and-true/15-synthetic-control/"}
---


- https://github.com/CausalInferenceLab/Causal-Inference-with-Python/blob/main/causal-inference-for-the-brave-and-true/15-Synthetic-Control.ipynb


> [!NOTE] Synthetic Control
> - 집계 수준의 데이터만 있는 경우 DID를 사용하여 추론할 수 없다
> - 제어 집단을 정의해야 하고, 하나의 제어 집단이 처치된 집단에 대한 counterfactual에 적합하지 않을 수 있으므로 한계가 있다
> - Synthetic Control : 여러 제어 집단을 결합하여 처치 집단과 유사하게 만드는 방법
> - Synthetic Control을 통해 처치 집단에 처치가 없었다면 어떤 일이 일어났을지 알 수 있다
> - Fisher's Exact Tests 개념을 사용하여 Synthetic Control을 추론하는 방법
> 	- 처치되지 않은 집단이 처치된 척하고 효과 계산
> 	- 처치 없이도 관찰할 수 있는 플라시보 효과
> 	- 추정한 처치 효과가 통계적으로 유의한지 확인



만일 데이터가 집계된 데이터만 있을 경우 
DID 추정량
$(E[Y(1)|D=1] - E[Y(1)|D=0]) - (E[Y(0)|D=1] - E[Y(0)|D=0])$ 

- [마케팅사례] (개입 전후 두 도시의 평균 예금 데이터만 있을 경우)
	- ![스크린샷 2024-02-08 오전 12.11.00.png|210](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%2012.11.00.png)
	- *13장 참고*
		  : Porgo Alegre와 Florianopolis의 고객데이터
		  2개의 다른 시점에 걸쳐져 있음 (마케팅 개입이 이루어지기 전과 후)
		  처치 효과 추정을 위해 회귀 분석으로 DID추정량과 표준오차를 알 수 있음

- 이때 DID 추정량은 $(87.06 - 206.16) - (46.01 - 171.64) = 6.53$ 
- 샘플 크기 = 4 (= DID 모델의 매개변수 개수)


Synthetic Control
	처치 그룹과 처치받지 않았지만 처치 그룹과 가장 유사한 그룹을 따로 찾을 필요가 없음
	처치 받지 않은 여러 그룹의 조합을 통해 가상의 통제 집단(Synthetic Control)을 만듦


[ex] 캘리포니아주에서 담배세 도입이 담배 소비에 미치는 영향 추정 
```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

%matplotlib inline

pd.set_option("display.max_columns", 6)
style.use("fivethirtyeight")
```

```python
cigar = (pd.read_csv("data/smoking.csv")
         .drop(columns=["lnincome","beer", "age15to24"]))

cigar.query("california").head()

```
![스크린샷 2024-02-08 오전 12.24.09.png|500](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%2012.24.09.png)
- `state` : 주 index (캘리포니아 : 3)
- `retprice` : 담배 소매 가격
- `cigsale` : 1인당 담배 한갑 판매량 ← 결과변수
- `califoria` : 캘리포니아 주 여부
- `after_treatment` : 사후 개입 기간 여부 (담배세 및 건강 보호법[^1])

(시간 경과에 따른 캘리포니아 및 기타 주의 담배 판매량)
```python
ax = plt.subplot(1, 1, 1)

(cigar
 .assign(california = np.where(cigar["california"], "California", "Other States"))
 .groupby(["year", "california"])
 ["cigsale"]
 .mean()
 .reset_index()
 .pivot("year", "california", "cigsale")
 .plot(ax=ax, figsize=(10,5)))

plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Cigarette Sales Trend")
plt.title("Gap in per-capita cigarette sales (in packs)")
plt.legend();
```
![Pasted image 20240208002706.png|500](/img/user/media/Pasted%20image%2020240208002706.png)
⇒ 캘리포니아 사람들은 전국 평균보다 담배를 덜 샀다
	1980년대 이후 담배소비가 감소

캘리포니아 주가 다른 주에 비해 담배 소비가 가속화된 것이 담배세가 영향을 준 것인가?


- $J + 1$ 집단 : 일반성을 잃지 않고, 1은 개입의 영향을 받는 장치
- $j=2,...,J+1$ : 처치되지 않은 집단의 집합 ("donor pool")
- 우리가 가진 데이터는 개입하기 전의 기간 $T_0$를 포함하여 $T$ 기간에 걸쳐있다

- $Y_{jt}$ : 각 집단 $j$와 기간 $t$에 대한 결과
- $Y^N_{jt}$: 각 집단 $j$와 기간 $t$에 대해 개입이 없는 잠재적 결과
- $Y^I_{jt}$ : 각 집단 $j$와 기간 $t$에 대해 개입으로 인한 잠재적 결과

- $\tau_{1t} = Y^I_{jt} - Y^N_{jt}$ 
	: $t>T_0$ 인 시간 $t$에서 처치 집단 $j=1$ 에 대한 효과
	 
- 집단 $j=1$ 은 처치된 것이기 때문에 $Y^I_{jt}$ 은 사실이지만 $Y^N_{jt}$ 는 사실이 아님
⇒ $Y^N_{jt}$ 를 어떻게 추정하는가

처치 효과가 기간별로 시간에 따라 달라지므로
![Pasted image 20240208004320.png](/img/user/media/Pasted%20image%2020240208004320.png)
처치효과 추정 ⇒ 처치되지 않았다면 $j=1$ 집단의 결과에 어떤일이 일어났을지를 추정하는 문제



donor pool은 처치되지 않은 집단보다 처치된 집단의 특성에 훨씬 가깝다는 것을 기억하자 → ❓
따라서 Synthetic Control은 대조군 범위에 있는 집단의 가중평균으로 정의된다

가중치 $\pmb{W}=(w_2, ..., w_{J+1})$ 가 주어졌을때,
 $Y^N_{jt}$ 의 Synthtic Control 추정치
	 $\hat{Y}^N_{jt} = \sum^{J+1}_{j=2} w_j Y_{jt}$

선형 회귀는 변수의 가중 평균으로 예측을 얻는 방법

각 변수가 일정 기간동안 더미일 경우 DID 회귀는 다음과 같은 행렬 곱
![Pasted image 20240208004826.png](/img/user/media/Pasted%20image%2020240208004826.png)


Synthetic Control의 경우에는 집단이 많지 않지만, 기간이 많음
→ 입력 행렬을 뒤집는다

![Pasted image 20240208004935.png](/img/user/media/Pasted%20image%2020240208004935.png)

기간당 하나 이상의 피쳐가 있는 경우 쌓을 수 있다
![Pasted image 20240208005005.png](/img/user/media/Pasted%20image%2020240208005005.png)


## Synthetic Control as Linear Regression

OLS 사용

```python
features = ["cigsale", "retprice"]

inverted = (cigar.query("~after_treatment") # filter pre-intervention period
            .pivot(index='state', columns="year")[features] # make one column per year and one row per state
            .T) # flip the table to have one column per state

inverted.head()
```
![스크린샷 2024-02-08 오전 12.53.41.png](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%2012.53.41.png)

Y 변수를 캘리포니아주로 정의하고 X를 다른 주로 정의할 수 있다
```python
y = inverted[3].values # state of california
X = inverted.drop(columns=3).values  # other states
```

회귀 실행
```python
from sklearn.linear_model import LinearRegression
weights_lr = LinearRegression(fit_intercept=False).fit(X, y).coef_
weights_lr.round(3)
```
![스크린샷 2024-02-08 오전 12.55.08.png|575](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%2012.55.08.png)
⇒ 처치된 집단과 donor pool 사이의 제곱 차를 최소화하는 가중치 집합 반환

donor pool에 있는 주의 행렬과 가중치 간 곱셈
(주 1의 결과에 -0.436, 주 2의 결과에 -1.038, 주 4의 결과에 0.679, ...를 곱함)
```python
calif_synth_lr = (cigar.query("~california")
                  .pivot(index='year', columns="state")["cigsale"]
                  .values.dot(weights_lr))
```

시각화
```python
plt.figure(figsize=(10,6))
plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"], label="California")
plt.plot(cigar.query("california")["year"], calif_synth_lr, label="Synthetic Control")
plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend();
```
![Pasted image 20240208005715.png|575](/img/user/media/Pasted%20image%2020240208005715.png)

개입 후 Synthetic Control의 담배 판매량이 캘리포니아보다 많음 (= 개입이 담배 수요를 낮춘다)
개입 전 기간에 데이터가 일치한다 → 과적합
개입 후 Synthetic Control의 결과 변수에 큰 분산

⇒ 왜 이런 일이 일어날까?
dornor pool에 38개 주가 있음
즉, 38개의 매개변수가 있음
→ T가 크더라도 N도 크기 때문에 선형 회귀 모델에 너무 많은 유연성을 제공한다
정규화 모델 (Ridge, Lasso) 로 해결할 수 있다


## 외삽ㄴ 보간ㅇ

아래와 같은 데이터가 있다고 가정

![스크린샷 2024-02-08 오전 1.02.01.png|193](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%201.02.01.png)

컨트롤 그룹의 선형 조합을 사용하여 처치된 집단을 재현하는 Synthetic Control을 구축한다고 할때
일치시킬 수 있는 집단은 3개이고 속성은 2개
1) 첫번째 컨트롤에는 2.25를 곱하고, 두번째 컨트롤에는 -2를 곱하고, 둘을 더한다
	두번째 곱셈이 -16의 매출과 -8의 가격을 가지게 됨
	-> control 2 집단을 의미 없는 데이터 영역으로 외삽

2) Synthetic Control을 보간만 하도록 제한
	: 가중치를 양수로 제한하고 합산을 1로 제한
	-> Sythentic Control은 dornor pool의 Convex combination
	보간을 할때, 아래 그림처럼 처치되지 않은 집단으로 정의된 convex hull에 처치 집단을 투영할 것

![Pasted image 20240208010951.png](/img/user/media/Pasted%20image%2020240208010951.png)
(1) 보간은 이 경우 처치된 집단과 완벽히 일치할 수 없다
: 가장 적은 판매량과 가장 높은 가격을 가진 집단이 취급되기 때문
Convex combination은 control 집단 사이에 있는 피쳐들만 정확하게 복제할 수 있다

(2) 보간이 희박하다 
: convex hull의 벽에 처치 집단을 투영할 것이며, 이 벽은 몇 개의 집단으로만 정의되기 때문에
보간은 많은 집단에 가중치 0을 할당한다


공식화

Synthetic Control
	$\hat{Y}^N_{jt} = \sum^{J+1}_{j=2} w_j Y_{jt}$

최소화하는 가중치 $\pmb{W}=(w_2, ..., w_{J+1})$ 사용
	$||\pmb{X}_1 - \pmb{X}_0 \pmb{W}|| = \bigg(\sum^k_{h=1}v_h \bigg(X_{h1} - \sum^{J+1}_{j=2} w_j X_{hj} \bigg)^2 \bigg)^{\frac{1}{2}}$

$w_2, ..., w_{J+1}$ 은 양수이고 합이 1
처치된 대조군과 Synthetic Control 간의 차이를 최소화할 때 각 변수의 중요성을 반영

$v$들이 다르면 최적의 가중치가 달라진다
$V$를 선택하는 방법 
1) 각 변수가 평균 0과 단위 분산을 갖도록 한다 (각 변수에 동일한 중요성 부여)
2) 예측하는데 도움이 되는 변수가 더 높은 중요성을 갖는 방식으로 $Y$ 선택

1)방법 코드

손실 함수 정의
```python
from typing import List
from operator import add
from toolz import reduce, partial

def loss_w(W, X, y) -> float:
    return np.sqrt(np.mean((y - X.dot(W))**2))
```

가중치를 1로 합산하도록 제한 (`lambda x: np.sum(x) - 1`)
최적화 범위를 0과 1사이로 설정
```python
from scipy.optimize import fmin_slsqp

def get_w(X, y):
    
    w_start = [1/X.shape[1]]*X.shape[1]

    weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights
```

Synthetic Control을 정의하는 가중치
```python
calif_weights = get_w(X, y)
print("Sum:", calif_weights.sum())
np.round(calif_weights, 4)
```
![스크린샷 2024-02-08 오전 1.21.24.png|625](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%201.21.24.png)

이 가중치를 사용하여 주 1,2,3에 0을 곱하고 주 4에 0.0852를 곱
가중치가 희박하고
모든 가중치의 합은 1
가중치는 0과 1사이에 있음
⇒ convex combination 제약 조건 충족


가중치 곱
```python
calif_synth = cigar.query("~california").pivot(index='year', columns="state")["cigsale"].values.dot(calif_weights)
```

결과 플롯
```python
plt.figure(figsize=(10,6))
plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"], label="California")
plt.plot(cigar.query("california")["year"], calif_synth, label="Synthetic Control")
plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Per-capita cigarette sales (in packs)")
plt.legend();
```
![Pasted image 20240208012331.png|575](/img/user/media/Pasted%20image%2020240208012331.png)

⇒ 과적합되지 않았음


Synthetic Control을 사용하여 처치효과 추정 : (처치효과) - (Synthetic Control 결과)
	$\tau_{1t} = Y^I_{jt} - Y^N_{jt}$

```python
plt.figure(figsize=(10,6))
plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"] - calif_synth,
         label="California Effect")
plt.vlines(x=1988, ymin=-30, ymax=7, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=2)
plt.title("State - Synthetic Across Time")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend();
```
![Pasted image 20240208012509.png|500](/img/user/media/Pasted%20image%2020240208012509.png)

⇒ 효과가 시간이 지날수록 커짐
	2000년까지 담배세가 담배 판매량을 25갑이나 줄임

→ 통계적으로 유의한가?


## 통계적 유의성

표본크기가 매우 작기 때문에 더 합리적이어야함
→ Fisher's Exact Test의 개념 사용

처치된 것을 치환하고 철저히 통제
처치된 집단이 하나기 때문에, 각 집단은 처치된 것으로 간주하고 다른 집단은 통제된 것으로 간주

![스크린샷 2024-02-08 오전 1.28.30.png|350](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%201.28.30.png)

각 주에 대해 하나의 Synthetic Control 및 효과 추정치를 갖게 됨
(캘리포니아가 아닌 다른 주에서 실제로 발생했다고 가정하고, 일어나지 않은 처치에 대한 효과가 무엇인지)

캘리포니아의 처치가 다른 가짜 처치에 비해 충분히 큰지 비교

실제로 처치 받지 않은 주들에 대해 처치를 받은 것처럼 가정하면 유의미한 처치 효과를 찾을 수 없다

`state` 로 해당 주에 대한 Synthetic Control을 추정하는 함수를 만듦
```python
def synthetic_control(state: int, data: pd.DataFrame) -> np.array:
    
    features = ["cigsale", "retprice"]
    
    inverted = (data.query("~after_treatment")
                .pivot(index='state', columns="year")[features]
                .T)
    
    y = inverted[state].values # treated
    X = inverted.drop(columns=state).values # donor pool

    weights = get_w(X, y)
    synthetic = (data.query(f"~(state=={state})")
                 .pivot(index='year', columns="state")["cigsale"]
                 .values.dot(weights))

    return (data
            .query(f"state=={state}")[["state", "year", "cigsale", "after_treatment"]]
            .assign(synthetic=synthetic))
```
⇒ 주, 연도, 판매 결과, 해당 주에 대한 결과를 포함하는 데이터 프레임 반환


첫 번째 주에 적용했을 때
```python
synthetic_control(1, cigar).head()
```
![스크린샷 2024-02-08 오전 1.32.23.png|400](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%201.32.23.png)


모든 주에 대한 결과를 얻기 위해 계산 병렬화
```python
from joblib import Parallel, delayed

control_pool = cigar["state"].unique()

parallel_fn = delayed(partial(synthetic_control, data=cigar))

synthetic_states = Parallel(n_jobs=8)(parallel_fn(state) for state in control_pool)
```

```python
synthetic_states[0].head()
```
![스크린샷 2024-02-08 오전 1.33.18.png|500](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%201.33.18.png)


모든 주에 대한 Synthetic control을 사용하여 모든 주에 대한 합성 상태와 실제 상태 사이의 간격 추정
(캘리포니아의 경우 이것이 처치효과, 다른 주는 실제로 일어나지 않은 Synthetic Control 처치 효과 추정)
```python
plt.figure(figsize=(12,7))
for state in synthetic_states:
    plt.plot(state["year"], state["cigsale"] - state["synthetic"], color="C5",alpha=0.4)

plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"] - calif_synth,
        label="California");

plt.vlines(x=1988, ymin=-50, ymax=120, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=3)
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.title("State - Synthetic Across Time")
plt.legend();
```
![Pasted image 20240208013423.png|575](/img/user/media/Pasted%20image%2020240208013423.png)

1) 개입 후의 분산이 개입 전의 분산보다 높다
	: Synthetic Control은 개입 전 기간의 차이를 최소화하도록 설계됐으므로 당연함
2) 개입 전 기간에도 잘 맞지 않는 집단이 있다
	: 일부 주에서 담배 소비량이 매우 높은 경우, 다른 주에서 convex combination은 절대 일치할 수 었으므로 가능함
	-> 이런 집단은 분석에서 제거하는 것이 좋음

사전 개입 오류에 대한 임계값 설정
	$MSE = \frac{1}{N}\sum\bigg(Y_t - \hat{Y}^{Synth}_t\bigg)^2$


높은 오류가 있는 집단을 제거
```python
def pre_treatment_error(state):
    pre_treat_error = (state.query("~after_treatment")["cigsale"] 
                       - state.query("~after_treatment")["synthetic"]) ** 2
    return pre_treat_error.mean()

plt.figure(figsize=(12,7))
for state in synthetic_states:
    
    # remove units with mean error above 80.
    if pre_treatment_error(state) < 80:
        plt.plot(state["year"], state["cigsale"] - state["synthetic"], color="C5",alpha=0.4)

plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"] - calif_synth,
        label="California");

plt.vlines(x=1988, ymin=-50, ymax=120, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=3)
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.title("Distribution of Effects")
plt.title("State - Synthetic Across Time (Large Pre-Treatment Errors Removed)")
plt.legend();
```
![Pasted image 20240208013816.png|575](/img/user/media/Pasted%20image%2020240208013816.png)
⇒ 캘리포니아 주의 영향이 극단적임을 알 수 있음
다른 주에서 처치가 일어났다고 가정해도, 캘리포니아와 같은 극단적인 효과를 거의 얻지 못할 것이다


P-value 도출
(캘리포니아의 영향보다 몇배나 낮은지 확인)
```python
calif_number = 3

effects = [state.query("year==2000").iloc[0]["cigsale"] - state.query("year==2000").iloc[0]["synthetic"]
           for state in synthetic_states
           if pre_treatment_error(state) < 80] # filter out noise

calif_effect = cigar.query("california & year==2000").iloc[0]["cigsale"] - calif_synth[-1] 

print("California Treatment Effect for the Year 2000:", calif_effect)
np.array(effects)
```
![스크린샷 2024-02-08 오전 1.40.07.png|600](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%201.40.07.png)

![스크린샷 2024-02-08 오전 1.40.29.png|575](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%201.40.29.png)

캘리포니아의 효과가 0보다 작다는 단측 가설을 검정
캘리포니아의 효과가 추정된 모든 효과보다 큰 비율인 P-value 추정
	$PV=\frac{1}{N}\sum \mathcal{1}\{\hat{\tau}_{Calif} > \hat{\tau}_j\}$

밝혀진 바와 같이, 2000년 캘리포니아의 치료 효과는 -24.8로, 개입으로 인해 담배 소비가 거의 25갑 감소했습니다.
우리가 추정한 다른 34가지 플라시보 효과 중에서 단 하나만 캘리포니아에서 발견한 효과보다 높습니다. 
따라서 P-value는 1/35


```python
np.mean(np.array(effects) < calif_effect)
```
![스크린샷 2024-02-08 오전 1.42.03.png|168](/img/user/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-08%20%EC%98%A4%EC%A0%84%201.42.03.png)


캘리포니아의 효과 값이 실제로 얼마나 극단적인지 분포 시각화
```python
_, bins, _ = plt.hist(effects, bins=20, color="C5", alpha=0.5);
plt.hist([calif_effect], bins=bins, color="C0", label="California")
plt.ylabel("Frquency")
plt.title("Distribution of Effects")
plt.legend();
```
![Pasted image 20240208014355.png](/img/user/media/Pasted%20image%2020240208014355.png)






---
[^1]: 1988년 캘리포니아는 [발의안 제 99호](https://en.wikipedia.org/wiki/1988_California_Proposition_99)로 잘 알려진 유명한 담배세 및 건강 보호법을 통과시켰습니다. “이 법안의 주요 효과는 캘리포니아 내 담배 판매에 대해 한 갑당 25센트의 주 소비세를 부과하는 것이며, 시가 및 씹는 담배와 같은 기타 상업용 담배 제품의 소매 판매에도 거의 동일한 소비세를 부과하는 것입니다. 담배 판매에 대한 추가 제한 사항에는 청소년이 접근할 수 있는 공공 장소에서 담배 자판기 사용 금지, 단일 담배 개별 판매 금지 등이 있습니다. 이 법으로 발생한 수익은 다양한 환경 및 건강 관리 프로그램과 금연 광고에 사용되었습니다.”