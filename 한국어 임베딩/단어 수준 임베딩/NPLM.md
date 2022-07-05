# NPLM

Neural Probabilistic Language Model를 살펴보자. NPLM은 선구자적 모델로 벤지오 연구팀이 2003년에 제안한 방법이다. 

## 기존 언어 모델의 단점

- n-gram 문장으로 변환한 후 OOV 단어를 포함한 문장의 나타날 확률은 0로 설정 됨 (back-off, smoothing 기술은 단순)
- 문장의 Long-Term dependency를 포착하기 어려움 (n-gram에서 n 값의 크기를 5이상 키우기 어려움)
- 단어/문장 간 유사도를 계산 X

## 모델
![image](https://user-images.githubusercontent.com/75521926/177001101-39148dcd-3fb5-4ebf-9007-ee2f420ab07c.png)

## 학습 과정
- 다음 단어를 예측하는 과정에서 학습한다.

- 네트워크의 출력은 각 단어의 출현 확률 $y$로 표현된다. 이때 $y$는 단어 $V$ 크기의 벡터로 단어들의 예측 확률 값을 표현한다. 즉, 목표 단어 $w_t$의 확률을 크게 되도록 예측한다.

  ​																$P(w_t|w_{t-1}, ..., w_{t-n+1}) = \frac{exp(y_{w_t})}{sum(exp(y_t))}$

- $y_i$는 $y_{w_t}$의 원소

- $y_{w_t}$ 계산

  - $x = [C(w_{t-1}), C(w_{t-2}), ..., C(w_{t-n+1})]$
  - $C(w_{t-1})$는 Embedding 과정을 거친 크기 $m$인 벡터
  - $x \ shape: (n-1) * m$
  - $y_{w_t} = b + Wx + Utanh(d + Hx)$
  - $y_{w_t} \ shape: |V| \ vector$


## 코드
https://kthworks.github.io/nlp/Neural-Probabilistic-Language-Model-(NPLM)-Pytorch%EB%A1%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0/
