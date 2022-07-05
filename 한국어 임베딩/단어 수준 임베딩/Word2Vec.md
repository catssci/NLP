# Word2Vec

2013 년 구글 연구팀이 발표한 기법으로 가장 널리 쓰이고 있는 단어 임베딩 모델이다.

## 모델의 종류

- CBOW: 주변의 context word를 이용하여 target word 하나를 맞추는 과정에서 학습
- Skip-gram: target word를 이용하여 주변 context word를 예측하는 과정에서 학습
- 기본적으로 Skip-gram의 성능이 더 좋다. 왜냐하면 같은 말뭉치로도 더 많은 데이터를 확보할 수 있기 때문이다.

## 모델의 변화

- target word에서 context word의 예측 확률를 높이도록 훈련
  - target word를 입력받아 context word의 확률을 출력하도록 모델을 학습
  - context word의 확률을 높이고, 나머지 word의 확률을 낮춘다.
  - 어휘 집합의 단어 수가 클수록 softmax의 계산량이 증가한다. (기본적으로 어휘 집합의 단어 수는 수십만 개)
- positive sample, negative sample 구분하는 binary classification 과정에서 훈련
  - negative sampling 방식을 적용하여 효율적인 학습을 진행한다.
  - 1 Step 당 1개의 positive sample, k개의 negative sample만 계산한다.

## Negative Sampling

- 확률 계산

​					$P_{negative}(w_i) = \frac{f(w_i)^\frac{3}{4}}{\sum^n_{j=0}f(w_j)^\frac{3}{4}}$, $f(w_i) = \frac{w_i 단어의 빈도수}{어휘집합의 크기}$

- 말뭉치에 자주 등장하지 않는 희귀한 단어가 negative sample로 잘 뽑힐 수 있도록 설계한다.
- example
  - $f(w_1) = 0.99, f(w_2) = 0.01$
  - $P(w_1) = 0.97, P(w_2) = 0.03$
  - negative sample로 뽑힐 확률이 자주 등장하는 $w_1$의 경우 0.99에서 0.97로 낮아지고, 희귀한 $w_2$의 경우 0.01에서 0.03으로 높아졌다.

## SubSampling

- 자주 등장하는 단어는 학습에서 제외한다.
- 확률 계산

​			$P_{subsampling}(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} , \ \ \  t = 1e-5$

- example
  - $w_1 = 0.01, w_2 = 0.0001$
  - $P(w_1) = 0.9684, P(w_2)= 0.6838$
  - 즉, 빈도가 높은 단어 $w_1$은 100번의 학습 중 96번 정도는 학습에서 제외된다. 그리고 빈도가 $w_1$보다 작은 $w_2$는 100번의 학습 중 68번 정도는 학습에서 제외된다.

## 모델 학습

- skip-gram 모델의 로그우도 함수

    		$L(\theta) = log{P(+|t_p, c_p)} + \sum^k_{i = 1}log{P(-|t_{n_i}, c_{n_i})}$

- $log{P(+|t_p, c_p)}= \frac{1}{1+exp(-u_tv_c)}$
- $log{P(-|t_n, c_n)}= 1 - \frac{exp(-u_tv_c)}{1+exp(-u_tv_c)}$
- positive sample t, c 의 벡터 간 유사도는 증가 시키고, negative sample t, c에서의 유사도는 낮게 되도록 학습한다.
- 모델의 파라미터는 $U, V$ 2개이다. $U$ 는 $|V| * d$ 크기의 가중치 행렬이고, $V$는 $d * |V|$ 크기의 가중치 행렬이다. 이를 이용하여 d차원의 임베딩 결과 또는 2d 차원의 임베딩 결과를 얻는다.

## Code

