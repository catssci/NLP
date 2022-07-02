# NPLM

Neural Probabilistic Language Model를 살펴보자. 선구자적 모델로 2003년에 벤지오 연구팀이 제안한 방법이다. 

## 기존 언어 모델의 단점

- OOV 문제가 발생한 단어를 n-gram 문장으로 변환하고 나타날 확률은 0로 설정 됨 (back-off, smoothing 기술은 단순)
- 문장의 Long-Term dependency를 포착하기 어려움 (n-gram에서 n 값의 크기를 5이상 키우기 어려움)
- 단어/문장 간 유사도를 계산 X

## 모델
![image](https://user-images.githubusercontent.com/75521926/177001101-39148dcd-3fb5-4ebf-9007-ee2f420ab07c.png)
