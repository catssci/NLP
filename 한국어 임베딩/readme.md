# 한국어 임베딩

### Embedding
자연어를 벡터로 바꾼 결과 또는 일련의 과정 전체
> 단어나 문장 각각을 벡터로 변환하여 벡터공간에 "끼워넣는다. (Embed)" 해서 **Embedding**
- Corpus 의미, 문장 정보 압축
- 사칙 연산 가능 -> 단어/문서 관련도 계산 가능
- 전이학습 (ELMo, BERT, GPT 등)

### 11가지 모델 학습
- 단어 수준 임베딩: NPLM, Word2Vec, FastText, LSA, GloVe, Swivel (6가지)
- 문장 수준 임베딩: LSA, Doc2Vec, LDA, ELMo, BERT (5가지)

### 임베딩의 발전 과정
1. 통계 기반 -> 뉴럴 네트워크 기반
2. 단어 수준 -> 문장 수준
3. 엔드투엔드 -> 프리트레인/파인 튜닝 방식

### 임베딩 기법의 분류
- 행렬 분해 모델
- 예측 기반 방법
- 토픽 기반 방법

### 임베딩 벡터의 의미 함축 분류
- Bag of Words
- Language Model
- Distributional Hypothesis

### 개발 환경
```
# gpu version
docker pull ratsgo/embedding-gpu
docker run -it --runtime=nvidia ratsgo/embedding-gpu bash
```
```
# cpu version
docker pull ratsgo/embedding-cpu
docker run -it ratsgo/embedding-cpu bash
```
