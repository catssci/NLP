# chatgpt api를 이용하여 chatbot 만들기 (using embedding)

# 01. embedding 사용 이유

1. fine-tuning 보다 비용이 적게 발생
2. fine-tuning 보다 더 작은 데이터를 사용하여 높은 성능을 보여줌

# 02. chatbot 만들기 과정 및 정리

1. 데이터 전처리
2. 데이터 임베딩
3. 입력된 쿼리로 부터 가장 유사한 데이터 n개 뽑기
4. 데이터 n개를 api의 `ChatCompletion`에 넣기
5. 원하는 결과를 얻기 위한 `prompt`를 작성
6. 결과 확인

### 과정 도식화
![embedding 과정 도식화](https://github.com/catssci/NLP/assets/75521926/68e4faa5-071b-4202-8715-7c4aa228b6a5)

# 03. reference
[openai-cookbook/examples/Question_answering_using_embeddings.ipynb at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)

[openai-cookbook/Recommendation_using_embeddings.ipynb at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb)

[openai-cookbook/examples/Obtain_dataset.ipynb at 41a5d394ca355e276ba21290696116c33f55ad9f · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/41a5d394ca355e276ba21290696116c33f55ad9f/examples/Obtain_dataset.ipynb)

[openai-cookbook/examples/User_and_product_embeddings.ipynb at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/User_and_product_embeddings.ipynb)
