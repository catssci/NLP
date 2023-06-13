# gpt api 사용 하기

# 요약

- 목적: 블로그 데이터를 학습한 gpt 만들기
- 활용 범위
    - 내부
        - 회사 내부의 질문에 대한 빠른 검색
        - 고객 응대 전 체크
    - 외부 (고객 지원)
        - 문의 메일 최소화
- 방법
    - openAI fine tuning 따라하기
- 결과
    - 좋은 성능을 보여주지는 못하고 있음..
- 추가 방향
    - **embedding을 이용한 검색, QnA 고려 + gpt를 이용한 텍스트 생성**
    - 다른 임베딩 방법 고려

# Reference

## fine-tuning

[OpenAI API](https://platform.openai.com/docs/guides/fine-tuning/advanced-usage)

[openai-cookbook/examples/fine-tuned_qa at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/tree/main/examples/fine-tuned_qa)

[openai-cookbook/Fine-tuned_classification.ipynb at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Fine-tuned_classification.ipynb)

## embedding

[openai-cookbook/Question_answering_using_embeddings.ipynb at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb?ref=mlq.ai)

[Building a Custom GPT-3 Q&A Bot Using Embeddings](https://www.mlq.ai/fine-tuning-gpt-3-question-answer-bot/)

# 세부 fine tuning

### 1. 데이터 전처리

- 사용 데이터: 회사 블로그 글
- 전처리
    - 블로그 제목 + 생성 날짜 + 본문 내용
    - fine tuning 모델은 token 개수가 제한적 (2049 tokens)
    - gpt-3.5-turbo 모델로 데이터 요약 (4096 tokens)

### 2. 학습 데이터 생성

- 블로그 글 데이터로 질문과 답변 데이터 생성
- 데이터 split (train, test)

### 3. fine tuning

```powershell
openai api fine_tunes.create -t "qa_train.jsonl" -v "qa_test.jsonl" --batch_size 16

'
Upload progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22.6k/22.6k [00:00<00:00, 12.2Mit/s]
Uploaded file from qa_train.jsonl: file-EVieU0wEjoq3YXIeZdIHXKye
Upload progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12.0k/12.0k [00:00<00:00, 15.4Mit/s]
Uploaded file from qa_test.jsonl: file-vBf46EJDVo4crCtIvOjaJFYv
Created fine-tune: ft-3N2woQxsz3OK4ddD8bKSEU3E
Streaming events until fine-tuning is complete...

(Ctrl-C will interrupt the stream, but not cancel the fine-tune)
[2023-06-13 15:15:14] Created fine-tune: ft-3N2woQxsz3OK4ddD8bKSEU3E
[2023-06-13 15:16:49] Fine-tune costs $0.22
[2023-06-13 15:16:50] Fine-tune enqueued. Queue number: 0
[2023-06-13 15:26:52] Fine-tune started
[2023-06-13 15:28:02] Completed epoch 1/4
[2023-06-13 15:28:03] Completed epoch 2/4
[2023-06-13 15:28:06] Completed epoch 3/4
[2023-06-13 15:28:07] Completed epoch 4/4

Job complete! Status: succeeded 🎉
Try out your fine-tuned model:

openai api completions.create -m curie:ft-personal-2023-06-13-06-28-34 -p <YOUR_PROMPT>
'

# 이어서 학습
# openai api fine_tunes.follow -i ft-3N2woQxsz3OK4ddD8bKSEU3E
```

### 4. 실제 테스트

```python
ft_qa = "curie:ft-personal-2023-06-13-06-28-34"

def apply_ft_qa_answer(question, answering_model):
    """
    Apply the fine tuned discriminator to a question
    """
    prompt = f"Question: {question}\nAnswer:"
    result = openai.Completion.create(model=answering_model, prompt=prompt, max_tokens=200, temperature=0, top_p=1, n=1)#, stop=['.','\n'])
    return result['choices'][0]['text']

apply_ft_qa_answer('가장 최근에 업데이트된 서비스는?', ft_qa)
```

### 5. 평가 및 정리

- 실제로 사용 할 수 있을 정도의 성능이 나오질 않음
- 일관된 데이터를 출력하고 있지 않음
- 사용 데이터가 적어 완전히 판단 할 수는 없지만 **어느정도 한계가 있는 듯!**
- 또한 한번 학습 시 많은 금액이 필요 (위의 데이터로 학습 한번 시 0.22$ cost 발생)
- **Embedding을 이용한 방법 연구 필요**

# plus

- gpt를 이용한 데이터베이스 시각화
  -> [Kanaries AI enhanced data exploration](https://kanaries.net/home)
