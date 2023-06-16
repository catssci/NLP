# prompt engineering method summary

# reference

[Best practices for prompt engineering with OpenAI API | OpenAI Help Center](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)

# 7가지 tips

## 1. propmt 시작 부분에 instruction과 context를 구분하여 입력

```markdown
Summarize the text below as a bullet point list of the most important points.

Text: """
{text input here}
"""
```

- 구분자는 `"""` or `###` 사용

## 2. 원하는 결과를 자세하게 서술

```markdown
Write a short inspiring poem about OpenAI, focusing on the recent DALL-E product launch (DALL-E is a text to image ML model) in the style of a {famous poet}
```

- outcome, length, format, style, etc…

## 3. 원하는 결과에 대한 예제를 서술

```markdown
Extract the important entities mentioned in the text below. First extract all company names, then extract all people names, then extract specific topics which fit the content and finally extract general overarching themes

Desired format:
Company names: <comma_separated_list_of_company_names>
People names: -||-
Specific topics: -||-
General themes: -||-

Text: {text}
```

- 원하는 결과에 대한 format과 예제를 서술해주자!

## 4. zero-shot, few-show, 둘 다 작동 X 시 fine-tune 이용하여 prompt를 시작

### Zero-shot

```markdown
Extract keywords from the below text.

Text: {text}

Keywords:
```

### Few-show

```markdown
Extract keywords from the corresponding texts below.

Text 1: Stripe provides APIs that web developers can use to integrate payment processing into their websites and mobile applications.
Keywords 1: Stripe, payment processing, APIs, web developers, websites, mobile applications
##
Text 2: OpenAI has trained cutting-edge language models that are very good at understanding and generating text. Our API provides access to these models and can be used to solve virtually any task that involves processing language.
Keywords 2: OpenAI, language models, text processing, API.
##
Text 3: {text}
Keywords 3:
```

## 5. fluffy, imprecise한 설명을 줄이기

```markdown
Use a 3 to 5 sentence paragraph to describe this product.
```

 `fairly short` `a few sentences only` `not too much more`

## 6. 하지말아야 할 것만 말하는 대신, 해야 할 것을 명령

```markdown
The following is a conversation between an Agent and a Customer. **The agent will attempt to diagnose the problem and suggest a solution, whilst refraining from asking any questions related to PII.** **Instead of asking for PII, such as username or password, refer the user to the help article www.samplewebsite.com/help/faq**

Customer: I can’t log in to my account.
Agent:
```

## 7. Code Generation: “Leading Words” 사용
```markdown
# Write a simple python function that
# 1. Ask me for a number in mile
# 2. It converts miles to kilometers
 
**import**
```
