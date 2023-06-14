# 1. token 개수 세기

- web

[OpenAI API](https://platform.openai.com/tokenizer)

- code

```python
import tiktoken  # for counting tokens

GPT_MODEL = "gpt-3.5-turbo"

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

test = """Many words map to one token, but some don't: indivisible."""

print(num_tokens(test))
print(len(test))

encoding = tiktoken.encoding_for_model(GPT_MODEL)
print(encoding.encode(test))
```

- reference

https://github.com/openai/tiktoken
