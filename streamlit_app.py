import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=100,     #トークン数の上限を設定
    temperature = 0.5,  # 0～2.0の値を指定
    messages=[
        {"role": "user", "content": "ChatGPTについて教えてください。"}
    ]
)
print(response.choices[0]["message"]["content"].strip())
print(response.usage["prompt_tokens"])
print(response.usage["completion_tokens"])
print(response.usage["total_tokens"])
