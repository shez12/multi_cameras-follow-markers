from ollama import chat
from ollama import ChatResponse


from ollama import chat
from ollama import ChatResponse

msg = [
            {
                'role': 'system',
                'content': '你是一个bade专属的智能聊天助手，帮助bade自动回复回答，并提供相关信息；你的回答语气是 正式、亲切、幽默、简短'
            },
        
            {
                'role': 'user',
                'content': '1+1=?'
            }

        ]


response: ChatResponse = chat(model='deepseek-r1:8b', messages=msg)

# Remove the thinking process from the response
content = response['message']['content']
if '<think>' in content and '</think>' in content:
    start = content.find('</think>') + len('</think>')
    content = content[start:].strip()

print(content)

# 或者直接访问响应对象的字段
# print(response.message.content)