from ollama import chat
from ollama import ChatResponse
import json
files = ["marker_tracker.py", "follow_aruco.py", "record_episode.py"]
overall_code = ""
for file in files:
    with open(file, 'r') as file:
        overall_code += file.read()


print(overall_code)


msg = [     {'role': 'system', 'content': '你是一个专业的机器人工程师，擅长回答对代码里的函数进行总结.在日常工作中，同一个项目文件下可能会有多个意义相近的函数，这些函数是可以合并的'},
            {
                'role': 'user',
                'content': '''这段代码是什么意思：''' + overall_code
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