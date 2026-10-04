from ddgs import DDGS
import os
import json
import requests

from datetime import date
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市未来几天的天气。适合温度、降水、穿衣等普通天气问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如北京、上海、大连"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索最新网络信息，用于天气预警、台风动态、航班延误等实时问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要搜索的完整问题"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def get_location(city):
    """把城市名称转换成经纬度"""

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "count": 1,
        "language": "zh",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if not data.get("results"):
        raise ValueError(f"找不到城市：{city}")

    location = data["results"][0]

    return {
        "name": location["name"],
        "latitude": location["latitude"],
        "longitude": location["longitude"]
    }


def get_weather(city):
    """查询指定城市未来天气"""
    

    location = get_location(city)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max"
        ],
        "timezone": "auto",
        "forecast_days": 3
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    return {
        "city": location["name"],
        "daily": data["daily"]
    }

def web_search(query):
    """搜索最新网络信息"""
    return DDGS().text(query, max_results=5)

def run_agent(question, history):
    messages = [
    {
        "role": "system",
        "content": (
            f"你是一个天气助手。今天是 {date.today().isoformat()}。"
            "结合之前的对话理解用户的省略和指代。"
            "需要天气信息时调用 get_weather 工具。"
            "普通天气、温度、降水、穿衣问题优先使用 get_weather。"
            "天气预警、台风动态、航班延误等实时信息使用 web_search。"
        )
    }
] + history + [
    {"role": "user", "content": question}
]
    """最基本的 Agent 执行循环"""

    for _ in range(3):

        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=tools,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )

        message = response.choices[0].message

        # 把模型的回复加入聊天历史
        messages.append(message)

        # 没有调用工具，说明模型已经有最终答案
        if not message.tool_calls:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": message.content})
            history[:] = history[-20:]
            return message.content
        

        # 模型要求调用工具
        for tool_call in message.tool_calls:

            arguments = json.loads(tool_call.function.arguments)

            if tool_call.function.name == "get_weather":
                city = arguments["city"]
                print(f"[Agent] 正在查询 {city} 的天气...")
                result = get_weather(city)

            elif tool_call.function.name == "web_search":
                query = arguments["query"]
                print(f"[Agent] 正在搜索网络：{query}")
                result = web_search(query)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                }
            )

            
    return "Agent 执行次数过多，请重新提问。"

def ask_llm(question):
    """向 DeepSeek 提问"""

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "system",
                "content": "你是一个简洁、友好的中文助手。"
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":

    history = []

    print("Weather Agent 已启动")
    print("输入 exit 可以退出")

    while True:

        question = input("\n你：").strip()

        if question.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break

        try:
            answer = run_agent(question, history)
            print(f"Agent：{answer}")

        except Exception as e:
            print(f"发生错误：{e}")
