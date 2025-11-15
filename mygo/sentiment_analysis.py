import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
# 初始化 Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def analyze_tone(text: str) -> dict:
    """
    使用 Gemini 分析對話文字的語氣與情緒。
    回傳：
    {
        "emotion": "開心 / 生氣 / 傲嬌 / 關心 / 不耐煩 / 冷淡 / 曖昧 / ...",
        "tone": "輕鬆、正式、敷衍、諷刺、撒嬌、崩潰、...",
        "intent": "詢問、拒絕、關心、試探、暗示、...",
        "confidence": 0.0~1.0
    }
    """

    prompt = f"""
你是一個專門分析聊天語氣的語言學家。

請分析以下訊息的「情緒、語氣、意圖」，以中文回答，並以 JSON 格式（不加註解）輸出：

訊息內容：
{text}

JSON 結構：
{{
  "emotion": "",
  "tone": "",
  "intent": "",
  "confidence": 0.0
}}
"""

    response = model.generate_content(prompt)
    return response.text
