import json
from sentiment_analysis import analyze_tone
from get_mygo_pic import find_images_by_text  # 你前面寫的模糊搜尋功能

# 讀 mapping JSON
with open("mapping_mygo.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)

def recommend_mygo_image(text):
    # Step 1：分析語氣
    tone_json = analyze_tone(text)
    tone_data = json.loads(tone_json)

    emotion = tone_data.get("emotion", "")
    tone = tone_data.get("tone", "")
    intent = tone_data.get("intent", "")

    keywords = []

    # Step 2：找 mapping
    for key in [emotion, tone, intent]:
        if key in mapping:
            keywords.extend(mapping[key])

    if not keywords:
        keywords = ["笑"]  # fallback 預設用可愛笑圖

    # Step 3：用模糊搜尋找 MyGO 圖片
    results = []
    for kw in keywords:
        images = find_images_by_text(kw, download=False, max_results=3)
        if images:
            results.extend(images)

    # 如果還是沒有
    if not results:
        return "找不到適合的表情包QQ"

    # 回傳前三個最好
    return results[:3]
