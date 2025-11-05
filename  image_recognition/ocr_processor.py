# image_recognition/ocr_processor.py
"""
OCR 模組 - 用於將聊天截圖轉成文字資料
支援中英混合辨識，包含基本前處理與錯誤處理
"""

import pytesseract
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
import os


def preprocess_image(image_path: str) -> np.ndarray:
    """
    讀取圖片並進行前處理，提升 OCR 準確率。
    包含：
      - 灰階化
      - 高斯模糊
      - 自適應閾值二值化
      - 邊緣去噪
    """
    # 使用 OpenCV 讀取圖片
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"找不到圖片檔案：{image_path}")

    # 轉灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 去雜訊
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 自適應閾值二值化
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    # 去除小雜點（開運算）
    kernel = np.ones((1, 1), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return clean


def extract_text(image_path: str, lang: str = "chi_sim+eng") -> str:
    """
    使用 Tesseract OCR 進行圖片文字辨識。
    預設語言為中英文混合。
    """
    try:
        preprocessed = preprocess_image(image_path)
        text = pytesseract.image_to_string(preprocessed, lang=lang)

        # 清理換行與多餘空白
        cleaned = " ".join(text.split())
        return cleaned

    except Exception as e:
        print(f"[ERROR] OCR 辨識失敗：{e}")
        return ""


def extract_chat_lines(image_path: str) -> list:
    """
    將 OCR 文字切割成一行一行的對話形式
    （方便後續對話分析模組處理）
    """
    text = extract_text(image_path)
    if not text:
        return []

    # 按句號、問號、換行符拆解
    lines = [
        line.strip()
        for line in text.replace("。", "\n").replace("?", "?\n").split("\n")
        if len(line.strip()) > 0
    ]
    return lines


if __name__ == "__main__":
    # 測試範例
    test_img = "example_chat.png"  # 你可以換成你的聊天截圖
    if os.path.exists(test_img):
        print("📷 開始 OCR 辨識...")
        lines = extract_chat_lines(test_img)
        print("\n辨識結果：")
        for i, line in enumerate(lines, 1):
            print(f"{i}. {line}")
    else:
        print("⚠️ 找不到測試圖片 example_chat.png")
