# %%
import os
import cv2
import json
import base64
import numpy as np
import requests
import pandas as pd
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "../Images/raw_images")
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USDA_API_KEY = os.getenv("USDA_API_KEY")

# ------------------------------
# DIP IMAGE PREPROCESSING
# ------------------------------

def preprocess_for_gemini(image_path):
    """High-quality enhancement without distorting the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")

    # Resize to model-friendly size
    img = cv2.resize(img, (768, 768))

    # ---- White Balance ----
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    # ---- Histogram Equalization on Y channel ----
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # ---- Gamma correction (light adjustment) ----
    gamma = 1.1
    img = np.power(img / 255.0, gamma)
    img = (img * 255).astype("uint8")

    return img

# ------------------------------
# GEMINI INGREDIENT + GRAMS RECOGNITION
# ------------------------------

def identify_food_with_gemini(image_path):
    """
    Use Gemini Vision to detect ingredients + estimated grams.
    Unified preprocessing + strict JSON extraction + manual fallback.
    """
    try:
        processed = preprocess_for_gemini(image_path)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed for {image_path}: {e}")
        processed = cv2.imread(image_path)

    # encode to base64
    _, buffer = cv2.imencode('.jpg', processed)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    prompt = """
    Identify the ingredients and approximate weight (grams) of each item in this meal image.

    Return ONLY a JSON array, like:
    [
      {"ingredient": "rice", "grams": 150},
      {"ingredient": "chicken", "grams": 80}
    ]

    NO explanation.
    NO markdown.
    NO text outside JSON.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0
    )

    # call LLM
    try:
        response = llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
            ])
        ])
        text = response.content.strip()
    except Exception as e:
        print("\n[Gemini Error] →", image_path, e)
        text = ""

    # ---------------------------------------
    # JSON PARSE
    # ---------------------------------------
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) > 0:
            print(f"Gemini recognized: {os.path.basename(image_path)}")
            return parsed
        else:
            raise ValueError("Invalid JSON")
    except:
        print(f"\nGemini failed — manual input required for {os.path.basename(image_path)}")

    # ---------------------------------------
    # MANUAL INPUT (no loop, returns ONCE)
    # ---------------------------------------
    print(f"\n Manual entry for {os.path.basename(image_path)}")
    manual_list = []

    while True:
        name = input("Ingredient name (Enter to finish): ").strip()
        if name == "":
            break

        grams = input(f"Weight (g) for {name}: ").strip()
        try:
            grams = int(float(grams))
        except:
            grams = None

        manual_list.append({
            "ingredient": name,
            "grams": grams
        })

    print("Manual entry saved:", manual_list)
    return manual_list   # THIS RETURN ENSURES NO LOOP

# -----------------------------
# Ingredient normalization dict
# -----------------------------
INGREDIENT_NORMALIZATION = {
    "green onions": "onions, spring or scallions",
    "spring onions": "onions, spring or scallions",
    "scallions": "onions, spring or scallions",

    "bok choy": "cabbage, chinese (pak-choi)",
    "pak choi": "cabbage, chinese (pak-choi)",
    "pok choi": "cabbage, chinese (pak-choi)",

    "chicken breast": "chicken breast",
    "chicken thigh": "chicken, dark meat",
    "fried chicken": "fried chicken",

    "rice": "cooked rice",
    "white rice": "cooked white rice",
    "brown rice": "cooked brown rice",

    "noodles": "cooked noodles",
    "udon": "udon noodles",

    "lettuce": "lettuce",
    "romaine lettuce": "romaine lettuce",

    "broccoli": "broccoli",
    "carrots": "carrots",
}


def normalize_ingredient(name: str) -> str:
    if not isinstance(name, str):
        return ""
    key = name.strip().lower()
    if key in INGREDIENT_NORMALIZATION:
        return INGREDIENT_NORMALIZATION[key]
    return key


# ------------------------------
# MANUAL MULTI-INGREDIENT INPUT
# ------------------------------

def manual_input(image_path):
    """
    Manual fallback for entering multiple ingredients + grams.
    """
    print(f"\n Manual entry for {os.path.basename(image_path)}")

    items = []
    while True:
        name = input("Ingredient name (Enter to finish): ").strip()
        if name == "":
            break

        grams = input(f"Weight (g) for {name}: ").strip()
        try:
            grams = float(grams)
        except:
            print("Invalid number. Try again.")
            continue

        items.append({"ingredient": name, "grams": grams})

    print(f"Manual result: {items}")
    return items


def usda_search(query: str):
    if not USDA_API_KEY:
        print("USDA_API_KEY missing, cannot query USDA.")
        return None

    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "api_key": USDA_API_KEY,
        "query": query,
        "pageSize": 1
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"USDA HTTP {resp.status_code} for '{query}'")
            return None

        data = resp.json()
        foods = data.get("foods", [])
        if not foods:
            return None

        food = foods[0]
        nutrients = food.get("foodNutrients", [])

        kcal_value = None
        for n in nutrients:
            if n.get("nutrientNumber") == "208" or n.get("nutrientName", "").lower().startswith("energy"):
                kcal_value = n.get("value")
                break

        if kcal_value is None:
            return None

        return {"kcal_per_100g": float(kcal_value)}

    except Exception as e:
        print(f"USDA query error for '{query}': {e}")
        return None

# ------------------------------
# CALORIE CALCULATION
# ------------------------------

def compute_kcal(ingredient_list):
    
    total_kcal = 0.0
    detail_list = []

    for item in ingredient_list:
        name = item.get("ingredient", "")
        grams = item.get("grams", 0) or 0

        norm_name = normalize_ingredient(name)

        food_data = usda_search(norm_name)

        if not food_data:
            print(f"USDA failed for {name}. Using default = 0 kcal")
            kcal = 0.0
        else:
            kcal_per_100g = food_data.get("kcal_per_100g", 0.0)
            kcal = (kcal_per_100g / 100.0) * grams

        total_kcal += kcal

        detail_list.append({
            "ingredient": name,
            "normalized": norm_name,
            "grams": grams,
            "kcal": kcal
        })

    return total_kcal, detail_list

# %%



