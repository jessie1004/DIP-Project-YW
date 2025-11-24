
from utils_00 import *
import pandas as pd
from tqdm import tqdm
import os
import json
import cv2
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../Data/linked_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "../Data/image_ingredients.csv")


PROCESSED_DIR = os.path.join(BASE_DIR, "../Images/processed_images")
os.makedirs(PROCESSED_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
records = []
processed = set()

print("Loaded dataset:", df.shape)

# ---- Gemini Vision free tier allows only 10 requests/min ----
REQUEST_INTERVAL = 7   # seconds, safe for 10 req/min


for idx, row in tqdm(df.iterrows(), total=len(df)):

    for meal_col in ["First Meal Path", "Second Meal Path", "Third Meal Path"]:

        img_path = row.get(meal_col)

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            continue

        if img_path in processed:
            continue

        processed.add(img_path)

        # --------------------------------------------
        # 1. preprocessing
        # --------------------------------------------
        try:
            enhanced = preprocess_for_gemini(img_path)
        except Exception as e:
            print("Preprocessing failed:", img_path, e)
            continue

        # save processed
        save_path = os.path.join(PROCESSED_DIR, os.path.basename(img_path))
        cv2.imwrite(save_path, enhanced)

        # --------------------------------------------
        # 2. Gemini recognition (with rate limit control)
        # --------------------------------------------
        try:
            ing = identify_food_with_gemini(save_path)
        except Exception as e:
            print("Gemini failed:", img_path, e)
            ing = []

        # ---- RATE LIMIT CONTROL ----
        print(f"Waiting {REQUEST_INTERVAL}s before next Gemini call...")
        time.sleep(REQUEST_INTERVAL)

        # --------------------------------------------
        # 3. Save record
        # --------------------------------------------
        records.append({
            "image": os.path.basename(img_path),
            "raw_image_path": img_path,
            "processed_image_path": save_path,
            "ingredients_json": json.dumps(ing)
        })


df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
df_out.head()


# %%



