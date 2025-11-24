from utils_00 import *
import pandas as pd
import os
import json
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LINKED = os.path.join(BASE_DIR, "../Data/linked_dataset.csv")
DATA_ING = os.path.join(BASE_DIR, "../Data/image_ingredients.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "../Data/user_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_linked = pd.read_csv(DATA_LINKED)
df_ing = pd.read_csv(DATA_ING)

print("Loaded linked dataset:", df_linked.shape)
print("Loaded ingredients:", df_ing.shape)

# ---------------------------------------------------------
# Build image_path → ingredient_list mapping
# ---------------------------------------------------------
image_to_ing = {}

for _, row in df_ing.iterrows():

    raw_path = row["raw_image_path"]   # <— KEY FIX

    try:
        ings = json.loads(row["ingredients_json"])
    except:
        ings = []

    image_to_ing[raw_path] = ings


# ---------------------------------------------------------
# Loop through users
# ---------------------------------------------------------
for user_id in df_linked["ID"].unique():

    print(f"\n===== Processing user {user_id} =====")

    user_df = df_linked[df_linked["ID"] == user_id].sort_values("Day")
    daily_records = []

    for _, r in user_df.iterrows():

        day_record = {"Day": r["Day"]}

        meal_paths = {
            "Breakfast": r["First Meal Path"],
            "Lunch":     r["Second Meal Path"],
            "Dinner":    r["Third Meal Path"]
        }

        for meal_name, path in meal_paths.items():

            if isinstance(path, str) and path in image_to_ing:
                ingredients = image_to_ing[path]
                total_kcal, detail = compute_kcal(ingredients)
            else:
                ingredients = []
                total_kcal = 0
                detail = []

            day_record[f"{meal_name}_Ingredients"] = json.dumps(ingredients)
            day_record[f"{meal_name}_Kcal"] = total_kcal
            day_record[f"{meal_name}_Detail"] = json.dumps(detail)

        day_record["Daily_Total_Kcal"] = (
            day_record["Breakfast_Kcal"] +
            day_record["Lunch_Kcal"] +
            day_record["Dinner_Kcal"]
        )

        daily_records.append(day_record)

    out_df = pd.DataFrame(daily_records)
    save_path = os.path.join(OUTPUT_DIR, f"{user_id}.csv")
    out_df.to_csv(save_path, index=False)

    print(f"Saved → {save_path}")





