# %%
import pandas as pd
import os

# --------------------------
# Config (modify paths only if needed)
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../Data/Smart Healthcare - Daily Lifestyle Dataset (Wearable device).csv")
IMAGE_DIR = os.path.join(BASE_DIR, "../Images/raw_images")

OUTPUT_PATH = os.path.join(BASE_DIR, "../Data/linked_dataset.csv")

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded:", df.shape)


# ============================================================
# Helper: convert meal image code → full path
# ============================================================
def code_to_image_path(code):
    """
    Convert image code (e.g., '001') → '/path/.../001.jpg'
    Handles:
       - numeric values (int or float): 1 → '001'
       - string without extension: '12' → '012.jpg'
       - string with extension → used directly
    """
    if pd.isna(code):
        return None

    # Convert to string
    code = str(code).strip()

    # If already like "003.jpg"
    if code.lower().endswith(".jpg"):
        file_name = code
    else:
        # Remove decimals like '1.0'
        if code.replace('.', '').isdigit():
            n = int(float(code))  # 1.0 → 1
            file_name = f"{n:03d}.jpg"
        else:
            # Unexpected but fallback
            file_name = code + ".jpg"

    full_path = os.path.join(IMAGE_DIR, file_name)
    return full_path if os.path.exists(full_path) else None


# ============================================================
# Apply mapping to the 3 image columns
# ============================================================
MEAL_COLUMNS = ["First Meal", "Second Meal", "Third Meal"]

for col in MEAL_COLUMNS:
    new_col = col + " Path"
    df[new_col] = df[col].apply(code_to_image_path)


# ============================================================
# Report missing images
# ============================================================
missing = df[[c+" Path" for c in MEAL_COLUMNS]].isna().sum()
print("\nMissing image counts:")
print(missing)


# ============================================================
# Save final linked dataset
# ============================================================
df.to_csv(OUTPUT_PATH, index=False)
print("\nSaved linked dataset: ", OUTPUT_PATH)

df.head()


# %%


