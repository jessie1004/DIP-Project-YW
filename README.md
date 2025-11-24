# Smartphone-Based Food Image Recognition and Dietary Suggestion through AI Agent

---

## 1. Introduction  
This repository provides the source codes and raw datasets associated with the paper Smartphone-Based Food Image Recognition and Dietary Suggestion through AI Agent. It presents an end-to-end pipeline for food image–based calorie estimation and personalized diet analysis. The project integrates Digital Image Processing with Large Language Models (Gemini) to produce interpretable nutrition outputs and personalized dietary recommendations.

Nowadays, the rise of diet-related health issues such as obesity, diabetes, and cardiovascular disease indicates the importance of accurate dietary monitoring. Even though there are many existed Food Image Recognition Application existed, they still need many manual steps from users and they don't provide a personlized summary and suggestion for the user. This paper aims to automate the whole process and provide each user with a personalized dietary report.

---

## 2. Project Objectives  
- Perform **image preprocessing** (resize, white balance, Gamma Correction) for food images.  
- Use **Gemini Vision** to recognize food items, extract ingredients, and estimate portion sizes.  
- Query the **USDA FoodData Central API** to retrieve calories per 100g of each ingredient.  
- Calculate total calories for every meal based on estimated portion sizes.  
- Aggregate 7-day, 3-meals-per-day caloric intake together with user basic information.  
- Apply **prompt engineering** to generate personalized diet reports using Gemini.

---

## 3. Data
The raw data can be divided into two parts. The first part includes users' demographic information (Age, Gender, Weight, Height, BMI), their lifettyle-related data (Step Count Distance Travel (Km), Blood Pressure Heart Rate (BPM), Blood Oxygen Level, Sleep Duration (minutes), Screen Time (minute), Earphone Time (minute)), and their meals for 7 days. The second part contains 20 food images correponding to the food label in the first dataset. In this paper, it combines two parts together to simulate users' dietary structure.

The users' data is from Kaggle: https://www.kaggle.com/datasets/mdimammahdi/smart-healthcare-dailylife-dataset-wearable-device
The images are from Huggingface: https://huggingface.co/datasets/Codatta/MM-Food-100K


## 4. Methods  

### 4.1 Image Preprocessing  
- Load images from `data/raw_images/`  
- Resize and preprocess images
- Save outputs to `data/processed_images/`

### 4.2 Food & Portion Recognition (Gemini)  
- Processed images are sent to Gemini with a structured prompt  
- Extract:  
  - Food names  
  - Ingredient list  
  - Estimated portion sizes (gram)  
- Save outputs to `data/`

### 4.3 Nutrition Lookup & Calorie Computation  
- For each ingredient:  
  - Query USDA FoodData Central  
  - Retrieve kcal per 100 g  
- Compute total calories:  
  `kcal = (portion_in_grams / 100) × kcal_per_100g`  
- Save daily total calorie CSV to `data/user_reports/`

### 4.4 Personalized Diet Report
- Construct detailed prompt:
  - Role-based prompting
  - Few-shot prompting
  - User profile  
  - Weekly calorie summary  
  - Trend patterns  
  - Meal-level results  
- Gemini generates:  
  - Personalized diet analysis  
  - Diet recommendations  
  - Suggested improvements  
- Save final report to `weekly_ai_reports/`

---

## 5. Repository Structure  

```text
Project/
│
├── food_tools/
│   ├── utils_00.py                           # Utility functions: DIP preprocessing, SAM segmentation, Gemini + USDA
│   ├── dataset_preprocessing_01.py           # Map meal codes → real image paths
│   ├── food_identification_02.ipynb          # Image preprocessing + Gemini ingredients extraction
│   ├── nutrition_estimation_03.ipynb         # Compute calories using USDA (ingredient-level)
│   └── langchain_agent_analysis_04.ipynb     # Weekly AI analysis using LangChain agent
│
├── Data/
│   ├── Smart Healthcare - Daily Lifestyle Dataset.csv
│   ├── linked_dataset.csv
│   ├── image_ingredients.csv
│   └── user_reports/                         # Daily calorie results per user
│
├── Images/
│   ├── raw_images                            # Raw meal images (001.jpg, 002.jpg...)
│   ├── processed_images                      # Preprocessed images
│                                            
├── weekly_ai_reports/                        # Final weekly reports for each user
│
└── README.md

```

---

## 6. Running the Pipeline  
This project is executed through five Jupyter notebooks located in `food_tools/`

You need:
- <GOOGLE_API_KEY>: Gemini Vision + Gemini Text
- <USDA_API_KEY>: Calorie Lookup

### Step 1 
```bash
python food_tools/utils_00.py
```

### Step 2 — Image Preprocessing 
```bash
python food_tools/data preprocessing_01.py
```

### Step 3 — Food Identification
```bash
python food_tools/food_identification_02.py
```

### Step 4 — Nutrition Estimation  
```bash
python food_tools/nutrition_estimation_03.py
```

### Step 5 — Diet report generation  
```bash
python food_tools/langchain_agent_analysis_04.py
```

Outputs are stored in `weekly_ai_reports/`.

---



