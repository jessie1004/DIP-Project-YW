from utils_00 import *
import os
import pandas as pd
import json
import numpy as np

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LINKED = os.path.join(BASE_DIR, "../Data/linked_dataset.csv")
USER_REPORT_DIR = os.path.join(BASE_DIR, "../Data/user_reports")

FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, "../weekly_ai_reports")
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

df_linked = pd.read_csv(DATA_LINKED)
print("Loaded linked dataset:", df_linked.shape)


def to_json_safe(obj) -> str:
    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    return json.dumps(obj, indent=2, default=_convert)


def load_weekly_kcal(user_id: str):
    path = os.path.join(USER_REPORT_DIR, f"{user_id}.csv")
    if not os.path.exists(path):
        return f"No weekly calorie report found for user {user_id}"
    df = pd.read_csv(path)
    return df.to_json(orient="records")

tools = [
    Tool(
        name="LoadWeeklyKcal",
        func=load_weekly_kcal,
        description="Load user's USDA-based 7-day calories (breakfast, lunch, dinner). Returns JSON string."
    )
]

# --------------------------
# LLM（Gemini）
# --------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.25,
)

# --------------------------
# Prompt
# --------------------------
SYSTEM_PROMPT = """
You are a senior nutrition scientist and AI health analyst.

Your job is to generate a 7-day diet analysis using:
- USDA ingredient-based calories
- Lifestyle metrics (sleep, steps, heart rate, screen time)
- Demographics (age, gender, BMI, height, weight)
- Meal structure (breakfast/lunch/dinner)

Your output MUST use the following Markdown structure:

#### 1. Weekly Calorie Overview
#### 2. Eating Pattern Insights
#### 3. Ingredient-Based Evaluation
#### 4. Lifestyle Interaction Analysis
#### 5. Personalized Weekly Recommendations

Be concise, scientific, and user-friendly.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# --------------------------
# Create ToolCalling Agent
# --------------------------
agent_core = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=PROMPT
)

agent = AgentExecutor(
    agent=agent_core,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# --------------------------
# Weekly Summary
# --------------------------
def generate_weekly_report(user_id: str) -> str:
    user_rows = df_linked[df_linked["ID"].astype(str) == str(user_id)].sort_values("Day")
    if user_rows.empty:
        return f"No rows found for user {user_id}"

    first = user_rows.iloc[0]

    demographics = {
        "ID": str(user_id),
        "Age": int(first["Age (years)"]),
        "Gender": str(first["Gender"]),
        "Height_m": float(first["Height (meter)"]),
        "Weight_kg": float(first["Weight (kg)"]),
        "BMI": float(first["BMI"]),
    }

    lifestyle = []
    for _, r in user_rows.iterrows():
        lifestyle.append({
            "Day": int(r["Day"]),
            "Steps": int(r["Step Count"]),
            "Sleep(min)": int(r["Sleep Duration (minutes)"]),
            "HeartRate": int(r["Heart Rate (BPM)"]),
            "ScreenTime(min)": int(r["Screen Time (minute)"]),
        })

    
    demographics_str = to_json_safe(demographics)
    lifestyle_str = to_json_safe(lifestyle)

    input_text = f"""
User Demographics (one person):
{demographics_str}

Lifestyle over 7 days:
{lifestyle_str}

You MUST first call the tool `LoadWeeklyKcal` with user_id="{user_id}"
to load the USDA-based calorie summary for 7 days (breakfast, lunch, dinner).

After you have the calorie data:
- Combine it with demographics and lifestyle info above.
- Then produce the structured weekly nutrition report following the required sections.
"""

    result = agent.invoke({"input": input_text})
    return result["output"]

# --------------------------
# weekly report
# --------------------------
for raw_id in df_linked["ID"].unique():
    user_id = str(raw_id)
    print(f"\n===== Generating weekly AI report for user {user_id} =====")
    summary = generate_weekly_report(user_id)

    save_path = os.path.join(FINAL_OUTPUT_DIR, f"{user_id}_weekly_report.txt")
    with open(save_path, "w") as f:
        f.write(summary)

    print(f"Saved → {save_path}")





