"""
generate_answers.py
--------------------
Yeh script Crop_quest2__2_.xlsx ke saare questions pe
chatbot se answer generate karke CSV mein save karta hai.

RESUME FEATURE: Agar bich mein band ho jaaye toh dobara chalao
- pehle se answer ho chuke questions skip ho jayenge

Usage:
    python generate_answers.py

Output:
    crop_answers.csv
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app import answer_question

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE  = "Crop_quest2 (2).xlsx"
OUTPUT_FILE = "crop_answers.csv"
DELAY       = 2  # seconds between requests

# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_excel(INPUT_FILE)
    total = len(df)
    print(f"Total questions: {total}")

    # Resume: agar CSV pehle se hai toh already done questions skip karo
    if os.path.exists(OUTPUT_FILE):
        done_df = pd.read_csv(OUTPUT_FILE)
        done_questions = set(done_df["Question"].tolist())
        answers = done_df.to_dict("records")
        print(f"Resuming... {len(done_questions)} already done, {total - len(done_questions)} remaining")
    else:
        done_questions = set()
        answers = []

    for i, row in df.iterrows():
        question  = row["QueryText"]
        crop      = row["CropName"]
        crop_type = row.get("CropType", "")
        q_type    = row.get("QType", "")

        # Skip agar pehle se ho gaya
        if question in done_questions:
            print(f"[{i+1}/{total}] SKIP: {crop}: {question[:50]}...")
            continue

        print(f"[{i+1}/{total}] {crop}: {question[:60]}...")

        try:
            answer = answer_question(question)
        except Exception as e:
            answer = f"ERROR: {str(e)}"
            print(f"  ❌ Error: {e}")
            # Rate limit error pe thoda zyada wait karo
            if "429" in str(e) or "rate" in str(e).lower():
                print("  ⏳ Rate limit hit — waiting 30 seconds...")
                time.sleep(30)

        answers.append({
            "CropName":  crop,
            "CropType":  crop_type,
            "QType":     q_type,
            "Question":  question,
            "Answer":    answer,
        })

        # Har 10 questions pe save
        if len(answers) % 10 == 0:
            pd.DataFrame(answers).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
            print(f"  ✅ Progress saved: {len(answers)} done")

        time.sleep(DELAY)

    # Final save
    pd.DataFrame(answers).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ Done! CSV saved: {OUTPUT_FILE} ({len(answers)} rows)")


if __name__ == "__main__":
    main()