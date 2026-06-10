"""
test_answers.py
----------------
Test karta hai ki kaunsa question KCC retrieval se aaya
aur kaunsa LLM se generate hua.

Usage:
    python test_answers.py
"""

from dotenv import load_dotenv
load_dotenv()

from app import answer_question, embed_model, kcc_embeddings, kcc_questions, kcc_answers
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# TEST QUESTIONS
# Exact KCC match, similar KCC, aur naye questions mix kiye
# ============================================================
test_questions = [
    # --- Exact KCC match hone chahiye (score ~0.99) ---
    "When is the best time to sow sugarcane?",
    "How do I control pests in my coriander crop?",
    "How do I control blast in tomatoes?",

    # --- Similar but not exact (score ~0.75-0.90) ---
    "What is the right time to plant sugarcane?",
    "How to remove insects from coriander?",
    "How to treat tomato blast disease?",

    # --- Naye farming questions (LLM se aane chahiye) ---
    "What is the best intercropping method for wheat and mustard?",
    "How to increase soil fertility organically?",
    "What are signs of nitrogen deficiency in rice?",

    # --- Hindi questions ---
    "गन्ना बोने का सबसे अच्छा समय कब है?",
    "मिर्च को सूखने से कैसे बचाएं?",
    "गेहूं में खाद कब डालें?",
]

THRESHOLD = 0.75

print("=" * 75)
print(f"{'#':<3} {'Source':<10} {'Score':<7} Question")
print("=" * 75)

for i, question in enumerate(test_questions, 1):
    query_emb    = embed_model.encode([question], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, kcc_embeddings)[0]
    best_idx     = int(np.argmax(similarities))
    best_score   = float(similarities[best_idx])

    if best_score >= THRESHOLD:
        source = "✅ KCC"
        answer = kcc_answers[best_idx]
        matched_q = kcc_questions[best_idx]
    else:
        source = "🤖 LLM"
        answer = answer_question(question)
        matched_q = "N/A"

    print(f"\n{i:<3} {source:<10} {best_score:.3f}  {question}")
    if source == "✅ KCC":
        print(f"    Matched: {matched_q[:70]}")
    print(f"    Answer:  {answer[:150]}...")
    print("-" * 75)

print("\n✅ Test complete!")