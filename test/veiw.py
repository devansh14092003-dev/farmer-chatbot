"""
test_and_save.py
-----------------
app.py se independent — sirf embed_model aur LLM use karta hai.
Results test_results.txt mein save karta hai.

Usage:
    python test_and_save.py
"""

import os
import numpy as np
import pickle
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

load_dotenv()

# ============================================================
# MODELS LOAD KARO
# ============================================================
print("Loading embed model...")
embed_model = SentenceTransformer("BAAI/bge-m3")

print("Loading KCC cache...")
with open("kcc_questions.pkl", "rb") as f:
    kcc_questions = pickle.load(f)
with open("kcc_answers.pkl", "rb") as f:
    kcc_answers = pickle.load(f)
kcc_embeddings = np.load("kcc_embeddings.npy")
print(f"KCC loaded: {len(kcc_questions)} entries")

client = InferenceClient(
    provider="featherless-ai",
    api_key=os.getenv("HF_API_KEY")
)

THRESHOLD = 0.75

# ============================================================
# ANSWER FUNCTION
# ============================================================
def answer_question(query):
    query_emb    = embed_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, kcc_embeddings)[0]
    best_idx     = int(np.argmax(similarities))
    best_score   = float(similarities[best_idx])

    if best_score >= THRESHOLD:
        return "KCC", best_score, kcc_questions[best_idx], kcc_answers[best_idx]

    # LLM generation
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": query}],
        max_tokens=900
    )
    return "LLM", best_score, "N/A", response.choices[0].message.content

# ============================================================
# TEST QUESTIONS
# ============================================================
test_questions = [
    "When is the best time to sow sugarcane?",
    "How do I control pests in my coriander crop?",
    "How do I control blast in tomatoes?",
    "What is the right time to plant sugarcane?",
    "How to remove insects from coriander?",
    "How to treat tomato blast disease?",
    "What is the best intercropping method for wheat and mustard?",
    "How to increase soil fertility organically?",
    "What are signs of nitrogen deficiency in rice?",
    "गन्ना बोने का सबसे अच्छा समय कब है?",
    "मिर्च को सूखने से कैसे बचाएं?",
    "गेहूं में खाद कब डालें?",
]

# ============================================================
# RUN TESTS
# ============================================================
print("\nRunning tests...")
results = []

for i, question in enumerate(test_questions, 1):
    print(f"[{i}/{len(test_questions)}] {question[:60]}...")
    source, score, matched_q, answer = answer_question(question)
    results.append({
        "num": i, "source": source, "score": score,
        "question": question, "matched_q": matched_q, "answer": answer
    })

# ============================================================
# SAVE test_results.txt
# ============================================================
kcc_count = sum(1 for r in results if r['source'] == 'KCC')
llm_count = len(results) - kcc_count

with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 75 + "\n")
    f.write("FARMER CHATBOT — TEST RESULTS\n")
    f.write(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Qs   : {len(results)}\n")
    f.write(f"KCC        : {kcc_count} | LLM: {llm_count}\n")
    f.write("=" * 75 + "\n\n")

    for r in results:
        f.write(f"Q{r['num']}. [{r['source']}] Score: {r['score']:.3f}\n")
        f.write(f"Question : {r['question']}\n")
        if r['source'] == "KCC":
            f.write(f"Matched  : {r['matched_q']}\n")
        f.write(f"Answer   : {r['answer']}\n")
        f.write("-" * 75 + "\n\n")

print(f"\n✅ test_results.txt saved! (KCC: {kcc_count}, LLM: {llm_count})")