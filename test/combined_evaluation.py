"""
evaluate.py
------------
Farmer Chatbot ka complete evaluation script.

Kya measure karta hai:
1. Retrieval Metrics  — KCC match accuracy, similarity scores, threshold analysis
2. Generation Metrics — BLEU, ROUGE-1, ROUGE-L, Semantic Similarity (LLM answers pe)

Install karo pehle:
    pip install rouge-score nltk sentence-transformers scikit-learn

Usage:
    python evaluate.py

Output:
    evaluation_report.txt
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PACKAGES
# ============================================================
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rs
import nltk
nltk.download('punkt', quiet=True)

# ============================================================
# LOAD MODELS & DATA
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

# English KCC dataset (ground truth)
kcc_df = pd.read_csv("KCCNewEnglishDataset.csv")

# LLM Client
client = InferenceClient(
    provider="featherless-ai",
    api_key=os.getenv("HF_API_KEY")
)

THRESHOLD = 0.75
EVAL_SAMPLE = 50  # kitne questions evaluate karne hain

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_kcc_answer(query):
    """KCC se best match answer do"""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    sims  = cosine_similarity(q_emb, kcc_embeddings)[0]
    idx   = int(np.argmax(sims))
    return kcc_questions[idx], kcc_answers[idx], float(sims[idx])

def get_llm_answer(query):
    """LLM se answer generate karo"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": query}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def bleu_score(reference, hypothesis):
    """BLEU score calculate karo"""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoother = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)

def rouge_scores(reference, hypothesis):
    """ROUGE-1 aur ROUGE-L calculate karo"""
    scorer = rs.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

def semantic_similarity(text1, text2):
    """Semantic similarity calculate karo"""
    embs = embed_model.encode([text1, text2], convert_to_numpy=True)
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])

# ============================================================
# PART 1: RETRIEVAL EVALUATION
# ============================================================
print("\n" + "="*60)
print("PART 1: RETRIEVAL EVALUATION")
print("="*60)

sample_df = kcc_df.sample(n=EVAL_SAMPLE, random_state=42)
retrieval_results = []

for _, row in sample_df.iterrows():
    query      = row['Query']
    true_ans   = row['KCC Ans']
    matched_q, matched_ans, score = get_kcc_answer(query)
    
    # Semantic similarity between true answer and retrieved answer
    sem_sim = semantic_similarity(true_ans, matched_ans)
    
    retrieval_results.append({
        'query':       query,
        'true_ans':    true_ans,
        'matched_q':   matched_q,
        'matched_ans': matched_ans,
        'cos_score':   score,
        'sem_sim':     sem_sim,
        'kcc_used':    score >= THRESHOLD
    })
    print(f"  Score: {score:.3f} | SemSim: {sem_sim:.3f} | {query[:50]}...")

ret_df = pd.DataFrame(retrieval_results)

# Stats
avg_cos    = ret_df['cos_score'].mean()
avg_sem    = ret_df['sem_sim'].mean()
kcc_pct    = ret_df['kcc_used'].sum() / len(ret_df) * 100
high_sim   = (ret_df['sem_sim'] >= 0.75).sum() / len(ret_df) * 100

print(f"\nRetrieval Stats:")
print(f"  Avg Cosine Similarity : {avg_cos:.3f}")
print(f"  Avg Semantic Similarity: {avg_sem:.3f}")
print(f"  KCC Used (>={THRESHOLD}): {kcc_pct:.1f}%")
print(f"  High Sem Sim (>=0.75)  : {high_sim:.1f}%")

# ============================================================
# PART 2: GENERATION EVALUATION
# ============================================================
print("\n" + "="*60)
print("PART 2: GENERATION EVALUATION (LLM vs KCC ground truth)")
print("="*60)

# 30 questions LLM ko do aur KCC answer se compare karo
gen_sample = kcc_df.sample(n=30, random_state=99)
gen_results = []

for i, (_, row) in enumerate(gen_sample.iterrows(), 1):
    query    = row['Query']
    true_ans = row['KCC Ans']
    
    print(f"[{i}/30] Generating: {query[:50]}...")
    llm_ans = get_llm_answer(query)
    
    if llm_ans.startswith("ERROR"):
        print(f"  ❌ Skipping due to error")
        continue

    b_score          = bleu_score(true_ans, llm_ans)
    r1_score, rl_score = rouge_scores(true_ans, llm_ans)
    sem              = semantic_similarity(true_ans, llm_ans)

    gen_results.append({
        'query':    query,
        'true_ans': true_ans,
        'llm_ans':  llm_ans,
        'bleu':     b_score,
        'rouge1':   r1_score,
        'rougeL':   rl_score,
        'sem_sim':  sem,
    })
    print(f"  BLEU: {b_score:.3f} | R1: {r1_score:.3f} | RL: {rl_score:.3f} | Sem: {sem:.3f}")

gen_df = pd.DataFrame(gen_results)

avg_bleu   = gen_df['bleu'].mean()
avg_r1     = gen_df['rouge1'].mean()
avg_rl     = gen_df['rougeL'].mean()
avg_sem_g  = gen_df['sem_sim'].mean()

print(f"\nGeneration Stats:")
print(f"  Avg BLEU Score       : {avg_bleu:.3f}")
print(f"  Avg ROUGE-1          : {avg_r1:.3f}")
print(f"  Avg ROUGE-L          : {avg_rl:.3f}")
print(f"  Avg Semantic Sim     : {avg_sem_g:.3f}")

# ============================================================
# SAVE REPORT
# ============================================================
with open("evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("FARMER CHATBOT — EVALUATION REPORT\n")
    f.write(f"Date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model       : BAAI/bge-m3 + Llama-3.1-8B-Instruct\n")
    f.write(f"KCC Entries : {len(kcc_questions)}\n")
    f.write(f"Threshold   : {THRESHOLD}\n")
    f.write("=" * 70 + "\n\n")

    f.write("─── PART 1: RETRIEVAL METRICS ───\n\n")
    f.write(f"Sample Size              : {EVAL_SAMPLE}\n")
    f.write(f"Avg Cosine Similarity    : {avg_cos:.4f}\n")
    f.write(f"Avg Semantic Similarity  : {avg_sem:.4f}\n")
    f.write(f"KCC Retrieved (>={THRESHOLD})   : {kcc_pct:.1f}%\n")
    f.write(f"High Sim Answers (>=0.75): {high_sim:.1f}%\n\n")

    f.write("─── PART 2: GENERATION METRICS ───\n\n")
    f.write(f"Sample Size              : {len(gen_df)}\n")
    f.write(f"Avg BLEU Score           : {avg_bleu:.4f}\n")
    f.write(f"Avg ROUGE-1              : {avg_r1:.4f}\n")
    f.write(f"Avg ROUGE-L              : {avg_rl:.4f}\n")
    f.write(f"Avg Semantic Similarity  : {avg_sem_g:.4f}\n\n")

    f.write("─── INTERPRETATION ───\n\n")
    f.write("Cosine/Semantic Sim: 0.75+ = Good retrieval\n")
    f.write("BLEU              : 0.3+ = Acceptable, 0.5+ = Good\n")
    f.write("ROUGE-1           : 0.4+ = Acceptable, 0.6+ = Good\n")
    f.write("ROUGE-L           : 0.3+ = Acceptable, 0.5+ = Good\n\n")

    f.write("─── DETAILED RETRIEVAL RESULTS ───\n\n")
    for r in retrieval_results:
        f.write(f"Q: {r['query']}\n")
        f.write(f"   Cos: {r['cos_score']:.3f} | Sem: {r['sem_sim']:.3f} | KCC: {r['kcc_used']}\n")
        f.write(f"   Matched: {r['matched_q'][:80]}\n\n")

    f.write("─── DETAILED GENERATION RESULTS ───\n\n")
    for r in gen_results:
        f.write(f"Q: {r['query']}\n")
        f.write(f"   BLEU: {r['bleu']:.3f} | R1: {r['rouge1']:.3f} | RL: {r['rougeL']:.3f} | Sem: {r['sem_sim']:.3f}\n")
        f.write(f"   True : {r['true_ans'][:100]}\n")
        f.write(f"   LLM  : {r['llm_ans'][:100]}\n\n")

print("\n✅ evaluation_report.txt saved!")