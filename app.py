
import re
import os
import pickle
import numpy as np
import faiss
import pdfplumber
import nltk
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from rank_bm25 import BM25Okapi

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from huggingface_hub import InferenceClient

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
embed_model = SentenceTransformer(
    "BAAI/bge-m3"
)

# ============================================================
# KCC DIRECT RETRIEVAL SETUP
# ============================================================
KCC_EN_FILE    = "KCCNewEnglishDataset.csv"
KCC_HI_FILE    = "KCCNewHindiDataset.xlsx"
KCC_EMB_FILE   = "kcc_embeddings.npy"   # cached embeddings
KCC_QUES_FILE  = "kcc_questions.pkl"    # cached questions
KCC_ANS_FILE   = "kcc_answers.pkl"      # cached answers

def load_kcc_data():
    # English dataset
    en_df = pd.read_csv(KCC_EN_FILE)
    en_q  = en_df["Query"].tolist()
    en_a  = en_df["KCC Ans"].tolist()

    # Hindi dataset
    hi_df = pd.read_excel(KCC_HI_FILE)
    hi_q  = hi_df["सवाल"].tolist()
    hi_a  = hi_df["केसीसी उत्तर"].tolist()

    # Dono combine karo
    all_q = en_q + hi_q
    all_a = en_a + hi_a
    return all_q, all_a

if os.path.exists(KCC_EMB_FILE) and os.path.exists(KCC_QUES_FILE):
    # Cached embeddings load karo (fast)
    print("Loading KCC embeddings from cache...")
    kcc_embeddings = np.load(KCC_EMB_FILE)
    with open(KCC_QUES_FILE, "rb") as f:
        kcc_questions = pickle.load(f)
    with open(KCC_ANS_FILE, "rb") as f:
        kcc_answers = pickle.load(f)
    print(f"KCC cache loaded: {len(kcc_questions)} entries")
else:
    # Pehli baar — embeddings banao aur save karo
    print("Building KCC embeddings (first run, will be cached)...")
    kcc_questions, kcc_answers = load_kcc_data()
    kcc_embeddings = embed_model.encode(
        kcc_questions,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64
    )
    np.save(KCC_EMB_FILE, kcc_embeddings)
    with open(KCC_QUES_FILE, "wb") as f:
        pickle.dump(kcc_questions, f)
    with open(KCC_ANS_FILE, "wb") as f:
        pickle.dump(kcc_answers, f)
    print(f"KCC embeddings saved: {len(kcc_questions)} entries")

# ======================================================
# NLTK
# ======================================================

nltk.download("punkt")


# ======================================================
# CONFIG
# ======================================================

PDF_PATH = "BasicAgri_book.pdf"

SKIP_INITIAL_PAGES = 23

CHUNK_SIZE = 420

CHUNK_OVERLAP = 80


# ======================================================
# CLEAN TEXT
# ======================================================

def clean_text(text):

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(
        r'[^\w\s.,%/():\-]',
        '',
        text
    )

    return text.strip()


# ======================================================
# BETTER SEMANTIC CHUNKING
# ======================================================

def semantic_chunk(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = clean_text(text)

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if len(chunk.split()) > 80:

            chunks.append(chunk)

        start += (
            chunk_size - overlap
        )

    return chunks


# ======================================================
# CLEAN FACT CHUNKS
# ======================================================

def extract_fact_chunks(text):

    facts = []

    sentences = re.split(
        r'(?<=[.])\s+',
        text
    )

    useful_keywords = [

        "kg",
        "cm",
        "spacing",
        "yield",
        "dose",
        "fertilizer",
        "%",
        "days",
        "seed rate",
        "variety",
        "irrigation",
        "weed",
        "disease"

    ]

    for s in sentences:

        s = clean_text(s)

        if len(s.split()) < 8:
            continue

        if len(s.split()) > 45:
            continue

        if any(
            c.isdigit()
            for c in s
        ):

            if any(
                k in s.lower()
                for k in useful_keywords
            ):

                facts.append(s)

    return facts


# ======================================================
# BUILD CHUNKS
# ======================================================

normal_chunks = []

fact_chunks = []

table_chunks = []


if os.path.exists("chunks.pkl"):

    print("Loading saved chunks...")

    with open(
        "chunks.pkl",
        "rb"
    ) as f:

        all_chunks = pickle.load(f)

else:

    print("Building optimized chunks...")


    with pdfplumber.open(PDF_PATH) as pdf:

        pages = pdf.pages[
            SKIP_INITIAL_PAGES:
        ]

        for page_no, page in enumerate(pages):

            actual_page = (
                page_no
                + SKIP_INITIAL_PAGES
                + 1
            )

            print(
                f"Processing page {actual_page}"
            )

            text = page.extract_text()

            if text:

                # ======================================
                # BIG SEMANTIC CHUNKS
                # ======================================

                s_chunks = semantic_chunk(text)

                normal_chunks += s_chunks


                # ======================================
                # FACT CHUNKS
                # ======================================

                f_chunks = extract_fact_chunks(text)

                fact_chunks += f_chunks


            # ==========================================
            # TABLE EXTRACTION
            # ==========================================

            tables = page.extract_tables()

            for table in tables:

                if not table:
                    continue

                if len(table) < 2:
                    continue


                # --------------------------------------
                # HEADERS
                # --------------------------------------

                headers = table[0]

                headers = [

                    clean_text(str(h))

                    for h in headers

                    if h

                ]


                # skip broken tables
                if len(headers) < 2:
                    continue


                # --------------------------------------
                # ROWS
                # --------------------------------------

                for row in table[1:]:

                    row = [

                        clean_text(str(c))

                        for c in row

                        if c

                    ]

                    if len(row) < 2:
                        continue


                    # ==================================
                    # STRUCTURED TABLE CHUNK
                    # ==================================

                    structured = []

                    for h, v in zip(
                        headers,
                        row
                    ):

                        structured.append(
                            f"{h}: {v}"
                        )


                    table_chunk = ". ".join(
                        structured
                    )

                    if len(
                        table_chunk.split()
                    ) > 6:

                        table_chunks.append(
                            table_chunk
                        )


    # ==================================================
    # MERGE ALL
    # ==================================================

    all_chunks = (

        normal_chunks
        + fact_chunks
        + table_chunks

    )


    # ==================================================
    # REMOVE DUPLICATES
    # ==================================================

    all_chunks = list(
        dict.fromkeys(all_chunks)
    )

    normal_chunks = list(
        dict.fromkeys(normal_chunks)
    )

    fact_chunks = list(
        dict.fromkeys(fact_chunks)
    )

    table_chunks = list(
        dict.fromkeys(table_chunks)
    )


    # ==================================================
    # SAVE
    # ==================================================

    with open(
        "chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            all_chunks,
            f
        )


    # ==================================================
    # DEBUG FILES
    # ==================================================

    def save_chunks(
        filename,
        chunks,
        title
    ):

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                f"===== {title} =====\n\n"
            )

            for i, c in enumerate(chunks):

                f.write(
                    f"\n--- {title} {i} ---\n"
                )

                f.write(c + "\n")


    save_chunks(
        "normal_chunks.txt",
        normal_chunks,
        "NORMAL"
    )

    save_chunks(
        "fact_chunks.txt",
        fact_chunks,
        "FACT"
    )

    save_chunks(
        "table_chunks.txt",
        table_chunks,
        "TABLE"
    )

    save_chunks(
        "all_chunks.txt",
        all_chunks,
        "FINAL"
    )


# ======================================================
# STATS
# ======================================================

print("\n========================")

print(
    "Normal Chunks:",
    len(normal_chunks)
)

print(
    "Fact Chunks:",
    len(fact_chunks)
)

print(
    "Table Chunks:",
    len(table_chunks)
)

print(
    "Total Chunks:",
    len(all_chunks)
)

print("========================\n")


# ======================================================
# SAMPLE CHUNKS
# ======================================================

print("\n===== SAMPLE CHUNKS =====")

for i in np.random.choice(
    len(all_chunks),
    min(10, len(all_chunks)),
    replace=False
):

    print(f"\n--- Chunk {i} ---\n")

    print(all_chunks[i])


# ======================================================
# EMBEDDING MODEL
# ======================================================

embed_model = SentenceTransformer(
    "BAAI/bge-m3"
)


# ======================================================
# RERANKER
# ======================================================

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# ======================================================
# BM25
# ======================================================

tokenized_chunks = [

    c.lower().split()

    for c in all_chunks

]

bm25 = BM25Okapi(
    tokenized_chunks
)


# ======================================================
# FAISS
# ======================================================

if os.path.exists(
    "faiss_index.bin"
):

    print("Loading FAISS index...")

    index = faiss.read_index(
        "faiss_index.bin"
    )

else:

    print("Creating embeddings...")

    embeddings = embed_model.encode(

        all_chunks,

        normalize_embeddings=True,

        convert_to_numpy=True,

        batch_size=8,

        show_progress_bar=True

    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        "faiss_index.bin"
    )


# ======================================================
# FILTER
# ======================================================

def filter_context(
    context_list,
    query
):

    keywords = query.lower().split()

    filtered = []

    for c in context_list:

        if any(
            w in c.lower()
            for w in keywords
        ):

            filtered.append(c)

    if filtered:
        return filtered

    return context_list


# ======================================================
# HYBRID RETRIEVAL
# ======================================================

def get_context(
    query,
    k=5
):

    # ==============================================
    # FAISS
    # ==============================================

    q = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q = np.array(
        q,
        dtype=np.float32
    )

    d, idx = index.search(
        q,
        k
    )

    faiss_chunks = [

        all_chunks[i]

        for i in idx[0]

    ]


    # ==============================================
    # BM25
    # ==============================================

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    bm_idx = np.argsort(
        bm25_scores
    )[-3:]

    bm_chunks = [

        all_chunks[i]

        for i in bm_idx

    ]


    # ==============================================
    # HYBRID MERGE
    # ==============================================

    merged = []

    for c in (
        faiss_chunks
        + bm_chunks
    ):

        if c not in merged:

            merged.append(c)


    context_list = filter_context(
        merged,
        query
    )


    # ==============================================
    # QUERY ROUTING
    # ==============================================

    numeric_words = [

        "rate",
        "dose",
        "spacing",
        "fertilizer",
        "kg",
        "cm",
        "yield",
        "%"

    ]

    if any(
        w in query.lower()
        for w in numeric_words
    ):

        context_list = [

            c for c in context_list

            if any(
                x in c.lower()
                for x in numeric_words
            )

        ] or context_list


    # ==============================================
    # RERANKING
    # ==============================================

    pairs = [

        [query, c]

        for c in context_list

    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(

        zip(
            scores,
            context_list
        ),

        reverse=True

    )

    context_list = [

        x[1]

        for x in ranked[:2]

    ]


    return "\n\n".join(
        context_list
    )


# ======================================================
# LLM
# ======================================================

client = InferenceClient(

    provider="featherless-ai",

    api_key=os.getenv("HF_API_KEY")

)


def generate_llm_answer(prompt):

    response = client.chat.completions.create(

        model=
        "meta-llama/Llama-3.1-8B-Instruct",

        messages=[

            {
                "role": "user",
                "content": prompt
            }

        ],

        max_tokens=900
    )

    return (
        response
        .choices[0]
        .message.content
    )


# ======================================================
# CHATBOT
# ======================================================

def agriculture_chatbot(query):

    context = get_context(query)

    print(
        "\n========== CONTEXT =========="
    )

    print(context)

    print(
        "========== END ==========\n"
    )


    prompt = f"""
You are an expert agriculture assistant named "Expert Kissan 🌾".

Your job is to help farmers with clear, practical, and easy-to-understand answers.

1.Behavior Rules:
-start with a friendly greeting
2.Answering rule
-always provide structured answer with clean bullet points and headings marked in bold format
-if the query is not related to the provided context then only use your knowledge to answer
-if the context contains mixed topics then use your knowledge to provide correct answer both from yourself and from context
-strictly do not give incorrect and doubtable answer if the topic is not in your capabilities then say cannot able to answer accordingly
-if the query ask specifications only then no need to extend the answer give one line answer suitable for the query
-very strictly follow the above points
 3.use bold text for headings and bullet points strictly

📚 Context:
{context}

❓ Question:
{query}

🌾 Answer:
"""

    return generate_llm_answer(
        prompt
    )


def answer_question(query):
    from sklearn.metrics.pairwise import cosine_similarity

    # KCC retrieval
    query_emb   = embed_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, kcc_embeddings)[0]
    best_idx    = int(np.argmax(similarities))
    best_score  = float(similarities[best_idx])

    print(f"KCC match score: {best_score:.3f} | Q: {kcc_questions[best_idx][:60]}")

    if best_score >= 0.75:
        print("✅ Direct KCC retrieval used")
        return kcc_answers[best_idx]

    print("🤖 LLM generation used")
    return agriculture_chatbot(query)


# ======================================================
# FLASK
# ======================================================

app = Flask(__name__)

CORS(app)


@app.route("/")
def home():

    return "Server running"


@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    print("Request received")

    user_message = (
        request.json.get(
            "message"
        )
    )

    reply = answer_question(
        user_message
    )

    return jsonify(
        {
            "reply": reply
        }
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )