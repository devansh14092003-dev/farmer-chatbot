import csv
import numpy as np

from sentence_transformers import SentenceTransformer

from app import agriculture_chatbot


# ------------------------
# Load embedding model
# ------------------------

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)


# ------------------------
# Cosine similarity
# ------------------------

def cosine(a,b):

    return np.dot(a,b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# ------------------------
# Counters
# ------------------------

total = 0
good = 0


# ------------------------
# Read CSV safely
# ------------------------

with open(
    "test_queries_semantic.csv",
    newline="",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(
        f,
        skipinitialspace=True
    )

    print("Detected Headers:",
          reader.fieldnames)


    for row in reader:

        # Safe column access
        query = row[
            reader.fieldnames[0]
        ].strip()

        expected = row[
            reader.fieldnames[1]
        ].strip()


        # Generate bot answer
        bot_answer = agriculture_chatbot(
            query
        )


        # Generate embeddings
        e = model.encode(expected)

        b = model.encode(bot_answer)


        # Similarity score
        score = cosine(e,b)


        print("\n===================")
        print("Query:", query)

        print(
            "Semantic Similarity:",
            round(score,3)
        )


        total += 1


        # Threshold
        if score >= 0.70:
            good += 1



# ------------------------
# Final accuracy
# ------------------------

accuracy = (
    good / total
) * 100


print("\nFINAL RESULT")
print("-------------------")

print(
    "Semantic Accuracy:",
    round(accuracy,2),
    "%"
)