import csv
import numpy as np

from sentence_transformers import SentenceTransformer

from app import get_context


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

    return np.dot(a,b)/(
      np.linalg.norm(a)*
      np.linalg.norm(b)
    )


# ------------------------
# Counters
# ------------------------

total = 0
good = 0

scores = []


# ------------------------
# Read CSV
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

    print(
      "Detected Headers:",
      reader.fieldnames
    )


    for row in reader:

        query = row[
          reader.fieldnames[0]
        ].strip()

        expected = row[
          reader.fieldnames[1]
        ].strip()



        # ------------------
        # Retrieved Context
        # ------------------

        context = get_context(
          query
        )



        # ------------------
        # Embed compare
        # ------------------

        e = model.encode(
          expected
        )

        c = model.encode(
          context
        )


        score = cosine(
          e,c
        )

        scores.append(
          score
        )


        print(
         "\n===================="
        )

        print(
         "Query:",
         query
        )


        print(
         "\nExpected Answer:"
        )

        print(
         expected
        )


        print(
         "\nRetrieved Context:"
        )

        print(
         context[:1200]
        )


        print(
         "\nContext Similarity:",
         round(score,3)
        )


        total += 1


        # threshold
        if score >= 0.70:
            good += 1



# ------------------------
# Final results
# ------------------------

accuracy = (
  good/total
)*100


avg_similarity = (
 sum(scores)/len(scores)
)


print(
 "\nFINAL RESULT"
)

print(
 "---------------------"
)

print(
 "Average Similarity:",
 round(avg_similarity,3)
)

print(
 "Context Retrieval Accuracy:",
 round(accuracy,2),
 "%"
)