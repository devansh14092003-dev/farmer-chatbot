import csv

# use your chatbot functions from app.py
from app import get_context, agriculture_chatbot


total = 0
retrieval_correct = 0
answer_correct = 0


def keyword_match(text, keywords):

    text = text.lower()

    matches = 0

    for kw in keywords:
        if kw.lower() in text:
            matches += 1

    return matches


with open("test_queries.csv", newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        query = row["query"]

        keywords = row["expected_keywords"].split(";")

        total += 1

        print("\n====================")
        print("QUERY:", query)

        # -------- Retrieval check --------
        context = get_context(query)

        r_matches = keyword_match(context, keywords)

        if r_matches >= 1:
            retrieval_correct += 1
            print("Retrieval: Relevant")
        else:
            print("Retrieval: Not relevant")

        # -------- Answer check --------
        answer = agriculture_chatbot(query)

        a_matches = keyword_match(answer, keywords)

        if a_matches >= 2:
            answer_correct += 1
            print("Answer: Correct")
        elif a_matches == 1:
            print("Answer: Partial")
        else:
            print("Answer: Wrong")

        print("====================")


# Final metrics

retrieval_accuracy = (retrieval_correct / total) * 100

answer_accuracy = (answer_correct / total) * 100


print("\nFINAL RESULTS")
print("----------------------")
print("Total Queries:", total)

print(
    "Retrieval Accuracy:",
    round(retrieval_accuracy,2),
    "%"
)

print(
    "Answer Accuracy:",
    round(answer_accuracy,2),
    "%"
)