"""
Retrieval evaluation for the Zepto RAG Support Assistant.

Run from the RAG_Project folder:
    C:\\venvs\\rag\\Scripts\\python.exe eval_retrieval.py

For each test question you list the policy file that SHOULD answer it.
The script checks whether that file appears in the top-1 and top-3
ChromaDB results and prints the hit rates for your resume and README.
"""

from rag import collection, embedding_model, document_ids

# ============================================================
# EDIT THIS LIST
# Write 15-20 realistic customer questions. For each one, put the exact
# file name (as shown by `dir docs`) of the policy that contains the answer.
# The first few entries are EXAMPLES: replace them with your own.
# ============================================================
TEST_CASES = [
    # doc_01: Delivery Policy
    ("How much do I pay for delivery if my cart is only INR 100?", "doc_01.txt"),
    ("Can I get my order sooner by paying a little extra?", "doc_01.txt"),
    ("Will Zepto deliver to an address that is not on its listed pin codes?", "doc_01.txt"),

    # doc_02: Returns & Refunds
    ("How many days do I have to send back an unopened packaged item?", "doc_02.txt"),
    ("How long until a refund reaches my bank account?", "doc_02.txt"),
    ("Can I return a shampoo bottle that I already opened?", "doc_02.txt"),

    # doc_03: Membership Tiers
    ("What does Zepto Pass cost per month and what do I get with it?", "doc_03.txt"),
    ("Which plan gives early access to limited-time deals?", "doc_03.txt"),
    ("If I cancel my membership midway, do I get money back for the current period?", "doc_03.txt"),

    # doc_04: Order Tracking
    ("How can I see where my rider is right now?", "doc_04.txt"),
    ("My order has not moved for 25 minutes past the estimated time. What should I do?", "doc_04.txt"),

    # doc_05: Order Cancellation
    ("Until when can I cancel an order I just placed?", "doc_05.txt"),
    ("My order was packed but no rider was available. Will I be charged anything?", "doc_05.txt"),

    # doc_06: Damaged or Missing Items
    ("An item was missing from my bag. How do I report it?", "doc_06.txt"),
    ("Do I need to upload a photo for an expensive order with a damaged item?", "doc_06.txt"),
    ("Will Zepto send a replacement without me returning the damaged product?", "doc_06.txt"),

    # doc_07: Gift Cards
    ("What amounts are gift cards sold in?", "doc_07.txt"),
    ("Can I use two gift cards in a single purchase?", "doc_07.txt"),
    ("How long is a gift card valid?", "doc_07.txt"),

    # doc_08: Customer Support Hours
    ("Is there a phone number I can call for help?", "doc_08.txt"),
    ("Is customer support available late at night?", "doc_08.txt"),
]
TOP_K = 3


def main():
    # Warn about expected files that do not exist in docs/
    unknown = {exp for _, exp in TEST_CASES if exp not in document_ids}
    if unknown:
        print("WARNING: these expected files are not in your docs folder:")
        for name in sorted(unknown):
            print("  -", name)
        print("Files that do exist:", document_ids, "\n")

    top1_hits = 0
    top3_hits = 0
    misses = []

    for question, expected in TEST_CASES:
        query_embedding = embedding_model.encode([question]).tolist()[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
        retrieved = results["ids"][0]

        in_top1 = retrieved[0] == expected
        in_top3 = expected in retrieved

        top1_hits += in_top1
        top3_hits += in_top3
        if not in_top3:
            misses.append((question, expected, retrieved))

        status = "TOP-1" if in_top1 else ("TOP-3" if in_top3 else "MISS ")
        print(f"[{status}] {question}")
        print(f"         expected: {expected} | retrieved: {retrieved}")

    n = len(TEST_CASES)
    print("\n" + "=" * 60)
    print(f"Questions tested        : {n}")
    print(f"Top-1 retrieval accuracy: {top1_hits}/{n} = {100 * top1_hits / n:.1f}%")
    print(f"Top-3 retrieval accuracy: {top3_hits}/{n} = {100 * top3_hits / n:.1f}%")
    print("=" * 60)

    if misses:
        print("\nQuestions where the correct policy was NOT in the top 3:")
        for question, expected, retrieved in misses:
            print(f"- {question}\n  expected {expected}, got {retrieved}")


if __name__ == "__main__":
    main()
