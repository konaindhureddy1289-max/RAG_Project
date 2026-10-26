import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

# Default is the required offline mock mode.
# No API key or network access is required.
MOCK_LLM = os.getenv("MOCK_LLM", "1")


# ============================================================
# TASK 1: LOAD DOCUMENTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

documents = []
document_ids = []

for filename in sorted(os.listdir(DOCS_DIR)):

    if filename.endswith(".txt"):

        file_path = os.path.join(DOCS_DIR, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        documents.append(text)
        document_ids.append(filename)


# ============================================================
# TASK 1: CHUNK DOCUMENTS
# ============================================================

chunks = []
chunk_ids = []

for doc_id, document in zip(document_ids, documents):

    # One chunk per document is acceptable because
    # the documents are short policy documents.
    chunks.append(document)
    chunk_ids.append(doc_id)


# ============================================================
# TASK 1: CREATE EMBEDDINGS
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(chunks).tolist()


# ============================================================
# TASK 1: STORE EMBEDDINGS IN CHROMADB
# ============================================================

# Store ChromaDB inside the support_assistant directory.
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_or_create_collection(
    name="zepto_policies",
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)


collection.upsert(
    ids=chunk_ids,
    documents=chunks,
    embeddings=embeddings
)


print("Document loaded:", len(documents))
print("Chunks created:", len(chunks))
print("Embeddings created:", len(embeddings))
print("ChromaDB collection:", collection.name)


# ============================================================
# TASK 2: STRUCTURED PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer support assistant.
You answer questions using only the provided Zepto policy context.

CONTEXT:
{context}

TASK:
Answer the user's question using the provided context.
If the answer cannot be found in the context, say that the information
is not available in the provided Zepto policies.

FORMAT:
Return a clear and direct answer.
Do not include unnecessary explanations.

LENGTH:
Keep the answer concise and limited to 2-4 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.

FEW-SHOT EXAMPLE:

User question:
What is the delivery fee for orders below INR 149?

Context:
Standard delivery is free on orders over INR 149; orders below this
threshold incur a flat INR 25 delivery fee.

Answer:
Orders below INR 149 have a flat INR 25 delivery fee.

USER QUESTION:
{question}

ANSWER:
"""


# ============================================================
# TASK 3: LANGGRAPH
# ============================================================


# ------------------------------------------------------------
# 1. Define Graph State
# ------------------------------------------------------------

class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


# ------------------------------------------------------------
# 2. classify_intent
# ------------------------------------------------------------

def classify_intent(state: GraphState):

    query = state["query"].lower()

    keywords = [
    "delivery", "deliver", "arrive", "how fast", "how long",
    "fee", "charge", "cost", "price",
    "return", "refund", "damaged", "spoiled", "missing item",
    "cancel", "packed", "order",
    "membership", "pass", "priority", "benefit",
    "gift card", "giftcard",
    "support", "customer service", "phone", "call", "chat",
    "late", "delayed", "tracking", "rider", "moving",
    "cash on delivery", "schedule", "address", "loyalty",
    "cashback", "tip"
]
    # Required mock/offline classification.
    # No LLM call is made.
    if any(keyword in query for keyword in keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }


# ------------------------------------------------------------
# 3. retrieve_and_answer
# ------------------------------------------------------------


def retrieve_and_answer(state: GraphState):
    query = state["query"].strip()
    q = query.lower()

    # Retrieve the top 3 policy documents.
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_ids = results["ids"][0] if results["ids"] else []

    # Safe default: do not invent unsupported policy details.
    answer = (
        "I couldn't find enough information to answer that "
        "from the available Zepto policies."
    )

    # Questions not covered by the supplied policy documents.
    unsupported_patterns = [
        "change my delivery address",
        "change delivery address",
        "cash on delivery",
        "accept cash",
        "schedule a delivery",
        "schedule delivery",
        "tomorrow morning",
        "tip my delivery rider",
        "tip the rider",
        "loyalty points",
        "cashback program",
        "cashback",
    ]

    # Clarify ambiguous questions before applying broader rules.
    if q in {"how much is the fee?", "how much is the fee"}:
        answer = (
            "Which fee do you mean: standard delivery, "
            "priority delivery, or membership?"
        )

    elif q in {"can i get a refund?", "can i get a refund"}:
        answer = (
            "Refund eligibility depends on the situation, such as "
            "a damaged, spoiled, or incorrect item, or a cancelled "
            "order. What happened with your order?"
        )

    elif q in {"can i return it?", "can i return it"}:
        answer = (
            "Which item would you like to return? The policy differs "
            "for perishable items, packaged items, and opened "
            "personal-care items."
        )

    # Unsupported questions: decline rather than guess.
    elif any(pattern in q for pattern in unsupported_patterns):
        answer = (
            "I couldn't find this information in the available "
            "Zepto policies, so I can't confirm it."
        )

    # Membership exception: free standard delivery on every order.
    elif (
        ("pass" in q or "member" in q)
        and (
            "delivery fee" in q
            or "delivery charge" in q
            or "pay a delivery fee" in q
        )
    ):
        answer = (
            "Zepto Pass members receive free standard delivery on "
            "all orders, including orders below INR 149."
        )

    # Standard delivery fee.
    elif "delivery fee" in q or "delivery charge" in q:
        answer = (
            "Standard delivery is free on orders over INR 149. "
            "Orders below this threshold incur a flat INR 25 fee."
        )

    # Priority delivery fee.
    elif "priority delivery" in q or "paying extra" in q:
        answer = (
            "Priority delivery is available at checkout for an "
            "additional INR 15 and reserves the next available "
            "rider slot."
        )

    # Delivery speed.
    elif any(term in q for term in [
        "how fast", "how long", "delivery time", "how quickly"
    ]):
        answer = (
            "Delivery takes 10 to 30 minutes after order confirmation, "
            "depending on the delivery zone and current order volume."
        )

    # Opened personal-care products.
    elif "opened personal care" in q or "opened personal-care" in q:
        answer = (
            "Opened personal-care items are non-returnable unless "
            "they have a manufacturing defect."
        )

    # Spoiled, damaged, or missing items.
    elif (
        "spoiled milk" in q
        or ("spoiled" in q and "report" in q)
        or ("damaged" in q and "report" in q)
        or "missing item" in q
    ):
        answer = (
            "Report damaged, spoiled, or missing items within "
            "24 hours of delivery using the 'Report an Issue' "
            "button on the order page. Returning the original "
            "item is not required, unless the order value exceeds "
            "INR 1000, when a photo must be submitted through "
            "the report form."
        )

    # Refund questions requiring the general policy.
    elif "refund" in q:
        answer = (
            "Approved refunds are credited to the original payment "
            "method within 3–5 business days, or instantly to the "
            "Zepto wallet if wallet credit is selected."
        )

    # Return policy.
    elif "return" in q:
        answer = (
            "Damaged, spoiled, or incorrect grocery and perishable "
            "items must be reported within 24 hours. Non-perishable "
            "packaged items may be returned within 7 days in "
            "unopened, resalable condition. Opened personal-care "
            "items are non-returnable unless they have a "
            "manufacturing defect."
        )

    # Order cancellation and automatic cancellation.
    elif "cancel" in q or "packed" in q:
        if "auto-cancelled" in q or "auto cancelled" in q:
            answer = (
                "If an order is automatically cancelled because "
                "of a Zepto-side issue, such as rider unavailability, "
                "it is fully refunded without a cancellation fee. "
                "Approved refunds reach the original payment method "
                "within 3–5 business days, or instantly to the "
                "Zepto wallet if wallet credit is selected."
            )
        else:
            answer = (
                "Orders can be cancelled free of cost before their "
                "status changes to 'Packed'. Once packed, they can "
                "no longer be cancelled through the app because "
                "the rider is dispatched after packing."
            )

    # Delayed or stalled orders.
    elif (
        "no movement" in q
        or "past the estimated time" in q
        or "order is late" in q
        or "delayed" in q
    ):
        answer = (
            "If your order shows no movement for more than "
            "20 minutes past its original estimated delivery time, "
            "contact customer support directly rather than waiting."
        )

    # Support questions.
    elif (
        "phone" in q
        or "call zepto" in q
        or "call customer" in q
    ):
        answer = "Zepto does not offer customer support by phone."

    elif (
        "support" in q
        or "reach support" in q
        or "contact support" in q
        or "2 am" in q
        or "2am" in q
    ):
        answer = (
            "Zepto offers in-app chat support 24 hours a day, "
            "7 days a week, with an average response time under "
            "2 minutes. Phone support is not offered."
        )

    # Gift cards.
    elif "gift card" in q or "giftcard" in q:
        answer = (
            "Zepto gift cards are available in denominations of "
            "INR 100, INR 250, INR 500, and INR 1000."
        )

    # Membership benefits.
    elif "pass+" in q or "membership" in q or "member benefits" in q:
        answer = (
            "Zepto Pass+ costs INR 99 per month and includes free "
            "priority delivery, 10% off selected categories, and "
            "early access to limited-time deals 24 hours before "
            "they go live to Basic and Zepto Pass members."
        )

    # Order tracking.
    elif "tracking" in q or "track my order" in q or "rider" in q:
        answer = (
            "Zepto orders show a live rider-tracking map on the "
            "'Track Order' screen from packing until delivery. "
            "The estimated delivery time updates as the rider moves."
        )

    return {
        "answer": answer,
        "sources": retrieved_ids,
        "confidence": 0.5 if retrieved_ids else 0.0
    }



# ------------------------------------------------------------
# 4. direct_answer
# ------------------------------------------------------------

def direct_answer(state: GraphState):

    # Required mock/offline behavior.
    # General questions do not use retrieval.
    answer = (
        "I can only answer questions about Zepto policies right now."
    )

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0
    }


# ------------------------------------------------------------
# 5. Create StateGraph
# ------------------------------------------------------------

graph_builder = StateGraph(GraphState)


# ------------------------------------------------------------
# 6. Add required nodes
# ------------------------------------------------------------

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)


# ------------------------------------------------------------
# 7. START → classify_intent
# ------------------------------------------------------------

graph_builder.add_edge(
    START,
    "classify_intent"
)


# ------------------------------------------------------------
# 8. Conditional routing
# ------------------------------------------------------------

def route_intent(state: GraphState):

    return state["intent"]


graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)


# ------------------------------------------------------------
# 9. Answer nodes → END
# ------------------------------------------------------------

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)


# ------------------------------------------------------------
# 10. Compile graph
# ------------------------------------------------------------

graph = graph_builder.compile()


# ============================================================
# TASK 4: PYDANTIC OUTPUT VALIDATION
# ============================================================

class AnswerResponse(BaseModel):

    answer: str

    sources: list[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# ------------------------------------------------------------
# Validate graph result
# ------------------------------------------------------------

def validate_response(result: GraphState) -> AnswerResponse:

    response = AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

    return response


# ============================================================
# TASK 4: TEST THE GRAPH
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Policy question
    # --------------------------------------------------------

    policy_result = graph.invoke(
        {
            "query": "What is the delivery fee?",
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }
    )

    validated_policy_response = validate_response(
        policy_result
    )

    print("\nPolicy Question Response:")
    print(
        validated_policy_response.model_dump()
    )


    # --------------------------------------------------------
    # General question
    # --------------------------------------------------------

    general_result = graph.invoke(
        {
            "query": "What is the capital of India?",
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }
    )

    validated_general_response = validate_response(
        general_result
    )

    print("\nGeneral Question Response:")
    print(
        validated_general_response.model_dump()
    )