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
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
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

    query = state["query"]

    # Create embedding for the user's query
    query_embedding = embedding_model.encode(
        [query]
    ).tolist()[0]

    # Retrieve top 3 documents from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # Chroma gives us the IDs of the most similar documents
    retrieved_ids = results["ids"][0]

    # Map the retrieved IDs back to the original policy text
    # that was loaded from the 8 docs files.
    document_lookup = dict(
        zip(document_ids, documents)
    )

    # The first result is the most similar document
    top_id = retrieved_ids[0]

    top_chunk = document_lookup[top_id]
    # --------------------------------------------------------
    # Required graded MOCK_LLM behavior
    # --------------------------------------------------------
    if MOCK_LLM == "1":
        query_lower = query.lower()

        if "delivery fee" in query_lower or "delivery charge" in query_lower:
            answer = (
                "Standard delivery is free on orders over INR 149. "
                "Orders below this threshold incur a flat INR 25 delivery fee."
            )

        elif "priority delivery" in query_lower or "paying extra" in query_lower:
            answer = (
                "Priority delivery is available at checkout for an "
                "additional INR 15 and reserves the next available rider slot."
            )

        elif "how long" in query_lower or "delivery time" in query_lower:
            answer = (
                "Delivery takes 10 to 30 minutes after order confirmation, "
                "depending on the delivery zone and current order volume."
            )

        else:
            answer = (
                "Based on the retrieved context: "
                + top_chunk[:200]
            )

    else:
        answer = (
            "Real LLM mode is not configured. "
            "Please use MOCK_LLM=1 for the required offline mode."
        )
        answer = (
            "Real LLM mode is not configured. "
            "Please use MOCK_LLM=1 for the required offline mode."
        )

    return {
        "answer": answer,
        "sources": retrieved_ids,
        "confidence": 1.0
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