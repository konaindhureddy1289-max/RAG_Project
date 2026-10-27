# RAG Support Assistant

A Retrieval-Augmented Generation (RAG) support assistant that retrieves relevant information from policy documents and provides responses to user queries. The project demonstrates document processing, semantic search, retrieval-based answering, intent routing, and API deployment.

## Live Demo

- **Live API (Swagger UI):** https://rag-project-86gm.onrender.com/docs
- **Source Code:** https://github.com/konaindhureddy1289-max/RAG_Project

Test the deployed API using the `POST /ask` endpoint to submit a policy-related question and view the answer, source documents, and confidence score.

## Overview

This project implements a support assistant using eight sample policy documents. It uses sentence embeddings and ChromaDB to retrieve relevant document chunks and applies a response pipeline to answer supported questions or decline out-of-scope requests.

The project is designed to demonstrate the core components of a document-based support assistant, from retrieval to API deployment.

## Features

- Processes and chunks policy documents for retrieval.
- Generates text embeddings using Sentence Transformers.
- Stores and searches document embeddings using ChromaDB.
- Retrieves the top 3 relevant results for a query.
- Uses LangGraph for intent-routing workflow.
- Exposes the assistant through a FastAPI application.
- Uses Pydantic for request and response validation.
- Includes an evaluation test set and evaluation results.
- Supports containerization using Docker.

## Tech Stack

- **Language:** Python
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Workflow Orchestration:** LangGraph
- **API Framework:** FastAPI
- **Data Validation:** Pydantic
- **Deployment:** Docker
- **Evaluation:** Python-based evaluation scripts and Excel test results

## Evaluation Results

The assistant was evaluated using a 20-question test set containing answerable questions and out-of-scope questions.

The following are preliminary results from the current manually labelled evaluation spreadsheet. They should be rechecked against the expected answers and retrieved evidence before being treated as validated performance.

| Metric | Preliminary Result |
|---|---:|
| Retrieval hit rate | 100% (15/15) |
| Answer accuracy | 100% (15/15) |
| Correct decline rate | 100% (5/5) |
| Groundedness | 75% (15/20) |
| Overall labelled accuracy | 100% (20/20) |
| Average response latency | 1.25 seconds |

### Evaluation Notes

- The test set contains 15 answerable or ambiguous questions and 5 out-of-scope questions.
- Results are based on manually assigned evaluation labels and are not independently validated.
- Retrieval relevance, answer correctness, and groundedness measure different aspects of performance.
- The reported results may change after further testing and correction of unsupported answers.
- Response latency depends on the execution environment and should not be interpreted as a production benchmark.

## Project Structure

```text
RAG_Project/
├── README.md
└── RAG_Project/
    ├── docs/
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    ├── rag.py
    ├── main.py
    ├── eval_retrieval.py
    ├── run_eval.py
    ├── requirements.txt
    ├── Dockerfile
    ├── rag_eval_results.xlsx
    └── rag_eval_test_set.xlsx
```

The inner `RAG_Project/` directory contains the application code and supporting files. The outer directory contains this README and the Git repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/konaindhureddy1289-max/RAG_Project.git
cd RAG_Project
```

### 2. Navigate to the application directory

```bash
cd RAG_Project
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Run the FastAPI application using the entry point configured in `main.py`:

```bash
uvicorn main:app --reload
```

If Uvicorn is not installed, install it or add it to `requirements.txt` as appropriate for the application.

Once the server starts, open:

```text
http://127.0.0.1:8000/docs
```

This opens the FastAPI interactive API documentation, provided the application exposes the default documentation route.

## Running the Evaluation

Run the evaluation scripts from the inner `RAG_Project/` directory:

```bash
python eval_retrieval.py
```

```bash
python run_eval.py
```

Check the scripts for their configured inputs and output paths before running them. The Excel files contain the current evaluation test set and results.

## Limitations

- Evaluation results are preliminary and based on manually assigned labels.
- Retrieval of relevant documents does not guarantee that every generated or rule-based answer is correct.
- The assistant's ability to answer depends on the content and coverage of the available policy documents.
- Out-of-scope handling and response quality require further testing.
- Production readiness, scalability, and performance under concurrent requests have not been established.

## Future Improvements

- Expand the evaluation dataset with more diverse questions.
- Automate retrieval relevance, answer correctness, and groundedness checks.
- Improve refusal handling when retrieved documents do not support an answer.
- Test latency and reliability across multiple runs.
- Add structured logging and monitoring.
- Explore production-grade deployment and evaluation practices.

## Disclaimer

This is an educational portfolio project using sample policy documents. It is not an official Zepto support system, and its responses should not be treated as official company policy.

## Author

**Kona Indhu**

GitHub: [konaindhureddy1289-max](https://github.com/konaindhureddy1289-max)
