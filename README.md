# Zepto RAG Support Assistant

A Retrieval-Augmented Generation (RAG) based customer support assistant that answers Zepto customer queries using information retrieved from policy documents.

## Features

- Processes eight Zepto policy documents.
- Splits document text into chunks for retrieval.
- Generates semantic embeddings using Sentence Transformers.
- Stores and retrieves document embeddings using ChromaDB.
- Uses LangGraph to organize the RAG workflow.
- Provides a FastAPI endpoint for submitting customer queries and returning answers with source information.

## Tech Stack

- **Language:** Python
- **Backend:** FastAPI, Uvicorn
- **RAG and Workflow:** LangGraph
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Validation:** Pydantic
- **Deployment:** Docker

## Project Structure

```text
RAG_Project/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   └── ...
├── main.py
├── rag.py
├── eval_retrieval.py
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Run Locally

1. Clone this repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd RAG_Project
   ```

2. Create and activate a virtual environment.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI application:

   ```bash
   uvicorn main:app --host 127.0.0.1 --port 7860
   ```

5. Open `http://127.0.0.1:7860/docs` to test the API.

## API Usage

**Endpoint:** `POST /ask`

Example request:

```json
{
  "query": "What is the delivery fee?"
}
```

The API returns an answer along with source information and a confidence value.

## Docker

Build the Docker image:

```bash
docker build -t zepto-support .
```

Run the container:

```bash
docker run -d --name zepto-support-app -p 7860:7860 zepto-support
```

Then visit `http://127.0.0.1:7860/docs`.

## Future Improvements

- Add a user-friendly chat interface.
- Improve retrieval evaluation.
- Deploy the API to a cloud hosting platform.

## Author

Kona Indhu
