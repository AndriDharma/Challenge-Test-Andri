-----

# Internal Agentic LLM Chatbot

This repository contains the source code for an internal, agentic Large Language Model (LLM) chatbot. The chatbot is designed to answer queries by intelligently routing requests to the appropriate data source, whether it's unstructured documents or structured databases.

-----

## 🚀 Overview

The core of this project is an agentic chatbot that leverages a powerful LLM (`Gemini 2.5 Pro`) to understand user intent. Based on the query, the agent decides between two primary tools:

  * **Retrieval-Augmented Generation (RAG)**: For querying unstructured data from documents like PDFs.
  * **Text-to-SQL**: For querying structured, tabular data from our BigQuery database.

The entire application is built on Google Cloud Platform, designed for scalability, security, and maintainability.

-----

## ✨ Features

  * **Agentic Routing**: Intelligently chooses between RAG and Text-to-SQL based on the user's query.
  * **Advanced RAG Pipeline**:
      * Uses `Gemini 2.5 Pro` for intelligent, context-aware document chunking.
      * Embeds text chunks using `gemini-embedding-001`.
      * Stores and retrieves vectors using a PGVector index on a managed Cloud SQL instance.
  * **Context-Aware Text-to-SQL**:
      * Generates accurate SQL queries by providing the LLM with database schema, data snippets, and detailed table descriptions.
  * **Persistent Memory**: Chat history is stored and retrieved from Google Cloud Storage, allowing for conversational context.
  * **Scalable Architecture**: Frontend (Streamlit) and Backend (FastAPI) are containerized and deployed on Cloud Run for automatic scaling.
  * **Automated CI/CD**: Container images are automatically built by Cloud Build and stored in Artifact Registry for streamlined deployments.

-----

## 🏗️ System Architecture

The application is split into two main parts: the core chatbot architecture and the deployment pipeline.

### Chatbot Architecture

The user interacts with a Streamlit frontend, which communicates with a FastAPI backend. The backend orchestrates the agentic logic, calling the Gemini LLM to process requests and interact with the data sources.

  * **Frontend**: Streamlit on Cloud Run
  * **Backend**: FastAPI on Cloud Run
  * **LLM**: Gemini 2.5 Pro
  * **Unstructured Data**: PGVector on Cloud SQL
  * **Structured Data**: BigQuery
  * **Chat Memory**: Google Cloud Storage
  * **Credentials**: Secret Manager
  * **Security**: Service Account

### Deployment Architecture

Development pushes trigger Google Cloud Build, which containerizes the application, stores the image in Artifact Registry, and deploys it to Cloud Run.

-----

## 📊 Data Handling

### RAG for Unstructured Data (PDFs)

Our RAG pipeline is designed to maximize retrieval accuracy:

1.  **AI-Powered Chunking**: Instead of fixed-size chunks, a `Gemini 2.5 Pro` model analyzes the structure of each PDF page to determine the most logical way to split the content.
2.  **AI-Powered Conversion**: The same model then converts these semantically meaningful chunks into clean text.
3.  **Embedding & Storage**: The text is embedded using the `gemini-embedding-001` model, and the resulting vectors are stored in a PGVector-enabled Cloud SQL database.

### Text-to-SQL for Structured Data (BigQuery)

To ensure the LLM generates correct SQL queries, we provide it with rich context about our database. The prompt sent to `Gemini 2.5 Pro` includes:

  * **Table Schema**: The detailed structure of the relevant tables.
  * **Data Snippet**: A few rows of sample data to illustrate the content.
  * **Detailed Description**: A natural language explanation of what each column represents and how to query the data.

-----

## ⚙️ Setup and Installation

This is a private repository intended for deployment on GCP. To run locally, you will need to configure your environment.

### Prerequisites:

  * Python 3.11+
  * Docker
  * Google Cloud SDK (`gcloud`)

### Local Configuration:

1.  Clone the repository.
2.  For both the `api/` and `streamlit/` directories, create a `.env` file (not available in this repo).
3.  Populate the `.env` files with the necessary credentials and configuration values for GCP services (Project ID, BigQuery dataset, Cloud SQL instance, etc).
4.  Install dependencies in each directory:
    ```bash
    pip install -r requirements.txt
    ```

-----

## 🚀 Deployment

Deployment is handled via Google Cloud Build and is triggered by running the deployment scripts.

1.  Navigate to the target directory:
    ```bash
    cd api/
    # or
    cd streamlit/
    ```
2.  Run the deployment script:
    ```bash
    bash deploy.sh
    ```

This script will submit a build job to Cloud Build, which will containerize the application, push the image to Artifact Registry, and deploy the new version to Cloud Run.

-----

## 🔐 Security

  * **Environment Variables**: All secrets, keys, and environment-specific configurations are managed via `.env` files.
  * **.gitignore**: The `.env` file is explicitly included in `.gitignore` and must not be committed to the repository.
  * **GCP IAM**: Access to GCP services is controlled by fine-grained IAM roles assigned to the Cloud Run service accounts.
  * **Credentials**: Cloud SQL and API credentials are stored in Google Cloud Secret Manager.
