# Legal Agent - AI-Powered Legal Assistant ⚖️

Welcome to Legal Agent, a chatbot project built to answer legal questions based on the Vietnamese Code of Systematized Legal Norms. The project uses an advanced Retrieval-Augmented Generation (RAG) model combined with Large Language Models (LLMs) to provide accurate, source-cited answers.

## Architecture Overview

The project is built with a modern microservices architecture, separating the user interface from the processing logic:

  * **Backend**: An API built with **FastAPI**, responsible for all AI agent logic, including query analysis, information retrieval from the vector database, and reasoning to generate answers.
  * **Frontend**: A user-friendly interface built with **Streamlit**, allowing users to easily interact with the chatbot.
  * **Docker**: The entire application is containerized using **Docker** and **Docker Compose**, making setup and deployment extremely simple and consistent across all environments.
  * **LangGraph & LangChain**: The agent's workflow is transparently and flexibly managed by LangGraph.
  * **Langfuse**: Ready for integration to trace and debug LLM chains.

-----

## Setup and Installation Guide

Follow the steps below to set up and run the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/hungtp2504/Legal-Agent.git
cd Legal-Agent
```

### Step 2: Download and Set Up Data

This is the most critical step. The agent needs the legal documents to learn from.

1.  [**Download the data here**](https://drive.google.com/file/d/19CS-zKRhniztrtDUTSfL2wmeyzqIfR3P/view?usp=sharing).
2.  Unzip the file you just downloaded. You will find two subfolders: `parsed_json_output` and `chroma_db`.
3.  Copy both of these folders and paste them into the project's `data/` directory. Your final structure should look like `data/parsed_json_output` and `data/chroma_db`.

### Step 3: Configure the Environment

Create a `.env` file in the project's root directory and enter your API keys.

```dotenv
OPENAI_API_KEY="..."
LANGFUSE_SECRET_KEY="..."
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### Step 4: Install Necessary Libraries

```bash
pip install -r frontend/requirements.txt -r backend/requirements.txt
```

-----

## Running the Application

After completing the setup, you have two options for running the application.

### Option 1: Using Docker Compose (Recommended)

This is the simplest and most stable method. Ensure you have Docker installed on your machine.

```bash
docker-compose up --build
```

After the build and startup process is complete:

  * **User Interface (Frontend)** will be available at: `http://localhost:8501`
  * **API Documentation (Backend)** will be available at: `http://localhost:8000/docs`

### Option 2: Running Locally via CLI (For Development)

This method is useful for development and debugging. You will need to open **two separate terminal windows**.

**Terminal 1: Run the Backend**

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Run the Frontend**

```bash
cd frontend
streamlit run app.py
```