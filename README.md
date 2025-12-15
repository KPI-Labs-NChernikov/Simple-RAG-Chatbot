# AWS Technical Support Bot

A lightweight RAG assistant for Amazon Web Services queries, powered by **Google Gemini 2.5 Flash** and **Gradio**.

## 🚀 Quick Start

**1. Add AWS documentation files**
in pdf format to data folder
```bash
mkdir ./data
```
```bash
cp /full/path/to/aws-docs1.pdf ./data/
```

**2. Install Dependencies**

```bash
pip install google-genai gradio langchain-chroma langchain-openai langchain-community pypdf langchain-text-splitters
```

**3. Set Gemini API Key**
You need a [Google AI Studio](https://aistudio.google.com/) key.

  * **Mac/Linux:** `export GEMINI_API_KEY="your_gemini_api_key"`
  * **Windows:** `$env:GEMINI_API_KEY="your_gemini_api_key"`

**4. Set OpenAI API Key (for embeddings)**
You need a [OpenAI Platform](https://platform.openai.com/api-keys/) key.

  * **Mac/Linux:** `export OPENAI_API_KEY="your_openai_api_key"`
  * **Windows:** `$env:OPENAI_API_KEY="your_openai_api_key"`

**5. Run db_uploader**
and wait until it completes ChromaDB setup.
```bash
python db_uploader.py
```

**6. Run the App**

```bash
python main.py
```

Open the URL displayed in your terminal (usually `http://127.0.0.1:7860`).

## 💡 What it does

  * **Expert Assistance:** Answers questions about AWS architecture (EC2, Lambda, S3, etc.), troubleshooting, and CI/CD.
  * **Contextual:** Remembers your chat history for debugging sessions.
  * **Focused:** It acts strictly as an AWS support agent and will decline unrelated topics (like cooking or general news).

## 🔧 Requirements

  * Python 3.9+
  * Google Gen AI SDK
  * OpenAI Embeddings
