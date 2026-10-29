# NLP Production Pipeline

A production-oriented Natural Language Processing (NLP) application that provides documentation-based question answering and automated support ticket triage through a REST API.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Tests](https://img.shields.io/badge/Tests-54%20passed-success)
![CI](https://github.com/akshay9063-skill/NLPProductionPipeline/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey)

## Overview

NLP Production Pipeline combines semantic search, document retrieval, question answering, and support ticket classification in a single API.

The project focuses on building an NLP system with testing, access control, document citations, evaluation, and automated continuous integration.

## Features

- **Documentation Question Answering:** Retrieve relevant documentation to answer user questions.
- **Semantic Search:** Use sentence embeddings to find relevant document chunks.
- **FAISS Retrieval:** Support efficient vector similarity search.
- **Source Citations:** Return document references with generated answers.
- **Access Control:** Restrict document retrieval according to the authenticated user's role.
- **Support Ticket Triage:** Classify support tickets and return category, priority, product area, and summary.
- **REST API:** Expose functionality through FastAPI endpoints.
- **Input Validation:** Validate incoming requests with Pydantic schemas.
- **Automated Testing:** Run the test suite locally and through GitHub Actions.
- **Evaluation:** Measure retrieval success, citation accuracy, and keyword coverage.
- **Logging:** Record API request metadata without logging request bodies or API keys.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| API framework | FastAPI |
| Request validation | Pydantic |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector search | FAISS |
| Numerical processing | NumPy |
| Configuration | python-dotenv |
| API testing | pytest and FastAPI TestClient |
| Continuous integration | GitHub Actions |

## Architecture

```text
Client
  |
  v
FastAPI Application
  |
  +---- GET /health
  |
  +---- POST /qa
  |       |
  |       v
  |   API Key Authentication
  |       |
  |       v
  |   Document Retrieval
  |       |
  |       v
  |   Relevant Context and Citations
  |
  +---- POST /triage
          |
          v
      API Key Authentication
          |
          v
      Ticket Classification
          |
          v
      Structured Triage Response

Document Ingestion
  |
  v
Documents -> Chunks -> Embeddings -> FAISS Index

GitHub Push
  |
  v
GitHub Actions
  |
  v
Automated Test Suite
