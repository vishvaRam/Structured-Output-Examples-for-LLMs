# Structured Output with Language Models

This repository demonstrates structured data extraction using various language models and frameworks. It includes examples of generating JSON outputs for name and age extraction from text prompts. The project leverages models like Qwen and frameworks such as LangChain, vLLM, and Outlines.

## Files Overview
- **[`vLLM.py`](vLLM.py)**: Uses the vLLM library for guided decoding with JSON schema.
- **[`vLLM_openai_client.py`](vLLM.py)**: Uses the vLLM's Openai client library to access vLLM server for guided decoding with JSON schema.
- **[`ollama.py`](ollama.py)**: Implements structured output using the Ollama chat API.
- **[`OllamaLLM_Batch_Processing.py`](OllamaLLM_Batch_Processing.py)**: Batch processes prompts with LangChain's OllamaLLM and Pydantic parsers.
- **[`OllamaLLM.py`](OllamaLLM.py)**: Single-prompt processing with LangChain's OllamaLLM.
- **[`chatOllama.py`](chatOllama.py)**: Chat-based structured output using LangChain's ChatOllama.
- **[`Outliner_for_transformers.py`](Outliner_for_transformers.py)**: Utilizes the Outlines library for JSON generation with transformer models.

## Key Features
- Structured JSON output using Pydantic schemas.
- Integration with multiple LLM frameworks.
- Examples for both single and batch processing.
