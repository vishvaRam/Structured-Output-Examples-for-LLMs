# Structured Output with Language Models

This repository demonstrates structured data extraction using various language models and frameworks. It includes examples of generating JSON outputs for name and age extraction from text prompts. The project leverages models like Qwen and frameworks such as LangChain, vLLM, and Outlines.

## Files Overview
- **[`Groq_Langchain.py`](Groq_Langchain.py)**: Uses the langchain Groq library for guided decoding with Pydantic JSON schema.
- **[`Gemini_Langchain.py`](Gemini_Langchain.py)**: Uses the langchain Gemini library for guided decoding with Pydantic JSON schema.
- **[`vLLM.py`](vLLM.py)**: Uses the vLLM library for guided decoding with JSON schema.
- **[`vLLM_openai_client.py`](vLLM.py)**: Uses the vLLM's Openai client library to access vLLM server for guided decoding with JSON schema.
- **[`ollama.py`](ollama.py)**: Implements structured output using the Ollama chat API.
- **[`OllamaLLM_Batch_Processing.py`](OllamaLLM_Batch_Processing.py)**: Batch processes prompts with LangChain's OllamaLLM and Pydantic parsers.
- **[`OllamaLLM.py`](OllamaLLM.py)**: Single-prompt processing with LangChain's OllamaLLM.
- **[`chatOllama.py`](chatOllama.py)**: Chat-based structured output using LangChain's ChatOllama.
- **[`Outliner_for_transformers.py`](Outliner_for_transformers.py)**: Utilizes the Outlines library for JSON generation with transformer models.
- **[`Outliner_for_transformers_vision.py`](Outliner_for_transformers_vision.py)**: Utilizes the Outlines library for JSON generation with transformer vision models (Note: This Outlines transformers_vision will only work on pytorch version 2.4, I tried 2.6 it was not working.)
- **[`Outliner_for_transformers_vision_batch.py`](Outliner_for_transformers_vision_batch.py)**: Utilizes the Outlines library for JSON generation with transformer vision models in batches.
  
## Key Features
- Structured JSON output using Pydantic schemas.
- Integration with multiple LLM frameworks.
- Examples for both single and batch processing.
