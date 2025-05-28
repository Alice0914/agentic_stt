# Agentic AI Call Summary System

This project showcases a **local agentic AI pipeline** that transcribes an audio file, summarizes the content, analyzes sentiment, generates a customer note, and evaluates the output using OpenAI's Whisper and GPT-4o.

---

## Files

- `stt_model.ipynb`: Main notebook implementing the pipeline.
- `Short_Record1.mp3`: Sample audio file for testing the system.

---

## Features

✅ Automatic speech-to-text (STT) using Whisper  
✅ Transcript cleaning and filler removal  
✅ Combined summarization + sentiment analysis  
✅ Customer note generation using GPT-4o  
✅ LLM-as-a-Judge style evaluation  

---

## Requirements

- Python 3.10+
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)

Install dependencies:
```bash
pip install openai langchain langgraph
