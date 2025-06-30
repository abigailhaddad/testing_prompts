# Prompt Testing Framework

A template for systematically testing prompts across models and configurations.

## Why Use This

When testing how well models and prompts predict your ground truth labels, you need a framework that tracks results and separates configuration (the parameters you want to change) from the rest of your code.

This template shows that workflow:
- [`config.json`](config.json) - Change prompts, models, temperatures
- [`.py files`](prompt_runner.py) - Reusable functions for experiments  
- [`notebook`](prompt_testing_demo.ipynb) - Step-by-step usage guide

We use binary classification (positive/negative) as the simplest case. Once you understand the pattern, you can extend it to multi-class or other tasks. Note: Generation tasks (like RAG) require manual evaluation or more complex success metrics, but you can still use this for the result generation, just not the automatic evaluation.

## What It Does

- Tests all combinations of prompts × models × temperatures on labeled data
- Uses structured output (Pydantic) for consistent API responses - no "please respond with only positive or negative"
- Saves results for comparison
- Shows which prompts perform best

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add API keys to `.env`. (Use the keys for whichever model providers you want to access):
   ```
   OPENAI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   ```

3. Run [`prompt_testing_demo.ipynb`](prompt_testing_demo.ipynb)

## Configuration

Edit [`config.json`](config.json) to:
- Add your prompts (use `{text}` placeholder)
- Choose models and temperatures to test
- Set your CSV column names
- Define valid response values (like ["positive", "negative"] for sentiment)

## Using Your Own Data

1. Create a CSV with text and label columns
2. Update [`config.json`](config.json) column names and response values
3. Run the notebook

**Note:** This repo includes a small sample (50 rows) of the IMDB dataset that we use to demonstrate the functionality. For the full 50,000 movie reviews, download from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## LiteLLM

This uses [LiteLLM](https://docs.litellm.ai/docs/completion/structured_outputs) for API calls, which lets you swap between providers (OpenAI, Anthropic, etc.) by changing the model name in config.json.