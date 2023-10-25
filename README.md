# Fake News Detection Model

## Overview
The Fake News Detection Model is designed to classify news articles as either "real" or "fake" based on their textual content and additional features. This model can be used to identify potentially misleading or false information in news articles.

## Features
- Text Preprocessing: The model performs text preprocessing steps, including lowercasing and stopword removal.
- Sentiment Analysis: It uses VADER SentimentIntensityAnalyzer to analyze the sentiment of the article's text.
- Additional Feature: The model also considers the length of the text as an additional feature.

## Requirements
- Python 3
- Required Python libraries mentioned in the code

## Usage
1. Clone this repository:

```bash
git clone <https://github.com/manicdon7/Fakenews-detection-AI.git>
cd fake-news-detection-model

Install the required Python libraries:

pip install -r requirements.txt

Run the model:

python fake_news_detector.py

Example
Consider the following example sentences:

True News Sentence:
"Scientists have discovered a new species of butterfly in the Amazon rainforest."

Fake News Sentence:
"Aliens have landed in New York City and are controlling our minds with their advanced technology."

Running the model with these sentences will classify the first as "real" and the second as "fake."

Note
This model is a simplified example and may not be suitable for production use. Further fine-tuning and extensive data are required for a robust solution.
The model is designed for classification and may not verify the recency of news articles. Additional checks are needed for real-time news validation.
License
This project is licensed under the MIT License - see the LICENSE.md file for details.
