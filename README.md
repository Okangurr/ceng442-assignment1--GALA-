# ceng442-assignment1--GALA-

CENG 442 - Assignment 1: Azerbaijani Text Preprocessing + Word Embeddings 


GitHub Repository: https://github.com/<org-or-user>/ceng442-assignment1-<groupname> 

Group Members:

<Okan Rıdvan Gür> 

<Enes Geldi> 

# 1. Data & Goal

The primary goal of this project is to clean 5 Azerbaijani text datasets sourced from various origins , standardize them for sentiment analysis , and train domain-aware Word2Vec and FastText word embedding models on this cleaned, combined corpus.

The 5 main datasets used are:

labeled-sentiment.xlsx (3-class) 
test_1_.xlsx (binary) 
train_3_.xlsx (binary) 
train-00000-of-00001.xlsx (3-class) 
merged_dataset_CSV_1_.xlsx (binary) 

The project supports both binary and tri-class labeling schemes. The map_sentiment_value function converts these labels into a standard sentiment_value (float) column, mapping them as Negative = 0.0, Neutral = 0.5, and Positive = 1.0.


Preserving the Neutral=0.5 value is important ; this allows sentiment scoring to be treated not only as a classification problem but also as a regression scale between 0.0 (most negative) and 1.0 (most positive), ensuring the model learns the neutral state as a clear midpoint between positive and negative sentiments.
