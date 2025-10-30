# ceng442-assignment1--GALA-

CENG 442 - Assignment 1: Azerbaijani Text Preprocessing + Word Embeddings 


GitHub Repository: https://github.com/<org-or-user>/ceng442-assignment1-<groupname> 

Group Members:

1-Okan Rıdvan Gür

2-Enes Geldi

## 1. Data & Goal
The primary goal of this project is to clean 5 Azerbaijani text datasets sourced from various origins, standardize them for sentiment analysis, and train domain-aware **Word2Vec** and **FastText** word embedding models on this cleaned, combined corpus.  
The 5 main datasets used are:  
- `labeled-sentiment.xlsx` (3-class)  
- `test_1_.xlsx` (binary)  
- `train_3_.xlsx` (binary)  
- `train-00000-of-00001.xlsx` (3-class)  
- `merged_dataset_CSV_1_.xlsx` (binary)  
The project supports both **binary** and **tri-class** labeling schemes. The `map_sentiment_value` function converts these labels into a standardized `sentiment_value` (float) column, mapping them as:  
- Negative = `0.0`  
- Neutral = `0.5`  
- Positive = `1.0`  
Preserving the **Neutral = 0.5** value is important, as this allows sentiment scoring to be treated not only as a classification problem but also as a **regression scale** between `0.0` (most negative) and `1.0` (most positive), ensuring the model learns the neutral state as a clear midpoint between positive and negative sentiments.


## 2. Preprocessing

The text cleaning process is structured around the normalize_text_az function in the preprocess.py script.  
This section covers the basic and simple cleaning steps required by Section 4 of the assignment.

---

### 🔹 Primary Basic Preprocessing Steps

1. **Character Normalization**  
   Potential encoding issues (mojibake) are fixed using the ftfy library, and HTML unescape is applied.

2. **Azerbaijani-Specific Lowercasing**  
   Texts are converted to lowercase using the lower_az function, which correctly handles Azerbaijani characters (İ → i, I → ı).

3. **Structural Noise Removal**  
   HTML tags (HTML_TAG_RE) are completely removed.

4. **Special Token Replacement**  
   - URLs (URL_RE) → replaced with “URL”  
   - Email addresses (EMAIL_RE) → replaced with “EMAIL”  
   - Phone numbers (PHONE_RE) → replaced with “PHONE”  
   - User mentions (@mention) (USER_RE) → replaced with “USER”

5. **Number Tokenization**  
   Numerical expressions (DIGIT_RE) are replaced with the “<NUM>” token.

6. **Character Noise Removal**  
   - Repeated characters (e.g., çooox) are reduced to two (çox) (REPEAT_CHARS).  
   - All punctuation and symbols outside the Azerbaijani alphabet (including ə, ğ, ı, ö, ü, ç, ş, x, q) and special tokens are removed (NONWORD_RE).  
   - Excess whitespace is collapsed to a single space (MULTI_SPACE).

7. **Final Filtering**  
   With min_token_len=2, all single-letter tokens are dropped, except for 'o' and 'e'.

---

### 🧹 Preprocessing Examples (Before / After)

#### Example 1 — General Cleaning  
Before:  
Salam @istifadechi! Bu məhsul ÇOOOX #YaxsiQiymet 👍 https://link.com/alishverish 100 AZN idi.  

After (Basic Rules):  
salam USER məhsul çox #yaxsiqiymet 👍 URL <NUM> azn  

Note: At this stage, the Emoji (👍) and Hashtag (#yaxsiqiymet) have not yet been processed.

---

#### Example 2 — Structural Cleaning  
Before:  
Cox pis <br> mehsul yaxsi deyil. Hec almağa dəyməz. 😠  

After (Basic Rules):  
cox pis məhsul yaxsi deyil hec almağa dəyməz 😠  

Note: At this stage, 'cox' has not been de-asciified, and negation scoping has not been applied.

---

### 📊 Dataset Statistics

While processing the datasets with the process_file function,  
rows containing empty (NaN) or duplicate text were removed using:

dropna(subset=[text_col])  
drop_duplicates(subset=[text_col])

Summary:  
Across all 5 datasets, a total of X empty and Y duplicate text rows were identified and removed from the main dataset.


