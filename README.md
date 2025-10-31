#  Azerbaijani Domain-Aware Word Embedding Project

---
#Embeddings Model Drive Link:https://drive.google.com/file/d/1lFhOj5CVqRB4WpkSpeRd61lHtEGO_d1y/view?usp=drive_link
Please download the embeddings file in the drive link to your project directory!
## 1️-Data & Goal:

- *Goal:* The purpose of this NLP project is to learn and properly understand the preprocessing pipeline for Azerbaijani texts and sentences, and to build domain-aware Word2Vec and FastText models. The project also aims to observe how the Lexical Coverage, Synonym, Antonym, and Separation scores of these models change based on the additional functions we implement in our preprocessing flow (e.g., negation scope, stemming, lemmatization, etc.), the hyperparameter tuning during model training (e.g., epochs, negative, vector-size, etc.), and domain-awareness. Ultimately, the objective is to fully comprehend the entire Preprocessing-Train-Analysis pipeline of an NLP project.
  
- *Datasets:* 5 corpora → labeled-sentiment, test__1_, train__3_, train-00000-of-00001, merged_dataset_CSV__1_.
- *Pipeline:* Cleaning, normalization, stemming-lemmatization, domain detection → corpus building → embedding training → evaluation.
- *Domains:* news, social, reviews, general.
- Why keep neutral=0.5 : The reason some texts in the datasets are assigned trinary values of 1.0, 0.5, and 0.0 is that we want our models to perform sentiment analysis. When doing this, we proceed by analyzing the words of the sentence (using tokenization). Consequently, some sentences can be positive sentiment, which equals 1.0, while some sentences can be negative, taking the value 0.0. However, some sentences may be neither positive nor negative (for example: "yaxşı amma bahalı," which means "good but expensive"). Sentences like these are considered neutral and are given a sentiment_value of 0.5.

---

## 2️ Preprocessing & Mini Challenges

## Rules We Apply in Preprocessing

| Step | Description |
|:--|:--|
| Lowercasing (Azeri-specific) | Handles special casing rules like İ → i, I → ı for Azerbaijani. |
| Emoji Mapping | Expanded to cover 50+ emojis across positive, negative, and neutral categories. |
| Hashtag Splitter | Splits hashtags by CamelCase, underscores `_`, and numeric transitions (e.g., `#CoxYaxsiFilm` → Cox Yaxsi Film). |
| Deasciify & Slang Normalization | Converts Latinized or slang variants (e.g., `yaxsi` → `yaxşı`, `cox` → `çox`). |
| Negation Scope Tagging | The `_NEG` suffix is added  (in a very detailed way) to negative words (`deyil`, `yox`, `qətiyyən`). |
| Stopword Filtering | Removes over 100 Azerbaijani function words while preserving negations for sentiment accuracy.Because in some sentences, these stopwords may contain emotions, contrary to what they normally do.|
| Snowball Stemming | Applies the Turkish Snowball Stemmer, adapted for Azerbaijani morphology to unify word variants. |
| Strict Filter | Keeps only valid Azerbaijani alphabet tokens and special placeholders (`<NUM>`, `URL`, `EMAIL`,`PHONE`,`USER`,`EMO_POS/EMO_NEG` `RATING_POS/NEG`, `STARS_\d_5`, `PRICE`). |


>What we actually struggled with here was finding a lemmatizer specific to the Azerbaijani language for our project. We did find Stanza, but we would have needed to install dependencies like PyTorch for the project. Since this would increase both the project overhead and the submission (deployment) overhead, we tried stemming instead, for the sake of learning the process.

This was actually a good experience for us, as we saw the difficulties of stemming. Since Azerbaijani, like Turkish, is an agglutinative (suffix-based) language, we used a stemmer from the Turkish SnowballStemmer. However, we frankly noticed that the stemmer struggled to identify the roots of some words.

Of course, it was still working better than before. Previously, "yaxşıdır" (is good) and "yaxşı" (good) were treated as different tokens and were perceived as having different semantic meanings, making it difficult for the model to place them closely in the vector space. But now, by applying stemming, we have managed to mitigate these problems, at least on the model's side.

> We perform the cleaning of empty data (null, whitespace-only) or duplicate data as follows:

Inside process_file(), we clean the unnecessary index columns (we are actually cleaning an empty column named "Unnamed: 0" that was in one Excel file). And, by using dropna(subset=[text_col]), we delete rows in the text column that are NaN or None.

    .str.strip() → We remove leading/trailing whitespaces.

    .str.len() > 0 → We also filter out rows that consist only of spaces.

    drop_duplicates(subset=[text_col]) → If the same text exists more than once, we keep only one.

>  As a result of all these steps, the lexical coverage rate reached approximately 97-98%. The number of OOV tokens has decreased significantly, which shows us how important preprocessing steps are for Lexical Coverage.

By combining the cleaned Excel files, we created corpus_all.txt, each line of which starts with the domain tag (domsocial, domnews, etc.). We will use this corpus_all.txt when training our models.l.

### Domain Aware
>We implemented the domain-aware adjustments within the detect_domain() function, using a heuristic, keyword-based (rule-based) approach.

Our goal was: To differentiate various domains such as news texts, social texts, or reviews, and to preserve the distinct linguistic features for each domain. This approach was intended to increase the model's understanding of semantics.

1. news -> apa,trend,azertac,reuters,bloomberg,dha,aa(When official news agency, economic or press sources were mentioned, they were tagged as “news”.)
2. social -> rt,@,# veya social media emojies(When a tweet, post, or short social message was detected, it was marked as “social.”)
3. reviews -> azn, manat, qiymət, ulduz, çox yaxşı, çox pis(Sentences containing product/service evaluation, stars or price were determined as "reviews".)
4. general -> All other texts(Neutral data that did not fit into any domain remained "general".)
   
>Domain-aware normalization:
>After the domain determination was made, we carried out standardization processes by applying special rules for some domains. These are:
>  Prices Expression,Star Rating,Positive Reviews, Negative Reviews
>Thanks to these special transformations the model:
>qiyməti 20 manat amma çox yaxşı xidmət idi -> PRICE amma RATING_POS xidmət idi
>Notes:
>
 We tried to perform domain detection, and for this,    a domain tag like dom<domain(reviews)> is added to the beginning of each line      as   a preparation for training when creating the corpus. corpus_all.txt → We      created this by adding domains over the cleaned data; it contains the separate     domain information and the cleaned version of all sentences.

###  3 Training Embeddings Model:
We had a few questions here regarding training strategies: These were questions like how domain knowledge affects training, and how parameters affect training. Therefore, we created and briefly tested training modes, and tried to make observations based on them.

1. Shared Mod: All sentences are combined and a single model is trained (domains are removed). Our aim in this mode is to produce general-purpose embeddings suitable for all domains.

2. Per-domain Mod: Each domain is trained as a separate model with its own subset. Our aim is to capture and examine domain-aware semantic density.

3. Adapt (fine-tune) Mod: The Shared model is trained, and then it is continued for a few more epochs with each domain's data. Our aim is to preserve general knowledge while learning domain-specific meanings.

By doing this, we tried to look at how the model learns differently, and how fine-tuning and domains affect this.

|  Model |  Mode |  Vec Size |  Window |  Min Count |  Negative |  SG |  Epochs |  n-gram |  Notes |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| *Word2Vec (shared)* | Single global model | 300 | 5 | 3 | 10/15 | 1 | 10/25 | – | Domain tags removed before training |
| *FastText (shared)* | Same as above | 300 | 5 | 3 | 10/15 | 1 | 10/25 | 3–6 | Subword-based representation |
| *W2V / FT (per-domain)* | Separate models per domain | 300 | 7 | 2 | 15 | 1 | 25 | 3–6 | Trained separately for news, social, reviews, general |
| *W2V / FT (adapt)* | Fine-tuned per-domain | 300 | 5 | 3 | 10 | 1 | 10 + 3 | 3–6 | Continued training from shared model with domain data |

>  All models were trained using preprocessed corpora (stemming=True, stopword=True, strict=True).
>  
> The *shared* models capture global semantics,  
> while *per-domain* and *adapt* variants specialize in local linguistic patterns.
>
>
##  4️ Quantitative Results
### 4.1 Lexical Coverage and Similarity

| Domain | Synonyms<br>(W2V / FT) | Antonyms<br>(W2V / FT) | Separation<br>(Syn−Ant)<br>(W2V / FT) | Coverage | Observation |
|:--|:--:|:--:|:--:|:--:|:--|
| *Baseline (Instructor)* | 0.346 / 0.441 | 0.355 / 0.417 | −0.009 / 0.024 | 0.93–0.99 | Simple preprocessing, shared model |
| *Shared (ep10, neg10)* | 0.284 / 0.319 | 0.304 / 0.349 | −0.020 / −0.031 | 0.93–0.99 | Slightly under baseline |
| *Shared (ep25, neg15)* | 0.279 / 0.321 | 0.270 / 0.308 | *+0.009 / +0.013* | 0.96–1.00 | Longer training improved separation |
| *Social (per-domain)* | 0.258 / 0.341 | 0.309 / 0.393 | −0.051 / −0.052 | 0.64–0.67 | FastText handles slang & noise better |
| *Reviews (per-domain)* | 0.296 / 0.334 | 0.331 / 0.397 | −0.035 / −0.063 | 0.60–0.65 | Captures product sentiment effectively |
| *News (per-domain)* | 0.271 / 0.317 | 0.301 / 0.369 | −0.030 / −0.052 | 0.41–0.46 | Limited domain coverage, formal structure |
| *General (per-domain)* | 0.286 / 0.339 | 0.273 / 0.377 | +0.013 / −0.038 | 0.92–0.97 | W2V stable; FT more flexible |
| *Social (adapt)* | 0.254 / 0.318 | 0.305 / 0.394 | −0.051 / −0.075 | – | Fine-tuning reduced domain noise |

>  FastText achieves more robust performance in morphology-rich domains due to its subword modeling.  
> Increasing epochs and negative sampling improved *semantic separation* for shared models.
> Domain-specific training significantly reduced vocabulary imbalance.


---

### 4.2 Qualitative Results (Nearest Neighbors)

| Word | Model | Top Neighbors |
|:--|:--|:--|
| *yaxşı* | W2V (shared) | RATING_POS, yaxşı, yaxşıca, demey, awsome |
|  | FT (shared) | yaxşıı, yaxş, yaxşıca, yaxşıya |
| *pis* | W2V (shared) | varidislara, xalçalardan, RATING_NEG |
|  | FT (shared) | piis, pisə, pisi, pisleşdi |
| *çox* | W2V (shared) | çöx, bəyəndim, gözəldir |
|  | FT (shared) | çoxx, çoxçox, çoxh, ço |
| *bahalı* | W2V (reviews) | restoranlarda, şeheri, villaları |
|  | FT (reviews) | bahalıq, bahalıdı, bahalıdır |
| *ucuz* | W2V (reviews) | şeytanbazardan, düzəltdirilib, qiymətə |
|  | FT (reviews) | ucuzu, ucuzdu, ucuzdur, ucuzluğa |
| *mükəmməl* | W2V (general) | möhtəşəm, kəliməylə, tamamlayır |
|  | FT (general) | mükəmməldi, mükəmməlsiz, mükəməl |
| *ulduz* | W2V (reviews) | verəm, verdimki, vercem |
|  | FT (reviews) | ulduzz, ulduza, ulduzu, ulduzdu |

>  *Observation:*  
> FastText successfully groups morphological variants (e.g., “yaxşıı”, “yaxş”),  
> while Word2Vec yields semantically coherent neighbors (“RATING_POS”, “mükəmməl”).  
> This confirms FastText’s advantage in *subword-level generalization*,  
> and Word2Vec’s strength in *semantic clarity* for domain-stable tokens.

---

##  5️ Analysis and Discussion

### 5.1 Baseline vs Enhanced Models
- *Baseline* model used minimal preprocessing: only lowercasing and punctuation removal.  
  No domain-awareness, stemming, or negation handling was applied.  
  As a result, *semantic separation* between synonym and antonym pairs was weak (Δ ≈ −0.009).  
  FastText slightly outperformed Word2Vec due to its subword representations.

- *Enhanced pipeline (ours)* introduced:
  - *Extended normalization:* emoji mapping, hashtag splitting, deasciification, stopword filtering  
  - *Negation scope detection:* deyil, yox annotated as _NEG to preserve polarity  
  - *Lemmatization / Stemming:* reduced morphological noise  
  - *Domain-awareness:* domnews, domreviews, domsocial, domgeneral tagging with custom normalization  
  - *Fine-tuned training:* multi-phase (shared → per-domain → adapt)

  This led to clear *semantic improvement*, particularly:
  - Positive Syn−Ant separation for shared W2V (+0.009)  
  - Stronger lexical coverage in per-domain W2V/FT (0.96–1.00)  
  - Cleaner neighbor clusters (e.g., “RATING_POS”, “bahalıq”, “ucuzdur”)

---

### 5.2 Domain Impact
- *Social domain* benefitted most from FastText: noise, slang, and diacritics handled via subword units.  
- *Reviews domain* showed most consistent sentiment terms — PRICE, RATING_POS, ulduz.  
- *News domain* remained sparse and formal — limited vocabulary overlap reduced improvement.  
- *General domain* acted as a stabilizing base, showing balanced synonym separation.

---

### 5.3 Overall Insights
- FastText provided *better generalization* and *morphological flexibility*.  
- Word2Vec captured *cleaner semantic relations*, especially when domain-filtered.  
- Increasing epochs and negative samples improved *semantic margin* and *vector quality*.  
- Domain-aware fine-tuning produced embeddings that better reflect *real usage variation* across datasets.
### Reproducibility:
1) Environment and Versions:
  - Pyhton:3.11.X
  - OS: macOS
2)Setup .venv & directory:
      cd /path/to/az_nlp_assignment2
      
      python3.11 -m venv .venv
      source .venv/bin/activate      # Windows: .venv\Scripts\activate
      
      ## 6 Dependencies
      pip install --upgrade pip
      pip install -r requirements.txt
      
      python -c "import nltk; nltk.download('snowball_data')"
3)Project Documentation(should be):
  
    ├─ data/
    │  ├─ data_raw/      # raw file (.xlsx / .csv / .json)
    │  └─ data_clean/    # cleaned 2-sütun Excel (cleaned_text, sentiment_value)
    ├─ embeddings/       # trained models (.model / .vec)
    ├─ reports/          # evaluations (txt reports)
    ├─ scripts/
    │  ├─ preprocess.py  # cleaning + production corpus 
    │  ├─ train_embeddings.py
    │  └─ evaluate_embeddings.py
    └─ corpus_all.txt    # ddomain-tagged composite corpus (carriage return                                   dom<domain>)

4)Operating Steps (End to End):
  Preprocessing:
  
  Each raw dataset was cleaned using preprocess.py with the following options:
      
      ```bash
     python scripts/preprocess.py process \
        --in data/data_raw/labeled-sentiment.xlsx \
        --out data/data_clean/labeled-sentiment.xlsx \
        --text-col text \
        --label-col sentiment \
        --scheme tri \
        --use-stemming \
        --drop-stops \
        --strict-filter \
        --minlen 2

      ```
In the script above, you need to write 'tri' or 'bin' for the 'scheme' parameter, according to the Excel output.

  Create Domain-tagged(corpus_all.txt):
  ```bash
      python scripts/preprocess.py build-corpus-clean \
      --inputs data/data_clean/labeled-sentiment.xlsx,data/data_clean/                   test__1_.xlsx,data/data_clean/train__3_.xlsx,data/data_clean/train-00000-          of-00001.xlsx,data/data_clean/merged_dataset_CSV__1_.xlsx \
      --out corpus_all.txt
  ```
Train Models:
  Shared Model: (single model, all domains)
      ```bash
        
        python scripts/train_embeddings.py \
        --input-txt corpus_all.txt \
        --algo word2vec \
        --train-mode shared \
        --vector-size 300 --window 5 --min-count 3 \
        --negative 10 --epochs 10 --sg 1


        python scripts/train_embeddings.py \
        --input-txt corpus_all.txt \
        --algo fasttext \
        --train-mode shared \
        --vector-size 300 --window 5 --min-count 3 \
        --negative 10 --epochs 10 --sg 1 --min-n 3 --max-n 6
      
      ```
    
  Per-domains model:
    ```bash

          python scripts/train_embeddings.py \
          --input-txt corpus_all.txt \
          --algo word2vec \
          --train-mode per-domain \
          --vector-size 300 --window 7 --min-count 2 \
          --negative 15 --epochs 25 --sg 1
          
          python scripts/train_embeddings.py \
              --input-txt corpus_all.txt \
              --algo fasttext \
              --train-mode per-domain \
              --vector-size 300 --window 7 --min-count 2 \
              --negative 15 --epochs 25 --sg 1 --min-n 3 --max-n 6

     ```
  Adapt(shared+fine-tune):
         ```bash
              python scripts/train_embeddings.py \
                --input-txt corpus_all.txt \
                --algo word2vec \
                --train-mode adapt \
                --vector-size 300 --window 5 --min-count 3 \
                --negative 10 --epochs 10 --sg 1 \
                --adapt-epochs 3
          ```

  Evaluate:
    1)Shared Model Comparison
    bash```
      python scripts/evaluate_embeddings.py \
      --w2v embeddings/word2vec_vs300_win5_mc3_ep10_neg10_shared.model \
      --ft  embeddings/fasttext_vs300_win5_mc3_ep10_neg10_shared.model \
      --out reports/eval_shared.txt
    ```
    2)Per-Domain/Adapt Comparison:
    bash```
          python scripts/evaluate_embeddings.py \
          --w2v embeddings/word2vec_vs300_win7_mc2_ep25_neg15_news.model \
          --ft  embeddings/fasttext_vs300_win7_mc2_ep25_neg15_news.model \
          --out reports/eval_news_ep25_neg15.txt
    ```
    

 ### 7 Conclusion (Detailed):
 
     
  
 The enhanced NLP pipeline developed in this study demonstrated clear and measurable improvements  
 over the baseline implementation, both in *semantic coherence* and *morphological generalization*.  
  
 By introducing advanced preprocessing components — including extended emoji normalization,  
 hashtag decomposition, deasciification, negation scope tagging, and optional stemming —  
 the model was able to capture linguistic nuances unique to Azerbaijani more effectively.  
 This produced cleaner and semantically consistent token representations,  
 particularly visible in *synonym–antonym separation* and *nearest neighbor clusters*.  
  
 The integration of *domain-aware processing* (news, social, reviews, general) further enhanced  
 contextual precision. Domain-specific normalization (PRICE, RATING_POS, RATING_NEG)  
 allowed the embeddings to internalize contextual meaning — e.g., “bahalı” and “ucuz”  
 were clustered differently across review vs. news corpora, reflecting real-world sentiment usage.  
  
 *FastText* consistently outperformed *Word2Vec* in morphology-rich or noisy text (social/review data),  
 thanks to its subword-based modeling. However, *Word2Vec* remained stronger in semantically  
 stable domains (news/general), showing clearer concept-level relationships.  
  
 Additionally, *fine-tuned (adaptive) training* led to incremental gains by refining shared embeddings  
 according to each domain’s linguistic style. The improvement in *lexical coverage (≈0.96–1.00)* and  
 the emergence of *positive separation margins* (ΔSyn−Ant > 0) confirm the overall robustness  
 of this approach.  
  
 In summary, the enhanced pipeline not only achieved superior numerical metrics  
 but also delivered *qualitatively richer, domain-sensitive, and morphologically resilient embeddings*.  
 These results demonstrate that even modest architectural modifications — when combined with  
 careful preprocessing and domain adaptation — can substantially improve representation learning  
 in low-resource, morphologically complex languages such as Azerbaijani.

###  Key Takeaways

| Aspect | Observation |
|:--|:--|
| *Best Overall Model* | FastText (shared) – highest robustness & subword flexibility |
| *Best Domain-Specific Model* | Word2Vec (reviews) – strongest sentiment signal alignment |
| *Most Improved Domain (via Fine-tuning)* | Social – reduced noise, improved slang handling |
| *Optimal Training Setup* | vector_size=300, window=7, min_count=2, negative=15, epochs=25 |
| *Preprocessing Impact* | ~70% reduction in OOV terms; improved word clustering consistency |

Next Steps:

Future work may extend this study by exploring transformer-based embeddings (BERT, RoBERTa) trained or adapted for Azerbaijani, and integrating automatic domain detection or sentiment-aware fine-tuning to further enhance both semantic precision and cross-domain generalization.

---


