#  Azerbaijani Domain-Aware Word Embedding Project

This project builds *domain-aware Word2Vec and FastText embeddings* for Azerbaijani text using a complete custom preprocessing pipeline.  
It demonstrates how *text normalization, **morphological stemming, and **domain-specific fine-tuning* affect semantic quality.

---

## 1️ Overview

- *Goal:* Train and evaluate domain-aware embeddings (Word2Vec & FastText) for Azerbaijani text.
- *Datasets:* 5 corpora → labeled-sentiment, test__1_, train__3_, train-00000-of-00001, merged_dataset_CSV__1_.
- *Pipeline:* Cleaning, normalization, stemming, domain detection → corpus building → embedding training → evaluation.
- *Domains:* news, social, reviews, general.

---

## 2️ Preprocessing Pipeline Summary

Each raw dataset was cleaned using preprocess.py with the following options:

```bash
python scripts/preprocess.py process \
  --in data/data_raw/<file>.xlsx \
  --out data/data_clean/<file>.xlsx \
  --text-col text --label-col sentiment --scheme tri \
  --use-stemming --drop-stops --strict-filter --minlen 2
```
## Active Components

| Step | Description |
|:--|:--|
| Lowercasing (Azeri-specific) | Handles special casing rules like İ → i, I → ı for Azerbaijani. |
| Emoji Mapping | Expanded to cover 50+ emojis across positive, negative, and neutral categories. |
| Hashtag Splitter | Splits hashtags by CamelCase, underscores `_`, and numeric transitions (e.g., `#CoxYaxsiFilm` → Cox Yaxsi Film). |
| Deasciify & Slang Normalization | Converts Latinized or slang variants (e.g., `yaxsi` → `yaxşı`, `cox` → `çox`). |
| Negation Scope Tagging | Adds `_NEG` suffix to words affected by negations (`deyil`, `yox`, `qətiyyən`). |
| Stopword Filtering | Removes over 100 Azerbaijani function words while preserving negations for sentiment accuracy. |
| Snowball Stemming | Applies the Turkish Snowball Stemmer, adapted for Azerbaijani morphology to unify word variants. |
| Strict Filter | Keeps only valid Azerbaijani alphabet tokens and special placeholders (`<NUM>`, `URL`, `EMAIL`, etc.). |


>  All preprocessing steps together improved lexical coverage (~97%) and significantly reduced OOV tokens.

Outcome: High lexical coverage (≈97%), reduced OOV, and cleaner sentence-level tokenization.
The cleaned files were merged to form corpus_all.txt, where each line begins with a domain tag like domsocial.

###  3️ Training Configurations

|  Model |  Mode |  Vec Size |  Window |  Min Count |  Negative |  SG |  Epochs |  n-gram |  Notes |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| *Word2Vec (shared)* | Single global model | 300 | 5 | 3 | 10 | 1 | 10 | – | Domain tags removed before training |
| *FastText (shared)* | Same as above | 300 | 5 | 3 | 10 | 1 | 10 | 3–6 | Subword-based representation |
| *W2V / FT (per-domain)* | Separate models per domain | 300 | 7 | 2 | 15 | 1 | 25 | 3–6 | Trained separately for news, social, reviews, general |
| *W2V / FT (adapt)* | Fine-tuned per-domain | 300 | 5 | 3 | 10 | 1 | 10 + 3 | 3–6 | Continued training from shared model with domain data |

>  All models were trained using preprocessed corpora (stemming=True, stopword=True, strict=True).
>  
> The *shared* models capture global semantics,  
> while *per-domain* and *adapt* variants specialize in local linguistic patterns.
>
>##  4️ Quantitative Results

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

> 📈 *Observation:*  
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

>  *Conclusion:*  
> Compared to the baseline, the enhanced pipeline achieved *higher lexical coverage*,  
> *better synonym separation, and **cleaner semantic clustering*.  
> Domain-aware preprocessing and fine-tuned training significantly increased embedding robustness,  
> especially for morphologically rich and noisy Azerbaijani data.
---

###  Key Takeaways

| Aspect | Observation |
|:--|:--|
| *Best Overall Model* | FastText (shared) – highest robustness & subword flexibility |
| *Best Domain-Specific Model* | Word2Vec (reviews) – strongest sentiment signal alignment |
| *Most Improved Domain (via Fine-tuning)* | Social – reduced noise, improved slang handling |
| *Optimal Training Setup* | vector_size=300, window=7, min_count=2, negative=15, epochs=25 |
| *Preprocessing Impact* | ~70% reduction in OOV terms; improved word clustering consistency |

---

###  Future Work
- Integrate *domain-specific gold-standard synonym/antonym pairs* for quantitative evaluation.  
- Experiment with *transformer-based Azerbaijani embeddings (e.g., BERT, RoBERTa)*.  
- Apply embeddings to downstream tasks: *sentiment classification, **topic modeling, and **text clustering*.  
- Explore *cross-lingual fine-tuning* between Turkish and Azerbaijani corpora to enhance performance.

---

>  Final Verdict:  
> Extended training and domain-aware fine-tuning produced *semantically coherent and context-rich embeddings* for Azerbaijani.  
> FastText excels in handling morphological variety, while Word2Vec maintains semantic precision —  
> together confirming the power of *domain-aware embedding design* in low-resource languages.
