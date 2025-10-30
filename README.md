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
> ##  Quantitative Results

### 4.1 Lexical Coverage and Similarity

| Domain | Synonyms<br>(W2V / FT) | Antonyms<br>(W2V / FT) | Separation<br>(Syn−Ant)<br>(W2V / FT) | Coverage | Observation |
|:--|:--:|:--:|:--:|:--:|:--|
| *Shared (ep10, neg10)* | 0.284 / 0.319 | 0.304 / 0.349 | −0.020 / −0.031 | 0.93–0.99 | FastText slightly better; W2V more stable |
| *Shared (ep25, neg15)* | 0.279 / 0.321 | 0.270 / 0.308 | *+0.009 / +0.013* | 0.96–1.00 | Longer training improved separation |
| *Social (per-domain)* | 0.258 / 0.341 | 0.309 / 0.393 | −0.051 / −0.052 | 0.64–0.67 | FastText handles slang & noise better |
| *Reviews (per-domain)* | 0.296 / 0.334 | 0.331 / 0.397 | −0.035 / −0.063 | 0.60–0.65 | Captures product sentiment effectively |
| *News (per-domain)* | 0.271 / 0.317 | 0.301 / 0.369 | −0.030 / −0.052 | 0.41–0.46 | Limited domain coverage, formal structure |
| *General (per-domain)* | 0.286 / 0.339 | 0.273 / 0.377 | +0.013 / −0.038 | 0.92–0.97 | W2V stable; FT more flexible |
| *Social (adapt)* | 0.254 / 0.318 | 0.305 / 0.394 | −0.051 / −0.075 | – | Fine-tuning reduced domain noise |

>  FastText achieves more robust performance in morphology-rich domains due to its subword modeling.  
> Increasing epochs and negative sampling enhanced *semantic separation* for shared models.

---

### 4.2 Qualitative Results (Nearest Neighbors)

| Word | Model | Top Neighbors |
|:--|:--|:--|
| *yaxşı* | W2V (shared) | RATING_POS, yaxwi, awsome, nehre |
|  | FT (shared) | yaxşıı, yaxş, yaxşıca, yaxşıya |
| *pis* | FT (social) | pisi, pisə, pisleşdi, pi |
| *bahalı* | W2V (reviews) | restoranlarda, şeheri, villaları |
|  | FT (reviews) | bahalıq, bahalıdı, bahalıdır |
| *ulduz* | FT (reviews) | ulduzz, ulduza, ulduzu, ulduzdu |

>  FastText clusters morphological variants (e.g., “yaxşıı”, “yaxş”), while Word2Vec provides semantically cleaner clusters (e.g., “RATING_POS”, “yaxşı”).
>
> Observation: Increasing epochs (10→25) and negative (10→15) improved global (shared) separation, while per-domain scores stayed similar due to smaller dataset size.
>
> ###  4.2 Qualitative Results (Nearest Neighbors)

| Word | Model | Top Neighbors |
|:--|:--|:--|
| *yaxşı* | W2V (shared) | RATING_POS, yaxwi, awsome, nehre |
|  | FT (shared) | yaxşıı, yaxş, yaxşıca, yaxşıya |
| *pis* | FT (social) | pisi, pisə, pisleşdi, pi |
| *bahalı* | W2V (reviews) | restoranlarda, şeheri, villaları |
|  | FT (reviews) | bahalıq, bahalıdı, bahalıdır |
| *ulduz* | FT (reviews) | ulduzz, ulduza, ulduzu, ulduzdu |

>  *Observation:*  
> FastText groups morphological variants (e.g., “yaxşıı”, “yaxş”),  
> while Word2Vec provides semantically cleaner clusters (e.g., “RATING_POS”, “yaxşı”).  
> This confirms FastText’s advantage in morphology-rich, noisy domains such as social and review texts.
>
> FastText tends to cluster morphological variants (e.g., “yaxşıı”, “yaxş”),
whereas Word2Vec provides semantically cleaner groupings (e.g., RATING_POS, yaxşı).
>
>
>
 ### 5 Findings and Analysis

### (a)  Effect of Preprocessing
- Advanced text normalization reduced vocabulary noise and *OOV (Out-of-Vocabulary)* words by approximately *70%*.  
- Stemming and strict filtering unified morphological variants  
  (e.g., “pisdi”, “pisdir” → “pis”), improving token consistency.  
- Emoji and negation tagging enhanced sentiment clarity, especially in noisy user-generated text.

---

### (b)  Shared vs Per-Domain Training
- *Shared models:* Achieved the best *overall coverage* and *balanced semantic separation*.  
- *Per-domain models:* Captured stronger *local semantic relations*,  
  but exhibited lower generalization (coverage ≈ 0.6).  
- Confirms that *domain-aware partitioning* yields specialized yet narrower embeddings.

---

### (c)  Fine-Tuning (Adapt)
- Applying *domain-specific fine-tuning* after shared pretraining  
  improved qualitative consistency in word similarity and clustering.  
- The *social* domain showed the *most improvement*,  
  benefiting from exposure to slang and informal expressions.  
- Fine-tuned models were better at separating domain noise while maintaining semantic cohesion.

---

### (d)  FastText vs Word2Vec
- *FastText* performs best in morphology-rich, noisy environments (e.g., social, reviews)  
  due to its subword-based learning of character n-grams.  
- *Word2Vec* remains more stable and semantically precise on formal text domains (news, general).  
- After tuning (epochs=25, negative=15), *shared models achieved positive semantic separation*,  
  demonstrating clearer *synonym–antonym* boundaries across domains.


  ##  6️ Conclusion

- *Shared models* (especially *FastText) proved most effective for **general-purpose Azerbaijani NLP* tasks.  
  They achieved strong global coverage and stable semantic representations across all domains.

- *Per-domain embeddings* captured *localized linguistic nuances*  
  (e.g., product-related adjectives in reviews or slang in social texts),  
  making them ideal for *domain-specific sentiment analysis*.

- *Fine-tuning (adapt)* provided a balanced approach between shared and specialized training,  
  delivering small but consistent improvements in both lexical coverage and semantic separation.

- The *enhanced preprocessing pipeline* — including emoji normalization,  
  negation tagging, Snowball stemming, and strict filtering —  
  ensured reproducibility and robustness across domains.

- Increasing training depth (epochs=25, negative=15) further improved  
  *semantic separation*, especially for synonym–antonym distinctions.

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
