#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mini Hyperparameter Search for Azerbaijani Embeddings
-----------------------------------------------------
Bu script, Word2Vec modellerini farklı hiperparametrelerle eğitir
ve evaluation (similarity separation) skorlarını ölçer.
"""

from pathlib import Path
import itertools
import pandas as pd
from gensim.models import Word2Vec
from evaluate_embeddings import pair_sim, ant_pairs, syn_pairs   # önceki script'ten import
from evaluate_embeddings import lexical_coverage, read_tokens

vector_sizes = [100, 200]
windows = [3, 5]
epochs_list = [10, 20]
min_counts = [2, 3]

CORPUS_PATH = Path("corpus_all.txt")
CLEAN_DIR = Path("data/data_clean")
FILES = [
    CLEAN_DIR / "labeled-sentiment.xlsx",
    CLEAN_DIR / "test__1_.xlsx",
]

OUT_CSV = Path("tuning_results.csv")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    sentences = [line.strip().split() for line in f if line.strip()]

results = []

for vs, win, ep, mc in itertools.product(vector_sizes, windows, epochs_list, min_counts):
    print(f"\n=== Training Word2Vec (vs={vs}, win={win}, ep={ep}, mc={mc}) ===")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vs,
        window=win,
        min_count=mc,
        sg=1,
        epochs=ep,
        workers=4,
        seed=42,
    )

    syn_score = pair_sim(model, syn_pairs)
    ant_score = pair_sim(model, ant_pairs)
    sep = syn_score - ant_score

    results.append({
        "vector_size": vs,
        "window": win,
        "epochs": ep,
        "min_count": mc,
        "synonyms": round(syn_score, 3),
        "antonyms": round(ant_score, 3),
        "separation": round(sep, 3),
    })

df = pd.DataFrame(results)
df = df.sort_values(by="separation", ascending=False)
print("\n=== Summary (sorted by separation) ===")
print(df)

df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"\n[OK] Saved tuning results to {OUT_CSV}")
