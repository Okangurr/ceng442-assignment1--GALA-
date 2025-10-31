#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Word2Vec / FastText embeddings on Azerbaijani data (domain-aware ready)
=============================================================================

Kullanım örnekleri
------------------
# 1) Cleaned Excel'lerden tek model (shared)
python scripts/train_embeddings.py --algo word2vec

# 2) corpus_all.txt'ten tek model (shared), dom etiketleri eğitimden çıkarılır
python scripts/train_embeddings.py --algo fasttext --input-txt corpus_all.txt

# 3) corpus_all.txt'ten per-domain 4 ayrı model
python scripts/train_embeddings.py --input-txt corpus_all.txt --train-mode per-domain

# 4) corpus_all.txt'ten shared + domain-adaptive fine-tuning
python scripts/train_embeddings.py --input-txt corpus_all.txt --train-mode adapt --adapt-epochs 3
"""

import argparse
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec, FastText



def read_cleaned_excels(clean_dir: Path):
    """data_clean klasöründeki tüm Excel'leri oku ve token listesi üret (dom yok)."""
    sentences = []
    for f in sorted(clean_dir.glob("*.xlsx")):
        df = pd.read_excel(f, usecols=["cleaned_text"])
        toks = (
            df["cleaned_text"]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda s: s.str.len() > 0]
            .str.split()
            .tolist()
        )
        sentences.extend(toks)
    return sentences


def parse_corpus_with_domains(path: Path):
    """
    corpus_all.txt'i oku → [(domain, tokens), ...]
    Satır başındaki 'dom<domain>' tokenini eğitimden çıkarır ama domain'i saklar.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            toks = ln.split()
            dom = "general"
            if toks and toks[0].startswith("dom"):
                dom = toks[0][3:]  # 'domreviews' -> 'reviews'
                toks = toks[1:]    # dom tokenini eğitimden at
            if toks:
                rows.append((dom, toks))
    return rows


def group_by_domain(rows):
    by_dom = {"news": [], "social": [], "reviews": [], "general": []}
    for dom, toks in rows:
        by_dom.setdefault(dom, []).append(toks)
    return by_dom


#Model kurma 

def build_model(sentences, algo="word2vec",
                vector_size=300, window=5, min_count=2,
                epochs=20, negative=8, sg=1, seed=42,
                min_n=3, max_n=6):
    """Verilen cümlelerle W2V/FT eğit ve modeli döndür."""
    if algo == "word2vec":
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            negative=negative,
            min_count=min_count,
            sg=sg,
            epochs=epochs,
            workers=4,
            seed=seed,
            alpha=0.05,
            min_alpha=0.0005,
        )
    elif algo == "fasttext":
        model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            negative=negative,
            min_count=min_count,
            sg=sg,
            epochs=epochs,
            workers=4,
            seed=seed,
            min_n=min_n,
            max_n=max_n,
            alpha=0.05,
            min_alpha=0.0005,
        )
    else:
        raise ValueError("Unsupported algo: choose 'word2vec' or 'fasttext'")
    return model


def save_model(model, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / f"{name}.model"))
    model.wv.save_word2vec_format(str(out_dir / f"{name}.vec"))
    print(f"[OK] Saved: {out_dir / (name + '.model')}")
    print(f"[OK] Saved: {out_dir / (name + '.vec')}")




def main():
    ap = argparse.ArgumentParser(description="Train Word2Vec / FastText (domain-aware training modes).")
    ap.add_argument("--input-txt", type=str, default=None,
                    help="Path to corpus_all.txt (opsiyonel; yoksa cleaned Excels okunur).")
    ap.add_argument("--algo", choices=["word2vec", "fasttext"], default="word2vec")
    ap.add_argument("--vector-size", type=int, default=300)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--negative", type=int, default=8)
    ap.add_argument("--sg", type=int, default=1, help="1=SkipGram, 0=CBOW")
    ap.add_argument("--min-n", type=int, default=3, help="(FastText) n-gram min")
    ap.add_argument("--max-n", type=int, default=6, help="(FastText) n-gram max")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="embeddings")

    # Domain-aware modlar
    ap.add_argument("--train-mode", choices=["shared", "per-domain", "adapt"], default="shared",
                    help="shared: tek model; per-domain: her domain için model; adapt: shared + domain fine-tune")
    ap.add_argument("--adapt-epochs", type=int, default=3,
                    help="--train-mode adapt için domain başına devam eğitim epoch sayısı")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    # --------- Veriyi oku ---------
    if args.input_txt:
        rows = parse_corpus_with_domains(Path(args.input_txt))  # [(dom, toks)]
        by_dom = group_by_domain(rows)
        all_sentences = [t for _, t in rows]
    else:
        # Cleaned Excel'den okuma (dom bilgisi yok -> shared)
        all_sentences = read_cleaned_excels(Path("data/data_clean"))
        by_dom = None  # per-domain/adapt modları yalnızca corpus_all ile mantıklı

    # --------- Eğitim modları ---------
    if args.train_mode == "shared":
        print(f"\n[INFO] Training SHARED model on {len(all_sentences):,} sentences (dom tokenleri eğitim dışı).")
        model = build_model(
            all_sentences, algo=args.algo,
            vector_size=args.vector_size, window=args.window, min_count=args.min_count,
            epochs=args.epochs, negative=args.negative, sg=args.sg, seed=args.seed,
            min_n=args.min_n, max_n=args.max_n
        )
        tag = f"{args.algo}_vs{args.vector_size}_win{args.window}_mc{args.min_count}_ep{args.epochs}_neg{args.negative}_shared"
        save_model(model, out_dir, tag)

    elif args.train_mode == "per-domain":
        if by_dom is None:
            raise SystemExit("[ERR] --train-mode per-domain için --input-txt (corpus_all.txt) gerekli.")
        for dom, sents in by_dom.items():
            if not sents:
                print(f"[WARN] No sentences for domain '{dom}', skipping.")
                continue
            print(f"\n[INFO] Training PER-DOMAIN model for '{dom}' on {len(sents):,} sentences.")
            model = build_model(
                sents, algo=args.algo,
                vector_size=args.vector_size, window=args.window, min_count=args.min_count,
                epochs=args.epochs, negative=args.negative, sg=args.sg, seed=args.seed,
                min_n=args.min_n, max_n=args.max_n
            )
            tag = f"{args.algo}_vs{args.vector_size}_win{args.window}_mc{args.min_count}_ep{args.epochs}_neg{args.negative}_{dom}"
            save_model(model, out_dir, tag)

    elif args.train_mode == "adapt":
        if by_dom is None:
            raise SystemExit("[ERR] --train-mode adapt için --input-txt (corpus_all.txt) gerekli.")
        # 1) Shared ön-eğitim
        print(f"\n[INFO] Training SHARED base model (for adapt) on {len(all_sentences):,} sentences.")
        base = build_model(
            all_sentences, algo=args.algo,
            vector_size=args.vector_size, window=args.window, min_count=args.min_count,
            epochs=args.epochs, negative=args.negative, sg=args.sg, seed=args.seed,
            min_n=args.min_n, max_n=args.max_n
        )
        base_tag = f"{args.algo}_vs{args.vector_size}_win{args.window}_mc{args.min_count}_ep{args.epochs}_neg{args.negative}_shared"
        save_model(base, out_dir, base_tag)

        # 2) Her domain için shared'ı yeniden yükleyip kısa bir devam eğitimi uygula
        base_path = out_dir / f"{base_tag}.model"
        for dom, sents in by_dom.items():
            if not sents:
                print(f"[WARN] No sentences for domain '{dom}', skipping adapt.")
                continue
            print(f"[INFO] Adapting model for '{dom}' with {len(sents):,} sentences (epochs={args.adapt_epochs})")
            model_dom = (Word2Vec.load if args.algo == "word2vec" else FastText.load)(str(base_path))
            model_dom.train(sents, total_examples=len(sents), epochs=args.adapt_epochs)
            tag = f"{base_tag}_{dom}_adapt{args.adapt_epochs}"
            save_model(model_dom, out_dir, tag)

    print("\n[DONE] Training workflow completed.\n")


if __name__ == "__main__":
    main()
