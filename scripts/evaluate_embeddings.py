#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Şimdi burada Eğittiğimiz Word2Vec ve FastText gömme modellerini (embeddings) değerlendireceğiz.
Yani aynı kelime uzayında(vektör temsilinde) iki farklı model,Azerice metinleri ne kadar iyi temsil ediyor? Bunu sorgulayacağız.
Aşağıdaki metrikleri hesaplayacağız:
    1- Lexical Coverage(Kapsama Oranı): Model,bizim oluşturduğumuz datasetteki kelimlerin ne kadarını tanıyor?
    2- Similarity Scores(Benzerlik Skorları): Model,benzer anlamlı kelimeleri yakın, zıt anlamlı kelimeleri ise uzak mı yerleştiriyor?
        Yani semantic ilişkileri doğru bir şekilde öğrenmiş mi?
    3- Nearest Neighbors(En Yakın Komşular): Bazı anahtar kelimeler için en yakın komşuları listeleyeceğiz ve niteliksel olarak inceleyeceğiz. Yani bir kelimenin yakınındaki kelimeler mantıklı mı diye bakacağız.

"""


from pathlib import Path
import argparse
import math
import pandas as pd
from gensim.models import Word2Vec, FastText
from numpy import dot
from numpy.linalg import norm

# ---- Burada dosya yollarını kolayca erişebilmek için tanımlıyoruz ----
CLEAN_DIR = Path("data/data_clean")
FILES = [
    CLEAN_DIR / "labeled-sentiment.xlsx",
    CLEAN_DIR / "test__1_.xlsx",
    CLEAN_DIR / "train__3_.xlsx",
    CLEAN_DIR / "train-00000-of-00001.xlsx",
    CLEAN_DIR / "merged_dataset_CSV__1_.xlsx",
]

# ---- seed words:Bu kelimeleri modelin kalitesini ölçmek için kullanacğız daha sonra(Nearest Neighbors) ----
seed_words = [
    "yaxşı", "pis", "çox", "bahalı", "ucuz", "mükəmməl", "dəhşət",
    "PRICE", "RATING_POS", "RATING_NEG", "ulduz", "Bakı"
]
#Burada synonym ve antonym çiftleri tanımlıyoruz. Bu çiftler üzerinden similarity skorlarını hesaplayacağız.
# Synonym çiftleri için skorun yüksek olması iyi, antonym çiftleri için ise düşük olması iyi bir şey model doğru çalışıyor anlamına geliyor.
syn_pairs = [
    ("yaxşı", "əla"),
    ("bahalı", "qiymətli"),
    ("ucuz", "sərfəli"),
]

ant_pairs = [
    ("yaxşı", "pis"),
    ("bahalı", "ucuz"),
    ("müsbət", "mənfi"),
]

# Lexical Covearage: Datasetteki kelimelerin ne kadarı modelin sözlüğünde(vocab) var?
# Burada dikkat etmemiz gereken nokta Word2Vec ve FastText'in OOV (Out-of-Vocabulary) kelimeleri farklı şekilde ele almasıdır.
#OOV kelimeler, eğitim sırasında görülmemiş ve modelin sözlüğünde bulunmayan kelimelerdir.

# Word2Vec, kelime bazlı çalışır ve mesela “qələm” varsa bilir ama “qələmlər” bilmiyorsa OOV (out-of-vocabulary) olur. OOV kelimeleri tamamen tanımaz ve bu kelimeler için vektör üretmez.
# FastText ise subword bazlı çalışır ve mesela “qələmlər” kelimesini bileşenlerine ayırarak tutar OOV kelimeleri subword (altkelime) bileşenlerine ayırarak vektör üretir.
#O yüzden genelde FastText'in lexical coverage'ı Word2Vec'ten daha yüksek olur.
# Ancak burada her iki model için de sadece vocab'e bakarak kapsama oranını hesaplayacağız.
#Yüksek Coverage, modelin datasetteki kelimeleri tanıdığını gösterir.
def lexical_coverage(model, tokens):
    vocab = model.wv.key_to_index
    if not tokens: return 0.0
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))
#read_tokens: Excel dosyasından cleaned_text feature'ındaki tüm tokenleri okur ve bir liste olarak döner.
def read_tokens(xlsx_path: Path):
    df = pd.read_excel(xlsx_path, usecols=["cleaned_text"])
    toks = []
    for row in df["cleaned_text"].astype(str):
        toks.extend(row.split())
    return toks
#cos: İki vektör arasındaki kosinüs benzerliğini hesaplar.
def cos(a, b):
    denom = (norm(a) * norm(b))
    if denom == 0: return float("nan")
    return float(dot(a, b) / denom)
#pair_sim fonskiyonu : Verilen kelime çiftleri listesindeki her çift için modelin similarity skorlarını hesaplar ve ortalamasını döner.
#Parametreler-> 
# model: Modelleri vereceğiz ayrı ayrı similarity skorlarını hesaplamak için.
# pairs: Kelime çiftleri listesi (syn_pairs veya ant_pairs) vereceğiz
#Return value-> Ortalama similarity skoru (float) olucak. Ve eğer tüm çiftler OOV ise NaN döner.
#Burada mesela W2V modelini ve synonym çiftlerini verince, modelin bu synonym çiftleri için hesapladığı similarity skorlarının ortalamasını döneeceğiz.
#Skoru daha yüksek olan model,synonym çiftlerini daha yakın yerleştiriyor demektir.Daha iyi öğrenmiş demektir anlamına gelir.
#Antonym çiftleri için ise skorun daha düşük olması istenir.
def pair_sim(model, pairs):
    vals = []
    for a, b in pairs:
        try:
            vals.append(model.wv.similarity(a, b))#model.wv.similarity gensim'in içinde cosine similarity hesaplayan fonksiyondur.
        except KeyError:
            # out-of-vocabulary: skip
            pass
    return sum(vals) / len(vals) if vals else float("nan")
#neighbors: Verilen kelime için modelin en yakın k komşusunu döner.
#Burada baktığımız şey şu: Mesela "yaxşı" kelimesi için modelin en yakın 5 komşusu nedir? Bu komşular mantıklı mı?
#Eğer model iyi çalışmışsa, "yaxşı" kelimesinin yakınında "əla", "mükəmməl", "gözəl" gibi olumlu anlamlı kelimeler olmalı.
#Parametreler->
# model: Word2Vec veya FastText modeli
# word: Anahtar kelime
# k: Döndürülecek en yakın komşu sayısı (default 5)
#Return value-> En yakın k komşu kelime listesi
def neighbors(model, word, k=5):#default olarak 5 komşu döneceğiz.
    try:
        return [w for w, _ in model.wv.most_similar(word, topn=k)]#modeş.wv.most_similar gensim'in içinde en yakın komşuları bulan fonksiyondur.Vektör uzayına bakar ve en yakın olanları döner.
    except KeyError:
        return []

# ---- corpus_all.txt için: dom etiketlerini oku, tokenları domain’e göre böl ----
def parse_corpus_with_domains(path: Path):
    """
    Returns:
        rows: list[(domain, tokens)]
        by_dom: dict[str, list[list[str]]]
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
                dom = toks[0][3:]  # domreviews -> reviews
                toks = toks[1:]    # dom tokenini at
            if toks:
                rows.append((dom, toks))
    by_dom = {}
    for dom, toks in rows:
        by_dom.setdefault(dom, []).append(toks)
    return rows, by_dom

def flatten(list_of_lists):
    out = []
    for lst in list_of_lists:
        out.extend(lst)
    return out    

def main():
    ap = argparse.ArgumentParser(description="Evaluate Azerbaijani Word2Vec & FastText embeddings.")
    ap.add_argument("--w2v", required=True, help="Path to Word2Vec .model file")
    ap.add_argument("--ft", required=True, help="Path to FastText .model file")
    ap.add_argument("--out", default="evaluation_report.txt", help="Output report path")
    ap.add_argument("--corpus", type=str, default=None,
                    help="Optional path to corpus_all.txt for DOMAIN-WISE evaluation")
    ap.add_argument("--nn-k", type=int, default=5, help="Top-N neighbors")
    args = ap.parse_args()

    W2V_PATH = Path(args.w2v)
    FT_PATH  = Path(args.ft)

    print(f"[INFO] Loading models:\n  W2V: {W2V_PATH}\n  FT : {FT_PATH}")
    w2v = Word2Vec.load(str(W2V_PATH))
    ft  = FastText.load(str(FT_PATH))

    report_lines = []

    # -------------------- GENEL RAPOR (cleaned Excels) --------------------
    report_lines.append("== Lexical coverage per dataset ==")
    for f in FILES:
        toks = read_tokens(f)
        cov_w2v = lexical_coverage(w2v, toks)
        cov_ft  = lexical_coverage(ft, toks)
        line = f"{f.name:32s}  W2V(vocab)={cov_w2v:.3f}  FT(vocab)={cov_ft:.3f}"
        print(line)
        report_lines.append(line)

    report_lines.append("\n== Similarity (higher better for synonyms; lower better for antonyms) ==")
    syn_w2v = pair_sim(w2v, syn_pairs)
    syn_ft  = pair_sim(ft,  syn_pairs)
    ant_w2v = pair_sim(w2v, ant_pairs)
    ant_ft  = pair_sim(ft,  ant_pairs)
    report_lines.append(f"Synonyms : W2V={syn_w2v:.3f}  FT={syn_ft:.3f}")
    report_lines.append(f"Antonyms : W2V={ant_w2v:.3f}  FT={ant_ft:.3f}")
    report_lines.append(f"Separation (Syn - Ant): W2V={(syn_w2v - ant_w2v):.3f}  FT={(syn_ft - ant_ft):.3f}")

    report_lines.append("\n== Nearest neighbors (qualitative) ==")
    for w in seed_words:
        n_w2v = neighbors(w2v, w, k=args.nn_k)
        n_ft  = neighbors(ft,  w, k=args.nn_k)
        report_lines.append(f" W2V NN for '{w}': {n_w2v}")
        report_lines.append(f"  FT NN for '{w}': {n_ft}")

    # -------------------- DOMAIN-WISE RAPOR (corpus_all.txt varsa) --------------------
    if args.corpus:
        report_lines.append("\n\n================ DOMAIN-WISE REPORT ================\n")
        rows, by_dom = parse_corpus_with_domains(Path(args.corpus))

        # 1) Coverage per domain
        report_lines.append("== Lexical coverage per domain ==")
        for dom in sorted(by_dom.keys()):
            toks_dom = flatten(by_dom[dom])  # list[list[str]] -> list[str]
            cov_w2v = lexical_coverage(w2v, toks_dom)
            cov_ft  = lexical_coverage(ft, toks_dom)
            line = f"{dom:10s}  W2V(vocab)={cov_w2v:.3f}  FT(vocab)={cov_ft:.3f}  (tokens={len(toks_dom):,})"
            print(line)
            report_lines.append(line)

        # 2) Similarity per domain
        report_lines.append("\n== Similarity per domain (syn/ant + separation) ==")
        for dom in sorted(by_dom.keys()):
            # not: pair_sim domain’e özgü kelime havuzunu zorunlu kılmaz;
            # ancak domain analizini tutarlı yapmak için OOV kelimeleri doğal olarak atlar.
            syn_w2v = pair_sim(w2v, syn_pairs)
            syn_ft  = pair_sim(ft,  syn_pairs)
            ant_w2v = pair_sim(w2v, ant_pairs)
            ant_ft  = pair_sim(ft,  ant_pairs)
            sep_w2v = (syn_w2v - ant_w2v) if (syn_w2v == syn_w2v and ant_w2v == ant_w2v) else float("nan")
            sep_ft  = (syn_ft  - ant_ft ) if (syn_ft  == syn_ft  and ant_ft  == ant_ft ) else float("nan")
            report_lines.append(f"{dom:10s}  Syn: W2V={syn_w2v:.3f} FT={syn_ft:.3f} | "
                                f"Ant: W2V={ant_w2v:.3f} FT={ant_ft:.3f} | "
                                f"Sep: W2V={sep_w2v:.3f} FT={sep_ft:.3f}")

        # 3) (Opsiyonel) Domain-wise nearest neighbors (aynı seed set ile)
        report_lines.append("\n== Nearest neighbors per domain (qualitative, shared models) ==")
        for dom in sorted(by_dom.keys()):
            report_lines.append(f"[Domain: {dom}]")
            for w in seed_words:
                report_lines.append(f"  W2V NN for '{w}': {neighbors(w2v, w, k=args.nn_k)}")
                report_lines.append(f"   FT NN for '{w}': {neighbors(ft,  w, k=args.nn_k)}")

    # Raporu yaz
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[OK] Evaluation complete. Report saved to {out_path}")

if __name__ == "__main__":
    main()
