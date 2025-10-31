#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Azerbaijani Text Preprocessing + Domain-aware utilities
======================================================

Komutlar
--------
1) Tek dosya temizle -> 2 sütunlu Excel (cleaned_text, sentiment_value)
   scheme=tri (neg=0.0, neu=0.5, pos=1.0) | scheme=binary (neg=0.0, pos=1.0)
   Örnek:
   python scripts/preprocess.py process \
     --in data_raw/labeled-sentiment.xlsx \
     --out data_clean/labeled-sentiment.xlsx \
     --text-col text \
     --label-col sentiment \
     --scheme tri \
     --strict-filter --minlen 2

2) Birleşik korpus üret (ham dosyalardan; domain-tagged, lowercase)
   Örnek:
   python scripts/preprocess.py build-corpus \
     --inputs data_raw/labeled-sentiment.xlsx,data_raw/test__1_.xlsx,data_raw/train__3_.xlsx,data_raw/train-00000-of-00001.xlsx,data_raw/merged_dataset_CSV__1_.xlsx \
     --text-cols text,text,text,text,text \
     --label-cols sentiment,label,label,labels,labels \
     --schemes tri,binary,binary,tri,binary \
     --out corpus_all.txt

3) (Opsiyonel) Temizlenmiş Excel’lerden korpus üret (cleaned_text’ten)
   Örnek:
   python scripts/preprocess.py build-corpus-clean \
     --inputs data_clean/labeled-sentiment.xlsx,data_clean/test__1_.xlsx,data_clean/train__3_.xlsx,data_clean/train-00000-of-00001.xlsx,data_clean/merged_dataset_CSV__1_.xlsx \
     --out corpus_all.txt

Notlar
------
- Excel çıktısı yalnızca `cleaned_text` ve `sentiment_value` sütunlarını içerir.
- Domain etiketi Excel'e yazılmaz; `corpus_all.txt` üretirken satır başına eklenir (domnews/domsocial/domreviews/domgeneral).
"""

import argparse, re, sys, json, html, unicodedata
from pathlib import Path
from typing import List
import pandas as pd



#Şimdi burada yapacağımız şey Stemming/Lemmatization mantığını eklemek
#Azerice Türkçe gibi sondan eklemeli bir dildir yani bir kelime onlarca farklı formda karşımıza çıkabilir.
#Yani bu kelimeler modele sanki farklı kelimelermiş gibi gelir ve farklı tokenler olarak öğrenir ve embedding vektör uzayında mesela yaxşı ile yaxşıdır'ı yakın yerleştirmeyi öğrenmekte zorlanabilir 
# Bu yüzden stemming/lemmatization işlemi kelime dağarcığını normalize eder.
# Yani aynı anlamı kelimeler tek temsile-cluster'a indirgenecek
# Daha semantic vektörler olucak “yaxşıdır”, “yaxşıydı”, “yaxşılaşdı” → aynı köke oturacak.
#Bu bizim modelimin daya iyi similarity sonucunun olacağı ve Noise(gürültü) azalacağından modelin daha temiz dil istatistikleri olacağını öngörüyoruz bu sayede
#Stemming yaparsak sadece sonda yakaladığı ek olarak aldığıladığı ekleri atacak bu kelimenin kökünü yanlış yakalayabileceği anlamına gelir Bunu da kullanabiliriz ama biz başlangıç olarak daha iyi ayrım yapan Lemmatization'ı kullanalım ve bakalım.Ama tabiki Lemmatization'da yavaş çalışabilir bakarak karar verelim ikisine de hangisi işimize gelirse
#Burada başlangıçta Snowball Stemmer deneyeceğiz çünkü lemmatization için Stanzanın PyTorch indirmemiz gerek
#Azerice bir nltk stemming kütüphanesi bulamadık ama Türkçe ile Azerice çok benzer olduğu için çalışır.

# --- Stemming (Türkçe tabanlı, Azerice ile uyumlu) ---
try:
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("turkish")
except Exception:
    stemmer = None

def stem_az(tokens: List[str]) -> List[str]:
    """Azeri kelimeleri Türkçe SnowballStemmer ile köklerine indirger."""
    if not stemmer:
        return tokens
    return [stemmer.stem(t) for t in tokens]


# --- optional ftfy (mojibake fix) ---
try:
    from ftfy import fix_text  # type: ignore
except Exception:
    def fix_text(s: str) -> str:
        return s

# ========== 1) Azerbaycanca lower + temel regexler ==========

def lower_az(s: str) -> str:
    """Azeri/Türk dillerine duyarlı küçük harf: İ→i, I→ı, sonra lower()."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("İ", "i").replace("I", "ı")
    return s.lower()

HTML_TAG_RE   = re.compile(r"<[^>]+>")
URL_RE        = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE      = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE      = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
USER_RE       = re.compile(r"@\w+")
MULTI_PUNCT   = re.compile(r"([!?.,;:])\1{1,}")
MULTI_SPACE   = re.compile(r"\s+")
REPEAT_CHARS  = re.compile(r"(.)\1{2,}", flags=re.UNICODE)  # ≥3 tekrarı 2'ye indir
DIGIT_RE      = re.compile(r"\b\d+(?:[.,]\d+)?\b")

# Azerbaycanca + x,q dahil, kelime karakter seti dışındaki her şeyi temizlemek için:
NONWORD_RE = re.compile(r"[^\wəğıöüçşxq\s]", flags=re.UNICODE)

# Token çıkarma için (kelime + özel placeholder’lar):
TOKEN_RE = re.compile(
    r"[A-Za-zəöğçışüİıÖĞÇŞƏXxQq]+|<NUM>|URL|EMAIL|PHONE|USER|EMO_(?:POS|NEG)",
    flags=re.UNICODE
)

# ========== 2) Mini: emoji, hashtag split, stopwords, negation-scope, deasciify ==========

EMO_MAP = {
    # --- POSITIVE EMOJIS ---
    "😀":"EMO_POS","😃":"EMO_POS","😄":"EMO_POS","😁":"EMO_POS","😆":"EMO_POS",
    "😅":"EMO_POS","😂":"EMO_POS","🤣":"EMO_POS","😊":"EMO_POS","🙂":"EMO_POS",
    "😉":"EMO_POS","😍":"EMO_POS","😘":"EMO_POS","😚":"EMO_POS","😋":"EMO_POS",
    "😎":"EMO_POS","🤩":"EMO_POS","😻":"EMO_POS","🥰":"EMO_POS","🤗":"EMO_POS",
    "💖":"EMO_POS","💗":"EMO_POS","💓":"EMO_POS","💞":"EMO_POS","💯":"EMO_POS",
    "👍":"EMO_POS","👌":"EMO_POS","✌️":"EMO_POS","👏":"EMO_POS","🙌":"EMO_POS",
    "🥳":"EMO_POS","🎉":"EMO_POS","🎊":"EMO_POS","🏆":"EMO_POS","💪":"EMO_POS",
    "🔥":"EMO_POS","⭐":"EMO_POS","🌟":"EMO_POS","🫶":"EMO_POS",

    # --- NEGATIVE EMOJIS ---
    "😞":"EMO_NEG","😟":"EMO_NEG","😠":"EMO_NEG","😡":"EMO_NEG","😔":"EMO_NEG",
    "😕":"EMO_NEG","🙁":"EMO_NEG","😣":"EMO_NEG","😖":"EMO_NEG","😫":"EMO_NEG",
    "😩":"EMO_NEG","😢":"EMO_NEG","😭":"EMO_NEG","😤":"EMO_NEG","🤬":"EMO_NEG",
    "👎":"EMO_NEG","😒":"EMO_NEG","😓":"EMO_NEG","🥺":"EMO_NEG","🤢":"EMO_NEG",
    "🤮":"EMO_NEG","😱":"EMO_NEG","😰":"EMO_NEG","😨":"EMO_NEG","😥":"EMO_NEG",
    "💔":"EMO_NEG","😶‍🌫️":"EMO_NEG","😵":"EMO_NEG","😵‍💫":"EMO_NEG",

    # --- NEUTRAL / MIXED EMOJIS ---
    "🤔":"EMO_NEU","😐":"EMO_NEU","😑":"EMO_NEU","🤨":"EMO_NEU",
    "🤯":"EMO_NEU","😶":"EMO_NEU","🤫":"EMO_NEU","🤭":"EMO_NEU",
    "🤝":"EMO_NEU","✋":"EMO_NEU","🤷":"EMO_NEU","🤦":"EMO_NEU",
}


SLANG_MAP = {
    "sagol":"sağol","cox":"çox","yaxsi":"yaxşı","tesekkur":"təşəkkür","salam":"salam","tamam":"tamam",
    # TR->AZ / varyant takviyeleri
    "iyi":"yaxşı","kötü":"pis","kotu":"pis","super":"süper","cok":"çox","çok":"çox",
    "yaxshi":"yaxşı","yaxşi":"yaxşı","yaxsı":"yaxşı","coox":"çox","coxx":"çox","cooox":"çox",
    "bahali":"bahalı","pahalı":"bahalı","pahali":"bahalı",
    "mukemmel":"mükəmməl","mukəmməl":"mükəmməl","mükemmel":"mükəmməl",
    "pisdi":"pisdir","pisdii":"pisdir"
}



CAMEL_SPLIT_RE = re.compile(r"([a-zəğıöüçşxq])([A-Z])")

def map_emojis(s: str) -> str:
    return "".join(EMO_MAP.get(ch, ch) for ch in s)


def split_hashtags(s: str) -> str:
    """
    Geliştirilmiş hashtag ayrıştırıcı:
    - '#' karakterini kaldırır
    - CamelCase veya PascalCase ayrımı yapar (#CoxYaxsiFilm → Cox Yaxsi Film)
    - Sayıları ve kelimeleri ayırır (#iphone14 → iphone 14)
    - Altçizgi '_' varsa ayırır (#cox_yaxsi → cox yaxsi)
    - Unicode Azerice karakterleri destekler
    """
    def _split(m):
        word = m.group(1)
        # CamelCase ayır
        word = CAMEL_SPLIT_RE.sub(r"\1 \2", word)
        # Alt çizgi temizle
        word = word.replace("_", " ")
        # Harf-sayı geçişlerinde boşluk ekle (ör: iphone14 -> iphone 14)
        word = re.sub(r"(?<=[A-Za-zəöğçışüİıÖĞÇŞƏXxQq])(?=\d)", " ", word)
        word = re.sub(r"(?<=\d)(?=[A-Za-zəöğçışüİıÖĞÇŞƏXxQq])", " ", word)
        return " " + word + " "
    
    # '#' ile başlayan tüm yapıları bul
    return re.sub(r"#([A-Za-z0-9_ğüşiöçıƏəİıŞşÇçĞğÖöÜüXxQq]+)", _split, s)

#Şimdi bizim sırada yapacağımız şey Stopwords filtering yani stopwords'leri atmak ama bu tür NLP sentiments analysis problemlerinde bu stopwordsleri atarken dikkat etmek gerekiyor çünkü bazı cümlelerde bu stopwordsler normalde olduğunun aksine duygu içerebiliyor.
#Bu yüzden stopwords fitering yaparken negation'ları korumak ve bunu detaylı dikkatki şekilde yapmak 
STOP_CANDIDATES = {
    # Bağlaçlar
    "və","amma","ancaq","lakin","amma","ya","ya da","də","da","ki",
    # Zamirler
    "mən","sən","o","biz","siz","onlar","məni","onu","bunu",
    # Belirleyiciler
    "bir","hər","bu","o","belə","bütün","başqa","öz","hansı",
    # Edatlar
    "ilə","üçün","kimi","gibi","barədə","üzrə","doğru","tərəf","sonra","əvvəl",
    # Zaman zarfları
    "indi","sonra","daha","artıq","biraz","çox","az","yenə","hələ",
    # Yardımcı fiiller
    "idi","dir","deyil","var","yoxdur","oldu","olar","etdi","edib",
    # Diğer dolgu kelimeler
    "nə","necə","çünki","bax","aha","demək","beləliklə"
}


DEASCI_SIMPLE = {
    "cox":"çox","yaxsi":"yaxşı","qebul":"qəbul","hec":"heç","duz":"düz","olke":"ölkə",
    "saglamliq":"sağlamlıq","muellim":"müəllim","tehsil":"təhsil"
}

def deasciify_token(tok: str) -> str:
    tok = SLANG_MAP.get(tok, tok)
    return DEASCI_SIMPLE.get(tok, tok)

#Negation Scope Nedir: Bir cümlenin içinde olumsuzluk belirten kelimelere(negators) denir. Bu kelimeler aslında kendineden önce veya sonra gelen kelimelerin anlamını terse çevirir.
#Şimdi burada bizim yapmamız gereken şey negationlar bildiğimiz gibi kendinden önceki(deyil) veya sonraki(yox,yoxdur,qətiyyən) kelimelere sıfatlara falan olumsuzluk anlamı verir ama bazıları da mesela başka bir negasyonlarla kullanılır(hiç gibi) bunlara da sadece işaretleme yerine flag tutmamız gerek
NEGATORS = {"yox","deyil","heç","qətiyyən","yoxdur","yoxdu","deyildi","heçə"}
def mark_negation_scope(tokens: List[str], window: int = 3) -> List[str]:
    """
    Azerice için doğal negation scope işaretleyici.
    ---------------------------------------------
    Mantık:
    - 'deyil'  → kendinden önceki kelimeyi olumsuz yapar
    - 'yox', 'yoxdur', 'qətiyyən' → kendinden sonraki kelimeyi olumsuz yapar
    - 'heç' → genişleyici ama kendi başına olumsuzluk oluşturmaz
    """
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # 1) 'deyil' → önceki kelimeyi olumsuz yap
        if tok == "deyil" and len(out) > 0:
            # önceki kelimenin placeholder olmadığından emin ol
            if out[-1] not in {"URL", "EMAIL", "PHONE", "USER"}:
                out[-1] = out[-1] + "_NEG"
            out.append(tok)

        # 2) 'yox', 'yoxdur', 'qətiyyən' → sonraki kelimeyi olumsuz yap
        elif tok in {"yox", "yoxdur", "qətiyyən"}:
            out.append(tok)
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt not in {"URL", "EMAIL", "PHONE", "USER"}:
                    out.append(nxt + "_NEG")
                i += 1  # bir kelime atla çünkü onu işledik

        # 3) 'heç' → işaretleme yapma ama koru (bağlam açısından önemli)
        elif tok == "heç":
            out.append(tok)

        # 4) diğer kelimeler olduğu gibi eklensin
        else:
            out.append(tok)

        i += 1
    return out



# ========== 3) Domain detection + domain-specific normalization ==========

import re

# --- sinyal kümeleri ---
NEWS_SOURCES = r"(apa|trend|azertac|report|reuters|bloomberg|bbc|cnnturk|dha|aa|oxu\.az)"
NEWS_TITLES  = r"(son dəqiqə|rəsmi açıqlama|bəyanat|nazirlik|tədbir|mənbə)"
NEWS_FORM    = r"(\b[\d]{1,2}\s+(yanvar|fevral|mart|aprel|may|iyun|iyul|avqust|sentyabr|oktyabr|noyabr|dekabr)\b)"

SOC_EMOJI    = r"[😂🤣😍🥰🔥⭐🌟👍👎💔👏🙌]"
SOC_CUES     = r"(\brt\b|#\w+|@\w+|story|dm|post|komment|status)"

REV_PRICE    = r"\b\d{1,4}\s*(azn|manat)\b"
REV_STARS    = r"\b([1-5])\s*(?:/|\/)?\s*5\b|\b(?:ulduz|stars?)\b"
REV_OPINION  = r"(çox (yaxşı|pis)|bərbad|mü-kəm-məl|super|süper|təklif|qənaət|zəif|keyfiyyət|servis|çatdırılma|kuryer)"

NEWS_HINTS   = re.compile(f"{NEWS_SOURCES}|{NEWS_TITLES}|{NEWS_FORM}", re.I)
SOC_HINTS    = re.compile(f"{SOC_EMOJI}|{SOC_CUES}", re.I)
REV_HINTS    = re.compile(f"{REV_PRICE}|{REV_STARS}|{REV_OPINION}", re.I)

def detect_domain(text: str) -> str:
    s = (text or "").lower()

    score = {"news":0, "social":0, "reviews":0}

    # haber sinyalleri
    score["news"]   += len(NEWS_HINTS.findall(s)) * 2
    score["news"]   += s.count("–") + s.count("—")  # haber başlıklarında sık

    # sosyal sinyaller
    score["social"] += len(SOC_HINTS.findall(s)) * 2
    score["social"] += s.count("http")             # link, sosyalda çok
    score["social"] += s.count("rt ") + s.count("@")

    # review sinyalleri
    score["reviews"]+= len(REV_HINTS.findall(s)) * 2
    if "ulduz" in s: score["reviews"] += 1
    if "qiymət" in s or "qiymet" in s: score["reviews"] += 1

    # karar
    dom = max(score.items(), key=lambda x: x[1])[0]
    return dom if score[dom] > 0 else "general"

PRICE_RE    = re.compile(r"\b\d{1,4}([.,]\d+)?\s*(azn|manat)\b", re.I)
STARS_RE    = re.compile(r"\b([1-5])\s*(?:/|\/)?\s*5\b|\bulduz\b", re.I)
POS_REVIEW  = re.compile(r"\b(çox yaxşı|mükəmməl|super|süper|əla)\b", re.I)
NEG_REVIEW  = re.compile(r"\b(çox pis|bərbad|dəhşət|zəif)\b", re.I)

DELIVERY_RE = re.compile(r"\b(çatdır(ıl|il)ma|kuryer|gecikmə|zamanında deyil)\b", re.I)
SERVICE_RE  = re.compile(r"\b(servis|xidmət|davranış|müştəri xidməti)\b", re.I)
DISCOUNT_RE = re.compile(r"\b(endirim|kampaniya|kupon|promo)\b", re.I)

def normalize_reviews(s: str) -> str:
    s = PRICE_RE.sub(" PRICE ", s)
    s = STARS_RE.sub(lambda m: " STARS_{}_5 ".format(m.group(1)) if m.group(1) else " STARS ", s)
    s = POS_REVIEW.sub(" RATING_POS ", s)
    s = NEG_REVIEW.sub(" RATING_NEG ", s)
    s = DELIVERY_RE.sub(" DELIVERY_ISSUE ", s)
    s = SERVICE_RE.sub(" SERVICE_ISSUE ", s)
    s = DISCOUNT_RE.sub(" DISCOUNT ", s)
    return " ".join(s.split())

SOC_ACTION  = re.compile(r"\b(story|dm|post|komment|status)\b", re.I)

def normalize_social(s: str) -> str:
    s = SOC_ACTION.sub(lambda m: f" SOC_{m.group(1).upper()} ", s)
    return " ".join(s.split())

NEWS_SRC_RE = re.compile(NEWS_SOURCES, re.I)
NEWS_INST   = re.compile(r"\b(nazirlik|məclis|prezident|nazir|rəsmi)\b", re.I)
LOC_RE      = re.compile(r"\b(bakı|gence|sumqayıt|şuşa|naxçıvan)\b", re.I)
DATE_RE     = re.compile(NEWS_FORM, re.I)

def normalize_news(s: str) -> str:
    s = NEWS_SRC_RE.sub(" NEWS_SRC ", s)
    s = NEWS_INST.sub(" NEWS_INST ", s)
    s = LOC_RE.sub(lambda m: f" LOC_{m.group(0).upper()} ", s)
    s = DATE_RE.sub(" <DATE> ", s)  # istersek
    return " ".join(s.split())

def domain_specific_normalize(cleaned: str, domain: str) -> str:
    s = cleaned
    if domain == "reviews":
        s = normalize_reviews(s)
    elif domain == "social":
        s = normalize_social(s)
    elif domain == "news":
        s = normalize_news(s)
    return s


def add_domain_tag(line: str, domain: str) -> str:
    return f"dom{domain} " + line

# ========== 4) Asıl normalize edici ==========

def normalize_text_az(s: str,
                      numbers_to_token: bool = True,
                      keep_sentence_punct: bool = False,
                      do_neg_scope: bool = True,
                      do_deasci: bool = True,
                      drop_stop_candidates: bool = True,#bu stopwordslerin kaldırılması bazen semantic anlamı öğrenmeyi olumsuz etkileyebilir çünkü karmaşık cümlelerde bağlaç eksikliği bağlamı hafif bozabilir. deneme yaparak burayı false veya true yapabilirz şu anda True yani stopwords filtering çalışacak
                      use_stemming:bool=False,#stemming
                      strict_filter: bool = False,
                      min_token_len: int = 2) -> str:
    if not isinstance(s, str):
        return ""
    # emoji -> EMO_*
    s = map_emojis(s)
    # decode issues
    s = fix_text(s)
    s = html.unescape(s)
    # HTML tag temizliği
    s = HTML_TAG_RE.sub(" ", s)
    # Hashtag split + '#' at
    s = split_hashtags(s)
    # Özel yapılar -> placeholder
    s = URL_RE.sub(" URL ", s)
    s = EMAIL_RE.sub(" EMAIL ", s)
    s = PHONE_RE.sub(" PHONE ", s)
    s = USER_RE.sub(" USER ", s)
    # Azeri-aware lower
    s = lower_az(s)
    # Çoklu noktalama/boşluk
    s = MULTI_PUNCT.sub(r"\1", s)

    # Sayılar -> <NUM>
    if numbers_to_token:
        s = DIGIT_RE.sub(" <NUM> ", s)

    # Noktalama temizliği (cümle sonu tutulsun mu?)
    if keep_sentence_punct:
        s_tmp = re.sub(r"[^\w\səğıöüçşxq<>\-!?\.]", " ", s)
    else:
        s_tmp = NONWORD_RE.sub(" ", s)
        s_tmp = re.sub(r"[!?\.]", " ", s_tmp)
    s = MULTI_SPACE.sub(" ", s_tmp).strip()

    # Tokenize
    tokens = TOKEN_RE.findall(s)

    # deasciify
    if do_deasci:
        tokens = [deasciify_token(t) for t in tokens]

    # stopword adaylarını (negasyon hariç) at
    if drop_stop_candidates:
        tokens = [t for t in tokens if t not in STOP_CANDIDATES or t in NEGATORS]

    # negation scope (_NEG)
    if do_neg_scope:
        tokens = mark_negation_scope(tokens, window=3)

    if use_stemming: 
        tokens = stem_az(tokens)    

    # strict whitelist (yalnızca AZ harfleri + placeholder'lar)
    if strict_filter:
        _allowed = re.compile(
            r"(?:<NUM>|URL|EMAIL|PHONE|USER|EMO_(?:POS|NEG)|"
            r"RATING_(?:POS|NEG)|STARS_\d_5|PRICE|[a-zəğıöüçşxq]+(?:_NEG)?)$"
        )
        tokens = [t for t in tokens if _allowed.fullmatch(t)]

    # kısa tokenları at (o/e ve placeholderlar hariç)
    tokens = [t for t in tokens if (len(t) >= min_token_len or t in {"o","e","<NUM>","URL","EMAIL","PHONE","USER"})]

    return " ".join(tokens).strip()

# ========== 5) Etiket eşleme ==========

def map_sentiment_value(v, scheme: str):
    """
    tri: negative->0.0, neutral->0.5, positive->1.0
    binary: negative->0.0, positive->1.0
    """
    try:
        fv = float(v)
        if scheme == "binary":
            return 1.0 if fv > 0 else 0.0
        return fv if fv in (0.0, 0.5, 1.0) else None
    except Exception:
        s = str(v).strip().lower()
        if scheme == "tri":
            if s in {"pos","positive","1","müsbət","musbet","good","pozitiv"}:  return 1.0
            if s in {"neu","neutral","0.5","1/2","neytral"}:                    return 0.5
            if s in {"neg","negative","0","-1","mənfi","menfi","bad","neqativ"}: return 0.0
            return None
        if s in {"pos","positive","1","müsbət","musbet","good","pozitiv"}:      return 1.0
        if s in {"neg","negative","0","-1","mənfi","menfi","bad","neqativ"}:    return 0.0
        return None

# ========== 6) IO yardımcıları ==========

def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        return pd.read_csv(path, sep=None, engine="python")
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported file type: {ext}")

# ========== 7) Tek dosya işleme (Excel 2 sütun) ==========

def process_file(in_path: str, text_col: str, label_col: str,
                 scheme: str, out_path: str,
                 drop_stops: bool = False,
                 strict_filter: bool = False,
                 minlen: int = 2,
                 use_stemming=False):
    p = Path(in_path)
    df = read_any(p)

    # gereksiz indeks sütunları vs. temizle
    for c in ("Unnamed: 0", "index"):
        if c in df.columns:
            df = df.drop(columns=[c])

    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Columns not found. Have: {list(df.columns)}; expected text_col='{text_col}' label_col='{label_col}'")

    # Boş metinleri ve duplikateleri at
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]
    df = df.drop_duplicates(subset=[text_col])

    # Domain tespiti orijinal metinden
    df["__domain__"] = df[text_col].astype(str).apply(detect_domain)

    # Temizleme + domain-özel normalize
    df["cleaned_text"] = df[text_col].astype(str).apply(lambda s: normalize_text_az(
        s,
        numbers_to_token=True,
        keep_sentence_punct=False,
        do_neg_scope=True,
        do_deasci=True,
        drop_stop_candidates=drop_stops,   # sentiment sinyalini korumak için
        strict_filter=strict_filter,
        min_token_len=minlen,
        use_stemming=use_stemming
    ))
    df["cleaned_text"] = df.apply(lambda r: domain_specific_normalize(r["cleaned_text"], r["__domain__"]), axis=1)

    # Etiket eşleme
    df["sentiment_value"] = df[label_col].apply(lambda v: map_sentiment_value(v, scheme))
    df = df.dropna(subset=["sentiment_value"])
    df["sentiment_value"] = df["sentiment_value"].astype(float)

    # Yalnızca 2 sütun (ödev gereği)
    out_df = df[["cleaned_text","sentiment_value"]].reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_path, index=False)
    print(f"[OK] Saved {len(out_df)} rows -> {out_path}")

# ========== 8) Birleşik korpus üretimi (ham dosyalardan) ==========

def build_corpus(inputs: List[str], text_cols: List[str], label_cols: List[str],
                 schemes: List[str], out_txt: str):
    """
    Ham dosyaları okur, her satırı cümlelere böler, temizler, domain belirler,
    satır başına 'dom<domain>' etiketi koyar; tümünü 'corpus_all.txt' olarak yazar.
    """
    assert len(inputs) == len(text_cols) == len(label_cols) == len(schemes)
    lines: List[str] = []
    for in_path, tcol, lcol, sch in zip(inputs, text_cols, label_cols, schemes):
        df = read_any(Path(in_path))
        if tcol not in df.columns or lcol not in df.columns:
            raise KeyError(f"Columns not found in {in_path}. Got {list(df.columns)}, expected {tcol}/{lcol}")
        # metni al, boş/duplicateleri at
        s = df[tcol].astype(str).dropna()
        s = s[s.str.strip().str.len() > 0].drop_duplicates()
        for raw in s.astype(str):
            dom = detect_domain(raw)
            # cümleleri ayır (.!?) kalsın, sonra temizlenir
            piece = normalize_text_az(raw, keep_sentence_punct=True)
            parts = re.split(r"[.!?]+", piece)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # noktalama ve fazla boşlukları kaldır + lower
                p = re.sub(r"[^\w\səğıöüçşxq<>\-]", " ", p)
                p = " ".join(p.split()).lower()
                if p:
                    p = domain_specific_normalize(p, dom)
                    lines.append(add_domain_tag(p, dom))

    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as w:
        for ln in lines:
            w.write(ln + "\n")
    print(f"[OK] Wrote {out_txt} with {len(lines)} lines")

# ========== 8b) (Opsiyonel) Cleaned Excel’lerden korpus ==========

def build_corpus_from_clean(inputs: List[str], out_txt: str):
    """
    Cleaned Excel'lerden (cleaned_text sütunu) domain-tagged korpus üretir.
    """
    lines: List[str] = []
    for p in inputs:
        df = pd.read_excel(p, usecols=["cleaned_text"])
        for sent in df["cleaned_text"].dropna().astype(str):
            sent = sent.strip()
            if not sent:
                continue
            dom = detect_domain(sent)
            lines.append(f"dom{dom} {sent}")
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as w:
        for ln in lines:
            w.write(ln + "\n")
    print(f"[OK] Wrote {out_txt} from cleaned files with {len(lines)} lines")

# ========== 9) CLI ==========

def main():
    ap = argparse.ArgumentParser(description="Azerbaijani preprocessing & corpus builder (domain-aware).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_p = sub.add_parser("process", help="Preprocess a single file into 2-column Excel")
    ap_p.add_argument("--in", dest="in_path", required=True)
    ap_p.add_argument("--out", dest="out_path", required=True)
    ap_p.add_argument("--text-col", required=True)
    ap_p.add_argument("--label-col", required=True)
    ap_p.add_argument("--scheme", choices=["tri","binary"], default="tri")
    ap_p.add_argument("--use-stemming", action="store_true",
                  help="Türkçe SnowballStemmer ile kelime köklerine indir (Azerice için uyumlu).")

    ap_p.add_argument("--drop-stops", action="store_true", help="Drop simple stopword candidates (negations kept)")
    # (Yeni) Gürültü kontrolü ve kısa token ayarı
    ap_p.add_argument("--strict-filter", action="store_true",
                      help="Only keep AZ letters + placeholders (filters noisy typos)")
    ap_p.add_argument("--minlen", type=int, default=2,
                      help="Minimum token length to keep (o/e istisna)")

    ap_c = sub.add_parser("build-corpus", help="Build domain-tagged corpus_all.txt from raw files")
    ap_c.add_argument("--inputs", required=True, help="Comma-separated file paths")
    ap_c.add_argument("--text-cols", required=True, help="Comma-separated text column names (match inputs)")
    ap_c.add_argument("--label-cols", required=True, help="Comma-separated label column names (match inputs)")
    ap_c.add_argument("--schemes", required=True, help="Comma-separated schemes per file (tri|binary)")
    ap_c.add_argument("--out", dest="out_txt", default="corpus_all.txt")

    # (Yeni, opsiyonel) Cleaned Excel'den corpus
    ap_c2 = sub.add_parser("build-corpus-clean", help="Build domain-tagged corpus from cleaned Excels (cleaned_text)")
    ap_c2.add_argument("--inputs", required=True, help="Comma-separated cleaned Excel paths")
    ap_c2.add_argument("--out", dest="out_txt", default="corpus_all.txt")

    args = ap.parse_args()

    if args.cmd == "process":
        process_file(args.in_path, args.text_col, args.label_col, args.scheme, args.out_path,
                     args.drop_stops,
                     strict_filter=args.strict_filter,
                     minlen=args.minlen,
                     use_stemming=args.use_stemming)

    elif args.cmd == "build-corpus":
        inputs  = [s.strip() for s in args.inputs.split(",") if s.strip()]
        tcols   = [s.strip() for s in args.text_cols.split(",")]
        lcols   = [s.strip() for s in args.label_cols.split(",")]
        schemes = [s.strip() for s in args.schemes.split(",")]
        build_corpus(inputs, tcols, lcols, schemes, args.out_txt)

    elif args.cmd == "build-corpus-clean":
        inputs = [s.strip() for s in args.inputs.split(",") if s.strip()]
        build_corpus_from_clean(inputs, args.out_txt)

if __name__ == "__main__":
    main()
