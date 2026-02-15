#!/usr/bin/env python3
# fj-retreat.py
#
# Usage:
#   python fj-retreat.py           # search mode (interactive), human readable
#   python fj-retreat.py --all     # full scan report, human readable
# Optional:
#   python fj-retreat.py --json    # print JSON instead of human readable

import re
import json
import argparse
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LEXICON = {
    "joy": {
        "joy",
        "rejoice",
        "glad",
        "praise",
        "peace",
        "grace",
        "love",
        "hope",
        "light",
    },
    "sadness": {"sad", "weep", "mourning", "grief", "lament", "sorrow"},
    "anger": {"anger", "wrath", "rage", "fury", "hate"},
    "fear": {"fear", "afraid", "anxious", "terror", "tremble"},
    "trust": {"trust", "faith", "believe", "amen", "steadfast", "mercy"},
    "reflection": {"meditate", "ponder", "wisdom", "understand", "discern", "clarity"},
}

STOPWORDS = set(
    """
a an the and or but if then else in on at to for from of with without into over under
is are was were be been being this that these those it its as by not no yes do does did
we you they i he she him her them our your their my mine yours ours theirs
""".split()
)


# ---------- Core NLP ----------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str):
    return [t for t in normalize(text).split() if t not in STOPWORDS and len(t) > 2]


def emotion_scores(text: str):
    toks = tokenize(text)
    counts = Counter(toks)

    raw = {emo: sum(counts[w] for w in words) for emo, words in LEXICON.items()}
    total = sum(raw.values()) or 1
    dist = {k: (v / total) for k, v in raw.items()}  # fraction of total emotion hits
    top = max(raw, key=raw.get)
    return {"raw": raw, "dist": dist, "top": top, "total_hits": sum(raw.values())}


def load_notes(path="notes.md") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_lines(notes: str):
    return [ln.strip() for ln in notes.splitlines() if ln.strip()]


def build_vectorizer(corpus):
    return TfidfVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z'][a-zA-Z']+\b",
    ).fit(corpus)


def top_keywords(tfidf_vec: np.ndarray, features: np.ndarray, n=18):
    idx = np.argsort(tfidf_vec)[::-1]
    out = []
    for i in idx:
        if tfidf_vec[i] <= 0:
            break
        out.append(features[i])
        if len(out) >= n:
            break
    return out


def retrieve(lines, vectorizer, query, k=7):
    X = vectorizer.transform(lines)
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).ravel()  # unitless, range [0,1]
    best = sims.argsort()[::-1][:k]
    return [(float(sims[i]), lines[i]) for i in best if sims[i] > 0]


def theme_similarity(doc_text: str, vectorizer):
    themes = {
        "Grace & Renewal": "grace truth renewed salvation mercy",
        "Kingdom & Calling": "kingdom proclamation fulfillment mission",
        "Lent & Practice": "lent preparation discipline prayer repentance",
        "Gospel as Message": "gospel message teachings proclaim",
        "Mindfulness & Clarity": "meditate clarity wisdom discern understand",
    }
    doc_v = vectorizer.transform([doc_text])
    theme_v = vectorizer.transform(list(themes.values()))
    sims = cosine_similarity(doc_v, theme_v).ravel()  # unitless [0,1]
    ranked = sorted(zip(themes.keys(), sims), key=lambda x: x[1], reverse=True)
    return [(name, float(score)) for name, score in ranked]


def scan_report(notes: str, lines, vectorizer):
    emo = emotion_scores(notes)

    feats = np.array(vectorizer.get_feature_names_out())
    doc_vec = vectorizer.transform([" ".join(lines)]).toarray().ravel()
    keywords = top_keywords(doc_vec, feats, n=18)

    toks = tokenize(notes)
    top_terms = Counter(toks).most_common(20)

    paras = [p.strip() for p in re.split(r"\n\s*\n", notes) if p.strip()]
    para_emotions = []
    for i, p in enumerate(paras[:12], start=1):
        es = emotion_scores(p)
        para_emotions.append(
            {
                "para": i,
                "top_emotion": es["top"],
                "dist": es["dist"],
                "total_hits": es["total_hits"],
                "preview": (p[:140] + "…") if len(p) > 140 else p,
            }
        )

    themes = theme_similarity(" ".join(lines), vectorizer)

    snippet_queries = {
        "Grace": "grace truth mercy renewed",
        "Kingdom": "kingdom fulfillment hands",
        "Gospel": "gospel proclaim message teachings",
        "Lent": "lent preparation connect life",
    }
    snippets_by_query = {
        k: [{"score": s, "text": t} for s, t in retrieve(lines, vectorizer, q, k=4)]
        for k, q in snippet_queries.items()
    }

    return {
        "stats": {"lines": len(lines), "tokens": len(toks), "paragraphs": len(paras)},
        "emotion": emo,
        "top_keywords": keywords,
        "top_terms": [{"term": t, "count": c} for t, c in top_terms],
        "theme_similarity": themes,
        "paragraph_emotions_sample": para_emotions,
        "snippets_by_query": snippets_by_query,
    }


# ---------- Human-readable formatting ----------
def fmt_ratio(x: float, denom: float = 1.0, digits=3) -> str:
    # Always show "x / 1.00" style so the scale is explicit
    return f"{x:.{digits}f} / {denom:.2f}"


def fmt_percent(frac: float, digits=1) -> str:
    # frac is 0..1 => percent
    return f"{(frac * 100):.{digits}f}%"


def print_all_report_human(r):
    print("\n=== FJ Retreat NLP Report (Human Readable, No LLM) ===\n")

    s = r["stats"]
    print(
        f"Text stats: {s['lines']} lines, {s['paragraphs']} paragraphs, {s['tokens']} tokens (after stopword filtering)."
    )

    emo = r["emotion"]
    print("\nEmotion signal (lexicon-based, weak heuristic):")
    print(
        f"  Total emotion word hits: {emo['total_hits']}  (if 0, lexicon didn’t match your wording)"
    )
    print(f"  Top emotion: {emo['top']}")
    # Dist is fraction-of-total-hits (NOT probability of true emotion)
    print("  Distribution (fraction of emotion hits):")
    for k, frac in sorted(emo["dist"].items(), key=lambda kv: kv[1], reverse=True):
        # show both percent and explicit fraction scale
        print(f"    - {k:10s}: {fmt_percent(frac)}  ({fmt_ratio(frac)})")

    print("\nTop keywords (TF-IDF terms/phrases, not a score):")
    print("  " + ", ".join(r["top_keywords"]))

    print("\nTop terms by frequency (counts, unit = occurrences):")
    for item in r["top_terms"][:12]:
        print(f"  - {item['term']}: {item['count']}")

    print("\nTheme similarity (cosine similarity on TF-IDF; unitless, max = 1.00):")
    for name, score in r["theme_similarity"]:
        print(f"  - {name:18s}: {fmt_ratio(score)}")

    print("\nParagraph emotion sample (first up to 12 paragraphs):")
    for p in r["paragraph_emotions_sample"]:
        print(
            f"\n  Paragraph {p['para']} — top emotion: {p['top_emotion']} (hits={p['total_hits']})"
        )
        # show top 2 dist entries
        dist_sorted = sorted(p["dist"].items(), key=lambda kv: kv[1], reverse=True)
        if p["total_hits"] == 0:
            print("    No lexicon hits in this paragraph (try expanding the lexicon).")
        else:
            for k, frac in dist_sorted[:2]:
                print(f"    {k}: {fmt_percent(frac)} ({fmt_ratio(frac)})")
        print(f"    Preview: {p['preview']}")

    print("\nRepresentative snippets (cosine similarity; unitless, max = 1.00):")
    for qname, snips in r["snippets_by_query"].items():
        print(f"\n  {qname}:")
        if not snips:
            print("    (no matches)")
            continue
        for s in snips:
            print(f"    - [{fmt_ratio(s['score'])}] {s['text']}")


def print_search_human(query, snippets):
    print(f"\n=== Search Results ===")
    print(
        "Score meaning: cosine similarity on TF-IDF (unitless), shown as score / 1.00\n"
    )
    print(f"Query: {query}\n")
    if not snippets:
        print("(No matching lines. Try different words or add more notes.)\n")
        return
    for score, text in snippets:
        print(f"- [{fmt_ratio(score)}] {text}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="FJ Retreat NLP (no LLM). Reads notes.md from CWD."
    )
    ap.add_argument(
        "--all", action="store_true", help="Scan whole text and print a full report"
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of human readable output",
    )
    ap.add_argument(
        "--file", default="notes.md", help="Path to notes markdown (default: notes.md)"
    )
    args = ap.parse_args()

    notes = load_notes(args.file)
    lines = split_lines(notes)
    corpus = lines + [" ".join(lines)]
    vectorizer = build_vectorizer(corpus)

    if args.all:
        report = scan_report(notes, lines, vectorizer)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_all_report_human(report)
        return

    # Search mode (default)
    while True:
        query = input("Search notes (blank to exit): ").strip()
        if not query:
            break
        snippets = retrieve(lines, vectorizer, query, k=7)
        if args.json:
            payload = {
                "query": query,
                "scale": "cosine similarity on TF-IDF; unitless; max=1.00",
                "snippets": [{"score": round(s, 6), "text": t} for s, t in snippets],
            }
            print(json.dumps(payload, indent=2))
            print()
        else:
            print_search_human(query, snippets)


if __name__ == "__main__":
    main()
