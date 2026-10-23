"""Document retrieval for the investigating agent.

BM25 over document chunks, with a metadata filter on document date. Lexical
retrieval is a deliberate first choice: it needs no model or API key, it is easy
to explain, and the Retriever class is the one place to swap in embeddings or a
hybrid later.

The date filter matters. An explanation for a September variance can be dated
before the month (an approval email) or just after (a vendor statement), but not
a year earlier. Without the filter, last year's audit invoice looks just as
relevant as this year's.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

STOP = set("a an and are as at be by for from has have in is it of on or that the this to was were will with we our".split())
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass
class Chunk:
    doc_id: str
    date: str
    doc_type: str
    title: str
    text: str
    idx: int = 0


def load_docs(docs_dir: Path) -> list[dict]:
    docs = []
    for p in sorted(Path(docs_dir).glob("*.txt")):
        head, _, body = p.read_text().partition("\n---\n")
        meta = dict(line.split(": ", 1) for line in head.strip().splitlines() if ": " in line)
        docs.append({"id": meta["id"], "date": meta["date"], "type": meta.get("type", ""),
                     "title": meta.get("title", ""), "text": body.strip()})
    return docs


def chunk_doc(doc: dict, max_chars: int = 700) -> list[Chunk]:
    """Split on blank lines and pack paragraphs up to max_chars. The title is
    repeated in every chunk so a chunk is understandable on its own."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc["text"]) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > max_chars:
            out.append(cur); cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return [Chunk(doc["id"], doc["date"], doc["type"], doc["title"], t, i) for i, t in enumerate(out)]


def window_for_month(month: str, before_days: int = 90, after_days: int = 45) -> tuple[str, str]:
    y, m = map(int, month.split("-"))
    start = date(y, m, 1)
    end = date(y + (m == 12), (m % 12) + 1, 1) - timedelta(days=1)
    return (start - timedelta(days=before_days)).isoformat(), (end + timedelta(days=after_days)).isoformat()


class LSAEmbedder:
    """Dependency-light embedding model: TF-IDF over words and character n-grams, reduced with
    truncated SVD (latent semantic analysis). It is fitted on the document corpus itself, so it
    learns which words co-occur ("slipping" ~ "slip" ~ "delay") and char n-grams bridge word forms.
    Runs offline in milliseconds. For production, swap in FastEmbedEmbedder (a real neural model)."""
    name = "lsa-local"

    def __init__(self, dims: int = 48):
        self.dims = dims

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack
        self._w = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        self._c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        X = hstack([self._w.fit_transform(texts), self._c.fit_transform(texts)]).tocsr()
        self._svd = TruncatedSVD(n_components=min(self.dims, max(2, X.shape[0] - 1), X.shape[1] - 1), random_state=0)
        self._svd.fit(X)
        return self

    def encode(self, texts: list[str]):
        import numpy as np
        from scipy.sparse import hstack
        X = hstack([self._w.transform(texts), self._c.transform(texts)]).tocsr()
        Z = self._svd.transform(X)
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.where(n == 0, 1, n)


class FastEmbedEmbedder:
    """Neural embeddings (BAAI/bge-small-en-v1.5, ~130 MB, CPU, no API key). Needs `pip install fastembed`
    and one-time internet access to download the model."""
    name = "bge-small-en-v1.5"

    def fit(self, texts):
        from fastembed import TextEmbedding
        self._m = TextEmbedding("BAAI/bge-small-en-v1.5")
        return self

    def encode(self, texts):
        import numpy as np
        Z = np.array(list(self._m.embed(texts)))
        return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def make_embedder(kind: str | None = None):
    """kind: 'lsa' | 'fastembed' | 'auto' (fastembed if it loads, else lsa). Env GVE_EMBEDDER overrides."""
    import os
    kind = (kind or os.environ.get("GVE_EMBEDDER") or "lsa").lower()
    if kind in ("fastembed", "auto"):
        try:
            FastEmbedEmbedder().fit(["probe"])
            return FastEmbedEmbedder()
        except Exception:
            if kind == "fastembed":
                raise
    return LSAEmbedder()


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: score = sum over rankers of 1/(k+rank). Rank-based, so BM25 scores and
    cosine similarities never need to be put on the same scale."""
    out: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r, 1):
            out[i] = out.get(i, 0.0) + 1.0 / (k + rank)
    return out


class Retriever:
    """mode: 'bm25' (keyword), 'dense' (embeddings), 'hybrid' (both, merged with reciprocal rank fusion)."""

    def __init__(self, docs_dir: Path | None = None, k1: float = 1.5, b: float = 0.75, mode: str = "bm25",
                 embedder=None, docs: list[dict] | None = None):
        """Index documents from a folder, from a list of dicts, or both (folder first)."""
        self.mode = mode
        self._embedder_arg, self._emb = embedder, None
        all_docs = (load_docs(docs_dir) if docs_dir else []) + list(docs or [])
        self.docs = {d["id"]: d for d in all_docs}
        self.chunks: list[Chunk] = [c for d in self.docs.values() for c in chunk_doc(d)]
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(f"{c.title} {c.text}")) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df = Counter(t for tf in self._tf for t in tf)
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def with_docs(self, extra: list[dict]) -> "Retriever":
        """A new retriever over this one's documents plus `extra`. The index is rebuilt (milliseconds at
        this scale); the original is untouched, so other sessions never see the extra documents."""
        return Retriever(docs=list(self.docs.values()) + list(extra), k1=self.k1, b=self.b, mode=self.mode,
                         embedder=type(self._embedder_arg)() if self._embedder_arg is not None else None)

    def _dense(self):
        if self._emb is None:
            e = self._embedder_arg or make_embedder()
            texts = [f"{c.title}. {c.text}" for c in self.chunks]
            self._emb = (e.fit(texts), None)
            self._emb = (e, e.encode(texts))
        return self._emb

    @property
    def embedder_name(self) -> str:
        return getattr(self._dense()[0], "name", "custom")

    def _score(self, q_tokens: list[str], i: int) -> float:
        tf, dl, s = self._tf[i], self._len[i], 0.0
        for t in q_tokens:
            if t in tf:
                f = tf[t]
                s += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avg))
        return s

    def search(self, query: str, k: int = 4, date_from: str | None = None, date_to: str | None = None,
               mode: str | None = None) -> list[dict]:
        mode = mode or self.mode
        q = tokenize(query)
        ok = [i for i, c in enumerate(self.chunks)
              if not (date_from and c.date < date_from) and not (date_to and c.date > date_to)]
        bm = sorted(((self._score(q, i), i) for i in ok), key=lambda x: -x[0])
        bm_rank = [i for s, i in bm if s > 0]
        dn_rank: list[int] = []
        if mode in ("dense", "hybrid") and ok:
            emb, M = self._dense()
            sims = M[ok] @ emb.encode([query])[0]
            order = sorted(zip(sims, ok), key=lambda x: -x[0])
            dn_rank = [i for s, i in order[:max(k * 3, 10)] if s > 0.05]
        if mode == "bm25":
            ranked = [(s, i) for s, i in bm if s > 0]
        elif mode == "dense":
            sc = dict(zip(ok, sims)) if ok else {}
            ranked = [(float(sc[i]), i) for i in dn_rank]
        else:
            f = rrf([bm_rank[:max(k * 3, 10)], dn_rank])
            ranked = sorted(((v, i) for i, v in f.items()), key=lambda x: -x[0])
        return [{"doc_id": self.chunks[i].doc_id, "date": self.chunks[i].date, "type": self.chunks[i].doc_type,
                 "title": self.chunks[i].title, "score": round(float(s), 4), "text": self.chunks[i].text}
                for s, i in ranked[:k]]
