"""检索：向量检索 + BM25，RRF 融合，可选重排；以及中文问题的术语改写。"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np
import yaml
from rank_bm25 import BM25Okapi

from .ingest import Chunk

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- 术语改写

CJK_RE = re.compile(r"[一-鿿]")


class QueryRewriter:
    """中文问题：按术语表补充英文检索词，使 BM25 和非多语言向量模型也能命中英文条文。"""

    def __init__(self, glossary_path: str | None):
        self.glossary: dict[str, str] = {}
        if glossary_path and os.path.exists(glossary_path):
            with open(glossary_path, encoding="utf-8") as f:
                self.glossary = yaml.safe_load(f) or {}
        # 长词优先，避免“净高”被“高度”抢先
        self._terms = sorted(self.glossary, key=len, reverse=True)

    def expand(self, query: str) -> tuple[str, list[str]]:
        if not CJK_RE.search(query):
            return query, []
        hits, rest = [], query
        for term in self._terms:
            if term in rest:
                hits.append(term)
                rest = rest.replace(term, " ")
        extra = " ".join(self.glossary[t] for t in hits)
        return (f"{query} {extra}".strip(), hits)


# ---------------------------------------------------------------- 向量化

class TfidfEmbedder:
    """离线退化方案：字符 n-gram TF-IDF。不需要下载模型，但不具备跨语言能力（靠术语表补足）。"""
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        self.fitted = False

    def fit(self, texts: list[str]):
        self.vec.fit(texts)
        self.fitted = True

    def encode(self, texts: list[str]):
        # 保持稀疏矩阵：全本 IRC 有 5000 多块、几十万个字符 n-gram，转成稠密矩阵会占用十几 GB 内存
        return self.vec.transform(texts).astype(np.float32)  # TfidfVectorizer 默认已做 L2 归一化


class BgeEmbedder:
    name = "bge-m3"

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # 延迟导入
        self.model = SentenceTransformer(model)

    def fit(self, texts):  # 无需拟合
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        many = len(texts) > 50
        if many:
            print(f"正在用 bge-m3 向量化 {len(texts)} 块（CPU 上较慢，完成后会缓存）…", flush=True)
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=8, show_progress_bar=many)
        return np.asarray(vecs, dtype=np.float32)


class ApiEmbedder:
    """OpenAI 兼容的 embedding 接口，如 DashScope text-embedding-v3。"""
    name = "api"

    def __init__(self, base_url: str, model: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.name = f"api:{model}"

    def fit(self, texts):
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 10):  # 小批量，避免触发限流
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + 10])
            out.extend(d.embedding for d in resp.data)
        return _normalize(np.asarray(out, dtype=np.float32))


def make_embedder(cfg: dict):
    backend = cfg.get("backend", "auto")
    if backend == "openai_compatible":
        return ApiEmbedder(cfg["base_url"], cfg["api_model"], os.environ[cfg["api_key_env"]])
    if backend in ("bge-m3", "auto"):
        try:
            return BgeEmbedder(cfg.get("model", "BAAI/bge-m3"))
        except Exception as e:  # 没装 sentence-transformers 或无法下载模型
            if backend == "bge-m3":
                raise
            log.warning("bge-m3 不可用（%s），退回本地 TF-IDF", type(e).__name__)
    return TfidfEmbedder()


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return m / n


# ---------------------------------------------------------------- 索引

TOKEN_RE = re.compile(r"[rnmgpe]\d{3,4}(?:\.\d+)*(?:\(\d+\))?|\d+x\d+|[a-z]+|\d+(?:[./-]\d+)*")
SECTION_TOKEN_RE = re.compile(r"^[rnmgpe]\d{3,4}")
STOP = set("a an the of to in on at for and or is are be must may not by with as any each this that from than where which what does do how".split())


def tokenize(text: str, expand_ids: bool = False) -> list[str]:
    toks = []
    for t in TOKEN_RE.findall(text.lower()):
        if t in STOP:
            continue
        if SECTION_TOKEN_RE.match(t):
            base = re.sub(r"\(\d+\)$", "", t)
            parts = base.split(".")
            # 索引侧展开条款号前缀：r602.3.2 → r602, r602.3, r602.3.2；表格 r602.3(5) 另加完整编号
            if expand_ids:
                toks.extend(".".join(parts[:i]) for i in range(1, len(parts) + 1))
                if base != t:
                    toks.append(t)
            else:
                toks.append(t)
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # 极简词干：studs → stud
        toks.append(t)
    return toks


@dataclass
class Hit:
    chunk: Chunk
    score: float = 0.0
    dense: float | None = None
    dense_rank: int | None = None
    bm25: float | None = None
    bm25_rank: int | None = None
    rerank: float | None = None
    sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 向量库

class MemoryVectorStore:
    """内存矩阵 + 点积（向量已归一化，即余弦相似度）。适合几千块以内。

    向量会缓存到 data/index/<指纹>.npy：语料和模型不变时，下次启动直接读取，
    不用再跑一遍 bge-m3（CPU 上给上千块向量化要好几分钟）。
    """
    name = "memory"

    def __init__(self, cache_dir: str | None = "data/index", **_):
        self.matrix = None
        self.cache_dir = cache_dir

    def _path(self, fingerprint: str) -> str | None:
        return os.path.join(self.cache_dir, f"{fingerprint}.npy") if self.cache_dir else None

    def has(self, fingerprint: str) -> bool:
        path = self._path(fingerprint)
        if path and os.path.exists(path):
            self.matrix = np.load(path)
            return True
        return False

    def add(self, vectors, chunks: list[Chunk], fingerprint: str):
        self.matrix = vectors
        path = self._path(fingerprint)
        if path and not _is_sparse(vectors):  # TF-IDF 计算很快，不缓存
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(path, vectors)

    def search(self, qvec, n: int) -> list[tuple[int, float]]:
        if _is_sparse(self.matrix):
            sims = np.asarray((self.matrix @ qvec.T).todense()).ravel()
        else:
            sims = self.matrix @ qvec
        order = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in order]


class MilvusVectorStore:
    """Milvus Lite：本地单文件向量库，无需 Docker；生产可把 uri 换成 Milvus 服务地址。

    集合名带“语料 + 模型”指纹，语料和模型都没变时直接复用已建好的索引，跳过全量向量化。
    """
    name = "milvus"

    def __init__(self, uri: str = "data/index/milvus.db", **_):
        from pymilvus import MilvusClient
        if "://" not in uri:
            os.makedirs(os.path.dirname(uri) or ".", exist_ok=True)
        self.client = MilvusClient(uri)
        self.collection = None

    def has(self, fingerprint: str) -> bool:
        name = f"irc_{fingerprint}"
        if self.client.has_collection(name):
            self.collection = name
            self.client.load_collection(name)
            return True
        return False

    def add(self, vectors: np.ndarray, chunks: list[Chunk], fingerprint: str):
        self.collection = f"irc_{fingerprint}"
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self.client.create_collection(self.collection, dimension=int(vectors.shape[1]),
                                      metric_type="IP", consistency_level="Strong")
        vectors = _dense(vectors)
        rows = [{"id": i, "vector": v.tolist(), "chunk_id": c.chunk_id} for i, (v, c) in enumerate(zip(vectors, chunks))]
        for i in range(0, len(rows), 500):
            self.client.insert(self.collection, rows[i:i + 500])

    def search(self, qvec: np.ndarray, n: int) -> list[tuple[int, float]]:
        qvec = _dense(qvec).ravel()
        res = self.client.search(self.collection, data=[qvec.tolist()], limit=n,
                                 search_params={"metric_type": "IP"})
        return [(int(r["id"]), float(r["distance"])) for r in res[0]]


def _is_sparse(m) -> bool:
    from scipy import sparse
    return sparse.issparse(m)


def _dense(m) -> np.ndarray:
    return m.toarray() if _is_sparse(m) else m


def make_vector_store(cfg: dict | None):
    cfg = cfg or {}
    if cfg.get("backend", "memory") == "milvus":
        return MilvusVectorStore(uri=cfg.get("uri", "data/index/milvus.db"))
    return MemoryVectorStore(cache_dir=cfg.get("cache_dir", "data/index"))


def corpus_fingerprint(chunks: list[Chunk], embedder_name: str) -> str:
    import hashlib
    h = hashlib.sha1(embedder_name.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:12]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder, cfg: dict, reranker=None, store=None):
        self.chunks = chunks
        self.embedder = embedder
        self.cfg = cfg
        self.reranker = reranker
        self.store = store or MemoryVectorStore(cache_dir=None)
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        fp = corpus_fingerprint(chunks, embedder.name)
        # TF-IDF 每次现算（很快，且是稀疏矩阵）；只有神经网络向量才读缓存
        self.reused_index = embedder.name != "tfidf" and self.store.has(fp)
        if not self.reused_index:
            self.store.add(embedder.encode(texts), chunks, fp)
        self.bm25 = BM25Okapi([tokenize(t, expand_ids=True) for t in texts])

    def dense_search(self, query: str, n: int):
        q = self.embedder.encode([query])[0]
        return self.store.search(q, n)

    def bm25_search(self, query: str, n: int):
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search(self, query: str, hybrid: bool | None = None, top_k: int | None = None) -> list[Hit]:
        cfg = self.cfg
        hybrid = cfg.get("hybrid", True) if hybrid is None else hybrid
        top_k = top_k or cfg.get("top_k", 5)
        n, k = cfg.get("candidates", 20), cfg.get("rrf_k", 60)

        hits: dict[int, Hit] = {}
        for rank, (i, s) in enumerate(self.dense_search(query, n)):
            h = hits.setdefault(i, Hit(self.chunks[i]))
            h.dense, h.dense_rank = s, rank + 1
            h.sources.append("Dense")
            h.score += 1 / (k + rank + 1) if hybrid else s
        if hybrid:
            for rank, (i, s) in enumerate(self.bm25_search(query, n)):
                h = hits.setdefault(i, Hit(self.chunks[i]))
                h.bm25, h.bm25_rank = s, rank + 1
                h.sources.append("BM25")
                h.score += 1 / (k + rank + 1)  # RRF：只看名次，不用对两路分数做归一化

        ranked = sorted(hits.values(), key=lambda h: -h.score)
        if self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked[: n])
        return ranked[:top_k]


class Reranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        for h, s in zip(hits, scores):
            h.rerank = float(s)
            h.sources.append("Rerank")
        return sorted(hits, key=lambda h: -h.rerank)
