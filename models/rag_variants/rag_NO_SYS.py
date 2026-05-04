# pip install chromadb sentence-transformers python-dotenv openai

import json
import os
import platform
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Local files (defaults match uploaded filenames)
VALORANT_ERROR_CODES_PATH = os.getenv("VALORANT_ERROR_CODES_PATH", "valorant_error_codes.jsonl")
VALORANT_KNOWN_ISSUES_PATH = os.getenv("VALORANT_KNOWN_ISSUES_PATH", "valorant_known_issues.jsonl")
MINECRAFT_NAMED_ERRORS_PATH = os.getenv("MINECRAFT_NAMED_ERRORS_PATH", "minecraft_named_errors.jsonl")
MINECRAFT_OFFICIAL_HELP_PATH = os.getenv("MINECRAFT_OFFICIAL_HELP_PATH", "minecraft_official_help.jsonl")
FORTNITE_ERRORS_SOLUTIONS_PATH = os.getenv("FORTNITE_ERRORS_SOLUTIONS_PATH", "fortnite_errors_solutions.jsonl")

# Chroma persistence
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "game_support_rag")
REBUILD_INDEX = os.getenv("REBUILD_INDEX", "1") not in ("0", "false", "False")

TOP_K = int(os.getenv("TOP_K", "6"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))          # chars
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))    # chars


# =========================================================
# UTILS
# =========================================================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Dosya yok: {path}")
        return docs

    bad = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1

    if bad:
        print(f"[WARN] {path}: {bad} bozuk satır atlandı.")
    return docs
def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Very simple char-based chunker (robust & language-agnostic)."""
    t = (text or "").strip()
    if not t:
        return []
    if chunk_size <= 0:
        return [t]
    overlap = max(0, min(overlap, chunk_size - 1))
    out = []
    start = 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        out.append(t[start:end].strip())
        if end >= len(t):
            break
        start = max(0, end - overlap)
    return [c for c in out if c]
def stable_hash_id(*parts: str) -> str:
    h = hashlib.sha256("||".join([p or "" for p in parts]).encode("utf-8", errors="ignore")).hexdigest()[:18]
    return h
def normalize_game_value(v: Optional[str]) -> str:
    """Return stable game key: valorant/minecraft/fortnite/unknown"""
    s = (v or "").strip().lower()
    if not s:
        return "unknown"
    if "valorant" in s or s == "val":
        return "valorant"
    if "minecraft" in s or s in ("mc", "bedrock", "java"):
        return "minecraft"
    if "fortnite" in s or s in ("fn", "epic"):
        return "fortnite"
    return s


# =========================================================
# LANGUAGE GUARDRAIL (same as before)
# =========================================================
_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TR_HINT_WORDS = {
    "ve", "veya", "ama", "de", "da", "ile", "için", "nasıl", "neden",
    "çözüm", "hata", "sorun", "ayar", "oyun", "bilgisayar", "lütfen",
}

def detect_text_lang(text: str) -> str:
    """Best-effort language detection: returns 'tr' or 'en'."""
    t = (text or "").strip()
    if not t:
        return "en"
    if any(ch in _TR_CHARS for ch in t):
        return "tr"
    low = t.lower()
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", low)
    if tokens:
        hit = sum(1 for tok in tokens if tok in _TR_HINT_WORDS)
        if hit >= 2:
            return "tr"
    return "en"
def language_system_rule(lang: str) -> str:
    """System-level hard rule to keep output language consistent."""
    lang = (lang or "en").lower()
    if lang.startswith("tr"):
        return "Yanıtı kesinlikle TÜRKÇE ver. İngilizce kelime veya cümle kullanma."
    return "Answer strictly in ENGLISH. Do not use Turkish words or Turkish characters."
def language_user_rule(lang: str) -> str:
    """User-visible reminder inside the prompt (still helpful, but system rule is stronger)."""
    lang = (lang or "en").lower()
    if lang.startswith("tr"):
        return "Cevabın dili soruyla aynı olmalı: TÜRKÇE."
    return "The answer language must match the question: ENGLISH."
def is_lang_match(text: str, target_lang: str) -> bool:
    return detect_text_lang(text) == ("tr" if (target_lang or "en").lower().startswith("tr") else "en")


# =========================================================
# NORMALIZATION (same as before)
# =========================================================
def normalize_minecraft_named_errors(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in raw_docs:
        error_name = (rec.get("error_name") or "").strip()
        what = (rec.get("what_it_means") or "").strip()
        summary = (rec.get("troubleshoot_summary") or "").strip()
        articles = rec.get("troubleshoot_articles") or []

        base_id = re.sub(r"\s+", "_", error_name).strip("_") if error_name else "unknown_error"

        if what:
            text = f"ERROR_NAME: {error_name}\n\nWHAT IT MEANS:\n{what}"
            for ci, chunk in enumerate(split_text_into_chunks(text)):
                out.append({
                    "type": "minecraft_named_error",
                    "game": "minecraft",
                    "source": "minecraft_named_errors",
                    "error_name": error_name,
                    "section": "what_it_means",
                    "title": f"{error_name} — What it means",
                    "text": chunk,
                    "chunk_id": ci,
                    "doc_id": f"mc_named::{base_id}::what::{ci}",
                })

        if summary:
            text = f"ERROR_NAME: {error_name}\n\nTROUBLESHOOT SUMMARY:\n{summary}"
            for ci, chunk in enumerate(split_text_into_chunks(text)):
                out.append({
                    "type": "minecraft_named_error",
                    "game": "minecraft",
                    "source": "minecraft_named_errors",
                    "error_name": error_name,
                    "section": "troubleshoot_summary",
                    "title": f"{error_name} — Troubleshoot summary",
                    "text": chunk,
                    "chunk_id": ci,
                    "doc_id": f"mc_named::{base_id}::summary::{ci}",
                })

        if isinstance(articles, list):
            for ai, a in enumerate(articles):
                url = (a.get("url") or "").strip() if isinstance(a, dict) else ""
                body = (a.get("text") or a.get("body") or "").strip() if isinstance(a, dict) else ""
                if not body:
                    continue
                title = (a.get("title") or "").strip() if isinstance(a, dict) else ""
                text = f"ERROR_NAME: {error_name}\n\nARTICLE: {title}\nURL: {url}\n\n{body}"
                for ci, chunk in enumerate(split_text_into_chunks(text)):
                    out.append({
                        "type": "minecraft_named_error",
                        "game": "minecraft",
                        "source": "minecraft_named_errors",
                        "error_name": error_name,
                        "section": "troubleshoot_article",
                        "title": f"{error_name} — {title or 'Troubleshoot article'}",
                        "url": url,
                        "text": chunk,
                        "chunk_id": ci,
                        "doc_id": f"mc_named::{base_id}::article{ai}::{ci}",
                    })
    return out
def normalize_minecraft_official_help(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in raw_docs:
        title = (rec.get("title") or rec.get("heading") or "").strip()
        url = (rec.get("url") or rec.get("link") or "").strip()
        body = (rec.get("body") or rec.get("text") or rec.get("content") or "").strip()
        if not (title or body):
            continue

        base = stable_hash_id(title, url, body[:200])
        combined = f"TITLE: {title}\nURL: {url}\n\n{body}".strip()

        for ci, chunk in enumerate(split_text_into_chunks(combined)):
            out.append({
                "type": "minecraft_official_help",
                "game": "minecraft",
                "source": (rec.get("source") or "help.minecraft.net"),
                "title": title,
                "url": url,
                "section": "official_help",
                "text": chunk,
                "chunk_id": ci,
                "doc_id": f"mc_help::{base}::{ci}",
            })
    return out
def normalize_fortnite_errors(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in raw_docs:
        doc_id = (rec.get("doc_id") or "").strip()
        code = (rec.get("code") or rec.get("error_code") or "").strip()
        title = (rec.get("title") or "").strip()
        category = (rec.get("category") or "").strip()

        symptoms = rec.get("symptoms") or []
        causes = rec.get("causes") or []
        fixes = rec.get("fixes") or rec.get("solutions") or []
        srcs = rec.get("sources") or []

        urls = []
        if isinstance(srcs, list):
            for s in srcs:
                if isinstance(s, dict) and s.get("url"):
                    urls.append(str(s["url"]).strip())
                elif isinstance(s, str):
                    urls.append(s.strip())
        urls = [u for u in urls if u]

        title_bits = ["Fortnite"]
        if code:
            title_bits.append(code)
        if title:
            title_bits.append(title)
        if category:
            title_bits.append(f"({category})")
        full_title = " — ".join([b for b in title_bits if b]).strip()

        parts = []
        if symptoms:
            parts.append("SYMPTOMS:\n- " + "\n- ".join([str(x).strip() for x in symptoms if str(x).strip()]))
        if causes:
            parts.append("POSSIBLE CAUSES:\n- " + "\n- ".join([str(x).strip() for x in causes if str(x).strip()]))
        if fixes:
            parts.append("RECOMMENDED FIXES:\n- " + "\n- ".join([str(x).strip() for x in fixes if str(x).strip()]))
        if urls:
            parts.append("OFFICIAL/REFERENCE SOURCES:\n- " + "\n- ".join(urls))

        combined = f"TITLE: {full_title}\nCODE: {code}\nCATEGORY: {category}\n\n" + "\n\n".join(parts)
        base = doc_id or f"fn::{stable_hash_id(full_title, code, category)}"

        for ci, chunk in enumerate(split_text_into_chunks(combined)):
            out.append({
                "type": "fortnite_error",
                "game": "fortnite",
                "source": (rec.get("source") or "fortnite_corpus"),
                "doc_id": f"{base}::{ci}",
                "title": full_title,
                "code": code,
                "category": category,
                "text": chunk,
                "chunk_id": ci,
            })
    return out
def normalize_valorant_docs(raw_docs: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in raw_docs:
        code = (rec.get("code") or "").strip()
        title = (rec.get("title") or "").strip()
        meaning = (rec.get("meaning") or "").strip()
        sol = (rec.get("official_solution") or rec.get("solution") or "").strip()
        url = (rec.get("url") or rec.get("article_url") or "").strip()

        base = (rec.get("doc_id") or "").strip()
        if not base:
            base = f"val::{stable_hash_id(doc_type, code, title, url)}"

        combined_parts = []
        if title:
            combined_parts.append(f"TITLE: {title}")
        if code:
            combined_parts.append(f"CODE: {code}")
        if meaning:
            combined_parts.append(f"MEANING: {meaning}")
        if url:
            combined_parts.append(f"URL: {url}")
        if sol:
            combined_parts.append(f"OFFICIAL SOLUTION:\n{sol}")

        combined = "\n".join(combined_parts).strip()
        if not combined:
            continue

        for ci, chunk in enumerate(split_text_into_chunks(combined)):
            out.append({
                "type": doc_type,
                "game": "valorant",
                "source": (rec.get("source") or "riot_valorant"),
                "doc_id": f"{base}::{ci}",
                "title": title,
                "code": code,
                "meaning": meaning,
                "url": url,
                "text": chunk,
                "chunk_id": ci,
            })
    return out


# =========================================================
# GAME ROUTING (same as before)
# =========================================================
_GAME_PATTERNS: Dict[str, List[str]] = {
    "valorant": [
        r"\bvalorant\b", r"\briot\b", r"\bvan\s?-?\s?\d+\b", r"\bval\s?[-_ ]?\d+\b",
        r"\bvanguard\b", r"\bvp?n\b",
    ],
    "minecraft": [
        r"\bminecraft\b", r"\bbedrock\b", r"\bjava edition\b", r"\bmojang\b",
        r"\b(netty|jvm|launcher)\b", r"\b(error code|named error)\b.*\b(minecraft)\b",
    ],
    "fortnite": [
        r"\bfortnite\b", r"\bepic\b", r"\beac\b", r"\bbattleye\b",
    ],
}

def route_game(query: str) -> Tuple[str, float]:
    q = (query or "").lower()
    scores: Dict[str, int] = {k: 0 for k in _GAME_PATTERNS.keys()}
    for game, pats in _GAME_PATTERNS.items():
        for pat in pats:
            if re.search(pat, q, flags=re.I):
                scores[game] += 1
    best_game = max(scores, key=lambda g: scores[g])
    best = scores[best_game]
    total = sum(scores.values())
    if best == 0:
        return "unknown", 0.0
    conf = best / max(1, total)
    return best_game, float(conf)
def infer_game_from_doc(doc: Dict[str, Any]) -> str:
    if doc.get("game"):
        return normalize_game_value(str(doc.get("game")))
    t = (doc.get("type") or "").lower()
    src = (doc.get("source") or "").lower()
    blob = " ".join([t, src, str(doc.get("title") or ""), str(doc.get("text") or "")]).lower()
    if "fortnite" in blob or "epic" in blob:
        return "fortnite"
    if "minecraft" in blob or "mojang" in blob:
        return "minecraft"
    if "valorant" in blob or "riot" in blob or "vanguard" in blob:
        return "valorant"
    return "unknown"


# =========================================================
# VALORANT HYBRID RETRIEVAL HELPERS
# (system_info ile ilgili canon map kaldırıldı)
# =========================================================
_VAL_CODE_PAT = re.compile(
    r"""(?ix)
    (?:\bVAN\s*-?\s*(\d{1,5})\b)
    |(?:\bError\s*Code\s*(\d{1,5})\b)
    |(?:\bCode\s*(\d{1,5})\b)
    """
)

def _canon_type(t: str) -> str:
    s = (t or "").strip().lower()
    if not s:
        return ""
    if s in {"error_code", "known_issue"}:
        return s
    if "error" in s and "code" in s:
        return "error_code"
    if "known" in s and "issue" in s:
        return "known_issue"
    return s
def _normalize_code_digits(code: str) -> str:
    if code is None:
        return ""
    return re.sub(r"\D+", "", str(code))
def extract_valorant_codes(query: str) -> List[str]:
    q = (query or "")
    hits = []
    for m in _VAL_CODE_PAT.finditer(q):
        for g in m.groups():
            if g:
                hits.append(g)
    if not hits:
        if re.search(r"(?i)\b(van|vanguard|valorant|error|code)\b", q):
            hits = re.findall(r"\b\d{1,5}\b", q)
    seen = set()
    out = []
    for h in hits:
        h = str(h).lstrip("0") or "0"
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out
def build_valorant_lexical_candidates(
    docs: List[Dict[str, Any]],
    codes: List[str],
    limit: int = 50,
) -> List[Tuple[Dict[str, Any], float]]:
    if not codes:
        return []
    cset = set(codes)
    scored: List[Tuple[Dict[str, Any], float]] = []

    for d in docs:
        if (d.get("game") or "") != "valorant":
            continue
        # Lexical code matching targets error_code docs
        if _canon_type(d.get("type")) != "error_code":
            continue

        dcode = _normalize_code_digits(d.get("code") or "")
        if not dcode:
            continue

        if dcode in cset:
            scored.append((d, 0.01))
            continue

        for c in codes:
            if c and c in dcode:
                scored.append((d, 0.08))
                break

    seen = set()
    out = []
    for d, dist in sorted(scored, key=lambda x: x[1]):
        did = str(d.get("doc_id"))
        if did in seen:
            continue
        seen.add(did)
        out.append((d, dist))
        if len(out) >= limit:
            break
    return out


class RAGIndex:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None
        self.docs: List[Dict[str, Any]] = []

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        parts = []
        if doc.get("type"):
            parts.append(f"TYPE: {doc['type']}")
        if doc.get("game"):
            parts.append(f"GAME: {doc['game']}")
        if doc.get("code"):
            parts.append(f"CODE: {doc['code']}")
        if doc.get("error_name"):
            parts.append(f"ERROR_NAME: {doc['error_name']}")
        if doc.get("title"):
            parts.append(f"TITLE: {doc['title']}")
        if doc.get("category"):
            parts.append(f"CATEGORY: {doc['category']}")
        if doc.get("section"):
            parts.append(f"SECTION: {doc['section']}")
        if doc.get("url"):
            parts.append(f"URL: {doc['url']}")
        if doc.get("meaning"):
            parts.append(f"MEANING: {doc['meaning']}")

        if doc.get("text"):
            parts.append(str(doc["text"]))
        elif doc.get("body"):
            parts.append(str(doc["body"]))

        return "\n".join([p for p in parts if p]).strip()
    def build_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        keys = [
            "game",
            "type",
            "source",
            "section",
            "url",
            "doc_id",
            "category",
            "title",
            "code",
            "error_name",
            "chunk_id",
        ]
        for k in keys:
            if k in doc and doc[k] is not None:
                v = doc[k]
                meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        return meta
    def _get_or_create_collection(self, rebuild: bool):
        if rebuild:
            try:
                self.client.delete_collection(CHROMA_COLLECTION)
                print(f"[INFO] Deleted existing collection: {CHROMA_COLLECTION}")
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    def build(self, docs: List[Dict[str, Any]]):
        for d in docs:
            d["game"] = infer_game_from_doc(d)

        self.docs = docs
        self._get_or_create_collection(REBUILD_INDEX)

        print(f"[INFO] Embeddingler hesaplanıyor... (n={len(docs)})")
        texts = [self.doc_to_text(d) for d in docs]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        ids = []
        metadatas = []
        for d, txt in zip(docs, texts):
            did = str(d.get("doc_id") or "")
            if not did:
                did = f"doc::{stable_hash_id(d.get('type',''), d.get('title',''), txt[:200])}"
                d["doc_id"] = did
            ids.append(did)
            metadatas.append(self.build_metadata(d))

        max_batch = 3000
        n = len(docs)
        print(f"[INFO] Chroma add batching: total={n}, batch_size={max_batch}")

        for start in range(0, n, max_batch):
            end = min(start + max_batch, n)
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"[INFO] Added batch: {start}:{end}")

        print(f"[INFO] ChromaDB index hazır. {len(docs)} vektör eklendi. (dir={CHROMA_DIR})")
    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
        if self.collection is None:
            raise RuntimeError("Index oluşturulmadı (collection None).")
        q = (query or "").strip()
        if not q:
            return []

        game, conf = route_game(q)
        q_emb = self.model.encode([q], convert_to_numpy=True)

        def _run(where_filter):
            res = self.collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "distances"],
            )
            metadatas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            out_local = []
            for meta, dist in zip(metadatas, dists):
                doc_id = meta.get("doc_id")
                doc = next((d for d in self.docs if str(d.get("doc_id")) == str(doc_id)), None)
                if doc:
                    out_local.append((doc, float(dist)))
            return out_local

        where = {"game": game} if (game != "unknown" and conf >= 0.34) else None

        # VALORANT HYBRID RETRIEVAL (same logic, no system_info)
        if where and game == "valorant":
            codes = extract_valorant_codes(q)
            lexical = build_valorant_lexical_candidates(self.docs, codes, limit=50)
            semantic = _run({"game": "valorant"})

            semantic_global = []
            if len(semantic) < max(3, top_k // 2):
                semantic_global = _run(None)

            def rrf(items, k0=60):
                scores = {}
                for rank, (doc, dist) in enumerate(items, start=1):
                    did = str(doc.get("doc_id"))
                    scores.setdefault(did, {"doc": doc, "score": 0.0})
                    scores[did]["score"] += 1.0 / (k0 + rank)
                return scores

            fused = {}
            for src in (lexical, semantic, semantic_global):
                part = rrf(src)
                for did, obj in part.items():
                    fused.setdefault(did, {"doc": obj["doc"], "score": 0.0})
                    fused[did]["score"] += obj["score"]

            if codes:
                cset = set(codes)
                for did, obj in fused.items():
                    dcode = _normalize_code_digits(obj["doc"].get("code") or "")
                    if dcode and dcode in cset:
                        obj["score"] += 0.25

            ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
            out = []
            for obj in ranked:
                score = obj["score"]
                dist = 1.0 / max(1e-9, score)
                out.append((obj["doc"], float(dist)))
            return out

        primary = _run(where) if where else _run(None)

        if where and len(primary) < max(3, top_k // 2):
            fallback = _run(None)
            seen = set()
            merged = []
            for d, sc in primary + fallback:
                did = str(d.get("doc_id"))
                if did in seen:
                    continue
                seen.add(did)
                merged.append((d, sc))
            return merged[:top_k]

        return primary


# =========================================================
# PROMPTING  (system_info kuralı kaldırıldı)
# =========================================================
def build_context_snippet(doc: Dict[str, Any]) -> str:
    parts = []
    if doc.get("game"):
        parts.append(f"[GAME] {doc['game']}")
    if doc.get("type"):
        parts.append(f"[TYPE] {doc['type']}")
    if doc.get("title"):
        parts.append(f"[TITLE] {doc['title']}")
    if doc.get("code"):
        parts.append(f"[CODE] {doc['code']}")
    if doc.get("error_name"):
        parts.append(f"[ERROR_NAME] {doc['error_name']}")
    if doc.get("category"):
        parts.append(f"[CATEGORY] {doc['category']}")
    if doc.get("section"):
        parts.append(f"[SECTION] {doc['section']}")
    if doc.get("url"):
        parts.append(f"[URL] {doc['url']}")

    if doc.get("text"):
        parts.append(f"[CONTENT]\n{str(doc['text'])[:1200]}")
    elif doc.get("body"):
        parts.append(f"[CONTENT]\n{str(doc['body'])[:1200]}")
    else:
        parts.append(f"[RAW]\n{str(doc)[:1200]}")

    return "\n".join(parts)


def build_rag_prompt(user_query: str, retrieved: List[Tuple[Dict[str, Any], float]], target_lang: Optional[str] = None) -> str:
    context = "\n\n".join(
        f"--- DOC {i+1} (distance={dist:.3f}) ---\n{build_context_snippet(doc)}"
        for i, (doc, dist) in enumerate(retrieved)
    )

    lang = (target_lang or detect_text_lang(user_query)).lower()

    if lang.startswith("tr"):
        assistant_role = "Fortnite, Minecraft ve VALORANT için oyun sorun giderme asistanısın."
        rules = "\n".join([
            language_user_rule(lang),
            "Kurallar:",
            "- Verilen bağlamdaki çözümleri kullan.",
            "- Bağlam yetersizse neyin eksik olduğunu söyle.",
            "- Adım adım uygulanabilir öneriler ver, gerekirse güvenlik notu ekle, çok uzun cevaplar verme.",
            "- Birden fazla oyun geçiyorsa her biri için ayrı yanıt ver.",
        ])
        write_line = "En iyi cevabı yaz:"
        uq = "Kullanıcı sorusu:"
    else:
        assistant_role = "You are a troubleshooting assistant for Fortnite, Minecraft, and VALORANT."
        rules = "\n".join([
            language_user_rule(lang),
            "Rules:",
            "- Prefer solutions in the provided context.",
            "- If the context is insufficient, say what is missing.",
            "- Give step-by-step actions and safety notes where relevant, do not give long answers.",
            "- If multiple games are mentioned, answer each separately.",
        ])
        write_line = "Write the best possible answer:"
        uq = "User question:"

    return f"""{assistant_role}

{rules}

{uq}
{user_query}

Context:
{context}

{write_line}
"""


def call_llm(prompt: str, target_lang: Optional[str] = None, max_rewrite_attempts: int = 1) -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[HATA] OPENAI_API_KEY bulunamadı. .env dosyanıza OPENAI_API_KEY ekleyin."

    lang = (target_lang or "en").lower()
    system_rule = language_system_rule(lang)

    client = OpenAI(api_key=api_key)

    def _call(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=messages,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        answer = _call([
            {"role": "system", "content": system_rule},
            {"role": "user", "content": prompt},
        ])

        attempts = 0
        while attempts < max_rewrite_attempts and answer and (not is_lang_match(answer, lang)):
            attempts += 1
            rewrite_instruction = (
                "Rewrite the answer strictly in ENGLISH. Do not include any Turkish words/characters. Keep the meaning and steps the same."
                if not lang.startswith("tr")
                else
                "Yanıtı yalnızca TÜRKÇE olarak yeniden yaz. İngilizce kelime/ifade kullanma. Anlamı ve adımları koru."
            )
            answer = _call([
                {"role": "system", "content": system_rule},
                {"role": "user", "content": rewrite_instruction + "\n\nANSWER:\n" + answer},
            ])

        return answer
    except Exception as e:
        return f"[HATA] LLM çağrısı başarısız: {type(e).__name__}: {e}"


# =========================================================
# CORPUS LOADING (system_docs tamamen kaldırıldı)
# =========================================================
def load_corpus() -> List[Dict[str, Any]]:
    val_error_raw = load_jsonl(VALORANT_ERROR_CODES_PATH)
    val_known_raw = load_jsonl(VALORANT_KNOWN_ISSUES_PATH)

    val_error_docs = normalize_valorant_docs(val_error_raw, doc_type="valorant_error_code")
    val_known_docs = normalize_valorant_docs(val_known_raw, doc_type="valorant_known_issue")

    named_raw = load_jsonl(MINECRAFT_NAMED_ERRORS_PATH)
    mc_named_docs = normalize_minecraft_named_errors(named_raw)

    mc_help_raw = load_jsonl(MINECRAFT_OFFICIAL_HELP_PATH)
    mc_help_docs = normalize_minecraft_official_help(mc_help_raw)

    fn_raw = load_jsonl(FORTNITE_ERRORS_SOLUTIONS_PATH)
    fn_docs = normalize_fortnite_errors(fn_raw)

    # system_docs = build_system_docs()  # REMOVED
    corpus = val_error_docs + val_known_docs + mc_named_docs + mc_help_docs + fn_docs

    print(
        f"[INFO] Corpus loaded: {len(corpus)} docs "
        f"(valorant_error={len(val_error_docs)}, "
        f"valorant_known_issue={len(val_known_docs)}, "
        f"mc_named_error={len(mc_named_docs)}, "
        f"mc_official_help={len(mc_help_docs)}, "
        f"fortnite_error={len(fn_docs)})"
    )
    return corpus


# =========================================================
# MAIN
# =========================================================
def main():
    print("[INFO] Loading corpus...")
    corpus = load_corpus()

    rag = RAGIndex()
    print("[INFO] Building / loading index...")
    rag.build(corpus)

    print("\n[READY] Soru sorabilirsiniz. Çıkmak için boş bırakın.\n")
    while True:
        q = input("Soru: ").strip()
        if not q:
            break
        retrieved = rag.search(q)
        target_lang = detect_text_lang(q)
        prompt = build_rag_prompt(q, retrieved, target_lang=target_lang)
        answer = call_llm(prompt, target_lang=target_lang)
        print("\n--- CEVAP ---")
        print(answer)
        print("-------------\n")


if __name__ == "__main__":
    main()
