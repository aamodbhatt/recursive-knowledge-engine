import hashlib
import html
import json
import math
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Set

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from app.text_utils import tokenize


class Retriever:
    TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".xml",
        ".html",
        ".htm",
        ".log",
        ".ini",
        ".cfg",
        ".conf",
        ".toml",
        ".env",
        ".rtf",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
    }
    SPECIAL_EXTENSIONS = {".pdf", ".docx"}
    SUPPORTED_EXTENSIONS = sorted(TEXT_EXTENSIONS | SPECIAL_EXTENSIONS)

    def __init__(
        self,
        embedding_model: str,
        index_path: Path,
        metadata_path: Path,
        chunk_size_words: int = 220,
        chunk_overlap_words: int = 40,
        query_cache_size: int = 256,
    ) -> None:
        self.embedding_model = embedding_model
        self.model: Optional[SentenceTransformer] = None
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.query_cache_size = query_cache_size
        self.lock = RLock()

        self.dimension: Optional[int] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict] = []
        self.last_embedding_at: Optional[datetime] = None
        self.chunk_hashes = set()
        self.chunk_hash_to_id: Dict[str, int] = {}
        self.query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.chunk_token_sets: List[Set[str]] = []
        self.inverted_index: Dict[str, Set[int]] = {}
        self.token_doc_freq: Dict[str, int] = {}

        self._load_state()

    def _load_state(self) -> None:
        with self.lock:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.dimension = int(self.index.d)

            if self.metadata_path.exists():
                self.chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                for item in self.chunks:
                    if "sources" not in item or not item.get("sources"):
                        item["sources"] = [item.get("source", "")]

                self.chunk_hashes = set()
                self.chunk_hash_to_id = {}
                for item in self.chunks:
                    chunk_id = int(item["chunk_id"])
                    chunk_hash = self._hash_text(str(item["text"]))
                    self.chunk_hashes.add(chunk_hash)
                    self.chunk_hash_to_id.setdefault(chunk_hash, chunk_id)
                self.last_embedding_at = datetime.fromtimestamp(
                    self.metadata_path.stat().st_mtime, tz=timezone.utc
                )
                self._rebuild_lexical_index()

    def _persist_state(self) -> None:
        if self.index is None:
            return

        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.chunks, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.embedding_model)

    def _index_chunk_tokens(self, chunk_id: int, text: str) -> None:
        token_set = set(tokenize(text))
        self.chunk_token_sets.append(token_set)
        for token in token_set:
            postings = self.inverted_index.setdefault(token, set())
            postings.add(chunk_id)
            self.token_doc_freq[token] = len(postings)

    def _rebuild_lexical_index(self) -> None:
        self.chunk_token_sets = []
        self.inverted_index = {}
        self.token_doc_freq = {}
        for chunk in self.chunks:
            self._index_chunk_tokens(int(chunk["chunk_id"]), str(chunk["text"]))

    def _ensure_index(self) -> None:
        if self.index is not None:
            return

        self._ensure_model()
        probe_embedding = self.model.encode(
            ["dimension probe"],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        self.dimension = int(probe_embedding.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        with self.lock:
            value = self.query_cache.get(key)
            if value is None:
                return None
            self.query_cache.move_to_end(key)
            return value

    def _cache_set(self, key: str, vector: np.ndarray) -> None:
        with self.lock:
            self.query_cache[key] = vector
            self.query_cache.move_to_end(key)
            if len(self.query_cache) > self.query_cache_size:
                self.query_cache.popitem(last=False)

    def _embed_queries(self, query: str) -> np.ndarray:
        cached = self._cache_get(query)
        if cached is not None:
            return cached

        self._ensure_model()
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        vector = np.asarray(vector, dtype="float32")
        self._cache_set(query, vector)
        return vector

    def _embed_texts(self, chunks: List[str]) -> np.ndarray:
        self._ensure_model()
        vectors = self.model.encode(
            chunks,
            normalize_embeddings=True,
            batch_size=min(32, len(chunks)),
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    @classmethod
    def is_supported_file(cls, filename: str) -> bool:
        return Path(filename).suffix.lower() in set(cls.SUPPORTED_EXTENSIONS)

    @staticmethod
    def _strip_markup(text: str) -> str:
        no_tags = re.sub(r"<[^>]+>", " ", text)
        clean = html.unescape(no_tags)
        return re.sub(r"\s+", " ", clean).strip()

    @staticmethod
    def _strip_rtf(text: str) -> str:
        clean = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
        clean = re.sub(r"\\[a-zA-Z]+\d*\s?", " ", clean)
        clean = re.sub(r"[{}]", " ", clean)
        return re.sub(r"\s+", " ", clean).strip()

    def _read_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix in self.TEXT_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if suffix in {".html", ".htm", ".xml"}:
                return self._strip_markup(text)
            if suffix == ".rtf":
                return self._strip_rtf(text)
            return text

        if suffix == ".docx":
            try:
                from docx import Document
            except Exception as exc:
                raise ValueError(
                    "DOCX parsing requires python-docx. Install dependencies and retry."
                ) from exc
            document = Document(str(file_path))
            lines = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
            return "\n".join(lines)

        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)

        supported = ", ".join(self.SUPPORTED_EXTENSIONS)
        raise ValueError(f"Unsupported file type. Supported formats: {supported}.")

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size_words, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start += self.chunk_size_words - self.chunk_overlap_words
        return chunks

    def _append_chunks(self, source: str, chunks: List[str]) -> int:
        if not chunks:
            return 0

        # Skip exact text duplicates so repeated uploads do not bloat FAISS.
        unique_chunks = []
        metadata_updated = False
        with self.lock:
            for chunk_text in chunks:
                chunk_hash = self._hash_text(chunk_text)
                existing_chunk_id = self.chunk_hash_to_id.get(chunk_hash)
                if existing_chunk_id is not None:
                    existing_chunk = self.chunks[existing_chunk_id]
                    sources = existing_chunk.setdefault("sources", [existing_chunk["source"]])
                    if source not in sources:
                        sources.append(source)
                        metadata_updated = True
                    continue
                self.chunk_hashes.add(chunk_hash)
                self.chunk_hash_to_id[chunk_hash] = len(self.chunks) + len(unique_chunks)
                unique_chunks.append(chunk_text)

        if not unique_chunks:
            if metadata_updated:
                with self.lock:
                    self._persist_state()
            return 0

        vectors = self._embed_texts(unique_chunks)
        self._ensure_index()

        with self.lock:
            start_id = len(self.chunks)
            for i, chunk_text in enumerate(unique_chunks):
                chunk_id = start_id + i
                self.chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": source,
                        "sources": [source],
                        "text": chunk_text,
                    }
                )
                self._index_chunk_tokens(chunk_id, chunk_text)
            self.index.add(vectors)
            self.last_embedding_at = datetime.now(timezone.utc)
            self._persist_state()

        return len(unique_chunks)

    def add_document(self, file_path: Path) -> int:
        text = self._read_file(file_path)
        chunks = self._chunk_text(text)
        return self._append_chunks(source=file_path.name, chunks=chunks)

    def has_data(self) -> bool:
        return len(self.chunks) > 0

    def total_chunks(self) -> int:
        return len(self.chunks)

    def has_sources(self, source_filters: Optional[List[str]]) -> bool:
        allowed = {
            str(item).strip()
            for item in (source_filters or [])
            if str(item).strip()
        }
        if not allowed:
            return False
        with self.lock:
            return any(bool(set(chunk.get("sources", [chunk.get("source", "")])) & allowed) for chunk in self.chunks)

    def is_model_ready(self) -> bool:
        return self.model is not None

    def stats(self) -> Dict:
        sources = set()
        for item in self.chunks:
            sources.update(item.get("sources", [item.get("source", "")]))
        sources.discard("")
        index_size_bytes = self.index_path.stat().st_size if self.index_path.exists() else 0
        return {
            "chunks_indexed": len(self.chunks),
            "documents_indexed": len(sources),
            "embedding_model_loaded": self.is_model_ready(),
            "query_cache_size": len(self.query_cache),
            "faiss_index_size_bytes": index_size_bytes,
            "last_embedding_at": self.last_embedding_at.isoformat() if self.last_embedding_at else None,
        }

    def clear_index(self) -> Dict:
        with self.lock:
            self.index = None
            self.dimension = None
            self.chunks = []
            self.chunk_hashes = set()
            self.chunk_hash_to_id = {}
            self.query_cache.clear()
            self.last_embedding_at = None
            self.chunk_token_sets = []
            self.inverted_index = {}
            self.token_doc_freq = {}

            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()

        return self.stats()

    @staticmethod
    def _clamp_score(value: float) -> float:
        return float(round(max(0.0, min(1.0, value)), 4))

    def _lexical_search(
        self,
        query: str,
        limit: int,
        allowed_ids: Optional[Set[int]] = None,
    ) -> Dict[int, float]:
        query_tokens = sorted(set(tokenize(query)))
        if not query_tokens or not self.chunks:
            return {}

        total_docs = max(1, len(allowed_ids) if allowed_ids is not None else len(self.chunks))
        scores: Dict[int, float] = {}
        with self.lock:
            for token in query_tokens:
                postings = self.inverted_index.get(token, set())
                if not postings:
                    continue
                if allowed_ids is not None:
                    postings = postings & allowed_ids
                    if not postings:
                        continue
                doc_freq = max(1, len(postings))
                idf = math.log((total_docs + 1) / (doc_freq + 0.5)) + 1.0
                for chunk_id in postings:
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + idf

        if not scores:
            return {}

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        max_score = ranked[0][1] if ranked else 1.0
        if max_score <= 0:
            return {}
        return {chunk_id: float(round(score / max_score, 4)) for chunk_id, score in ranked}

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filters: Optional[List[str]] = None,
    ) -> List[Dict]:
        if not self.has_data():
            return []

        self._ensure_index()
        allowed_sources = {
            str(item).strip()
            for item in (source_filters or [])
            if str(item).strip()
        }
        allowed_ids: Optional[Set[int]] = None
        if allowed_sources:
            with self.lock:
                allowed_ids = {
                    idx
                    for idx, chunk in enumerate(self.chunks)
                    if bool(set(chunk.get("sources", [chunk.get("source", "")])) & allowed_sources)
                }
            if not allowed_ids:
                return []

        dense_pool_size = min(
            len(self.chunks),
            len(self.chunks) if allowed_ids is not None else max(top_k * 4, top_k),
        )
        query_vec = self._embed_queries(query)

        with self.lock:
            dense_scores, dense_idxs = self.index.search(query_vec, dense_pool_size)

        dense_ranked: List[tuple[int, float]] = []
        for score, idx in zip(dense_scores[0].tolist(), dense_idxs[0].tolist()):
            if idx < 0:
                continue
            if allowed_ids is not None and idx not in allowed_ids:
                continue
            dense_ranked.append((idx, self._clamp_score(score)))

        lexical_scores = self._lexical_search(query, dense_pool_size, allowed_ids=allowed_ids)
        dense_score_map = {idx: score for idx, score in dense_ranked}
        dense_rank_map = {idx: rank for rank, (idx, _score) in enumerate(dense_ranked, start=1)}
        lexical_rank_map = {
            idx: rank
            for rank, (idx, _score) in enumerate(
                sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True),
                start=1,
            )
        }

        candidate_ids = set(dense_rank_map.keys()) | set(lexical_scores.keys())
        fused: List[tuple[int, float]] = []
        for idx in candidate_ids:
            dense_score = dense_score_map.get(idx, 0.0)
            lexical_score = lexical_scores.get(idx, 0.0)
            dense_rrf = 1.0 / (60 + dense_rank_map[idx]) if idx in dense_rank_map else 0.0
            lexical_rrf = 1.0 / (60 + lexical_rank_map[idx]) if idx in lexical_rank_map else 0.0
            rrf_score = dense_rrf + lexical_rrf
            fused_score = (0.62 * dense_score) + (0.28 * lexical_score) + (0.10 * min(1.0, rrf_score * 60))
            fused.append((idx, self._clamp_score(fused_score)))

        fused.sort(key=lambda item: item[1], reverse=True)
        final_hits: List[Dict] = []
        for idx, fused_score in fused[:top_k]:
            metadata = self.chunks[idx]
            chunk_sources = metadata.get("sources", [metadata.get("source", "")])
            chosen_source = metadata.get("source", "")
            if allowed_sources:
                for source_name in chunk_sources:
                    if source_name in allowed_sources:
                        chosen_source = source_name
                        break
            final_hits.append(
                {
                    "chunk_id": metadata["chunk_id"],
                    "source": chosen_source,
                    "text": metadata["text"],
                    "score": fused_score,
                }
            )
        return final_hits
