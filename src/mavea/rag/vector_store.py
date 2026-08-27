"""Chroma 向量库封装。

延迟加载 BGE Embedding 模型，避免导入时就下载/加载大模型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from mavea.config import get_settings

logger = structlog.get_logger(__name__)


class VectorStore:
    """Chroma 向量库封装，用于剪辑模板检索。"""

    def __init__(self, collection_name: str | None = None):
        self._settings = get_settings()
        self._collection_name = collection_name or self._settings.rag.collection_name
        self._client = None
        self._collection = None
        self._embedding_model = None
        self._initialized = False
        self.embed_ok = True  # 向量模型是否可用（不可用时上层自动降级为纯 BM25）

    def _load_embed_model(self):
        """加载句向量模型。
        优先 local_files_only（模型已缓存时秒加载、完全不联网，避免 HF 在线校验
        在弱网下反复重试卡十几分钟）；本地没有才联网下载一次。
        """
        from sentence_transformers import SentenceTransformer
        name = self._settings.rag.embedding_model
        try:
            return SentenceTransformer(name, local_files_only=True)
        except Exception:
            return SentenceTransformer(name)  # 首次使用，需联网下载

    def _ensure_init(self) -> None:
        """延迟初始化：首次使用时才加载 Chroma 和 Embedding 模型。"""
        if self._initialized:
            return

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir = Path(self._settings.rag.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vector_store.init", collection=self._collection_name, count=self._collection.count())
        self._initialized = True

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """使用 BGE 模型生成向量；模型不可用时置 embed_ok=False 并抛出，由上层降级。"""
        if self._embedding_model is None:
            try:
                self._embedding_model = self._load_embed_model()
            except Exception as e:  # 离线/下载失败：不卡死，交给上层走纯 BM25
                self.embed_ok = False
                logger.warning("vector_store.embed_unavailable", error=str(e)[:120])
                raise
        return self._embedding_model.encode(texts, normalize_embeddings=True).tolist()

    def add_templates(self, templates: list[dict[str, Any]]) -> None:
        """批量添加模板。

        Args:
            templates: 模板 dict 列表，每个包含 name/scenario/description/editing_params
        """
        self._ensure_init()
        ids = []
        documents = []
        metadatas = []

        for t in templates:
            ids.append(t["name"])
            # 检索文本：名称+场景+描述+结构描述
            doc_text = (
                f"{t['name']} {t['scenario']} {t['description']} "
                + " ".join(s.get("description", "") for s in t.get("structure", []))
            )
            documents.append(doc_text)
            metadatas.append({
                "scenario": t["scenario"],
                "target_duration": t.get("target_duration", 30),
                "content_json": json.dumps(t, ensure_ascii=False),
            })

        embeddings = self._get_embeddings(documents)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("vector_store.added", count=len(templates))

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """向量检索。

        Returns:
            list of {name, content, score, metadata}
        """
        self._ensure_init()
        if self._collection.count() == 0:
            return []

        query_embedding = self._get_embeddings([text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "name": results["ids"][0][i],
                "content": results["documents"][0][i],
                "score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
                "metadata": results["metadatas"][0][i],
            })
        return items

    def count(self) -> int:
        self._ensure_init()
        return self._collection.count()

    def clear(self) -> None:
        """清空集合（用于重建索引）。"""
        self._ensure_init()
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
