"""素材标签索引：将分析 Agent 生成的素材标签存入向量库，支持语义检索。"""

from __future__ import annotations

from typing import Any

import structlog

from mavea.config import get_settings

logger = structlog.get_logger(__name__)


class MaterialIndex:
    """素材标签语义索引。规划 Agent 可用"找一个产品特写镜头"这类查询检索素材。"""

    def __init__(self):
        self._settings = get_settings()
        self._store = None
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        from mavea.rag.vector_store import VectorStore
        self._store = VectorStore(collection_name="mavea_materials")
        self._initialized = True

    def index_materials(self, materials: list[dict[str, Any]]) -> None:
        """批量索引素材。

        Args:
            materials: list of {id, type, description, tags, scenes}
        """
        self._ensure_init()
        ids = []
        documents = []
        metadatas = []

        for mat in materials:
            for i, scene in enumerate(mat.get("scenes", [])):
                doc_id = f"{mat['id']}_scene_{i:03d}"
                tags = " ".join(scene.get("tags", []))
                doc = f"{scene.get('description', '')} {tags}"
                ids.append(doc_id)
                documents.append(doc)
                metadatas.append({
                    "material_id": mat["id"],
                    "scene_index": i,
                    "start": scene.get("start", 0),
                    "end": scene.get("end", 0),
                    "tags": tags,
                })

        if ids:
            self._store._ensure_init()
            embeddings = self._store._get_embeddings(documents)
            self._store._collection.upsert(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas,
            )
            logger.info("material_index.indexed", scenes=len(ids))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """语义检索素材镜头。

        Returns:
            list of {material_id, scene_index, start, end, description, score}
        """
        self._ensure_init()
        results = self._store.query(query, top_k=top_k)
        return [
            {
                "material_id": r["metadata"]["material_id"],
                "scene_index": r["metadata"]["scene_index"],
                "start": r["metadata"]["start"],
                "end": r["metadata"]["end"],
                "description": r["content"],
                "score": r["score"],
            }
            for r in results
        ]
