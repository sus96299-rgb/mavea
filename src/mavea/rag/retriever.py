"""混合检索器：向量检索 + BM25 关键词检索 + BGE Reranker 重排序。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from mavea.config import get_settings
from mavea.rag.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class TemplateRetriever:
    """剪辑模板混合检索器。"""

    def __init__(self):
        self._settings = get_settings()
        self._vector_store = VectorStore()
        self._bm25 = None
        self._bm25_corpus: list[dict[str, Any]] = []
        self._reranker = None
        self._templates_loaded = False
        self._vector_ok = True  # 向量库/模型不可用时自动降级为纯 BM25

    def load_templates(self, templates_dir: Path | None = None) -> int:
        """从 templates 目录加载所有 JSON 模板并构建索引。

        Returns:
            加载的模板数量
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        templates = []
        for f in sorted(templates_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                templates.append(json.load(fh))

        if not templates:
            logger.warning("retriever.no_templates", dir=str(templates_dir))
            return 0

        # 向量索引：模型不可用/离线时自动降级为纯 BM25，绝不阻塞主流程
        try:
            if self._vector_store.count() < len(templates):
                self._vector_store.add_templates(templates)
            else:
                logger.info("retriever.templates_cached", count=self._vector_store.count())
            self._vector_ok = getattr(self._vector_store, "embed_ok", True)
        except Exception as e:
            self._vector_ok = False
            logger.warning("retriever.vector_disabled_fallback_bm25", error=str(e)[:120])

        # BM25 索引（内存索引，每次重建但很快）
        self._build_bm25(templates)

        self._templates_loaded = True
        logger.info("retriever.loaded", count=len(templates))
        return len(templates)

    def _build_bm25(self, templates: list[dict[str, Any]]) -> None:
        """构建 BM25 关键词索引。"""
        from rank_bm25 import BM25Okapi

        self._bm25_corpus = templates
        # 简单中文分词：按字符 + 按空格
        tokenized = []
        for t in templates:
            text = (
                f"{t['name']} {t['scenario']} {t['description']} "
                + " ".join(s.get("shot_type", "") + " " + s.get("description", "")
                           for s in t.get("structure", []))
            )
            # 中文按字分词，英文按词
            tokens = list(text.lower().replace("，", " ").replace("。", " "))
            tokenized.append(tokens)
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 关键词检索。"""
        if self._bm25 is None or not self._bm25_corpus:
            return []

        tokens = list(query.lower())
        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:
                t = self._bm25_corpus[idx]
                results.append({
                    "name": t["name"],
                    "content": t["description"],
                    "score": float(scores[idx]),
                    "metadata": {"content_json": json.dumps(t, ensure_ascii=False)},
                })
        return results

    def _rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        """使用 BGE Reranker 重排序。"""
        if not candidates:
            return []

        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker
                model_name = self._settings.rag.reranker_model
                try:
                    # 已缓存则离线秒加载，避免弱网下 HF 在线校验反复重试
                    self._reranker = FlagReranker(
                        model_name, use_fp16=False, local_files_only=True
                    )
                except TypeError:
                    # 旧版 FlagEmbedding 不支持 local_files_only 参数
                    self._reranker = FlagReranker(model_name, use_fp16=False)
                except Exception:
                    self._reranker = FlagReranker(model_name, use_fp16=False)
            except Exception as e:
                logger.warning("reranker.load_failed", error=str(e)[:120])
                # Reranker 不可用时按融合分排序（纯 BM25/向量结果依然可用）
                return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]

        pairs = [[query, c["content"]] for c in candidates]
        scores = self._reranker.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [scores]

        for c, s in zip(candidates, scores, strict=False):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """混合检索：向量 + BM25 融合 → Rerank。

        Returns:
            list of {name, content, score, metadata, template}
        """
        if not self._templates_loaded:
            self.load_templates()

        k = top_k or self._settings.rag.top_k
        vec_weight = self._settings.rag.vector_weight
        bm25_weight = self._settings.rag.bm25_weight

        # 向量检索
        vec_results = []
        if self._vector_ok:
            try:
                vec_results = self._vector_store.query(query, top_k=k * 2)
            except Exception as e:
                self._vector_ok = False
                logger.warning("retriever.vector_query_failed_bm25_only", error=str(e)[:120])

        # BM25 检索
        bm25_results = self._bm25_search(query, top_k=k * 2)

        # 融合（按 name 去重，加权分数）
        fused: dict[str, dict[str, Any]] = {}
        for r in vec_results:
            fused[r["name"]] = {**r, "score": r["score"] * vec_weight}
        for r in bm25_results:
            if r["name"] in fused:
                fused[r["name"]]["score"] += r["score"] * bm25_weight
            else:
                fused[r["name"]] = {**r, "score": r["score"] * bm25_weight}

        candidates = list(fused.values())
        if not candidates:
            return []

        # Rerank
        reranked = self._rerank(query, candidates, self._settings.rag.rerank_top_n)

        # 解析模板 JSON
        for r in reranked:
            if "metadata" in r and "content_json" in r["metadata"]:
                try:
                    r["template"] = json.loads(r["metadata"]["content_json"])
                except json.JSONDecodeError:
                    r["template"] = None

        return reranked


# 全局单例
_retriever: TemplateRetriever | None = None


def get_retriever() -> TemplateRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TemplateRetriever()
    return _retriever
