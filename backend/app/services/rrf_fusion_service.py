from __future__ import annotations

from collections.abc import Sequence

from app.schemas.vector_search_result import VectorSearchResult


class RRFFusionService:
    """使用 Reciprocal Rank Fusion 合并多路检索排名。"""

    def __init__(self, rank_constant: int = 60) -> None:
        if rank_constant <= 0:
            raise ValueError("rank_constant must be greater than zero")
        self.rank_constant = rank_constant

    def fuse(
        self,
        rankings: Sequence[Sequence[VectorSearchResult]],
        top_k: int,
    ) -> list[VectorSearchResult]:
        """按 Chunk ID 聚合多路排名并返回统一排序结果。"""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        non_empty_rankings = [ranking for ranking in rankings if ranking]
        if not non_empty_rankings:
            return []

        scores: dict[int, float] = {}
        candidates: dict[int, VectorSearchResult] = {}

        for ranking in non_empty_rankings:
            for rank, result in enumerate(ranking, start=1):
                candidates.setdefault(result.chunk_id, result)
                scores[result.chunk_id] = (
                    scores.get(result.chunk_id, 0.0)
                    + 1.0 / (self.rank_constant + rank)
                )

        # 归一化到约 [0, 1]，方便调试接口观察；最终排序仍完全由 RRF 决定。
        maximum_score = (
            len(non_empty_rankings)
            / (self.rank_constant + 1)
        )

        fused = [
            candidates[chunk_id].model_copy(
                update={"score": score / maximum_score}
            )
            for chunk_id, score in scores.items()
        ]
        fused.sort(key=lambda result: (-result.score, result.chunk_id))
        return fused[:top_k]
