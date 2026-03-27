
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, List, Optional

import torch

from lodd_sv.verification.uncertainty import uncertainty_score
from lodd_sv.verification.knowledge_base import KnowledgeBase


@dataclass
class RouterResult:
    triggered: bool
    uncertainty_value: float
    retrieved_fact: Optional[str] = None
    query_used: Optional[str] = None


class StatisticalRouter:

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        threshold: float = 2.0,
        uncertainty_method: str = "entropy",
        aggregate: str = "max",
    ) -> None:
        self.kb = knowledge_base
        self.threshold = threshold
        self.uncertainty_method = uncertainty_method
        self.aggregate = aggregate

    def format_context_as_query(self, token_ids: torch.Tensor, id_to_token: Optional[Callable[[int], str]] = None) -> str:
        ids = token_ids.cpu().tolist()
        if isinstance(ids[0], list):
            ids = ids[0]
        recent = ids[-5:] if len(ids) >= 5 else ids
        if id_to_token:
            return " ".join(id_to_token(i) for i in recent)
        return " ".join(str(i) for i in recent)

    def __call__(
        self,
        probs: torch.Tensor,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        id_to_token: Optional[Callable[[int], str]] = None,
    ) -> RouterResult:
        u = uncertainty_score(probs, mask=mask, method=self.uncertainty_method, aggregate=self.aggregate)
        u_val = u[0].item()


        if not math.isfinite(u_val):
            u_val = float("inf")
        triggered = u_val > self.threshold
        retrieved_fact = None
        query_used = None
        if triggered:
            query_used = self.format_context_as_query(token_ids, id_to_token=id_to_token)
            results = self.kb.query(query_used, limit=1)
            if results:
                retrieved_fact = results[0].get("text") or results[0].get("object") or str(results[0])
        return RouterResult(
            triggered=triggered,
            uncertainty_value=u_val,
            retrieved_fact=retrieved_fact,
            query_used=query_used,
        )

    def inject_fact_into_logits(
        self,
        logits: torch.Tensor,
        fact_tokens: List[int],
        position: int,
        boost: float = 10.0,
    ) -> torch.Tensor:
        out = logits.clone()
        for tid in fact_tokens:
            if tid < out.size(-1):
                out[:, position, tid] = out[:, position, tid] + boost
        return out
