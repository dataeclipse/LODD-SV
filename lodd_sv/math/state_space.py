
from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch


class DiscreteStateSpace:

    def __init__(self, state_dim: int) -> None:
        self.state_dim = state_dim

    def sample_uniform(self, shape: tuple, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.state_dim, size=shape, device=device)

    def one_hot(self, states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            states.clamp(0, self.state_dim - 1), num_classes=self.state_dim
        ).float()


class VocabularyStateSpace(DiscreteStateSpace):

    def __init__(
        self,
        vocab_size: int,
        pad_id: Optional[int] = None,
        unk_id: Optional[int] = None,
        token_to_id: Optional[Dict[str, int]] = None,
        id_to_token: Optional[Dict[int, str]] = None,
    ) -> None:
        super().__init__(state_dim=vocab_size)
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.unk_id = unk_id
        self._token_to_id = token_to_id or {}
        self._id_to_token = id_to_token or {}

    def state_to_token(self, state_id: int) -> str:
        return self._id_to_token.get(state_id, f"<id_{state_id}>")

    def token_to_state(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_id or 0)

    def encode_tokens(self, tokens: List[str]) -> torch.Tensor:
        ids = [self.token_to_state(t) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def decode_states(self, states: torch.Tensor) -> List[str]:
        states = states.cpu().tolist()
        if isinstance(states[0], list):
            return [[self.state_to_token(s) for s in seq] for seq in states]
        return [self.state_to_token(s) for s in states]
