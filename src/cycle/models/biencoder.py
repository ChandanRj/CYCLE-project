"""Bi-encoder: two independent BERT towers for mention and entity (Eq. 1-5)."""

import os

import torch
import torch.nn as nn
from transformers import AutoModel

# Allow a local snapshot path to override the HF hub model name — useful when
# the cached model is in .bin format and the transformers version rejects torch.load.
# Set CYCLE_BERT_PATH=/path/to/local/snapshot to activate.
_BERT_PATH_OVERRIDE = os.environ.get("CYCLE_BERT_PATH")


class _BertTower(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        path = _BERT_PATH_OVERRIDE or model_name
        self.bert_model = AutoModel.from_pretrained(path)  # must be named bert_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != 0).long()
        return self.bert_model(
            input_ids=input_ids, attention_mask=mask
        ).last_hidden_state[:, 0, :]   # [CLS] token, Eq. 3/4


class BiEncoder(nn.Module):
    def __init__(self, bert_model: str) -> None:
        super().__init__()
        self.context_encoder   = _BertTower(bert_model)  # T1 — mention encoder, Eq. 3
        self.candidate_encoder = _BertTower(bert_model)  # T2 — entity encoder,  Eq. 4
        self.output_dim: int   = self.context_encoder.bert_model.config.hidden_size

    def encode_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of mention contexts (Eq. 1/3)."""
        return self.context_encoder(input_ids)

    def encode_candidate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of entity descriptions (Eq. 2/4)."""
        return self.candidate_encoder(input_ids)
