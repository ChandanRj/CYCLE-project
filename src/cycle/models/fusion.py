"""Score fusion: additive tanh-gated combination of BERT scores and graph embeddings.

Not specified by a paper equation — implementation detail for bridging the graph
module's hidden_dim-space embeddings into the BERT cosine-score space.
Design rationale from configs/paper.yaml:
  "the paper does not specify how graph embeddings reach the EL scores; we use an
   additive tanh-fusion, QID-aligned and trained"
"""

import torch
import torch.nn as nn


class ScoreFusion(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        bert_dim: int,
        weight: float,
        projection: str,
        alignment: str,
        mode: str,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        self.alignment = alignment
        self.mode      = mode
        self.weight    = float(weight)
        # Must be named .linear — evaluate.py replaces it by direct attribute assignment
        # before load_state_dict to handle checkpoints with mismatched bert_dim.
        self.linear    = nn.Linear(graph_dim, bert_dim, bias=False)

        if projection == "learned":
            nn.init.zeros_(self.linear.weight)
        elif projection == "random_frozen":
            nn.init.normal_(
                self.linear.weight,
                generator=torch.Generator().manual_seed(init_seed),
            )
            for p in self.linear.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        scores:   torch.Tensor,  # (B, B) in-batch similarity scores
        cand_emb: torch.Tensor,  # (B, bert_dim) candidate embeddings
        z:        torch.Tensor,  # (B, graph_dim) graph embeddings for this batch
    ) -> torch.Tensor:
        """Additive tanh-gated fusion.

        gate ≈ 1 for weight=45 (tanh(45) ≈ 1); makes the linear term fully active.
        """
        gate     = torch.tensor(self.weight, dtype=scores.dtype, device=scores.device).tanh()
        ctx_bias = self.linear(z.float()).to(scores.dtype)   # (B, bert_dim)
        return scores + gate * (ctx_bias @ cand_emb.t())

    def select_batch_embeddings(
        self,
        node_embeds: torch.Tensor,  # (num_nodes, graph_dim)
        label_idx:   torch.Tensor,  # (B,) entity node ids for this batch
        step:        int,
        batch_size:  int,
    ) -> torch.Tensor:              # (B, graph_dim)
        """Return graph embeddings for the current batch.

        alignment=qid:        direct lookup by label_id (paper mode).
        alignment=sequential: sequential window through node_embeds, independent
                              of which entities are actually in the batch (baseline).
        """
        if self.alignment == "qid":
            return node_embeds[label_idx]
        n, b  = node_embeds.size(0), label_idx.size(0)
        start = (step * batch_size) % n
        end   = start + b
        if end <= n:
            return node_embeds[start:end]
        return torch.cat([node_embeds[start:], node_embeds[:end - n]], dim=0)

    def enhance_candidates(
        self,
        cand_emb: torch.Tensor,          # (N, bert_dim)
        z:        torch.Tensor,          # (N, graph_dim)
        present:  torch.Tensor | None = None,  # (N,) float mask (1=known, 0=unknown)
    ) -> torch.Tensor:                   # (N, bert_dim)
        """Add projected graph delta to candidate encodings (evaluation / pool enhancement)."""
        delta = self.linear(z.float()).to(cand_emb.dtype)
        if present is not None:
            delta = delta * present.unsqueeze(-1)
        return cand_emb + delta
