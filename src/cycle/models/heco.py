"""HeCo graph module — used only by configs/baseline.yaml (samples2015 source).

Wang et al., "Self-supervised Heterogeneous Graph Pre-training Through
Dependency Maximization", KDD 2021.

Two views:
  sc (network schema): GCN propagation through each meta-path adjacency matrix
  mp (meta-path):      attention aggregation over sampled per-type neighbor lists

Only pretrain_heco() and the returned sc-view embeddings are consumed by train.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeCo(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        feats_dim_list: list[int],   # one per entry in inputs["feats"]
        feat_drop: float,
        attn_drop: float,
        num_paths: int,              # len(inputs["mps"])
        sample_rate: list[int],      # neighbors to sample per nei type
        nei_num: int,                # len(inputs["nei"])
        tau: float,
        lam: float,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.tau         = tau
        self.lam         = lam
        self.sample_rate = sample_rate
        self.nei_num     = nei_num

        # Type-specific feature projection: feats_dim_list[t] → hidden_dim
        # For samples2015: [2914, 828, 828]
        self.fc_list = nn.ModuleList([
            nn.Linear(fd, hidden_dim, bias=True) for fd in feats_dim_list
        ])
        self._feat_drop = nn.Dropout(feat_drop)
        self._attn_drop = nn.Dropout(attn_drop)

        # SC view: learnable semantic attention query vector
        self.sc_attn = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.sc_attn)

        # MP view: intra-type attention + semantic attention query
        self.mp_intra = nn.ModuleList([
            nn.Linear(hidden_dim * 2, 1, bias=False) for _ in range(nei_num)
        ])
        self.mp_attn = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.mp_attn)

    # ------------------------------------------------------------------
    # SC (network schema) view
    # ------------------------------------------------------------------

    def _sc_view(self, h: torch.Tensor, mps: list) -> torch.Tensor:
        """GCN-style aggregation over meta-path adjacency matrices.

        For each meta-path Φ:
          h^Φ = mp^Φ @ h                       (sparse mm)
          β_Φ = (1/n) Σ_i q^T tanh(h^Φ_i)     (global semantic score)
        z_sc = Σ_Φ softmax(β)_Φ · h^Φ
        """
        path_embs = [torch.sparse.mm(mp, h) for mp in mps]  # [(n, hidden_dim), ...]

        betas = torch.stack([
            (torch.tanh(e) * self.sc_attn).sum(dim=-1).mean()
            for e in path_embs
        ])                                              # (num_paths,)
        w = self._attn_drop(F.softmax(betas, dim=0))   # (num_paths,)

        z_sc = sum(w[i] * path_embs[i] for i in range(len(path_embs)))
        return F.elu(z_sc)                              # (n, hidden_dim)

    # ------------------------------------------------------------------
    # MP (meta-path) view
    # ------------------------------------------------------------------

    def _mp_view(self, z_list: list, nei: list) -> torch.Tensor:
        """Attention aggregation over sampled per-type neighbor lists.

        For neighbor type t:
          - sample up to sample_rate[t] neighbors per node
          - α_j = softmax(LeakyReLU(a_t^T [z_list[0]_i || z_list[t+1]_j]))
          - h_t_i = Σ_j α_j · z_list[t+1]_j   (type-specific neighbor features)
        Combine types with the same global semantic attention form as the sc view.
        """
        device    = z_list[0].device
        n         = z_list[0].shape[0]
        type_embs = []

        for t in range(self.nei_num):
            S    = self.sample_rate[t]
            pad  = torch.zeros(n, S, dtype=torch.long, device=device)
            mask = torch.zeros(n, S, dtype=torch.bool,  device=device)

            for i, nbrs in enumerate(nei[t]):
                nbrs = nbrs.to(device)
                m    = len(nbrs)
                if m == 0:
                    continue
                chosen = nbrs[torch.randperm(m)[:S]] if m > S else nbrs
                k = len(chosen)
                pad[i,  :k] = chosen
                mask[i, :k] = True

            z_nbr  = z_list[t + 1][pad]                         # (n, S, hidden_dim)
            z_self = z_list[0].unsqueeze(1).expand(-1, S, -1)   # (n, S, hidden_dim)

            e = self.mp_intra[t](
                torch.cat([z_self, z_nbr], dim=-1)
            ).squeeze(-1)                                        # (n, S)
            e = F.leaky_relu(e, 0.2)
            e = e.masked_fill(~mask, float('-inf'))
            alpha = self._attn_drop(F.softmax(e, dim=1))        # (n, S)

            h_t = (alpha.unsqueeze(-1) * z_nbr).sum(1)          # (n, hidden_dim)
            isolated = ~mask.any(1)
            if isolated.any():                                   # fallback to anchor features
                h_t = torch.where(isolated.unsqueeze(-1), z_list[0], h_t)
            type_embs.append(h_t)

        betas = torch.stack([
            (torch.tanh(e) * self.mp_attn).sum(dim=-1).mean()
            for e in type_embs
        ])                                             # (nei_num,)
        w = self._attn_drop(F.softmax(betas, dim=0))  # (nei_num,)

        z_mp = sum(w[t] * type_embs[t] for t in range(self.nei_num))
        return F.elu(z_mp)                             # (n, hidden_dim)

    # ------------------------------------------------------------------
    # forward + contrastive loss
    # ------------------------------------------------------------------

    def forward(self, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        feats, mps, nei = inputs["feats"], inputs["mps"], inputs["nei"]
        z_list = [F.elu(fc(self._feat_drop(f))) for fc, f in zip(self.fc_list, feats)]
        return self._sc_view(z_list[0], mps), self._mp_view(z_list, nei)

    def contrastive_loss(
        self,
        z_sc: torch.Tensor,
        z_mp: torch.Tensor,
        pos:  torch.Tensor,  # (n, n) sparse 0/1 positive-pair matrix
    ) -> torch.Tensor:
        """Symmetric InfoNCE: lam * L(sc→mp) + (1-lam) * L(mp→sc)."""
        return (self.lam * self._infonce(z_sc, z_mp, pos)
                + (1 - self.lam) * self._infonce(z_mp, z_sc, pos))

    def _infonce(
        self,
        anchor: torch.Tensor,  # (n, hidden_dim)
        target: torch.Tensor,  # (n, hidden_dim)
        pos:    torch.Tensor,  # (n, n) sparse 0/1 positive-pair matrix
    ) -> torch.Tensor:
        """Sum-then-log InfoNCE; denominator = all n nodes.

        pos.to_dense() is 828×828 ≈ 2.7 MB — negligible.
        Every node has ≥1 positive via kNN(k=4), but the has_pos guard is kept
        for correctness with arbitrary pos matrices.
        """
        sim      = (anchor @ target.t()) / self.tau         # (n, n)
        log_Z    = torch.logsumexp(sim, dim=1)              # (n,)
        pos_mask = pos.to_dense().bool()                    # (n, n)
        sim_pos  = sim.masked_fill(~pos_mask, float('-inf'))
        log_num  = torch.logsumexp(sim_pos, dim=1)          # (n,)
        has_pos  = pos_mask.any(dim=1)
        if not has_pos.any():
            return anchor.sum() * 0
        return (-(log_num - log_Z))[has_pos].mean()


# ------------------------------------------------------------------
# pretraining entry point (called by train.py)
# ------------------------------------------------------------------

def pretrain_heco(
    heco: HeCo,
    inputs: dict,
    lr: float,
    weight_decay: float,
    patience: int,
    max_epochs: int,
    logger,
) -> torch.Tensor:
    """Pretrain HeCo; return sc-view embeddings of shape (num_nodes, hidden_dim).

    Early stopping on training loss with epsilon=1e-6 tolerance to avoid
    premature stopping on noisy loss curves (e.g. debug_n=1024 subsamples).
    Restores best weights before computing the final embeddings.
    """
    pos       = inputs["pos"]
    optimizer = torch.optim.Adam(heco.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss  = float('inf')
    no_improve = 0
    best_state = {k: v.clone() for k, v in heco.state_dict().items()}

    heco.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        z_sc, z_mp = heco(inputs)
        loss = heco.contrastive_loss(z_sc, z_mp, pos)
        loss.backward()
        optimizer.step()

        l = loss.item()
        if l < best_loss - 1e-6:   # epsilon avoids spurious early stop on noisy curves
            best_loss  = l
            no_improve = 0
            best_state = {k: v.clone() for k, v in heco.state_dict().items()}
        else:
            no_improve += 1

        if (epoch + 1) % 500 == 0:
            logger.info(f"HeCo epoch {epoch + 1:5d}: loss={l:.4f}  best={best_loss:.4f}")

        if no_improve >= patience:
            logger.info(
                f"HeCo early stop at epoch {epoch + 1} "
                f"(no improvement for {patience} epochs, best {best_loss:.4f})"
            )
            break

    heco.load_state_dict(best_state)
    heco.eval()
    with torch.no_grad():
        z_sc, _ = heco(inputs)
    return z_sc.detach()   # sc-view embeddings; fusion.source=rel maps to this
