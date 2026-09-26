"""Graph module: cross-year contrastive learning (CYCLE paper, Sec. 5.2).

Eq. 6/7  — attention aggregation on raw X (feature/kNN and relation views)
Eq. 8/9  — shared ELU MLP: fc_in (W^1 feat→hidden), fc_out (W^2 hidden→hidden)
Eq. 10   — P_i^r = newly added relation edges t-1→t (cross-year positives)
Eq. 11   — N_i^r = removed relation edges t-1→t (cross-year negatives for L_f);
           EXTENSION: relation_extra_negatives > 0 substitutes sampled relation
           non-neighbors where N_i^r is empty (always, in GCL-TempEL)
Eq. 13   — L_f: anchor=z_f, target=z_r, positive=P_i^r, denom=P_i^r ∪ N_i^r
Eq. 14   — L_r: anchor=z_r, target=z_f, positive=P_i^f (kNN), denom≈all n
Eq. 15   — joint objective assembled in train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class PaperGraphModule(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        feat_drop: float,
        attn_drop: float,
        tau: float,
        neighbor_threshold: int,
        isolated_fallback: str,
        feature_negatives: str,
        relation_extra_negatives: int = 0,
        agg_chunk: int = 1024,
    ):
        super().__init__()
        self.tau = tau
        self.K = neighbor_threshold
        self.isolated_fallback = isolated_fallback
        self.feature_negatives = feature_negatives
        self.relation_extra_negatives = relation_extra_negatives
        self._agg_chunk = agg_chunk  # process this many nodes at a time in _aggregate

        # Attention on raw X per Eq. 6/7: a^T [x_i || x_j], x_i ∈ R^feat_dim.
        # AMBIG-A: standard GAT applies W before attention; Eq. 6/7 uses x directly.
        # With feat_dim≈2929 attn_fc is Linear(5858, 1) — fine on a 32 GB GPU.
        self.attn_fc = nn.Linear(feat_dim * 2, 1, bias=False)

        # Shared ELU MLP (Eq. 8/9), applied AFTER aggregation, same weights for
        # both views.
        self.fc_in  = nn.Linear(feat_dim,   hidden_dim)  # W^1  Eq. 8
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)  # W^2  Eq. 9

        self._feat_drop = nn.Dropout(feat_drop)
        self._attn_drop = nn.Dropout(attn_drop)

        # Set by resample_neighbors() each epoch; None until first call.
        self._knn_nbrs: torch.Tensor | None = None
        self._knn_mask: torch.Tensor | None = None
        self._rel_nbrs: torch.Tensor | None = None
        self._rel_mask: torch.Tensor | None = None
        self._pos_idx:  torch.Tensor | None = None  # (n, K) cross-year positive indices
        self._pos_mask: torch.Tensor | None = None  # (n, K) bool valid mask
        self._neg_idx:  torch.Tensor | None = None  # (n, K) cross-year negative indices
        self._neg_mask: torch.Tensor | None = None  # (n, K) bool valid mask
        self._rel_forbidden: torch.Tensor | None = None  # sorted i*n+j keys never usable as extra negatives

    # ------------------------------------------------------------------
    # epoch-start neighbor sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _make_nbr_table(
        nbr_lists: list,
        n: int,
        K: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (n, K) padded aggregation neighbor table + valid mask.

        nbr_lists[i] is a CPU LongTensor of neighbor ids.
        If |N_i| > K: sample K uniformly.  If <= K: use all, zero-pad the rest.
        """
        pad  = torch.zeros(n, K, dtype=torch.long)
        mask = torch.zeros(n, K, dtype=torch.bool)
        for i, lst in enumerate(nbr_lists):
            m = len(lst)
            if m == 0:
                continue
            if m > K:
                perm   = torch.randperm(m, generator=generator)[:K]
                chosen = lst[perm]
            else:
                chosen = lst
            k = len(chosen)
            pad[i,  :k] = chosen
            mask[i, :k] = True
        return pad.to(device), mask.to(device)

    @staticmethod
    def _make_pos_table(
        src: torch.Tensor | None,
        dst: torch.Tensor | None,
        n: int,
        K: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build (n, K) padded cross-year positive/negative index table + valid mask.

        pos_src/pos_dst (and neg_src/neg_dst) are already bidirectionalized.
        Sampling rule (Eq. 10/11): if |list_i| > K subsample K; otherwise use all.
        Returns (None, None) when no cross-year pairs exist.
        """
        if src is None or src.numel() == 0:
            return None, None
        lists: list[list[int]] = [[] for _ in range(n)]
        for s, d in zip(src.cpu().tolist(), dst.cpu().tolist()):
            lists[s].append(d)
        pad  = torch.zeros(n, K, dtype=torch.long)
        mask = torch.zeros(n, K, dtype=torch.bool)
        for i, lst in enumerate(lists):
            m = len(lst)
            if m == 0:
                continue
            if m > K:
                perm   = torch.randperm(m, generator=generator)[:K]
                chosen = [lst[p] for p in perm.tolist()]
            else:
                chosen = lst
            k = len(chosen)
            pad[i,  :k] = torch.tensor(chosen, dtype=torch.long)
            mask[i, :k] = True
        return pad.to(device), mask.to(device)

    def _sample_relation_non_neighbors(
        self,
        graph: dict,
        n: int,
        R: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:                # (n, R) CPU LongTensor
        """Sample R distinct non-neighbors of each node in the current-year relation graph.

        Excluded: the node itself, its relation neighbors, and its cross-year
        positives (P_i^r ⊆ relation neighbors already; excluded explicitly anyway).
        Rejection sampling — the relation graph is sparse, so redraws are rare.
        """
        if self._rel_forbidden is None:
            nbrs = graph["rel_neighbors"]
            src = torch.cat([torch.full((len(nb),), i, dtype=torch.long) for i, nb in enumerate(nbrs)])
            dst = torch.cat([nb.long() for nb in nbrs])
            ar = torch.arange(n)
            keys = [src * n + dst, ar * n + ar]
            if graph.get("pos_src") is not None:
                keys.append(graph["pos_src"].cpu() * n + graph["pos_dst"].cpu())
            self._rel_forbidden = torch.unique(torch.cat(keys))  # sorted
        rows = torch.arange(n).unsqueeze(1)
        idx = torch.randint(n, (n, R), generator=generator)
        while True:
            bad = torch.isin(rows * n + idx, self._rel_forbidden)
            same = idx.unsqueeze(2) == idx.unsqueeze(1)          # (n, R, R)
            bad |= torch.triu(same, diagonal=1).any(dim=1)       # repeats of an earlier entry
            if not bad.any():
                return idx
            idx[bad] = torch.randint(n, (int(bad.sum()),), generator=generator)

    def resample_neighbors(
        self,
        graph: dict,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        """Rebuild sampled aggregation + cross-year positive/negative tables.

        Called once per epoch by train.py before any forward/loss calls.
        """
        n = graph["num_nodes"]
        K = self.K
        self._knn_nbrs, self._knn_mask = self._make_nbr_table(
            graph["knn_neighbors"], n, K, device, generator
        )
        self._rel_nbrs, self._rel_mask = self._make_nbr_table(
            graph["rel_neighbors"], n, K, device, generator
        )
        self._pos_idx, self._pos_mask = self._make_pos_table(
            graph.get("pos_src"), graph.get("pos_dst"), n, K, device, generator
        )
        self._neg_idx, self._neg_mask = self._make_pos_table(
            graph.get("neg_src"), graph.get("neg_dst"), n, K, device, generator
        )
        R = self.relation_extra_negatives
        if R > 0:
            # EXTENSION beyond Eq. 11: real newly-removed-edge negatives are
            # structurally absent in this dataset (see diagnostic); substitute
            # negatives sampled from non-neighbors, mirroring the paper's own
            # N_i^f definition for the feature graph (Sec 5.3).
            # Used only for nodes whose strict N_i^r is empty.
            extra = self._sample_relation_non_neighbors(graph, n, R, generator).to(device)
            if self._neg_idx is None:
                self._neg_idx = extra
                self._neg_mask = torch.ones(n, R, dtype=torch.bool, device=device)
            else:
                strict_empty = ~self._neg_mask.any(dim=1, keepdim=True)
                self._neg_idx = torch.cat([self._neg_idx, extra], dim=1)
                self._neg_mask = torch.cat([self._neg_mask, strict_empty.expand(n, R)], dim=1)

    # ------------------------------------------------------------------
    # attention aggregation + MLP projection (Eq. 6/7/8/9)
    # ------------------------------------------------------------------

    def _aggregate_and_project(
        self,
        X: torch.Tensor,          # (n, feat_dim) dense, dropout already applied
        nbr_idx: torch.Tensor,    # (n, K) neighbor indices (padded with 0)
        nbr_mask: torch.Tensor,   # (n, K) valid entry mask
    ) -> torch.Tensor:            # (n, hidden_dim)
        """GAT aggregation (Eq. 6/7) + shared MLP (Eq. 8/9), using per-chunk gradient
        checkpointing so peak memory is O(chunk * K * feat_dim) not O(n * K * feat_dim).

        With n=10373, K=8, feat_dim=2929:
          Full-batch X_nbr ≈ 927 MiB — checkpoint avoids retaining it between chunks.
          Split-weight attention avoids the (c, K, 2*feat_dim) cat intermediate.
          Peak per chunk ≈ chunk * K * feat_dim * 4 bytes (one X_nbr_c at a time).
        """
        n  = X.shape[0]
        C  = self._agg_chunk
        z_chunks = []

        for start in range(0, n, C):
            end = min(start + C, n)
            idx_c  = nbr_idx[start:end]   # (c, K)
            mask_c = nbr_mask[start:end]  # (c, K)

            # Closure captures start/end by value via default-arg trick.
            def _chunk(X, idx_c, mask_c, _s=start, _e=end, _self=self):
                X_c     = X[_s:_e]          # (c, feat_dim) — view of X, no extra alloc
                X_nbr_c = X[idx_c]          # (c, K, feat_dim) — only alloc in checkpoint

                # Attention without (c,K,2*feat_dim) cat: split attn_fc weight.
                W        = _self.attn_fc.weight            # (1, 2*feat_dim)
                feat_dim = X_c.shape[1]
                e_self   = X_c @ W[:, :feat_dim].t()       # (c, 1)
                e_nbr    = (X_nbr_c @ W[:, feat_dim:].t()).squeeze(-1)  # (c, K)

                e = F.leaky_relu(e_self + e_nbr, negative_slope=0.2)  # (c, K)
                e = e.masked_fill(~mask_c, float('-inf'))
                alpha = _self._attn_drop(F.softmax(e, dim=1))          # (c, K)

                # bmm: (c,1,K) @ (c,K,feat_dim) → (c,feat_dim)
                h_c = torch.bmm(alpha.unsqueeze(1), X_nbr_c).squeeze(1)

                if _self.isolated_fallback == "self":
                    isolated_c = ~mask_c.any(dim=1)
                    if isolated_c.any():
                        h_c = torch.where(isolated_c.unsqueeze(-1), X_c, h_c)

                return F.elu(_self.fc_out(F.elu(_self.fc_in(h_c))))    # (c, hidden_dim)

            if self.training:
                # Gradient checkpointing: X_nbr_c is NOT saved; recomputed on backward.
                # Saves ~chunk*K*feat_dim*4 bytes × (n/chunk) activations per aggregate call.
                z_c = grad_checkpoint(_chunk, X, idx_c, mask_c, use_reentrant=False)
            else:
                z_c = _chunk(X, idx_c, mask_c)
            z_chunks.append(z_c)

        return torch.cat(z_chunks, dim=0)  # (n, hidden_dim)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, graph: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (z_r, z_f) for all n nodes.

        X.to_dense() ≈ 121 MB; _aggregate_and_project runs in checkpointed chunks so
        peak forward memory is ~(agg_chunk * K * feat_dim * 4) bytes per aggregate call.
        """
        assert self._knn_nbrs is not None, "call resample_neighbors() before forward()"

        X = graph["X"].to_dense()    # row-normalised bag-of-words (n, feat_dim)
        X = self._feat_drop(X)

        # Feature view: kNN graph, Eq. 6/7 → shared MLP Eq. 8/9
        z_f = self._aggregate_and_project(X, self._knn_nbrs, self._knn_mask)  # (n, hidden_dim)

        # Relation view: relation graph, same attention weights + same MLP
        z_r = self._aggregate_and_project(X, self._rel_nbrs, self._rel_mask)  # (n, hidden_dim)

        return z_r, z_f

    # ------------------------------------------------------------------
    # InfoNCE losses (Eq. 13/14)
    # ------------------------------------------------------------------

    def _infonce_f(
        self,
        anchor:   torch.Tensor,  # (n, hidden_dim)
        target:   torch.Tensor,  # (n, hidden_dim)
        pos_idx:  torch.Tensor,  # (n, K) positive indices (padded with 0)
        pos_mask: torch.Tensor,  # (n, K) bool valid mask
        neg_idx:  torch.Tensor,  # (n, K) negative indices (padded with 0)
        neg_mask: torch.Tensor,  # (n, K) bool valid mask
    ) -> torch.Tensor:
        """L_f inner: sum-then-log, denominator = P_i^r ∪ N_i^r (Eq. 13).

        Avoids the (n, n) similarity matrix by directly gathering target embeddings
        for the K positive and K negative indices, then computing dot products via bmm.
        Peak memory: 2 × (n, K, hidden_dim) ≈ 2 × n × K × hidden_dim × 4 bytes.
        """
        # Gather target embeddings for positives and negatives — no (n,n) matrix.
        T_pos = target[pos_idx]                                               # (n, K, d)
        T_neg = target[neg_idx]                                               # (n, K, d)
        a = anchor.unsqueeze(1)                                               # (n, 1, d)
        p_sim = torch.bmm(a, T_pos.transpose(-2, -1)).squeeze(1) / self.tau  # (n, K)
        q_sim = torch.bmm(a, T_neg.transpose(-2, -1)).squeeze(1) / self.tau  # (n, K)
        p_sim = p_sim.masked_fill(~pos_mask, float('-inf'))
        q_sim = q_sim.masked_fill(~neg_mask, float('-inf'))

        log_num = torch.logsumexp(p_sim, dim=1)                               # (n,)
        log_den = torch.logsumexp(torch.cat([p_sim, q_sim], dim=1), dim=1)   # (n,)

        per_node = -(log_num - log_den)                        # (n,)
        return per_node[pos_mask.any(dim=1)].mean()

    def _infonce_r(
        self,
        anchor:   torch.Tensor,  # (n, hidden_dim)
        target:   torch.Tensor,  # (n, hidden_dim)
        knn_idx:  torch.Tensor,  # (n, K) kNN positive indices (padded with 0)
        knn_mask: torch.Tensor,  # (n, K) bool valid mask
    ) -> torch.Tensor:
        """L_r inner: sum-then-log, denominator = all n nodes (Eq. 14 / Sec 5.3).

        Chunked over anchor rows to avoid the (n, n) = 430 MB similarity matrix.
        Each chunk produces (c, n) which is freed before the next chunk.
        Per-chunk output log_num_c, log_den_c are (c,) scalars — accumulated cheaply.
        """
        n  = anchor.shape[0]
        C  = self._agg_chunk
        T_knn = target[knn_idx]   # (n, K, hidden_dim) — always gathered once; 21 MB for K=8, d=64

        log_num_chunks: list[torch.Tensor] = []
        log_den_chunks: list[torch.Tensor] = []

        for start in range(0, n, C):
            end      = min(start + C, n)
            sim_c    = (anchor[start:end] @ target.t()) / self.tau            # (c, n)
            knn_c    = torch.bmm(
                anchor[start:end].unsqueeze(1),
                T_knn[start:end].transpose(-2, -1),
            ).squeeze(1) / self.tau                                           # (c, K)
            knn_c    = knn_c.masked_fill(~knn_mask[start:end], float('-inf'))

            log_num_chunks.append(torch.logsumexp(knn_c, dim=1))             # (c,)
            log_den_chunks.append(torch.logsumexp(sim_c, dim=1))             # (c,)

        log_num  = torch.cat(log_num_chunks)                                  # (n,)
        log_den  = torch.cat(log_den_chunks)                                  # (n,)
        per_node = -(log_num - log_den)
        return per_node[knn_mask.any(dim=1)].mean()

    def loss_f(self, z_r: torch.Tensor, z_f: torch.Tensor, graph: dict) -> torch.Tensor:
        """L_f: anchor=z_f, target=z_r, positives=P_i^r, denom=P_i^r ∪ N_i^r (Eq. 13).

        N_i^r includes the substitute non-neighbor negatives when
        relation_extra_negatives > 0 (see resample_neighbors).
        """
        if self._pos_idx is None:
            return z_f.sum() * 0    # no cross-year pairs for this year
        has_pos = self._pos_mask.any(dim=1)
        if not has_pos.any():       # all n nodes lack cross-year positives (e.g. debug_n subsample)
            return z_f.sum() * 0    # zero scalar on computation graph; other loss terms unaffected
        neg_idx  = self._neg_idx  if self._neg_idx  is not None else self._pos_idx
        neg_mask = (self._neg_mask if self._neg_mask is not None
                    else torch.zeros_like(self._pos_mask))
        return self._infonce_f(z_f, z_r, self._pos_idx, self._pos_mask, neg_idx, neg_mask)

    def loss_r(self, z_r: torch.Tensor, z_f: torch.Tensor, graph: dict) -> torch.Tensor:
        """L_r: anchor=z_r, target=z_f, positives=P_i^f (kNN), denom≈all n (Eq. 14)."""
        has_pos = self._knn_mask.any(dim=1)
        if not has_pos.any():       # all nodes isolated in feature graph
            return z_r.sum() * 0
        return self._infonce_r(z_r, z_f, self._knn_nbrs, self._knn_mask)
