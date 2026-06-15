"""Recursive Logit route choice model (Fosgerau et al., 2013)."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.special

from rcm.entities import ObservedTrip, RoadNetwork

log = logging.getLogger(__name__)


class ConvergenceWarning(UserWarning):
    """Issued when Bellman value iteration does not converge."""


def _build_log_transition_matrix(
    network: RoadNetwork,
    beta: np.ndarray,
    link_features: np.ndarray,
    dest_node: int | None = None,
) -> scipy.sparse.csr_matrix:
    """Build (n_links+1) x (n_links+1) CSR log-weight transition matrix.

    Entry M[i, j] = beta @ x_j  (log-weight) for link pairs (i, j) where
    ``to_node(i) == from_node(j)``.  The terminal transition entry is 0.0
    (= log 1).  When *dest_node* is supplied, links whose ``to_node`` equals
    *dest_node* additionally receive a transition to the virtual terminal at
    index ``n_links``.
    """
    n_links = len(network.links)
    n = n_links + 1  # last index = virtual terminal

    utils = link_features @ beta  # (n_links,)

    from_nodes = np.array([lk.from_node for lk in network.links])
    to_nodes = np.array([lk.to_node for lk in network.links])

    argsort_from = np.argsort(from_nodes, kind="stable")
    sorted_from_nodes = from_nodes[argsort_from]

    lefts = np.searchsorted(sorted_from_nodes, to_nodes, side="left")
    rights = np.searchsorted(sorted_from_nodes, to_nodes, side="right")
    counts = rights - lefts

    total_edges = int(counts.sum())
    if total_edges > 0:
        cum_counts = np.empty(n_links + 1, dtype=np.intp)
        cum_counts[0] = 0
        np.cumsum(counts, out=cum_counts[1:])
        rows_ll = np.repeat(np.arange(n_links), counts)
        starts = np.repeat(lefts, counts)
        local_offsets = np.arange(total_edges) - np.repeat(cum_counts[:-1], counts)
        cols_ll = argsort_from[starts + local_offsets]
        data_ll = utils[cols_ll]
    else:
        rows_ll = np.empty(0, dtype=np.intp)
        cols_ll = np.empty(0, dtype=np.intp)
        data_ll = np.empty(0, dtype=float)

    if dest_node is not None:
        dest_links = np.where(to_nodes == dest_node)[0]
        if len(dest_links) > 0:
            rows = np.concatenate([rows_ll, dest_links])
            cols = np.concatenate(
                [cols_ll, np.full(len(dest_links), n_links, dtype=np.intp)]
            )
            data = np.concatenate([data_ll, np.zeros(len(dest_links))])
        else:
            rows, cols, data = rows_ll, cols_ll, data_ll
    else:
        rows, cols, data = rows_ll, cols_ll, data_ll

    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)


def _solve_bellman(
    log_M: scipy.sparse.csr_matrix,
    dest_link_idx: int,
    *,
    gamma: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 500,
) -> np.ndarray:
    """Solve discounted Bellman equation V[a] = logsumexp_j(log_M[a,j] + gamma*V[j])."""
    n = log_M.shape[0]
    V = np.zeros(n, dtype=float)
    V[dest_link_idx] = 0.0

    log_M_csr = log_M.tocsr()
    indptr = log_M_csr.indptr
    col_indices = log_M_csr.indices
    base_data = log_M_csr.data

    row_nnz = np.diff(indptr)
    non_empty = row_nnz > 0
    row_starts = indptr[:-1][non_empty]

    for _ in range(max_iter):
        V_old = V.copy()

        combined = base_data + gamma * V[col_indices]

        row_max = np.full(n, -np.inf)
        if row_starts.size:
            row_max[non_empty] = np.maximum.reduceat(combined, row_starts)

        row_max_finite = np.where(np.isfinite(row_max), row_max, 0.0)

        exp_shifted = np.exp(combined - np.repeat(row_max_finite, row_nnz))
        row_sum = np.zeros(n)
        if row_starts.size:
            row_sum[non_empty] = np.add.reduceat(exp_shifted, row_starts)

        with np.errstate(divide="ignore"):
            V_new = np.where(row_sum > 0.0, row_max_finite + np.log(row_sum), -np.inf)
        V_new[dest_link_idx] = 0.0

        finite_mask = np.isfinite(V_new) & np.isfinite(V_old)
        diff = (
            float(np.max(np.abs(V_new[finite_mask] - V_old[finite_mask])))
            if finite_mask.any()
            else 0.0
        )
        V = V_new
        if diff < eps:
            return V

    warnings.warn("Bellman iteration did not converge", ConvergenceWarning)
    return V


def _conditional_log_likelihood(
    trips: list[ObservedTrip],
    network: RoadNetwork,
    params: np.ndarray,
    link_features: np.ndarray,
) -> float:
    """Compute conditional log-likelihood for the discounted RL model.

    ``params[-1]`` is ``gamma_raw`` (unconstrained); actual discount is
    ``gamma = sigmoid(gamma_raw) ∈ (0, 1)``.
    """
    beta = params[:-1]
    gamma = float(scipy.special.expit(params[-1]))

    n_links = len(network.links)
    virtual_idx = n_links

    destinations = list({t.destination for t in trips})
    V_cache: dict[int, np.ndarray] = {}
    for dest in destinations:
        log_M = _build_log_transition_matrix(network, beta, link_features, dest_node=dest)
        V_cache[dest] = _solve_bellman(log_M, dest_link_idx=virtual_idx, gamma=gamma)

    link_by_id = {lk.link_id: i for i, lk in enumerate(network.links)}
    utils = (link_features @ beta).astype(float)

    from_node_to_idxs: dict[int, list[int]] = {}
    for i, lk in enumerate(network.links):
        from_node_to_idxs.setdefault(lk.from_node, []).append(i)

    total_ll = 0.0
    for trip in trips:
        V = V_cache[trip.destination]
        link_ids = trip.chosen_route.link_ids
        if not link_ids:
            continue

        origin = trip.origin
        origin_idxs = from_node_to_idxs.get(origin)
        if not origin_idxs:
            continue
        origin_arr = np.array(origin_idxs)
        V_origin = float(scipy.special.logsumexp(utils[origin_arr] + gamma * V[origin_arr]))

        route_idxs = [link_by_id[lid] for lid in link_ids]
        route_utils = float(utils[route_idxs].sum())
        route_V = float(V[route_idxs].sum())
        total_ll += route_utils + (gamma - 1.0) * route_V - V_origin

    return total_ll


class RecursiveLogit:
    """Recursive Logit route choice model (Fosgerau et al., 2013)."""

    def __init__(
        self,
        max_iter: int = 500,
        conv_eps: float = 1e-6,
    ) -> None:
        self.max_iter = max_iter
        self.conv_eps = conv_eps
        self._beta: np.ndarray | None = None
        self._beta_se: np.ndarray | None = None
        self._train_ll: float | None = None
        self._ll_null: float | None = None
        self._V_cache: dict[int, np.ndarray] | None = None
        self._link_features: np.ndarray | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._network: RoadNetwork | None = None
        self._result: scipy.optimize.OptimizeResult | None = None
        self._extra_features_raw: np.ndarray | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        trips: list[ObservedTrip],
        network: RoadNetwork,
        *,
        extra_link_features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """Fit RL model by maximising conditional log-likelihood (L-BFGS-B).

        Parameters
        ----------
        trips:
            List of observed pedestrian trips.
        network:
            Road network with links in a consistent order.
        extra_link_features:
            Optional ``(n_links, n_extra)`` array of additional per-link
            explanatory variables aligned with ``network.links`` order.
            Default features are link length only.
        feature_names:
            Human-readable names for columns in ``extra_link_features``.
            When provided, used in ``summary()`` output instead of generic
            ``beta[k]`` labels.  Length must equal ``n_extra``.
        """
        self._network = network
        raw_features = np.array([[lk.length_m] for lk in network.links], dtype=float)

        if extra_link_features is not None:
            self._extra_features_raw = extra_link_features.astype(float)
            raw_features = np.concatenate([raw_features, self._extra_features_raw], axis=1)
            if feature_names is not None:
                if len(feature_names) != extra_link_features.shape[1]:
                    raise ValueError(
                        f"feature_names length ({len(feature_names)}) must equal "
                        f"extra_link_features columns ({extra_link_features.shape[1]})"
                    )
                self._feature_names = ["length_m"] + list(feature_names)
            else:
                self._feature_names = None
        else:
            self._extra_features_raw = None
            self._feature_names = ["length_m"] if feature_names is None else None

        feat_mean = raw_features.mean(axis=0)
        feat_std = raw_features.std(axis=0)
        feat_std = np.where(feat_std == 0.0, 1.0, feat_std)
        self._feat_mean = feat_mean
        self._feat_std = feat_std

        link_features = (raw_features - feat_mean) / feat_std
        self._link_features = link_features
        n_feat = link_features.shape[1]

        null_params = np.zeros(n_feat + 1, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._ll_null = _conditional_log_likelihood(trips, network, null_params, link_features)

        def neg_ll(params: np.ndarray) -> float:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    ll = _conditional_log_likelihood(trips, network, params, link_features)
                return -ll if np.isfinite(ll) else 1e10
            except Exception:
                return 1e10

        x0 = np.zeros(n_feat + 1, dtype=float)
        x0[-1] = 2.0  # gamma_raw initial: sigmoid(2.0) ≈ 0.88
        result = scipy.optimize.minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": self.conv_eps},
        )
        self._result = result
        self._beta = result.x.copy()

        # Numerical Hessian for standard errors
        h = 1e-4
        n_par = len(result.x)
        H = np.zeros((n_par, n_par))
        for i in range(n_par):
            for j in range(n_par):
                xpp = result.x.copy(); xpp[i] += h; xpp[j] += h
                xpn = result.x.copy(); xpn[i] += h; xpn[j] -= h
                xnp = result.x.copy(); xnp[i] -= h; xnp[j] += h
                xnn = result.x.copy(); xnn[i] -= h; xnn[j] -= h
                H[i, j] = (neg_ll(xpp) - neg_ll(xpn) - neg_ll(xnp) + neg_ll(xnn)) / (4.0 * h**2)
        try:
            H_inv = np.linalg.inv(H)
            se = np.sqrt(np.abs(np.diag(H_inv)))
        except np.linalg.LinAlgError:
            se = np.full(n_par, float("nan"))
        self._beta_se = se

        self._train_ll = -float(result.fun)

        gamma = float(scipy.special.expit(self._beta[-1]))
        n_links = len(network.links)
        virtual_idx = n_links
        self._V_cache = {}
        for dest in {t.destination for t in trips}:
            log_M = _build_log_transition_matrix(
                network, self._beta[:-1], link_features, dest_node=dest
            )
            self._V_cache[dest] = _solve_bellman(log_M, dest_link_idx=virtual_idx, gamma=gamma)

    def _build_features(self, network: RoadNetwork) -> np.ndarray:
        if self._link_features is not None and self._network is network:
            return self._link_features
        raw = np.array([[lk.length_m] for lk in network.links], dtype=float)
        if self._extra_features_raw is not None:
            raw = np.concatenate([raw, self._extra_features_raw], axis=1)
        if self._feat_mean is not None and self._feat_std is not None:
            return (raw - self._feat_mean) / self._feat_std
        return raw

    def predict(self, origin: int, destination: int, network: RoadNetwork) -> list[int]:
        """Greedily follow the highest-value link from origin to destination."""
        if self._beta is None:
            raise RuntimeError("Model not fitted yet")
        link_features = self._build_features(network)
        n_links = len(network.links)
        virtual_idx = n_links
        gamma = float(scipy.special.expit(self._beta[-1]))
        if destination not in (self._V_cache or {}):
            log_M = _build_log_transition_matrix(
                network, self._beta[:-1], link_features, dest_node=destination
            )
            V = _solve_bellman(log_M, dest_link_idx=virtual_idx, gamma=gamma)
        else:
            assert self._V_cache is not None
            V = self._V_cache[destination]

        link_by_id = {lk.link_id: i for i, lk in enumerate(network.links)}
        utils = (link_features @ self._beta[:-1]).astype(float)

        from_node_to_links: dict[int, list] = {}
        for lk in network.links:
            from_node_to_links.setdefault(lk.from_node, []).append(lk)

        route_links: list[int] = []
        current_node = origin
        visited_nodes: set[int] = {current_node}

        for _ in range(len(network.links) + 1):
            if current_node == destination:
                break
            outgoing = from_node_to_links.get(current_node)
            if not outgoing:
                break
            best_link = max(
                outgoing,
                key=lambda lk: utils[link_by_id[lk.link_id]] + V[link_by_id[lk.link_id]],
            )
            route_links.append(best_link.link_id)
            current_node = best_link.to_node
            if current_node in visited_nodes:
                break
            visited_nodes.add(current_node)

        return route_links

    def log_likelihood(self, trips: list[ObservedTrip], network: RoadNetwork) -> float:
        """Compute log-likelihood on given trips."""
        if self._beta is None:
            raise RuntimeError("Model not fitted yet")
        return _conditional_log_likelihood(
            trips, network, self._beta, self._build_features(network)
        )

    def summary(self) -> dict[str, Any]:
        """Print estimation summary and return results dict.

        Returns
        -------
        dict with keys ``beta``, ``beta_se``, ``t_values``, ``p_values``,
        ``gamma``, ``gamma_se``, ``gamma_t``, ``gamma_p``, ``ll_null``,
        ``ll_final``, ``rho_squared``, ``adj_rho_squared``.
        """
        if (
            self._beta is None
            or self._beta_se is None
            or self._train_ll is None
            or self._ll_null is None
        ):
            raise RuntimeError("Model not fitted yet")

        import scipy.stats

        beta = self._beta[:-1]
        beta_se = self._beta_se[:-1]
        with np.errstate(invalid="ignore"):
            t_values = np.where(beta_se > 0, beta / beta_se, np.nan)
        p_values = np.where(
            np.isfinite(t_values),
            2.0 * scipy.stats.norm.sf(np.abs(t_values)),
            np.nan,
        )

        gamma = float(scipy.special.expit(self._beta[-1]))
        gamma_raw_se = float(self._beta_se[-1])
        gamma_se = gamma_raw_se * gamma * (1.0 - gamma)
        gamma_t = gamma / gamma_se if gamma_se > 0 else float("nan")
        gamma_p = (
            float(2.0 * scipy.stats.norm.sf(abs(gamma_t)))
            if np.isfinite(gamma_t)
            else float("nan")
        )

        ll_null = self._ll_null
        ll_final = self._train_ll
        n_params = len(self._beta)
        rho_sq = 1.0 - ll_final / ll_null if ll_null != 0.0 else float("nan")
        rho_sq_adj = (
            1.0 - (ll_final - n_params) / ll_null if ll_null != 0.0 else float("nan")
        )

        def _stars(p: float) -> str:
            if not np.isfinite(p):
                return "   "
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "** "
            if p < 0.05:
                return "*  "
            return "   "

        # Determine parameter labels
        if self._feature_names is not None:
            param_labels = self._feature_names
        else:
            param_labels = [f"beta[{k}]" for k in range(len(beta))]

        w = 72
        print("=" * w)
        print("Recursive Logit — Estimation Summary")
        print("=" * w)
        print(f"  LL(0)  [initial log-likelihood]:   {ll_null:>12.4f}")
        print(f"  LL(*)  [final log-likelihood]:     {ll_final:>12.4f}")
        print(f"  Rho-squared        (McFadden R²):  {rho_sq:>12.4f}")
        print(f"  Adj. rho-squared:                  {rho_sq_adj:>12.4f}")
        print("-" * w)
        header = f"  {'Parameter':<20}  {'Estimate':>10}  {'Std. Err.':>10}"
        header += f"  {'t-value':>9}  {'p-value':>9}  {'':3}"
        print(header)
        print("-" * w)
        for k in range(len(beta)):
            label = param_labels[k] if k < len(param_labels) else f"beta[{k}]"
            b = float(beta[k])
            se = float(beta_se[k])
            t = float(t_values[k])
            p = float(p_values[k])
            row = f"  {label:<20}  {b:>10.4f}  {se:>10.4f}"
            row += f"  {t:>9.4f}  {p:>9.4f}  {_stars(p)}"
            print(row)
        gamma_row = f"  {'gamma':<20}  {gamma:>10.4f}  {gamma_se:>10.4f}"
        gamma_row += f"  {gamma_t:>9.4f}  {gamma_p:>9.4f}  {_stars(gamma_p)}"
        print(gamma_row)
        print("=" * w)
        print("Signif. codes:  *** p<0.001  ** p<0.01  * p<0.05")

        return {
            "beta": beta.tolist(),
            "beta_se": beta_se.tolist(),
            "t_values": t_values.tolist(),
            "p_values": p_values.tolist(),
            "gamma": gamma,
            "gamma_se": gamma_se,
            "gamma_t": gamma_t,
            "gamma_p": gamma_p,
            "ll_null": ll_null,
            "ll_final": ll_final,
            "rho_squared": rho_sq,
            "adj_rho_squared": rho_sq_adj,
        }

    @property
    def n_params(self) -> int:
        return 1 if self._beta is None else len(self._beta)

    @property
    def aic(self) -> float:
        if self._train_ll is None:
            raise RuntimeError("Model not fitted yet")
        return -2.0 * self._train_ll + 2.0 * self.n_params

    def save(self, path: Path | str) -> None:
        """Save fitted weights to *path* directory."""
        if self._beta is None:
            raise RuntimeError("Model not fitted yet; call fit() before save().")
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        save_dict: dict = dict(
            beta=self._beta,
            beta_se=self._beta_se if self._beta_se is not None else np.array([float("nan")]),
            train_ll=np.array([self._train_ll if self._train_ll is not None else float("nan")]),
            ll_null=np.array([self._ll_null if self._ll_null is not None else float("nan")]),
            feat_mean=self._feat_mean if self._feat_mean is not None else np.array([float("nan")]),
            feat_std=self._feat_std if self._feat_std is not None else np.array([float("nan")]),
        )
        if self._extra_features_raw is not None:
            save_dict["extra_features_raw"] = self._extra_features_raw
        np.savez(dest / "weights.npz", **save_dict)
        meta = {"feature_names": self._feature_names}
        (dest / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> RecursiveLogit:
        """Load a previously saved model from *path* directory."""
        src = Path(path)
        model = cls()
        data = np.load(src / "weights.npz")
        model._beta = data["beta"]
        model._beta_se = data["beta_se"]
        model._train_ll = float(data["train_ll"][0])
        model._ll_null = float(data["ll_null"][0]) if "ll_null" in data else None
        model._feat_mean = data["feat_mean"] if "feat_mean" in data else None
        model._feat_std = data["feat_std"] if "feat_std" in data else None
        model._extra_features_raw = (
            data["extra_features_raw"] if "extra_features_raw" in data else None
        )
        meta_path = src / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            model._feature_names = meta.get("feature_names")
        return model
