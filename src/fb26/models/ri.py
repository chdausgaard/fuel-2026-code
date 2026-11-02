"""Randomization inference for P1 (memo §7.1).

Assignment model: within each PADD group, every permutation of the states'
shock deviations is equally likely, independently across groups. Permuting
within a group leaves its mean unchanged, so counties inherit the permuted
deviation and B̃^(k) = A d^(k), with A[i, s] = E_c when row i is in state s.
Every control is invariant under the permutations, so the residualized
treatment of draw k is R d^(k) with R = M_Z A, computed once.

Test statistic: the studentized coefficient (state-clustered CR1 t).
Confidence sets invert the test under constant effects: Y^b = Y - b·B̃ (μ is
absorbed by the controls). For each draw the coefficient's numerator is linear
in b and the squared cluster-score sum is quadratic in b, so p(b) costs
O(K) for any b once the draws are summarized.
"""
from dataclasses import dataclass, field

import numpy as np

from fb26.models.p1 import Residualizer, cr1_factor

SEED = 20261019
K_DRAWS = 9_999
CHUNK = 500


def permuted_deviations(dev: np.ndarray, groups: np.ndarray, k: int,
                        rng: np.random.Generator) -> np.ndarray:
    """(k, n_states) matrix; row j permutes dev within each group."""
    out = np.empty((k, dev.size))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[:, idx] = dev[rng.permuted(np.tile(idx, (k, 1)), axis=1)]
    return out


@dataclass
class Draws:
    """Per-draw sufficient statistics; t_k(b) = (a - b·c) / sqrt(f·(s0 + 2b·s1 + b²·s2))."""
    a: np.ndarray
    c: np.ndarray
    s0: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    f: float

    def t(self, b: float) -> np.ndarray:
        return (self.a - b * self.c) / np.sqrt(self.f * (self.s0 + 2 * b * self.s1 + b * b * self.s2))


def summarize(x: np.ndarray, y: np.ndarray, x_obs: np.ndarray, w: np.ndarray,
              cmat: np.ndarray, f: float) -> Draws:
    """x is (n, m): residualized treatment for m draws; cmat is the (G, n) cluster indicator."""
    wx = w[:, None] * x
    a = wx.T @ y
    c = wx.T @ x_obs
    den = np.sum(wx * x, axis=0)
    p = cmat @ (wx * y[:, None])
    p_obs = cmat @ (wx * x_obs[:, None])
    q = cmat @ (wx * x)
    alpha = p - (a / den) * q
    gamma = -p_obs + (c / den) * q
    return Draws(a, c, np.sum(alpha ** 2, axis=0), np.sum(alpha * gamma, axis=0),
                 np.sum(gamma ** 2, axis=0), f)


def _concat(parts: list[Draws]) -> Draws:
    return Draws(*(np.concatenate([getattr(p, k) for p in parts]) for k in ('a', 'c', 's0', 's1', 's2')),
                 parts[0].f)


@dataclass
class RIResult:
    t_obs: float
    p_value: float
    ci95: tuple[float, float]
    ci90: tuple[float, float]
    eps_star: float
    k_draws: int
    seed: int
    flags: list[str] = field(default_factory=list)


class Inversion:
    """p(b) and inverted confidence sets from the observed and permuted draw summaries."""

    def __init__(self, obs: Draws, draws: Draws, k: int):
        self.obs, self.draws, self.k = obs, draws, k

    def p(self, b: float = 0.0) -> float:
        t_obs = abs(self.obs.t(b)[0])
        return (1 + np.sum(np.abs(self.draws.t(b)) >= t_obs * (1 - 1e-12))) / (self.k + 1)

    def _edge(self, inside: float, outside: float, alpha: float) -> float:
        for _ in range(60):
            mid = (inside + outside) / 2
            inside, outside = (mid, outside) if self.p(mid) > alpha else (inside, mid)
        return (inside + outside) / 2

    def ci(self, alpha: float, center: float, scale: float,
           n_grid: int = 2001) -> tuple[tuple[float, float], list[str]]:
        """Inverted-test confidence set {b : p(b) > alpha}, reported as its hull."""
        flags = []
        for width in (10, 20, 40):
            grid = np.linspace(center - width * scale, center + width * scale, n_grid)
            keep = np.array([self.p(b) > alpha for b in grid])
            if not keep.any():
                return (np.nan, np.nan), ['ci_empty']
            if not (keep[0] or keep[-1]):
                break
        else:
            flags.append('ci_unbounded_on_grid')
        idx = np.flatnonzero(keep)
        if idx.size != idx[-1] - idx[0] + 1:
            flags.append('ci_nonconvex')
        lo = grid[idx[0]] if idx[0] == 0 else self._edge(grid[idx[0]], grid[idx[0] - 1], alpha)
        hi = grid[idx[-1]] if idx[-1] == n_grid - 1 else self._edge(grid[idx[-1]], grid[idx[-1] + 1], alpha)
        return (float(lo), float(hi)), flags


class Design:
    """The parts of an RI problem that do not depend on the assignment.

    rows: y (outcome), e (exposure E_c), state (row -> state index 0..S-1),
    z (controls), w (weights), cluster labels (states); n_states = S.
    """

    def __init__(self, y, e, state, z, w, clusters, n_states: int):
        res = Residualizer(z, w)
        a_mat = np.zeros((len(y), n_states))
        a_mat[np.arange(len(y)), state] = e
        self.r = res(a_mat)
        self.y = res(np.asarray(y, float))
        self.w = np.asarray(w, float)
        codes = np.unique(np.asarray(clusters), return_inverse=True)[1]
        g = int(codes.max()) + 1
        self.cmat = (codes[None, :] == np.arange(g)[:, None]).astype(float)
        self.f = cr1_factor(len(y), res.rank + 1, g)

    def inversion(self, dev: np.ndarray, groups: np.ndarray, k: int, rng: np.random.Generator) -> Inversion:
        """Observed assignment dev (length S) against k within-group permutations of it."""
        x_obs = self.r @ dev
        obs = summarize(x_obs[:, None], self.y, x_obs, self.w, self.cmat, self.f)
        parts = []
        for start in range(0, k, CHUNK):
            d = permuted_deviations(dev, groups, min(CHUNK, k - start), rng)
            parts.append(summarize(self.r @ d.T, self.y, x_obs, self.w, self.cmat, self.f))
        return Inversion(obs, _concat(parts), k)


def center_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """values minus their group means (unweighted)."""
    out = np.asarray(values, float).copy()
    for g in np.unique(groups):
        idx = groups == g
        out[idx] -= out[idx].mean()
    return out


class TwoLayerDesign:
    """RI design with a state layer and a sub-state layer (memo §5.1, O6).

    The permuted vector is [state deviations; demeaned cluster means of u]. Layer 1 permutes the state
    deviations within PADD groups, layer 2 the cluster means within state × stratum; the groups array
    labels both (labels must differ across layers). A row's B̃ = e × (d[state] + d[S + unit]), with
    unit = -1 for rows without a sub-state unit (their layer-2 part is 0). The county's fixed
    within-cluster residual drops out of B̃ and is not an argument. With n_units = 0 this is Design.
    """

    def __init__(self, y, e, state, z, w, clusters, n_states: int, unit=None, n_units: int = 0):
        self.res = res = Residualizer(z, w)
        n = len(y)
        self.e = np.asarray(e, float)
        self.state = np.asarray(state)
        self.unit = np.full(n, -1) if unit is None else np.asarray(unit)
        self.n_states, self.n_units = n_states, n_units
        self.y = res(np.asarray(y, float))
        self.w = np.asarray(w, float)
        codes = np.unique(np.asarray(clusters), return_inverse=True)[1]
        g = int(codes.max()) + 1
        self.cmat = (codes[None, :] == np.arange(g)[:, None]).astype(float)
        self.f = cr1_factor(n, res.rank + 1, g)
        self.r = None
        if n_states + n_units <= res.rank:      # precomputed R = M_Z A is cheaper than residualizing each chunk
            a_mat = np.zeros((n, n_states + n_units))
            a_mat[np.arange(n), self.state] = self.e
            has = self.unit >= 0
            a_mat[np.flatnonzero(has), n_states + self.unit[has]] = self.e[has]
            self.r = res(a_mat)

    def residual_treatment(self, d: np.ndarray) -> np.ndarray:
        """(n, m) residualized B̃ for the m assignments in the rows of d."""
        if self.r is not None:
            return self.r @ d.T
        v = self.e[:, None] * d[:, self.state].T
        has = np.flatnonzero(self.unit >= 0)
        v[has] += self.e[has, None] * d[:, self.n_states + self.unit[has]].T
        return self.res(v)

    def inversion(self, dev: np.ndarray, groups: np.ndarray, k: int, rng: np.random.Generator) -> Inversion:
        """Observed assignment dev (length S + L) against k independent two-layer permutations of it."""
        x_obs = self.residual_treatment(dev[None, :])[:, 0]
        obs = summarize(x_obs[:, None], self.y, x_obs, self.w, self.cmat, self.f)
        parts = []
        for start in range(0, k, CHUNK):
            d = permuted_deviations(dev, groups, min(CHUNK, k - start), rng)
            parts.append(summarize(self.residual_treatment(d), self.y, x_obs, self.w, self.cmat, self.f))
        return Inversion(obs, _concat(parts), k)


class RandomizationTest(Inversion):
    """RI over within-group permutations of state shock deviations.

    Arguments as in Design, plus dev (observed deviations, length S) and
    groups (PADD group per state). States without rows still enter the
    permutations (memo §3: all 51 units).
    """

    def __init__(self, y, e, state, z, w, clusters, dev, groups,
                 k: int = K_DRAWS, seed: int = SEED):
        self.dev, self.groups, self.seed = np.asarray(dev, float), np.asarray(groups), seed
        design = Design(y, e, state, z, w, clusters, self.dev.size)
        inv = design.inversion(self.dev, self.groups, k, np.random.default_rng(seed))
        super().__init__(inv.obs, inv.draws, k)

    def run(self, beta_hat: float, se: float) -> RIResult:
        ci95, f95 = self.ci(0.05, beta_hat, se)
        ci90, f90 = self.ci(0.10, beta_hat, se)
        return RIResult(float(self.obs.t(0.0)[0]), float(self.p(0.0)), ci95, ci90,
                        float(max(abs(ci90[0]), abs(ci90[1]))), self.k, self.seed,
                        sorted(set(f95 + f90)))


class TwoLayerRandomizationTest(RandomizationTest):
    """Two-layer counterpart with the same test-inversion report.

    dev and groups contain the full assignment pool, including units without
    outcome rows. Group labels must distinguish the two layers.
    """

    def __init__(self, y, e, state, z, w, clusters, dev, groups, n_states: int,
                 unit=None, *, k: int = K_DRAWS, seed: int = SEED):
        self.dev, self.groups, self.seed = np.asarray(dev, float), np.asarray(groups), seed
        design = TwoLayerDesign(y, e, state, z, w, clusters, n_states, unit=unit,
                                n_units=self.dev.size - n_states)
        inv = design.inversion(self.dev, self.groups, k, np.random.default_rng(seed))
        Inversion.__init__(self, inv.obs, inv.draws, k)
