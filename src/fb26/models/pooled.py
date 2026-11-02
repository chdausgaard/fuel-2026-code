"""Pooled validation estimate θ_hist (memo §6.3).

    Δv_{c,r,t} = α_{r,t} + θ·(I_t·B̃_{c,t}) + [cycle-specific design and precision controls] + ε
    I_t = +1 under a Democratic president (2022, 2024), −1 under a Republican one (2018, 2020)

Rows from each cycle's V_t frame (primary exposure, C1) are stacked. Controls
enter cycle by cycle (block-diagonal), so race FE are race × cycle. SEs cluster
by state across cycles. RI treats each (cycle, state) as a unit: deviations are
permuted within cycle × PADD group, independently across cycles (the memo's
approximation, since windows share baseline years). Prediction: θ > 0.

    python -m fb26.models.pooled
"""
import uuid
from datetime import datetime, timezone

import numpy as np
import polars as pl

from fb26.models import p1, ri, validate
from fb26.util import paths, runlog

CYCLES = (2018, 2020, 2022, 2024)
SIGN = {2018: -1, 2020: -1, 2022: 1, 2024: 1, 2026: -1}
RISING = (2018, 2022, 2024)  # national mean shock > 0 (shocks.parquet); 2020 is the falling-price cycle


def stack(cycles=CYCLES, g_method: str = validate.PRIMARY, *, block2026=None, blocks=None):
    """Stack state-level blocks; a supplied 2026 (frame, shocks) is appended.

    blocks optionally supplies historical (frame, shocks) pairs, so callers
    can use frozen inputs without invoking the historical input loader.
    """
    cycles = tuple(cycles)
    if block2026 is not None and 2026 not in cycles:
        cycles += (2026,)
    supplied = dict(blocks or {})
    if block2026 is not None:
        supplied[2026] = block2026
    ys, es, xs, ws, cl, units, zs, devs, groups = [], [], [], [], [], [], [], [], []
    offset = 0
    for t in cycles:
        if t in supplied:
            df, shocks = supplied[t]
        elif t == 2026:
            raise ValueError('2026 requires a prebuilt state-level block')
        else:
            df, shocks, _ = validate.frame(t, g_method)
        if t == 2026:
            index = {s: i for i, s in enumerate(shocks['state_fips'].to_list())}
            state = df['state_fips'].replace_strict(index).to_numpy()
            expected = df['e_c'].to_numpy() * shocks['deviation'].to_numpy()[state]
            if not np.allclose(df['b_tilde'].to_numpy(), expected, rtol=1e-10, atol=1e-12):
                raise ValueError('2026 pooled block must contain state-level treatment')
        z = validate.design_matrix(df, precision=validate.C1)
        index = {s: i for i, s in enumerate(shocks['state_fips'].to_list())}
        ys.append(df['dv'].to_numpy())
        es.append(SIGN[t] * df['e_c'].to_numpy())
        xs.append(SIGN[t] * df['b_tilde'].to_numpy())
        ws.append(df['w'].to_numpy())
        cl.append(df['state_fips'].to_numpy())
        units.append(offset + df['state_fips'].replace_strict(index).to_numpy())
        zs.append(z)
        devs.append(shocks['deviation'].to_numpy())
        groups.append(np.array([f'{t}_{g}' for g in shocks['padd_group'].to_list()]))
        offset += shocks.height
    n = sum(len(y) for y in ys)
    z = np.zeros((n, sum(m.shape[1] for m in zs)))
    r = c = 0
    for m in zs:
        z[r:r + m.shape[0], c:c + m.shape[1]] = m
        r, c = r + m.shape[0], c + m.shape[1]
    cat = np.concatenate
    return dict(y=cat(ys), e=cat(es), x=cat(xs), w=cat(ws), clusters=cat(cl), unit=cat(units), z=z,
                dev=cat(devs), groups=cat(groups))


def estimate(cycles=CYCLES, k: int = ri.K_DRAWS, seed: int = ri.SEED,
             *, block2026=None, blocks=None) -> dict:
    cycles = tuple(cycles)
    if block2026 is not None and 2026 not in cycles:
        cycles += (2026,)
    s = stack(cycles, block2026=block2026, blocks=blocks)
    fit = p1.fit(s['y'], s['x'], s['z'], s['w'], s['clusters'])
    design = ri.Design(s['y'], s['e'], s['unit'], s['z'], s['w'], s['clusters'], s['dev'].size)
    inv = design.inversion(s['dev'], s['groups'], k, np.random.default_rng(seed))
    ci95, f95 = inv.ci(0.05, fit.beta, fit.se_cr1)
    ci90, f90 = inv.ci(0.10, fit.beta, fit.se_cr1)
    return {'cycles': ';'.join(map(str, cycles)), 'theta': fit.beta, 'se_cr1': fit.se_cr1, 't_cr1': fit.t,
            'p_ri': inv.p(0.0), 'ci95_lo': ci95[0], 'ci95_hi': ci95[1], 'ci90_lo': ci90[0], 'ci90_hi': ci90[1],
            'eps_star': max(abs(ci90[0]), abs(ci90[1])), 'n_rows': fit.n, 'n_states': fit.n_clusters,
            'n_params': fit.n_params, 'k_draws': k, 'seed': seed, 'ri_flags': ';'.join(sorted(set(f95 + f90)))}


def main() -> None:
    table = pl.DataFrame([estimate(CYCLES), estimate(RISING)])
    out = paths.repo_root() / 'reports' / 'validation' / 'theta_hist.csv'
    table.write_csv(out)
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id='theta_hist',
        data_snapshot_ids='exposure_county.parquet;shocks.parquet;results_county.parquet',
        output_path=str(out.relative_to(paths.repo_root())),
        notes=f'C1; K={ri.K_DRAWS}; seed={ri.SEED}; all cycles and rising-price cycles; memo D18')
    print(table.select('cycles', 'theta', 'se_cr1', 'p_ri', 'ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi', 'n_rows'))


if __name__ == '__main__':
    main()
