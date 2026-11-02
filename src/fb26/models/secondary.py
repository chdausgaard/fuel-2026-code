"""Secondary office-gap estimator and shared inference helpers.

Pass an H1 analysis frame and its complete state shock table explicitly. The
primary caller should supply precision=validate.C1, just as for H1.
"""
from datetime import date

import numpy as np
import polars as pl

from fb26.models import p1, ri, validate
from fb26.util import paths

B_DRAWS = 9_999
SEED = 20261020
WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5), np.sqrt(.5), 1., np.sqrt(1.5)])


def incumbent_join_check(race_ids, incumbents: pl.DataFrame | None = None) -> dict:
    """Report missing exact race IDs and unresolved parties; never strip SPECIAL."""
    if incumbents is None:
        incumbents = pl.read_csv(paths.repo_root() / 'config/race_incumbents.csv')
    if incumbents['race_id'].n_unique() != incumbents.height:
        raise ValueError('duplicate incumbent race_id')
    lookup = {r['race_id']: r for r in incumbents.iter_rows(named=True)}
    missing, ambiguous, signs = [], {}, {}
    for race in dict.fromkeys(race_ids):
        if race not in lookup:
            missing.append(race)
            continue
        row = lookup[race]
        party = row['party']
        if str(row.get('incumbent_name', '')).strip().lower() in ('vacant', 'vacancy'):
            ambiguous[race] = 'vacant seat'
            continue
        if party == 'I':
            party = row.get('caucus')
        if party not in ('R', 'D'):
            ambiguous[race] = f"unresolved party={row['party']!r}, caucus={row.get('caucus')!r}"
        else:
            signs[race] = 1 if party == 'R' else -1
    return {'missing_races': missing, 'ambiguous_races': ambiguous, 'signs': signs}


def incumbent_sign(race_ids, incumbents: pl.DataFrame | None = None) -> np.ndarray:
    """+1 for Republican incumbency, -1 for Democratic (including I caucus)."""
    race_ids = list(race_ids)
    check = incumbent_join_check(race_ids, incumbents)
    if check['missing_races'] or check['ambiguous_races']:
        raise ValueError(f"incumbent join failed: {check}")
    return np.array([check['signs'][r] for r in race_ids], dtype=float)


def office_gap_frame(df: pl.DataFrame, incumbents: pl.DataFrame | None = None) -> pl.DataFrame:
    """One county pair, signed by the sitting governor, including open seats."""
    races = df.select('state_fips', 'race_id').unique().with_columns(
        pl.when(pl.col('race_id').str.contains(r'_GOV(?:_SPECIAL)?$')).then(pl.lit('GOVERNOR'))
        .when(pl.col('race_id').str.contains(r'_SEN(?:_SPECIAL)?$')).then(pl.lit('SENATE'))
        .otherwise(None).alias('office')).drop_nulls('office')
    if races.group_by('state_fips', 'office').len().filter(pl.col('len') > 1).height:
        raise ValueError('multiple included races for one office in a state')
    paired = races.group_by('state_fips').len().filter(pl.col('len') == 2)['state_fips']
    df = df.filter(pl.col('state_fips').is_in(paired.implode()))
    gov = df.filter(pl.col('race_id').str.contains(r'_GOV(?:_SPECIAL)?$'))
    sen = df.filter(pl.col('race_id').str.contains(r'_SEN(?:_SPECIAL)?$'))
    if gov.is_empty():
        return gov
    if incumbents is None:
        incumbents = pl.read_csv(paths.repo_root() / 'config/race_incumbents.csv')
    incumbents = incumbents.filter(pl.col('office') == 'GOVERNOR')
    gov = gov.with_columns(pl.Series('governor_sign', incumbent_sign(gov['race_id'], incumbents)))
    columns = ['state_fips', 'geo_id']
    values = ['r2'] + (['mock_source_race_id'] if 'mock_source_race_id' in df.columns else [])
    out = gov.join(sen.select(*columns, *values), on=columns, suffix='_sen', validate='1:1')
    if 'mock_source_race_id' in out.columns:
        same = out.filter(pl.col('mock_source_race_id') == pl.col('mock_source_race_id_sen'))['state_fips']
        out = out.filter(~pl.col('state_fips').is_in(same.implode()))
    return out.with_columns((pl.col('governor_sign') * (pl.col('r2_sen') - pl.col('r2'))).alias('dv'),
                            pl.col('v_c').alias('w'), pl.col('state_fips').alias('race_id'))


def office_gap_statistics(y, x, signs, z, w, clusters):
    """Batched WLS with both treatment columns rebuilt for each assignment."""
    res = p1.Residualizer(z, w)
    y, h, x = res(y), res(signs[:, None] * x), res(x)
    hh = np.sum(w[:, None] * h * h, axis=0)
    xh = np.sum(w[:, None] * x * h, axis=0) / hh
    xr = x - h * xh
    den = np.sum(w[:, None] * xr * xr, axis=0)
    if np.any(hh <= 0) or np.any(den <= 1e-20):
        raise ValueError('office gap treatment slopes unidentified')
    beta = np.sum(w[:, None] * xr * y[:, None], axis=0) / den
    interaction = np.sum(w[:, None] * h * (y[:, None] - x * beta), axis=0) / hh
    errors = y[:, None] - x * beta - h * interaction
    codes = p1.cluster_codes(clusters)
    scores = np.array([np.sum((w[:, None] * xr * errors)[codes == g], axis=0)
                       for g in np.unique(codes)])
    n_params = res.rank + 2
    se = np.sqrt(p1.cr1_factor(len(y), n_params, len(np.unique(codes)))
                 * np.sum(scores ** 2, axis=0)) / den
    return beta, interaction, se, beta / se, n_params


def estimate_office_gap(df: pl.DataFrame, shocks: pl.DataFrame, k: int,
                        *, assignment=None, seed: int = ri.SEED) -> dict:
    """Average of the two governor-party slopes; two-sided studentized RI."""
    z = validate.design_matrix(df, precision=validate.C1)
    y, e, x, w, signs = (df[c].to_numpy() for c in ('dv', 'e_c', 'b_tilde', 'w', 'governor_sign'))
    clusters = df['state_fips'].to_numpy()
    states = shocks['state_fips'].to_list() if assignment is None else assignment.states
    state = np.array([states.index(s) for s in clusters])
    dev = shocks['deviation'].to_numpy() if assignment is None else assignment.dev
    groups = shocks['padd_group'].to_numpy() if assignment is None else assignment.groups
    unit = np.full(df.height, -1) if assignment is None else df['unit'].to_numpy()
    def treatment(d):
        out = e[:, None] * d[:, state].T
        has = unit >= 0
        out[has] += e[has, None] * d[:, len(states) + unit[has]].T
        return out
    if not np.allclose(x, treatment(dev[None, :])[:, 0]):
        raise ValueError('not_run: treatment differs from randomization assignment')
    beta, interaction, se, t, n_params = office_gap_statistics(y, x[:, None], signs, z, w, clusters)
    rng, exceed = np.random.default_rng(seed), 0
    for start in range(0, k, ri.CHUNK):
        draws = ri.permuted_deviations(dev, groups, min(ri.CHUNK, k - start), rng)
        ts = office_gap_statistics(y, treatment(draws), signs, z, w, clusters)[3]
        exceed += np.sum(np.abs(ts) >= abs(t[0]) * (1 - 1e-12))
    return {'beta': float(beta[0]), 'beta_interaction': float(interaction[0]),
            'slope_republican_governor': float(beta[0] + interaction[0]),
            'slope_democratic_governor': float(beta[0] - interaction[0]),
            'se_cr1': float(se[0]), 't_cr1': float(t[0]), 'p_ri': float((1 + exceed) / (k + 1)),
            'n_rows': df.height, 'n_states': df['state_fips'].n_unique(), 'n_params': n_params,
            'n_counties': df['geo_id'].n_unique(), 'k_draws': k, 'seed': seed}


class WildClusterTest(ri.Inversion):
    """Restricted WLS wild cluster bootstrap-t, Webb six-point weights.

    Each candidate null is imposed in the bootstrap DGP. Restricted residuals
    are multiplied by one Webb draw per state; every bootstrap regression is
    unrestricted and studentized with state CR1. Linear/quadratic sufficient
    statistics allow genuine test inversion with the same draws at every null.
    """

    def __init__(self, y, x, w, clusters, contrast, b: int = B_DRAWS, seed: int = SEED):
        y, x, w, contrast = (np.asarray(v, float) for v in (y, x, w, contrast))
        if b < 1 or np.any(w <= 0) or not all(np.isfinite(v).all() for v in (y, x, w, contrast)):
            raise ValueError('finite inputs, positive weights and bootstrap count required')
        sw = np.sqrt(w)
        a = x * sw[:, None]
        inverse = np.linalg.pinv(a)
        rank = np.linalg.matrix_rank(a)
        if not np.allclose(contrast @ inverse @ a, contrast, atol=1e-8):
            raise ValueError('contrast is not identified')
        beta = inverse @ (sw * y)
        direction = inverse.T @ contrast
        norm = direction @ direction
        if norm <= 0:
            raise ValueError('zero contrast')
        codes = p1.cluster_codes(clusters)
        g = codes.max() + 1
        if g < 2 or len(y) <= rank:
            raise ValueError('insufficient clusters or residual degrees of freedom')
        cmat = (codes[None, :] == np.arange(g)[:, None]).astype(float)
        factor = p1.cr1_factor(len(y), rank, g)
        resid = sw * y - a @ beta
        score = cmat @ (direction * resid)
        self.estimate = float(contrast @ beta)
        self.se = float(np.sqrt(factor * np.sum(score ** 2)))
        self.beta, self.rank, self.seed = beta, rank, seed
        # Restricted residual for null h: resid + direction*(estimate-h)/norm.
        e0 = resid + direction * self.estimate / norm
        e1 = -direction / norm
        rng = np.random.default_rng(seed)
        pieces = []
        for start in range(0, b, ri.CHUNK):
            webb = rng.choice(WEBB, size=(min(ri.CHUNK, b - start), g)).T
            u0, u1 = e0[:, None] * webb[codes], e1[:, None] * webb[codes]
            n0, n1 = direction @ u0, direction @ u1
            s0 = cmat @ (direction[:, None] * (u0 - a @ (inverse @ u0)))
            s1 = cmat @ (direction[:, None] * (u1 - a @ (inverse @ u1)))
            pieces.append(ri.Draws(n0, -n1, np.sum(s0 ** 2, axis=0),
                                  np.sum(s0 * s1, axis=0), np.sum(s1 ** 2, axis=0), factor))
        obs = ri.Draws(np.array([self.estimate]), np.ones(1), np.array([self.se ** 2]),
                       np.zeros(1), np.zeros(1), 1.)
        super().__init__(obs, ri._concat(pieces), b)

    def report(self) -> dict:
        ci95, flags95 = self.ci(.05, self.estimate, self.se)
        ci90, flags90 = self.ci(.10, self.estimate, self.se)
        return {'estimate': self.estimate, 'se_cr1': self.se, 'p_wild': self.p(),
                'ci95': ci95, 'ci90': ci90, 'b_draws': self.k, 'seed': self.seed,
                'flags': sorted(set(flags95 + flags90))}


def _holiday_states(windows, holidays: pl.DataFrame | None) -> set[str]:
    if holidays is None:
        holidays = pl.read_csv(paths.repo_root() / 'config/gas_tax_holidays.csv',
                               schema_overrides={'state_fips': pl.String})
    overlaps = pl.lit(False)
    for start, end in windows:
        overlaps |= ((pl.col('start_date').cast(pl.String) <= end)
                     & (pl.col('end_date').is_null()
                        | (pl.col('end_date').cast(pl.String) >= start)))
    return set(holidays.filter(overlaps)['state_fips'])


def holiday_states(holidays: pl.DataFrame | None = None) -> set[str]:
    """States with a holiday in either the 2024 or 2026 election window."""
    return _holiday_states([('2024-10-15', '2024-11-04'),
                            ('2026-10-13', '2026-11-02')], holidays)


def tax_cut_relief(holidays, prices, window):
    """Window-average dollars per gallon, with each overlapping tax row accounted for."""
    start, end = (date.fromisoformat(str(window[c])) for c in ('start', 'end'))
    days = (end - start).days + 1
    means = {r['state_abbr']: r for r in prices.iter_rows(named=True)}
    states = {}
    for r in holidays.iter_rows(named=True):
        first = max(start, date.fromisoformat(str(r['start_date'])))
        last = min(end, date.fromisoformat(str(r['end_date']))) if r['end_date'] else end
        active = (last - first).days + 1
        if active <= 0:
            continue
        cents, percent = r['cut_cents_per_gal'], r['cut_percent_of_price']
        if cents is None and percent is None:
            raise ValueError('missing numeric tax cut')
        price = means.get(r['state_abbr'], {})
        if percent is not None and (price.get('aaa_mean') is None or not np.isfinite(price['aaa_mean'])):
            raise ValueError('missing AAA window price for percentage cut')
        cut = (cents or 0.) / 100 + (percent or 0.) / 100 * (price.get('aaa_mean', .184) - .184)
        contribution = cut * active / days
        state = states.setdefault(r['state_fips'], dict(state_abbr=r['state_abbr'], relief=0., components=[]))
        state['relief'] += contribution
        notes = r.get('notes') or ''
        state['components'].append(dict(tax=r['tax'], reduction=r['reduction'],
            start_date=str(r['start_date']), end_date=str(r['end_date']) if r['end_date'] else None,
            cut_cents_per_gal=cents, cut_percent_of_price=percent, days_in_force=active,
            window_days=days, relief_cents_per_gal=100 * contribution,
            aaa_mean_price=price.get('aaa_mean'), price_days=price.get('aaa_n'), notes=notes))
    return states


TAX_CUT_SOFTENING = dict(label='How much the tax cuts may have softened the swing (rough calculation)',
    inference='none; rescaling of H1',
    price_note='AAA means use observed daily prices in the 2026 window; price_days records coverage.',
    assumptions=[
        'The whole tax cut reached the pump.',
        'Voters respond to a cent of tax relief as to a cent of any price change.',
        'The main estimate from differences between nearby markets carries over to a change shared by a whole state.'])


def tax_cut_softening(h1, holidays, prices, window, exposure, history, income):
    """Arithmetic translation of H1, using all counties rather than the race sample."""
    row = dict(TAX_CUT_SOFTENING)
    if h1['status'] != 'run':
        return {**row, 'status': 'not_run: H1 did not run; ' + h1['status']}
    if any(x is None for x in (holidays, prices, window, exposure, history)):
        return {**row, 'status': 'not_run: tax cut inputs unavailable'}
    relief = tax_cut_relief(holidays, prices, window)
    votes = validate._r2(history.filter((pl.col('year') == 2024) & (pl.col('office') == 'PRESIDENT')), ['geo_id'])
    counties = exposure.filter(pl.col('g_method') == validate.PRIMARY).join(
        votes.select('geo_id', 'two_party'), on='geo_id', how='left', validate='1:1')
    entries = {}
    for state, cut in sorted(relief.items()):
        df = counties.filter(pl.col('state_fips') == state)
        if df.is_empty() or df.select(pl.any_horizontal(
                pl.col('e_c').is_null() | ~pl.col('e_c').is_finite(),
                pl.col('two_party').is_null() | ~pl.col('two_party').is_finite()
                | (pl.col('two_party') <= 0))).to_series().any():
            raise ValueError('county exposure or presidential weights unavailable')
        burden = float(np.average(df['e_c'].to_numpy() * cut['relief'], weights=df['two_party'].to_numpy()))
        entries[state] = dict(state_abbr=cut['state_abbr'], n_counties=df.height,
            relief_cents_per_gal=100 * cut['relief'], components=cut['components'],
            delta_b=burden, monthly_dollars=burden * income / 100,
            softening=-h1['beta'] * burden,
            softening_at_h1_ci95_lo=-h1['ci95_lo'] * burden,
            softening_at_h1_ci95_hi=-h1['ci95_hi'] * burden)
    return {**row, 'status': 'run', 'states': entries,
            'weight': 'county 2024 two-party presidential vote', 'monthly_income': income,
            'interval_note': 'Values at the two ends of H1’s registered 95% interval; no new interval or test.'}
