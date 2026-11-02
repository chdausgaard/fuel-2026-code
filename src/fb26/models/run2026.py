"""2026 orchestration. No fetching; supplied snapshot times determine outputs.

MOCK statistics exist only in memory. The output boundary masks before serializing,
including run-log notes. Real-outcome entry points are never exercised by tests.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4
import os
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import polars as pl
import yaml

from fb26.models import balance, frame2026, level2026, p1, pooled, power, ri, secondary, validate
from fb26.util import paths, runlog

LABEL = 'MOCK — NOT 2026 RESULTS'
PREFIX = 'MOCK-NOT-2026-RESULTS_'
FAMILIES = ('H1', 'H1_state', 'turnout', 'ladder_1', 'ladder_2', 'ladder_3', 'ladder_4', 'office_gap', 'tax_cut_softening', 'secondary_measures',
            'robustness', 'competitive_by_state', 'balance',
            'exhibits', 'house')
GUARDS = ('prices', 'level', 'race_incumbents')
H1_STATE_LABEL = 'H1-state: state-wide price differences'
SECONDARY_MEASURES = ('sa',)   # registered secondary measure: the since-January change; ky12 stays in the code, unregistered
COUNTS = {'n_rows', 'n_states', 'n_params', 'n_races', 'n_counties', 'k_draws', 'b_draws',
          'seed', 'n_markets', 'n_groups', 'n_units', 'n_covariates', 'n_local_counties'}
DECIMAL = re.compile(r'[-+]?(\d+\.\d*|\.\d+)([eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+|[-+]?(nan|inf)', re.IGNORECASE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('snapshot timestamps must be UTC')
    return parsed


def mask(value, key=''):
    """Conservative recursive mask: only named counts/seeds retain numeric values."""
    if key in ('ri_flags', 'wild_flags', 'flags', 'identified', 'ri_wild_disagree_5pct'):
        return 'MASKED'
    if isinstance(value, dict):
        return {k: mask(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [mask(v, key) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return value.item() if key in COUNTS and isinstance(value, np.generic) else value if key in COUNTS else 'MASKED'
    # Numbers read from CSV arrive as strings; integer codes such as FIPS stay.
    if isinstance(value, str) and DECIMAL.fullmatch(value.strip()) and key not in COUNTS:
        return 'MASKED'
    return value


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config['mu_pt']['source'] not in ('seds', 'hf_nodrift'):
        raise ValueError('unknown mu_pt.source')
    if config['primary'] != {'measure': 'w21', 'controls': 'C1', 'exposure': 'commute_vehicles'}:
        raise ValueError('stage 1 requires the registered primary specification')
    if config['ri']['k'] < 1 or config['wild']['b'] < 1:
        raise ValueError('positive simulation counts required')
    if not config['translation']['monthly_income'] > 0:
        raise ValueError('translation.monthly_income must be positive')
    return config


def monthly_income(config=None):
    """The one income figure of every per-dollar translation (analysis.yaml, translation.monthly_income)."""
    if config is None:
        config = yaml.safe_load((paths.config_dir() / 'analysis.yaml').read_text())
    return float(config['translation']['monthly_income'])


def per_10_dollars(value, income=None):
    """Vote points per extra $10 a month, from a coefficient or bound in points per point of monthly income:
    $10 is 1000 / income points of income, so beta = 1 is 0.150 points at $6,659."""
    return value * 1000 / (monthly_income() if income is None else income)


def race_slopes(df, column):
    """One control per race: `column` inside the race, 0 elsewhere. A race whose values are all missing gets none."""
    race, x = df['race_id'].to_numpy(), df[column].to_numpy().astype(float)
    out = []
    for r in np.unique(race):
        inside = race == r
        if not np.isnan(x[inside]).all():
            out.append(np.where(inside, x, 0.))
    return out


def market_clusters(df):
    """Repeated races share a cluster; counties without a local unit stand alone."""
    units = df['unit'] if 'unit' in df.columns else np.full(df.height, -1)
    return np.array([f'm:{u}' if u >= 0 else f'c:{geo}'
                     for u, geo in zip(units, df['geo_id'])])


def design_names(df, precision, lag, pt, slopes):
    names = [f'race:{r}' for r in np.unique(df['race_id'].to_numpy())]
    names += [f'exposure:{g}' for g in np.unique(df['padd_group'].to_numpy())]
    names += [c for c, use in [('mu_pt', pt), ('b_tilde_lag', lag)] if use and df[c].null_count() == 0]
    names += ['oil_gas_emp_share', *precision]
    names += [f'{c}:{r}' for c in slopes for r in np.unique(df['race_id'].to_numpy())
              if not np.isnan(df.filter(pl.col('race_id') == r)[c].to_numpy().astype(float)).all()]
    return names


def competitive_states():
    ratings = pl.read_csv(paths.config_dir() / 'competitive_senate_2026.csv')
    states = pl.read_csv(paths.config_dir() / 'states.csv', schema_overrides={'state_fips': pl.String})
    return ratings.join(states.select(pl.col('state_abbr').alias('state_po'), 'state_fips'),
                        on='state_po', how='left', validate='1:1').sort('state_po')


def varying_market_groups(df, assignment):
    if assignment is None or 'unit' not in df.columns:
        return 0
    units = df.filter(pl.col('unit') >= 0)['unit'].unique().to_numpy()
    indices = len(assignment.states) + units
    groups = np.unique(assignment.groups[indices])
    return sum(np.ptp(assignment.dev[assignment.groups == g]) > 0 for g in groups)


def office_races(df, office):
    """Races of one office ('SENATE' or 'GOVERNOR'), weighted as in R12: a state's weights sum to its baseline votes."""
    sub = df.filter(pl.col('race_id').map_elements(lambda r: race_parts(r)[1], return_dtype=pl.String) == office)
    if 'v_c' in sub.columns:   # divided by the state's number of included races of the office
        sub = sub.with_columns((pl.col('v_c') / pl.col('race_id').n_unique().over('state_fips')).alias('w'))
    return sub


def competitive_by_state(df, shocks, assignment, kwargs, level):
    """Exploratory one-state rows on each competitive state's Senate race(s) alone (Ohio's special counts)."""
    rows = {}
    for state in competitive_states().iter_rows(named=True):
        sub = df.filter(pl.col('state_fips') == state['state_fips']) if df is not None else None
        if sub is None or sub.is_empty():
            continue
        try:
            sub = office_races(sub, 'SENATE')
        except Exception:
            sub = sub.clear()   # race ids without an office: nothing to estimate
        if sub.is_empty():
            rows[state['state_po']] = {'status': 'not_run: no Senate race in the state', 'exploratory': True,
                                       'level_note': level}
            continue
        n_groups = varying_market_groups(sub, assignment)
        result = ({'status': 'not_run: too few comparison groups in the state'} if n_groups < 2 else
                  attempted('single-state estimation failed', estimate_h1, sub, shocks, assignment=assignment,
                            **kwargs, cluster='market', wild=False, drop_collinear=True))
        rows[state['state_po']] = {**result, 'exploratory': True, 'level_note': level,
                                  'n_groups': int(n_groups), 'n_rows': sub.height,
                                  'n_counties': sub['geo_id'].n_unique(),
                                  'n_markets': len(np.unique(market_clusters(sub)))}
    return family_rows(rows)


def estimate_h1(df, shocks, *, assignment=None, k=ri.K_DRAWS, seed=ri.SEED,
                b=secondary.B_DRAWS, wild_seed=secondary.SEED,
                precision=validate.C1, lag=True, pt=True, minimal=False, outcome='dv', slopes=(),
                price_only=False, cluster='state', wild=True, drop_collinear=False):
    """Same P1 fit and RI as validation; optional independent sub-state layer.

    `outcome` names the outcome column; `slopes` names columns that enter with a separate slope in every race.
    `price_only` drops exposure from the treatment and its permutations; the controls stay.
    Exploratory single-state fits use market clusters, skip wild inference, and drop
    collinear controls in design order (race effects and exposure are kept first)."""
    if minimal:
        df = df.with_columns(pl.lit(0.).alias('oil_gas_emp_share'))
        precision, lag, pt = (), False, False
    z = validate.design_matrix(df, precision=precision, lag=lag, pt=pt)
    if slopes:
        z = np.column_stack([z] + [c for column in slopes for c in race_slopes(df, column)])
    y, e, x, w = (df[c].to_numpy() for c in (outcome, 'e_c', 'b_tilde', 'w'))
    if cluster not in ('state', 'market'):
        raise ValueError('unknown cluster')
    clusters = df['state_fips'].to_numpy()
    state_labels = clusters
    if cluster == 'market':
        clusters = market_clusters(df)
    states = shocks['state_fips'].to_list() if assignment is None else assignment.states
    state = np.array([states.index(s) for s in state_labels])
    dev = shocks['deviation'].to_numpy() if assignment is None else assignment.dev
    shock = dev[state].copy()
    if assignment is not None:
        unit = df['unit'].to_numpy()
        has = unit >= 0
        shock[has] += dev[len(states) + unit[has]]
    if not np.allclose(x, e * shock):
        raise ValueError('not_run: treatment differs from randomization assignment')
    main_x = x
    if price_only:
        x, e = shock, np.ones_like(e)
    dropped = []
    if drop_collinear:
        names = design_names(df, precision, lag, pt, slopes)
        keep = []
        for j, name in enumerate(names):
            candidate = z[:, keep + [j]] * np.sqrt(w)[:, None]
            if p1.Residualizer(candidate, np.ones(len(w))).rank > len(keep):
                keep.append(j)
            else:
                dropped.append(name)
        z = z[:, keep]
        if p1.Residualizer(np.column_stack([x, z]), w).rank != z.shape[1] + 1:
            raise ValueError('not_run: design matrix is not full column rank')
    fit = p1.fit(y, x, z, w, clusters)
    if assignment is None:
        test = ri.RandomizationTest(y, e, state, z, w, clusters, shocks['deviation'].to_numpy(),
                                     shocks['padd_group'].to_numpy(), k=k, seed=seed)
    else:
        test = ri.TwoLayerRandomizationTest(y, e, state, z, w, clusters, assignment.dev,
                                            assignment.groups, len(states), df['unit'].to_numpy(),
                                            k=k, seed=seed)
    result = test.run(fit.beta, fit.se_cr1)
    design = np.column_stack([x, z])
    contrast = np.zeros(design.shape[1]); contrast[0] = 1
    wild_result = (secondary.WildClusterTest(y, design, w, clusters, contrast, b=b, seed=wild_seed).report()
                   if wild else dict(p_wild=None, ci95=None, ci90=None, flags='skipped: single-state analysis'))
    row = {'beta': fit.beta, 'se_cr1': fit.se_cr1, 't_cr1': fit.t, 'p_ri': result.p_value,
           'ci95_lo': result.ci95[0], 'ci95_hi': result.ci95[1],
           'ci90_lo': result.ci90[0], 'ci90_hi': result.ci90[1], 'eps_star': result.eps_star,
           'ri_flags': ';'.join(result.flags), 'p_wild': wild_result['p_wild'],
           'wild_ci95': wild_result['ci95'], 'wild_ci90': wild_result['ci90'], 'wild_flags': wild_result['flags'],
           'ri_wild_disagree_5pct': ((result.p_value <= .05) != (wild_result['p_wild'] <= .05)) if wild else None,
           'n_rows': fit.n, 'n_states': df['state_fips'].n_unique(), 'n_params': fit.n_params,
           'n_races': df['race_id'].n_unique(), 'n_counties': df['geo_id'].n_unique(),
           'k_draws': k, 'seed': seed, 'b_draws': b}
    if not wild:
        row['b_draws'] = None
        row['wild_skipped'] = True
    if drop_collinear:
        row['dropped_columns'] = dropped
        row['design_full_rank'] = True
    if cluster == 'market':
        row['cluster'] = 'market'
        row['n_markets'] = fit.n_clusters
    if price_only:      # dollars of recentered rise; the correlation says how far this row can differ from the main one
        row.update({f'{key}_per_10_cents': row[key] / 10
                    for key in ('beta', 'se_cr1', 'ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi')})
        res = p1.Residualizer(z, w)
        a, c = res(main_x), res(x)
        row['corr_with_main_treatment'] = (np.average(a * c, weights=w)
                                           / np.sqrt(np.average(a * a, weights=w) * np.average(c * c, weights=w)))
    return row


def estimate_ladder(df, step, income=None):
    """Illustrative weighted slopes; conventional state-clustered CR1 intervals, without RI."""
    race = df['race_id'].to_numpy()
    z = np.ones((df.height, 1)) if step == 1 else np.column_stack([
        (race == r).astype(float) for r in np.unique(race)])
    if step == 3:
        e, padd = df['e_c'].to_numpy(), df['padd_group'].to_numpy()
        z = np.column_stack([z] + [e * (padd == g) for g in np.unique(padd)])
    x = df['b_tilde' if step == 3 else 'b_raw'].to_numpy()
    fit = p1.fit(df['dv'].to_numpy(), x, z, df['w'].to_numpy(), df['state_fips'].to_numpy())
    result = {'beta': fit.beta, 'se_cr1': fit.se_cr1, 't_cr1': fit.t,
              'ci95_lo': fit.beta - 1.96 * fit.se_cr1, 'ci95_hi': fit.beta + 1.96 * fit.se_cr1,
              'n_rows': fit.n, 'n_states': fit.n_clusters, 'n_params': fit.n_params,
              'n_races': df['race_id'].n_unique(), 'n_counties': df['geo_id'].n_unique(),
              'inference': 'state-clustered CR1; illustrative only',
              'units': 'vote percentage points per percentage point of monthly income'}
    result.update({f'{key}_per_10_dollars': per_10_dollars(result[key], income)
                   for key in ('beta', 'se_cr1', 'ci95_lo', 'ci95_hi')})
    return result


def estimate_turnout_moderation(df, income=None):
    """Turnout on the treatment interacted with the centred 2024 Republican share; state-clustered CR1, no RI
    (the randomization test permutes a single treatment term)."""
    y, x, w, clusters = df['turnout'].to_numpy(), df['b_tilde'].to_numpy(), df['w'].to_numpy(), df['state_fips'].to_numpy()
    r = df['r2_base'].to_numpy() - np.average(df['r2_base'].to_numpy(), weights=w)
    z = validate.design_matrix(df, precision=validate.C1)
    main = p1.fit(y, x, np.column_stack([z, r, x * r]), w, clusters)
    fit = p1.fit(y, x * r, np.column_stack([z, r, x]), w, clusters)
    result = {'beta': main.beta, 'se_cr1': main.se_cr1, 'interaction': fit.beta, 'interaction_se_cr1': fit.se_cr1,
              'interaction_ci95_lo': fit.beta - 1.96 * fit.se_cr1, 'interaction_ci95_hi': fit.beta + 1.96 * fit.se_cr1,
              'n_rows': fit.n, 'n_states': fit.n_clusters, 'n_params': fit.n_params,
              'moderator': '2024 Republican two-party share in points, centred at its weighted mean',
              'inference': 'state-clustered CR1'}
    result.update({f'{key}_per_10_dollars': per_10_dollars(result[key], income)
                   for key in ('beta', 'interaction', 'interaction_ci95_lo', 'interaction_ci95_hi')})
    return result


def attempted(reason, function, *args, **kwargs):
    """Keep an unavailable secondary result from interrupting the main analysis."""
    try:
        return {'status': 'run', **function(*args, **kwargs)}
    except Exception:
        return {'status': 'not_run: ' + reason}


def family_rows(rows):
    return {'status': 'run' if any(r['status'] == 'run' for r in rows.values())
            else 'not_run: no available estimates', 'rows': rows}


def leave_out(df):
    """Point estimates only; retain the original treatment and recentering."""
    rows = {}
    for column, label in [('state_fips', 'state'), ('padd_group', 'padd')]:
        drops = []
        for value in sorted(df[column].unique().to_list()):
            subset = df.filter(pl.col(column) != value)
            def fit():
                result = p1.fit(subset['dv'].to_numpy(), subset['b_tilde'].to_numpy(),
                                validate.design_matrix(subset, precision=validate.C1),
                                subset['w'].to_numpy(), subset['state_fips'].to_numpy())
                return {'beta': result.beta, 'n_rows': result.n, 'n_states': result.n_clusters}
            drops.append({'dropped': str(value), **attempted('leave-out estimation failed', fit)})
        rows[label] = drops
    if not any(r['status'] == 'run' for drops in rows.values() for r in drops):
        raise ValueError('no leave-out estimates')
    return rows


def race_parts(race_id):
    """('FL', 'SENATE', True) for '2026_FL_SEN_SPECIAL'."""
    _, state, office, *special = race_id.split('_')
    return state, 'SENATE' if office == 'SEN' else 'GOVERNOR', bool(special)


def previous_midterm_races(history, race_ids):
    """F6 baseline race: the same office in 2022, else the state's other 2022 office, else none; a regular race before a
    special one, then the first race id (the 2022 rule of placebo._build_extras)."""
    src = history.filter((pl.col('year') == 2022) & pl.col('office').is_in(['SENATE', 'GOVERNOR'])
                         & pl.col('party').is_in(['D', 'R']) & pl.col('included_d3').fill_null(False))
    cand = src.select('race_id', 'office', 'special', 'state_po').unique().sort('race_id')
    rows = []
    for race_id in race_ids:
        state, office, _ = race_parts(race_id)
        c = cand.filter(pl.col('state_po') == state)
        same = c.filter(pl.col('office') == office)
        options = same if same.height else c
        if options.is_empty():
            rows.append({'race_id': race_id, 'prev_race_id': None, 'prev_rule': 'none'})
            continue
        regular = options.filter(~pl.col('special').fill_null(False))
        options = regular if regular.height else options
        rows.append({'race_id': race_id, 'prev_race_id': options['race_id'][0],
                     'prev_rule': 'same_office' if same.height else 'other_office'})
    return pl.DataFrame(rows, schema={'race_id': pl.String, 'prev_race_id': pl.String, 'prev_rule': pl.String})


def same_office_baselines(history, hist, race_ids):
    """F3 baseline race: the previous race for the same seat.

    Governor: the 2022 governor race (Vermont and New Hampshire also held 2024 races; 2022 is used for consistency).
    Senate: the 2020 race of the seat for a regular 2026 race; the 2022 regular race for a special (Florida, Ohio).
    A seat without a usable previous race keeps F1 (`f3_race_id` null). 2020 comes from the historical file, 2022 from
    the project's results (D3 races)."""
    usable_2020 = set(hist.filter(pl.col('year') == 2020)['race_id'].unique().to_list())
    in_2022 = set(history.filter((pl.col('year') == 2022) & pl.col('party').is_in(['D', 'R'])
                                 & pl.col('included_d3').fill_null(False))['race_id'].unique().to_list())
    rows = []
    for race_id in race_ids:
        state, office, special = race_parts(race_id)
        if office == 'GOVERNOR':
            base, source = f'2022_{state}_GOV', '2022 results'
        elif special:
            base, source = f'2022_{state}_SEN', '2022 results'
        else:
            base, source = f'2020_{state}_SEN', '2020 historical file'
        usable = base in (in_2022 if source == '2022 results' else usable_2020)
        rows.append({'race_id': race_id, 'f3_race_id': base if usable else None, 'f3_source': source if usable else '',
                     'f3_status': 'same-office baseline' if usable else 'F1 kept: no usable previous race for the seat'})
    return pl.DataFrame(rows, schema={'race_id': pl.String, 'f3_race_id': pl.String, 'f3_source': pl.String,
                                      'f3_status': pl.String})


def past_shares(history, hist, race_ids):
    """County Republican two-party share (points) of the given past races (race_id, geo_id, r2_past)."""
    a = validate._r2(history.filter((pl.col('year') == 2022) & pl.col('race_id').is_in(race_ids)
                                    & pl.col('party').is_in(['D', 'R'])), ['geo_id', 'race_id'])
    b = hist.filter(pl.col('race_id').is_in(race_ids)).select(
        'geo_id', 'race_id', (pl.col('rep_votes') + pl.col('dem_votes')).alias('two_party'),
        (100 * pl.col('rep_votes') / (pl.col('rep_votes') + pl.col('dem_votes'))).alias('r2'))
    both = [x.select('geo_id', 'race_id', pl.col('r2').cast(pl.Float64), pl.col('two_party').cast(pl.Float64)) for x in (a, b)]
    return (pl.concat(both)
            .filter(pl.col('two_party') > 0).select('geo_id', 'race_id', pl.col('r2').alias('r2_past')))


def with_past_races(df, history, hist):
    """Adds r2_prev (F6), f3_race_id and r2_f3 (F3) and returns the mapping table of the frame's races."""
    race_ids = sorted(df['race_id'].unique().to_list())
    prev = previous_midterm_races(history, race_ids)
    f3 = same_office_baselines(history, hist, race_ids)
    shares = past_shares(history, hist, sorted(set(prev['prev_race_id'].drop_nulls()) | set(f3['f3_race_id'].drop_nulls())))
    df = (df.join(prev, on='race_id', how='left', validate='m:1').join(f3, on='race_id', how='left', validate='m:1')
          .join(shares.rename({'race_id': 'prev_race_id', 'r2_past': 'r2_prev'}), on=['prev_race_id', 'geo_id'],
                how='left', validate='m:1')
          .join(shares.rename({'race_id': 'f3_race_id', 'r2_past': 'r2_f3'}), on=['f3_race_id', 'geo_id'],
                how='left', validate='m:1'))
    # a county missing from a mapped previous race takes the race's weighted mean share, so it adds no information
    mean = (pl.col('w') * pl.col('r2_prev')).sum().over('race_id') / pl.col('w').filter(pl.col('r2_prev').is_not_null()).sum().over('race_id')
    n_filled = df.filter(pl.col('prev_race_id').is_not_null() & pl.col('r2_prev').is_null()).height
    df = df.with_columns(pl.when(pl.col('prev_race_id').is_not_null() & pl.col('r2_prev').is_null()).then(mean)
                         .otherwise(pl.col('r2_prev')).alias('r2_prev'))
    lost = df.filter(pl.col('f3_race_id').is_not_null() & pl.col('r2_f3').is_null()).group_by('race_id').len()
    mapping = (prev.join(f3, on='race_id').join(lost.rename({'len': 'f3_counties_missing_in_base'}), on='race_id', how='left')
               .with_columns(pl.col('f3_counties_missing_in_base').fill_null(0)).sort('race_id'))
    return df, mapping, n_filled


def estimate_forms(df, shocks, asg, kwargs, past):
    """Robustness rows F5, F6 and F3 (memo decision 2) and R12 (Senate and governor separately)."""
    rows = {}
    def f5():
        return estimate_h1(df, shocks, assignment=asg, **kwargs, outcome='r2', slopes=('r2_base',))
    rows['F5'] = attempted('F5 estimation failed', f5)
    try:
        d, mapping, n_filled = with_past_races(df, *past)
    except Exception:
        for name in ('F6', 'F3'):
            rows[name] = {'status': 'not_run: past-election inputs unavailable'}
    else:
        def f6():
            return {**estimate_h1(d, shocks, assignment=asg, **kwargs, outcome='r2', slopes=('r2_base', 'r2_prev')),
                    'races_without_previous_race': mapping.filter(pl.col('prev_rule') == 'none')['race_id'].to_list(),
                    'n_rows_filled_with_race_mean': n_filled}
        rows['F6'] = attempted('F6 estimation failed', f6)
        def f3():
            has = pl.col('f3_race_id').is_not_null()
            keep = d.filter(~has | pl.col('r2_f3').is_not_null())
            keep = keep.with_columns(pl.when(has).then(pl.col('r2') - pl.col('r2_f3')).otherwise(pl.col('r2') - pl.col('r2_base'))
                                     .alias('dv_f3'))
            return {**estimate_h1(keep, shocks, assignment=asg, **kwargs, outcome='dv_f3'),
                    'n_rows_dropped_no_base': d.height - keep.height,
                    'races_f1_kept': mapping.filter(pl.col('f3_race_id').is_null())['race_id'].to_list(),
                    'mapping': mapping.select('race_id', 'f3_race_id', 'f3_source', 'f3_status',
                                              'f3_counties_missing_in_base').to_dicts()}
        rows['F3'] = attempted('F3 estimation failed', f3)
    for name, office in (('R12_senate', 'SENATE'), ('R12_governor', 'GOVERNOR')):
        def one(office=office):
            return estimate_h1(office_races(df, office), shocks, assignment=asg, **kwargs)
        rows[name] = attempted('office-specific estimation failed', one)
    return rows


def robustness(primary, state, assignment, shocks, kwargs, *, level, vehicles, holidays, past=None,
               metro_lag=None, r_metro=None):
    df, asg = primary, assignment
    used_level = level
    if level != 'state' and (asg is None or len(asg.dev) == len(asg.states)):
        df, asg, used_level = state, None, 'state'
    note = used_level if used_level == level else 'state; sub-state shocks unavailable'
    specs = {'R2': {'minimal': True}, 'R3': {'precision': validate.C2}, 'R5': {'pt': False}}
    rows = {name: {**attempted('robustness estimation failed', estimate_h1, df, shocks,
                              assignment=asg, **kwargs, **controls), 'level_note': note}
            for name, controls in specs.items()}
    def exclude_holidays():
        if holidays is None:
            raise ValueError('missing holidays')
        excluded = sorted(secondary.holiday_states(holidays))
        result = estimate_h1(df.filter(~pl.col('state_fips').is_in(excluded)), shocks,
                             assignment=asg, **kwargs)
        return {**result, 'excluded_states': excluded}
    rows['R7'] = {**attempted('holiday exclusion unavailable', exclude_holidays), 'level_note': note}
    competitive = competitive_states()
    fips = competitive['state_fips'].to_list()
    state_table = pl.read_csv(paths.config_dir() / 'states.csv', schema_overrides={'state_fips': pl.String})
    for name, inside in [('competitive_senate', True), ('other_senate', False)]:
        # Senate races only (Ohio's special counts), weighted as R12_senate; the two groups split the Senate counties
        def group(inside=inside):
            sub = office_races(df, 'SENATE')
            sub = sub.filter(pl.col('state_fips').is_in(fips) == inside)
            result = estimate_h1(sub, shocks, assignment=asg, **kwargs)
            present = sorted(set(sub['state_fips']))
            return {**result, 'states': sorted(state_table.filter(pl.col('state_fips').is_in(present))['state_abbr'])}
        rows[name] = {**attempted('state subset estimation failed', group), 'level_note': note}
    def competitive_difference():
        # Registered (plan §8): competitive Senate races minus all other Senate races. The two estimates rest on
        # different states, so the standard error combines their state-clustered (CR1) standard errors. With
        # few states in a group, CR1 errors are too small against the normal curve, so the difference is read
        # against a t-distribution with (states in the smaller group - 1) degrees of freedom.
        from scipy import stats
        a, b = rows['competitive_senate'], rows['other_senate']
        if a['status'] != 'run' or b['status'] != 'run':
            raise ValueError('a group estimate is unavailable')
        diff, se = a['beta'] - b['beta'], float(np.hypot(a['se_cr1'], b['se_cr1']))
        dof = max(int(min(a['n_states'], b['n_states'])) - 1, 1)
        t95, t90 = float(stats.t.ppf(.975, dof)), float(stats.t.ppf(.95, dof))
        return {'beta': diff, 'se_cr1': se, 't_cr1': diff / se, 'df': dof,
                'p_t': float(2 * stats.t.sf(abs(diff / se), dof)),
                'ci95_lo': diff - t95 * se, 'ci95_hi': diff + t95 * se,
                'ci90_lo': diff - t90 * se, 'ci90_hi': diff + t90 * se,
                'method': 'competitive minus other Senate races; combined state-clustered standard errors of two disjoint '
                          'groups; t-distribution with (states in the smaller group - 1) degrees of freedom'}
    rows['competitive_difference'] = {**attempted('difference unavailable', competitive_difference), 'level_note': note}
    for name, row in estimate_forms(df, shocks, asg, kwargs, past).items():
        rows[name] = {**row, 'level_note': note}
    rows['R13'] = {**attempted('leave-out estimation failed', leave_out, df), 'level_note': note}
    if asg is None:
        rows['R_price_only'] = {'status': 'not_run: no sub-state prices; race effects absorb the state layer'}
    else:
        rows['R_price_only'] = {**attempted('price-only estimation failed', estimate_h1, df, shocks,
                                            assignment=asg, **kwargs, price_only=True), 'level_note': note}
    if vehicles is None:
        rows['R19'] = {'status': 'not_run: vehicles exposure unavailable'}
    else:
        vdf, vasg, vshocks, vlevel = vehicles
        rows['R19'] = {**attempted('vehicles estimation failed', estimate_h1, vdf, vshocks,
                                 assignment=vasg, **kwargs), 'level_note': vlevel}
    if r_metro is None:
        rows['R_metro'] = {'status': 'not_run: metro robustness row unavailable'}
    else:
        mdf, masg, mlevel = r_metro
        rows['R_metro'] = {**attempted('metro robustness estimation failed', estimate_h1, mdf, shocks,
                                      assignment=masg, **kwargs), 'level_note': mlevel}
    if metro_lag is None:
        rows['R_metro_lag'] = {'status': 'not_run: no sub-state level for the metro layer of the lagged burden'}
    else:   # the primary frame rebuilt with the metro layer inside B̃^lag (the main run keeps the state-only lag)
        ldf, lasg, llevel = metro_lag
        rows['R_metro_lag'] = {**attempted('metro-lag estimation failed', estimate_h1, ldf, shocks,
                                         assignment=lasg, **kwargs), 'level_note': llevel}
    return family_rows(rows)


def analyze(primary, state, assignment, shocks, config, *, incumbents=None,
            measures=None, vehicles=None, holidays=None, level='state',
            balance_inputs=None, past=None, metro_lag=None, r_metro=None, tax_cut_inputs=None):
    """Pure orchestration over supplied frames; missing families are explicit.

    metro_lag: (frame, assignment, level) of the primary run with the metro layer in the lagged burden.
    r_metro: (frame, assignment, level) of the metro block design with metro prices (robustness row)."""
    rows = {family: {'status': 'not_run: deferred to later stage'} for family in FAMILIES}
    k, seed = config['ri']['k'], config['ri']['seed']
    kwargs = dict(k=k, seed=seed, b=config['wild']['b'], wild_seed=config['wild']['seed'])
    for family, df, asg in [('H1', primary, assignment), ('H1_state', state, None)]:
        if df is None or df.height == 0:
            rows[family] = {'status': 'not_run: no eligible analysis rows'}
        else:
            rows[family] = {'status': 'run', **estimate_h1(df, shocks, assignment=asg, **kwargs)}
    tax_inputs = tax_cut_inputs or dict(prices=None, window=None, exposure=None, history=None)
    rows['tax_cut_softening'] = {**secondary.TAX_CUT_SOFTENING,
        **attempted('tax cut rescaling unavailable', secondary.tax_cut_softening,
                    rows['H1'], holidays, income=monthly_income(config), **tax_inputs)}
    for step, label in enumerate(('All counties', 'Within races', 'Unusual part of the rise only'), start=1):
        rows[f'ladder_{step}'] = ({**attempted('ladder estimation failed', estimate_ladder, primary, step,
                                             monthly_income(config)), 'label': label}
                                  if primary is not None and primary.height else
                                  {'status': 'not_run: no eligible analysis rows', 'label': label})
    rows['ladder_4'] = {**rows['H1'], 'reference': 'H1', 'label': 'Final analysis'}
    if primary is None or primary.is_empty():
        rows['turnout'] = {'status': 'not_run: no eligible analysis rows'}
    elif ('turnout' not in primary.columns or primary['turnout'].null_count()
          or not primary['turnout'].is_finite().all()):
        rows['turnout'] = {'status': 'not_run: all-candidate turnout totals unavailable'}
    else:
        rows['turnout'] = attempted('turnout estimation failed', estimate_h1, primary, shocks,
                                     assignment=assignment, outcome='turnout', **kwargs)
        rows['turnout']['moderation'] = attempted('turnout moderation failed', estimate_turnout_moderation,
                                                  primary, monthly_income(config))
    rows['turnout'].update(outcome='total race votes as percent of baseline presidential votes',
                           alternative='two-sided', predicted_direction=None)
    for name in ('ladder_4', 'turnout'):
        result = rows[name]
        if result['status'] == 'run':
            result.update({f'{key}_per_10_dollars': per_10_dollars(result[key], monthly_income(config))
                           for key in ('beta', 'se_cr1', 'ci95_lo', 'ci95_hi')})

    rows['H1_state'].update(
        label=H1_STATE_LABEL, level_note='state level; past-elections comparison figure input',
        test='randomization inference over within-PADD permutations of the state shocks (state layer only)')
    gaps = {}
    for label, df, asg in [('primary', primary, assignment), ('state', state, None)]:
        gap = secondary.office_gap_frame(df, incumbents) if df is not None else None
        if gap is None or gap.is_empty():
            gaps[label] = {'status': 'not_run: no eligible paired races', 'n_states': 0}
        else:
            gaps[label] = attempted('office gap estimation failed', secondary.estimate_office_gap,
                                     gap, shocks, k, assignment=asg, seed=seed)
            gaps[label]['n_states'] = gap['state_fips'].n_unique()
    rows['office_gap'] = {
        'status': 'run' if any(v['status'] == 'run' for v in gaps.values()) else
                  'not_run: office gap family unavailable',
        'label': "Governor race against Senate race, against the governor's party",
        'outcome': "governor race shift against the governor's party minus Senate race shift against that party",
        'sign_convention': "a positive estimate means a larger extra fuel cost moved the governor race further against the governor's party than the Senate race",
        'alternative': 'two-sided',
        'test': 'studentized average slope; signed treatment recomputed in every reshuffle', **gaps}
    measure_rows = {}
    for measure in SECONDARY_MEASURES:
        if measures is None or measure not in measures:
            measure_rows[measure] = {'status': 'not_run: secondary measure input unavailable'}
            continue
        df, asg, sh, used_level = measures[measure]
        measure_rows[measure] = {
            **attempted('secondary measure estimation failed', estimate_h1, df, sh,
                        assignment=asg, **kwargs),
            'level_note': used_level if used_level == level else 'state; sub-state coverage insufficient'}
        if df is not None:
            measure_rows[measure]['omitted_controls'] = [
                c for c in ('mu_pt', 'b_tilde_lag') if df[c].null_count()]
    rows['secondary_measures'] = family_rows(measure_rows)
    rows['robustness'] = robustness(primary, state, assignment, shocks, kwargs, level=level,
                                    vehicles=vehicles, holidays=holidays, past=past, metro_lag=metro_lag,
                                    r_metro=r_metro)
    rows['competitive_by_state'] = competitive_by_state(primary, shocks, assignment, kwargs, level)
    if balance_inputs is None:
        rows['balance'] = {'status': 'not_run: balance inputs unavailable'}
    else:
        def run_balance():
            states = balance_inputs['state_assignment']
            layers = {'state': states}
            if assignment is not None and len(assignment.dev) > len(assignment.states):
                layers['D22 level: ' + level] = assignment
            return balance.run(balance_inputs['outcomes'], layers, balance_inputs['exposure'],
                               balance_inputs['history'], incumbents, k, seed, catch=True)
        rows['balance'] = attempted('balance estimation failed', run_balance)
    return rows


def live_guards(shocks, level_path, *, config, substate, hf, results, races, incumbents):
    """Check the inputs actually used, the level file and the race/incumbent join."""
    frame2026.require_final(shocks.filter(pl.col('year') == 2026), 'provisional state shocks')
    frame2026.require_final(substate.filter((pl.col('cycle') == 2026) & (pl.col('measure') == 'w21')),
                            'provisional sub-state shocks')
    if config['mu_pt']['source'] == 'hf_nodrift':
        if hf is None:
            raise ValueError('live guard: missing pass-through input')
        frame2026.require_final(hf.filter((pl.col('year') == 2026) & (pl.col('measure') == 'w21')),
                                'provisional pass-through input')
    level_path = Path(level_path)
    if not level_path.is_file():
        raise ValueError('live guard: missing level file')
    level = json.loads(level_path.read_text())
    if level.get('forced', False):
        raise ValueError('live guard: forced level refused')
    if level.get('level') not in ('state', 'metro', 'county', 'county_metro'):
        raise ValueError('live guard: invalid level')
    selected, _ = frame2026.select_races(results, races, mock=False)
    check = secondary.incumbent_join_check(selected, incumbents)
    if check['missing_races'] or check['ambiguous_races']:
        raise ValueError('live guard: incumbent join failed')
    return {guard: 'passed' for guard in GUARDS}, level


def mock_failure(exc):
    """Expose fixed causes only; library exceptions may contain outcome values."""
    message = str(exc)
    if isinstance(exc, FileNotFoundError):
        cause = 'missing input file'
    elif message.startswith(('not_run:', 'live guard:', 'refusing')) and not any(c.isdigit() for c in message):
        cause = message
    else:
        cause = 'pipeline operation failed'
    print(f'{LABEL}: {type(exc).__name__}: {cause}; no statistical values released', file=sys.stderr)


def write_outputs(rows, metadata, directory, *, mock, config_paths, input_paths,
                  git_sha='', git_dirty=False, log=True):
    """One serialization boundary for tables, JSON, markdown and log notes."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('refusing to overwrite an analysis run')
    config_digest = runlog.config_hash(config_paths)
    snapshots = {os.path.relpath(p, paths.repo_root()): sha(p) for p in input_paths if Path(p).is_file()}
    snapshots['snapshot_id'] = metadata['snapshot_id']
    safe = mask({'metadata': metadata, 'families': rows}) if mock else {'metadata': metadata, 'families': rows}
    label = metadata['label']
    safe['label'] = label
    prefix = PREFIX if mock else ''
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'{prefix}stage1.json'
    target.write_text(json.dumps(safe, sort_keys=True, indent=2, default=str, ensure_ascii=False) + '\n', encoding='utf-8')
    flat = []
    for family, row in safe['families'].items():
        flat.append({'label': label, 'spec': family, **safe['metadata'], **row})
    # Nested records stay JSON strings in CSV, preserving all stamps on every row.
    table = [{k: json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list, tuple)) else v
              for k, v in row.items()} for row in flat]
    pl.DataFrame(table, infer_schema_length=None, strict=False).write_csv(directory / f'{prefix}stage1.csv')
    lines = [f'# {label}', '', json.dumps(safe['metadata'], sort_keys=True, default=str), '', '| Family | Result |', '| --- | --- |']
    lines += [f'| {family} | {json.dumps(row, sort_keys=True, default=str)} |' for family, row in safe['families'].items()]
    (directory / f'{prefix}stage1.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if log:
        invocation = uuid4().hex
        when = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        note_keys = ('label', 'snapshot_id', 'snapshot_utc', 'level', 'forced', 'n_units',
                     'n_local_counties', 'substate_source', 'mock_price_window',
                     'k_draws', 'b_draws', 'seed', 'wild_seed', 'states_in', 'states_out')
        brief = {k: metadata[k] for k in note_keys if k in metadata}
        if mock:
            brief['input_checks'] = 'waived (MOCK)'
        for family, row in safe['families'].items():
            notes = json.dumps({**brief, 'status': row['status']}, sort_keys=True, ensure_ascii=False)
            runlog.append_run_log(run_id=invocation + '_' + family,
                utc_timestamp=when, git_sha=git_sha, git_dirty=git_dirty,
                config_hash=config_digest, spec_id=f'{"MOCK_" if mock else ""}2026_{family}',
                data_snapshot_ids=json.dumps(snapshots, sort_keys=True),
                output_path=os.path.relpath(target, paths.repo_root()), notes=notes)
    return safe


def output_rows_for_update(rows: dict, *, final: bool) -> dict:
    """Withhold the entire turnout family until the final update (§7)."""
    if final:
        return rows
    return {**rows, 'turnout': {'status': 'withheld: reported only at the final update (--final)'}}


def exhibit_inputs(primary, state):
    """Frames and stored validation results the exhibits need; each is optional."""
    root = paths.repo_root() / 'reports/validation'
    inputs = {'primary': primary, 'state': state, 'cycles': [], 'validation': {}}
    for cycle in pooled.CYCLES:
        try:
            df, shocks, _ = validate.frame(cycle, validate.PRIMARY)
            inputs['cycles'].append((cycle, df, shocks))
            v = pl.read_csv(root / f'V_{cycle}.csv').filter(pl.col('spec') == 'primary')
            inputs['validation'][cycle] = v.to_dicts()[0]
        except Exception:
            pass
    return inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--mock', action='store_true'); mode.add_argument('--live', action='store_true')
    parser.add_argument('--label', help='Free-text label for this results snapshot')
    parser.add_argument('--final', action='store_true',
                        help='Mark this as the final update; includes turnout in the output rows')
    parser.add_argument('--config', type=Path, default=paths.config_dir() / 'analysis.yaml')
    args = parser.parse_args(argv)
    if args.live and not args.label:
        parser.error('--label is required in live mode')
    # Warnings can contain fit values; they are suppressed before any MOCK computation.
    with warnings.catch_warnings():
        if args.mock:
            warnings.simplefilter('ignore')
        try:
            return run(args)
        except Exception as exc:
            if args.mock:
                mock_failure(exc)
                return 1
            raise


def run(args):
    config = load_config(args.config)
    root, processed = paths.repo_root(), paths.processed_dir()
    input_paths = []
    def read(name, directory=processed):
        path = directory / name
        input_paths.append(path)
        return pl.read_parquet(path)
    shocks = read('shocks.parquet')
    substate = read('substate_shocks.parquet')
    exposure = read('exposure_county.parquet')
    races_path = root / 'config/races_2026.csv'
    incumbents_path = root / 'config/race_incumbents.csv'
    races = pl.read_csv(races_path, schema_overrides={'state_fips': pl.String})
    incumbents = pl.read_csv(incumbents_path)
    history = read('results_county.parquet')
    level_path = root / config['outputs']['level_file']
    hf = read('g_pt_hf.parquet') if config['mu_pt']['source'] == 'hf_nodrift' else None
    directory = paths.interim_dir() if args.mock else processed
    prefix = PREFIX if args.mock else ''
    results = read(prefix + 'results_2026_county.parquet', directory)
    completeness = read(prefix + 'completeness_2026.parquet', directory)
    if results['snapshot_id'].n_unique() != 1:
        raise ValueError('not_run: exactly one results snapshot required')
    times = results['snapshot_utc'] if 'snapshot_utc' in results.columns else results['retrieved_at_utc']
    snapshot = {'snapshot_id': results['snapshot_id'][0], 'snapshot_utc': str(max(times, key=utc)),
                'label': LABEL if args.mock else args.label}
    mock_window = None
    if args.mock:
        from fb26.models import mock2026
        results = mock2026.restore_outcomes(results, history)
        mapping = read(PREFIX + 'mapping_2026.parquet', directory)
        mock_cfg = config.get('mock') or {}
        window = mock_cfg.get('window', 'current')
        if window != 'current':
            shocks, substate, hf, mock_window = mock2026.stand_in_prices(shocks, substate, hf, window)
        # 'auto': the main design (county markets) runs at the level the rule sets; the metro source forces metro.
        forced = 'metro' if (mock_cfg.get('level', 'auto') == 'metro'
                             or power.substate_source_default() == 'metro') else None
        level = level2026.choose_level(exposure, races, substate, forced_level=forced, mock=True,
                                       thresholds=config['level'])
        guards = {g: 'waived (MOCK)' for g in GUARDS}
    else:
        guards, level = live_guards(shocks, level_path, config=config, substate=substate, hf=hf,
                                   results=results, races=races, incumbents=incumbents)
        mapping = []
    params = dict(results=results, history=history, completeness=completeness, races=races,
                  exposure=exposure, shocks=shocks, substate=substate, mock=args.mock,
                  mu_pt_source=config['mu_pt']['source'], hf=hf, thresholds=config['level'],
                  pct_expected_counted=config['completeness']['pct_expected_counted'])
    primary, assignment, info = frame2026.frame(level=level['level'], **params)
    if not args.mock:
        actual_races = results.filter(pl.col('race_id').is_in(info['included_races'])).select(
            'race_id', 'state_fips').unique().with_columns(pl.lit(True).alias('proposed_in_sample'))
        actual_level = level2026.choose_level(exposure, actual_races, substate, thresholds=config['level'])
        if actual_level['level'] != level['level']:
            raise ValueError('live guard: final D3 race universe changes the recorded D22 level')
    state, state_assignment, _ = frame2026.frame(level='state', **params)
    shock_pool = state_assignment.state_table.select('state_fips', 'padd_group', pl.col('dev').alias('deviation'))
    measures = {}
    measure_files = {'sa': 'shocks_r9a.parquet', 'ky12': 'shocks_ky12.parquet'}
    for measure, filename in [(m, measure_files[m]) for m in SECONDARY_MEASURES]:
        try:
            measure_shocks = read(filename)
            used_level = frame2026.measure_level(exposure, races if args.mock else actual_races,
                                                  substate, level['level'],
                                                  measure, config['level'])
            mdf, masg, _ = frame2026.frame(
                level=used_level, measure=measure, **{**params, 'shocks': measure_shocks})
            msh = masg.state_table.select('state_fips', 'padd_group', pl.col('dev').alias('deviation'))
            measures[measure] = (mdf, masg, msh, used_level)
        except Exception:
            # Missing or unusable inputs become fixed not_run rows in analyze().
            pass
    vehicles = None
    try:
        used_level = level['level'] if len(assignment.dev) > len(assignment.states) else 'state'
        vdf, vasg, _ = frame2026.frame(level=used_level, g_method='vehicles', **params)
        vehicles = (vdf, vasg, shock_pool, used_level)
    except Exception:
        pass
    metro_lag = None
    try:
        if len(assignment.dev) > len(assignment.states):
            ldf, lasg, _ = frame2026.frame(level=level['level'], metro_lag=True, **params)
            metro_lag = (ldf, lasg, level['level'])
    except Exception:
        pass
    r_metro = None
    source = power.substate_source_default()
    if source == 'county_markets':
        try:
            rdf, rasg, _ = frame2026.frame(level='metro', source='metro', **params)
            r_metro = (rdf, rasg, 'metro')
        except Exception:
            pass
    holidays = None
    holiday_path = root / 'config/gas_tax_holidays.csv'
    input_paths.append(holiday_path)
    try:
        holidays = pl.read_csv(holiday_path, schema_overrides={'state_fips': pl.String})
    except Exception:
        pass
    tax_cut_inputs = None
    try:
        from fb26.etl import shocks as state_prices
        window = yaml.safe_load((root / 'config/windows.yaml').read_text())['elections'][2026]
        input_paths.extend(paths.interim_dir() / f for f in state_prices.PRICE_FILES)
        prices = state_prices.aaa_window_table(state_prices.load_aaa_prices(), {2026: window}, 2026)
        tax_cut_inputs = dict(prices=prices, window=window, exposure=exposure, history=history)
    except Exception:
        pass
    balance_inputs = None
    try:
        balance_inputs = dict(outcomes=balance.load_outcomes(exposure, catch=True), exposure=exposure,
                              history=history, state_assignment=state_assignment)
    except Exception:
        pass
    past = None
    try:
        past = (history, read('results_hist.parquet').filter(pl.col('usable')))
    except Exception:
        pass
    rows = analyze(primary, state, assignment, shock_pool, config, incumbents=incumbents,
                   measures=measures, vehicles=vehicles, holidays=holidays,
                   level=level['level'], balance_inputs=balance_inputs, past=past,
                   metro_lag=metro_lag, r_metro=r_metro, tax_cut_inputs=tax_cut_inputs)
    metadata = {**snapshot, 'states_in': info['states_in'], 'states_out': info['states_out'],
                'level': level['level'], 'forced': level.get('forced', False),
                'n_units': len(assignment.dev) - len(assignment.states),
                'level_note': ('forced metro; registered coverage only' if level.get('forced') else
                               'per-market rule; registered coverage only') if args.mock else 'D22 rule',
                'n_local_counties': int(assignment.counties.height),
                'substate_source': power.substate_source_default(),
                'mu_pt_source': config['mu_pt']['source'], 'completeness': info,
                'mock_mapping': mapping.to_dicts() if isinstance(mapping, pl.DataFrame) else mapping,
                'guards': guards,
                'k_draws': config['ri']['k'], 'b_draws': config['wild']['b'],
                'seed': config['ri']['seed'], 'wild_seed': config['wild']['seed']}
    if mock_window:
        metadata['mock_price_window'] = mock_window
    out = root / config['outputs']['mock' if args.mock else 'live'] / uuid4().hex
    configs = [args.config, root / 'config/windows.yaml', root / 'config/gas_tax_holidays.csv', races_path,
               root / 'config/competitive_senate_2026.csv', root / 'config/states.csv']
    if level_path.exists() and not args.mock:
        configs.append(level_path)
    from fb26.report import exhibits2026
    exhibit_files = {}
    try:
        rows['exhibits'], exhibit_files = exhibits2026.build(
            rows, exhibit_inputs(primary, state), metadata, mock=args.mock)
    except Exception:
        rows['exhibits'] = {'status': 'not_run: exhibits unavailable'}
    output_rows = output_rows_for_update(rows, final=args.final)
    write_outputs(output_rows, metadata, out, mock=args.mock, config_paths=configs,
                  input_paths=input_paths + [races_path, incumbents_path],
                  git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty())
    exhibits2026.write_files(exhibit_files, out / 'exhibits')
    print(f'{LABEL if args.mock else snapshot["label"]}: stage 1 written to {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
