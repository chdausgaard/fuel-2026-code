"""Pre-specified exhibits 2, 4 and 5 and the result-statement templates (memo §10.1, §10.3).

Tables and figure data only (CSV plus a short markdown file); figures are drawn from them later.
Every file carries the run label, snapshot ID, snapshot time (UTC) and the states in and out. In MOCK
runs the files keep the MOCK prefix and label, every estimate passes through run2026.mask, and the
template choice itself is masked (it would reveal the sign and significance of the stand-in estimate).

    build(rows, inputs, metadata, mock=...) -> (family row, {file name: text})
    write_files(files, directory)
"""
from pathlib import Path

import numpy as np
import polars as pl

from fb26.models import p1, pooled, validate
from fb26.models.run2026 import H1_STATE_LABEL, PREFIX, mask, per_10_dollars
COMPETITIVE_LABEL = 'Competitive Senate races (Cook toss-up or lean, Oct 30)'
OTHER_SENATE_LABEL = 'Other Senate races'

TEMPLATES = {
    'A': ('Counties with heavier fuel-cost burdens swung further against Republicans than comparable counties '
          "in the same race: {against} points against the president's party per point of monthly income "
          '(95% CI [{lo}, {hi}]), or {against10} points '
          'per extra $10 a month (95% CI [{lo10}, {hi10}]). This fits a '
          'pocketbook or local channel. It is not the total effect of gas prices: national effects shared by '
          'all counties are not identified. The state-price analysis, a separate question, gives {against_state} '
          "points against the president's party per point of monthly income."),
    'B': ('We find no detectable difference in swing between more and less burdened counties in the same race '
          "({against} points against the president's party per point of monthly income; 95% CI [{lo}, {hi}]). "
          'The data rule out effects larger than {eps} points per point of '
          'income ({eps10} points per extra $10 a month). National '
          'effects of gas prices, through news, mood or views of the president, are outside what this design '
          'can measure.'),
    'C': ('Contrary to the registered prediction, more burdened counties swung toward Republicans relative to '
          "comparable counties ({against} points against the president's party per point of monthly income; "
          '95% CI [{lo}, {hi}]). We report this as registered; see the twelve-election '
          'check.'),
}


def choose_template(beta: float, lo: float, hi: float) -> str:
    """A: beta < 0 and the 95% CI excludes 0; C: beta > 0 and it excludes 0; B otherwise."""
    if beta < 0 and hi < 0:
        return 'A'
    if beta > 0 and lo > 0:
        return 'C'
    return 'B'


def fmt(value) -> str:
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def stamp_lines(metadata: dict) -> list[str]:
    return [f"Snapshot {metadata['snapshot_id']} at {metadata['snapshot_utc']} (UTC).",
            f"States in: {', '.join(metadata['states_in'])}. States out: {', '.join(metadata['states_out']) or 'none'}."]


def table_text(rows: list[dict], metadata: dict) -> str:
    stamp = {'label': metadata['label'], 'snapshot_id': metadata['snapshot_id'],
             'snapshot_utc': metadata['snapshot_utc'], 'states_in': ';'.join(metadata['states_in']),
             'states_out': ';'.join(metadata['states_out'])}
    return pl.DataFrame([{**stamp, **row} for row in rows], infer_schema_length=None, strict=False).write_csv()


def markdown_table(header: list[str], body: list[list]) -> list[str]:
    return (['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |']
            + ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in body])


def document(title: str, metadata: dict, *parts: list[str]) -> str:
    lines = [f"# {metadata['label']}", '', f'## {title}', '', *stamp_lines(metadata), '']
    for part in parts:
        lines += part + ['']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------- exhibit 2

def tercile_gap(df: pl.DataFrame) -> pl.DataFrame:
    """Per state: weighted mean race-demeaned swing in the top minus the bottom third of counties by E_c."""
    w = pl.col('w')
    d = df.with_columns((pl.col('dv') - (w * pl.col('dv')).sum().over('race_id') / w.sum().over('race_id')).alias('dvd'))
    counties = (d.group_by('state_fips', 'geo_id').agg(pl.col('e_c').first())
                .with_columns(pl.col('e_c').rank('ordinal').over('state_fips').alias('rank'),
                              (pl.len().over('state_fips') // 3).alias('third'))
                .with_columns(pl.when((pl.col('third') >= 1) & (pl.col('rank') <= pl.col('third'))).then(pl.lit('low'))
                              .when((pl.col('third') >= 1) & (pl.col('rank') > pl.len().over('state_fips') - pl.col('third')))
                              .then(pl.lit('high')).alias('tercile')))
    cell = (d.join(counties.select('state_fips', 'geo_id', 'tercile'), on=['state_fips', 'geo_id'])
            .filter(pl.col('tercile').is_not_null())
            .group_by('state_fips', 'tercile').agg(((w * pl.col('dvd')).sum() / w.sum()).alias('m')))
    wide = cell.pivot(on='tercile', index='state_fips', values='m')
    if not {'low', 'high'} <= set(wide.columns):
        return pl.DataFrame(schema={'state_fips': pl.Utf8, 'tercile_gap': pl.Float64})
    return wide.select('state_fips', (pl.col('high') - pl.col('low')).alias('tercile_gap'))


def design_points(df: pl.DataFrame, precision: tuple) -> tuple[pl.DataFrame, float, float]:
    """State points (d_s, ȳ⊥, x̄⊥, S_s, tercile gap) and β̂ from the fit and from the BHJ identity."""
    w, e = df['w'].to_numpy(), df['e_c'].to_numpy()
    res = p1.Residualizer(validate.design_matrix(df, precision=precision), w)
    y, x = res(df['dv'].to_numpy()), res(df['b_tilde'].to_numpy())
    beta_fit = float(np.sum(w * x * y) / np.sum(w * x * x))
    pts = (df.select('state_fips', pl.col('padd_group').cast(pl.Utf8), pl.col('deviation').alias('d_s'))
           .with_columns(pl.Series('s', w * e), pl.Series('sy', w * e * y), pl.Series('sx', w * e * x))
           .group_by('state_fips').agg(pl.col('padd_group').first(), pl.col('d_s').first(),
                                       pl.col('s').sum().alias('S_s'), pl.col('sy').sum(), pl.col('sx').sum())
           .with_columns((pl.col('sy') / pl.col('S_s')).alias('y_bar_perp'),
                         (pl.col('sx') / pl.col('S_s')).alias('x_bar_perp'))
           .join(tercile_gap(df), on='state_fips', how='left')
           .with_columns((pl.col('tercile_gap') - pl.col('tercile_gap').mean().over('padd_group'))
                         .alias('tercile_gap_centered'))
           .sort('state_fips'))
    s, d = pts['S_s'].to_numpy(), pts['d_s'].to_numpy()
    beta_id = float(np.sum(s * d * pts['y_bar_perp'].to_numpy()) / np.sum(s * d * pts['x_bar_perp'].to_numpy()))
    return pts.drop('sy', 'sx'), beta_fit, beta_id


def exhibit2(rows: dict, inputs: dict) -> tuple[list[dict], list[dict]]:
    frames = [(str(cycle), df) for cycle, df, _ in inputs['cycles']] + [('2026', inputs['state'])]
    points, identity = [], []
    for cycle, df in frames:
        precision = validate.C1 if df['c1_swing'].null_count() == 0 else ()
        pts, beta_fit, beta_id = design_points(df, precision)
        points += [{'cycle': cycle, **r} for r in pts.iter_rows(named=True)]
        row = {'cycle': cycle, 'beta_fit': beta_fit, 'beta_identity': beta_id, 'abs_diff': abs(beta_fit - beta_id)}
        if cycle == '2026' and rows['H1_state']['status'] == 'run':
            row['beta_h1_state'] = rows['H1_state']['beta']
            row['abs_diff_h1_state'] = abs(beta_id - rows['H1_state']['beta'])
        identity.append(row)
    return points, identity


# ---------------------------------------------------------------------------- exhibit 4

STAT_ROWS = [('beta', 'β̂'), ('p_ri', 'RI p'), ('ci95', '95% CI'), ('ci90', '90% CI'), ('eps_star', 'ε*'),
             ('beta_per_10_dollars', 'β̂, points per $10 a month'), ('ci95_per_10_dollars', '95% CI, points per $10 a month'),
             ('eps_star_per_10_dollars', 'ε*, points per $10 a month'), ('n_eff_state_layer', 'N_eff (state layer)'),
             ('n_states', 'states'), ('n_races', 'races'), ('n_counties', 'counties')]


def exhibit4(rows: dict, inputs: dict) -> list[dict]:
    robust = rows['robustness']['rows'] if rows['robustness']['status'] == 'run' else {}
    from fb26.models.power import substate_source_default
    source = substate_source_default()
    primary_label = 'Primary, county prices, local markets' if source == 'county_markets' else 'Primary, D22 level'
    columns = [('R2 (minimal)', robust.get('R2', {'status': 'not_run: R2 unavailable'}), None),
               (primary_label, rows['H1'], rows['H1'].get('level_note')),
               (H1_STATE_LABEL, rows['H1_state'], rows['H1_state'].get('level_note')),
               ('R3 (extended)', robust.get('R3', {'status': 'not_run: R3 unavailable'}), None)]
    columns += [(label, robust[name], None) for name, label in (
        ('competitive_senate', COMPETITIVE_LABEL),
        ('other_senate', OTHER_SENATE_LABEL)) if name in robust]
    n_eff = validate.diagnostics(inputs['state'])['n_eff']
    table = []
    for name, r, note in columns:
        row = {'column': name, 'status': r['status'], 'level_note': note or r.get('level_note', '')}
        if r['status'] == 'run':
            row.update({k: r[k] for k in ('beta', 'p_ri', 'eps_star', 'n_states', 'n_races', 'n_counties')},
                       ci95_lo=r['ci95_lo'], ci95_hi=r['ci95_hi'], ci90_lo=r['ci90_lo'], ci90_hi=r['ci90_hi'],
                       n_eff_state_layer=n_eff if name not in (
                           COMPETITIVE_LABEL, OTHER_SENATE_LABEL) else '')
            row.update({f'{k}_per_10_dollars': per_10_dollars(r[k])
                        for k in ('beta', 'eps_star', 'ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi')})
        table.append(row)
    return table


def exhibit4_markdown(table: list[dict]) -> list[str]:
    def cell(r, key):
        if r['status'] != 'run':
            return r['status'] if key == 'beta' else ''
        if key in ('ci95', 'ci90'):
            return f"[{fmt(r[key + '_lo'])}, {fmt(r[key + '_hi'])}]"
        if key == 'ci95_per_10_dollars':
            return f"[{fmt(r['ci95_lo_per_10_dollars'])}, {fmt(r['ci95_hi_per_10_dollars'])}]"
        return r[key]
    body = [[label, *(cell(r, key) for r in table)] for key, label in STAT_ROWS]
    return markdown_table(['', *(r['column'] for r in table)], body)


def competitive_difference(rows: dict) -> list[dict]:
    """Registered (plan §8): competitive Senate races minus all other Senate races, with little power."""
    robust = rows['robustness']['rows'] if rows['robustness']['status'] == 'run' else {}
    d = robust.get('competitive_difference')
    if d is None:
        return []
    row = {'comparison': 'Competitive minus other Senate races', 'status': d['status']}
    if d['status'] == 'run':
        row.update({f'{k}_per_10_dollars': per_10_dollars(d[k])
                    for k in ('beta', 'se_cr1', 'ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi')})
        row.update(p_t=d['p_t'], df=d['df'])
    return [row]


def competitive_difference_parts(diff: list[dict]) -> list[list[str]]:
    if not diff:
        return []
    header = list(dict.fromkeys(k for r in diff for k in r))
    return [['Registered comparison of the two Senate-race columns, in points per $10 a month. The two '
             'estimates rest on different states, so the standard error combines their state-clustered standard '
             'errors, read against a t-distribution with the smaller group\'s number of states minus one as degrees of '
             'freedom. This comparison has little power.'],
            markdown_table(header, [[r.get(k, '') for k in header] for r in diff])]


def exploratory_competitive_states(rows):
    """Explicit publication allowlist: never serialize state-specific price summaries."""
    keys = ('beta', 'se_cr1', 'p_ri', 'ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi',
            'n_rows', 'n_counties', 'n_markets', 'n_groups')
    return [{'state': state, 'status': row['status'], 'exploratory': True,
             **{k: row[k] for k in keys if k in row}}
            for state, row in sorted(rows.get('competitive_by_state', {}).get('rows', {}).items())]


# ---------------------------------------------------------------------------- exhibit 5

def leave_out_ranges(rows: dict) -> list[dict]:
    """Min and max of the leave-out-one point estimates (§8: one state, one supply region)."""
    robust = rows.get('robustness', {})
    if robust.get('status') != 'run':
        return []
    r13 = robust.get('rows', {}).get('R13', {})
    if r13.get('status') != 'run':
        return []
    ranges = []
    for key, label in (('state', 'state'), ('padd', 'region')):
        betas = [drop['beta'] for drop in r13.get(key, [])
                 if drop.get('status') == 'run' and 'beta' in drop]
        if betas:
            lo, hi = min(betas), max(betas)
            ranges.append({'entry': f'leave_out_{label}', 'cycle': '2026', 'I_t': '',
                           'estimate': '', 'ci95_lo': '', 'ci95_hi': '', 'ci90_lo': '', 'ci90_hi': '',
                           'range_lo': lo, 'range_hi': hi, 'n_drops': len(betas),
                           'range_lo_per_10_dollars': per_10_dollars(lo),
                           'range_hi_per_10_dollars': per_10_dollars(hi)})
    return ranges


def exhibit5(rows: dict, inputs: dict) -> list[dict]:
    table = []
    for cycle in pooled.CYCLES:
        v = inputs.get('validation', {}).get(cycle)
        if v is None:
            continue
        sign = pooled.SIGN[cycle]
        lo95, hi95 = sorted((sign * v['ci95_lo'], sign * v['ci95_hi']))
        lo90, hi90 = sorted((sign * v['ci90_lo'], sign * v['ci90_hi']))
        table.append({'entry': f'I_t x beta_{cycle}', 'cycle': str(cycle), 'I_t': f'{sign:+d}',
                      'estimate': sign * v['beta'], 'ci95_lo': lo95, 'ci95_hi': hi95, 'ci90_lo': lo90, 'ci90_hi': hi90})
    h = rows['H1_state']
    if h['status'] != 'run':
        raise ValueError('not_run: state-level estimate unavailable')
    table.append({'entry': 'beta_2026_state', 'cycle': '2026', 'I_t': f"{pooled.SIGN[2026]:+d}",
                  'estimate': h['beta'], 'ci95_lo': h['ci95_lo'], 'ci95_hi': h['ci95_hi'],
                  'ci90_lo': h['ci90_lo'], 'ci90_hi': h['ci90_hi']})
    table.extend(leave_out_ranges(rows))
    return table


# ---------------------------------------------------------------------------- result templates

def result_template(rows: dict, inputs: dict, mock: bool) -> tuple[str, str]:
    h = rows['H1']
    if h['status'] != 'run':
        raise ValueError('not_run: primary estimate unavailable')
    choice = choose_template(h['beta'], h['ci95_lo'], h['ci95_hi'])
    state = rows['H1_state']
    values = {'against': -h['beta'], 'lo': -h['ci95_hi'], 'hi': -h['ci95_lo'], 'eps': h['eps_star'],
              'against10': per_10_dollars(-h['beta']), 'lo10': per_10_dollars(-h['ci95_hi']),
              'hi10': per_10_dollars(-h['ci95_lo']), 'eps10': per_10_dollars(h['eps_star']),
              'against_state': -state['beta'] if state['status'] == 'run' else 'n/a (state-level estimate not run)'}
    if mock:
        values = {k: 'MASKED' for k in values}
        text = ['Template selection is masked in MOCK runs. Fill-ins of each template:', '']
        text += [f'- **{k}.** ' + TEMPLATES[k].format(**values) for k in TEMPLATES]
        return 'MASKED', '\n'.join(text)
    text = [f'Draft {choice}, chosen by the sign of the primary estimate and whether its 95% CI excludes 0. '
            'A starting point only: the published text is written by hand.', '',
            TEMPLATES[choice].format(**{k: fmt(v) for k, v in values.items()})]
    return choice, '\n'.join(text)


# ---------------------------------------------------------------------------- assembly

def build(rows: dict, inputs: dict, metadata: dict, *, mock: bool) -> tuple[dict, dict]:
    """Family row and files; a failing exhibit becomes a fixed not_run string."""
    prefix = PREFIX if mock else ''
    safe = mask if mock else (lambda value, key='': value)
    files, family = {}, {}

    def ex2():
        points, identity = exhibit2(rows, inputs)
        points, identity = safe(points, 'points'), safe(identity, 'identity')
        files[prefix + 'exhibit2_design.csv'] = table_text(points, metadata)
        files[prefix + 'exhibit2_identity.csv'] = table_text(identity, metadata)
        files[prefix + 'exhibit2.md'] = document(
            'Exhibit 2: the design in one picture (state layer)', metadata,
            ['One point per state and cycle: d_s (deviation from the PADD-group mean), the exposure-weighted '
             'residualized swing ȳ⊥, the residualized treatment x̄⊥ and S_s (point size) are in '
             f'`{prefix}exhibit2_design.csv`, with the coarsened twin (tercile gap minus its PADD-group mean). '
             'The identity β̂ = Σ S·d·ȳ⊥ / Σ S·d·x̄⊥ is checked below. State layer of the design only.'],
            markdown_table(list(identity[0]), [list(r.values()) for r in identity]))

    def ex4():
        table = safe(exhibit4(rows, inputs), 'table')
        files[prefix + 'exhibit4_main_table.csv'] = table_text(table, metadata)
        files[prefix + 'exhibit4.md'] = document(
            'Exhibit 4: main table', metadata,
            ['R2 and R3 are at the D22 level unless a level note says otherwise. N_eff is the effective number of '
             'shocks of the state layer. The state-level column is the H1-state question, state-wide price '
             'differences, with its own reshuffling test (within PADD groups). Points per $10 a month use one '
             'income figure, the household-weighted median monthly income of the analysis counties '
             '(analysis.yaml, translation).'], exhibit4_markdown(table),
            *competitive_difference_parts(safe(competitive_difference(rows), 'table')))

    def exploratory():
        table = safe(exploratory_competitive_states(rows), 'table')
        if not table:
            raise ValueError('no competitive states in the frame')
        title = 'Exploratory: each competitive Senate race alone (noisy; few comparison groups per state)'
        files[prefix + 'exploratory_competitive_states.csv'] = table_text(table, metadata)
        header = list(dict.fromkeys(k for r in table for k in r))
        files[prefix + 'exploratory_competitive_states.md'] = document(
            title, metadata, ['CR1 standard errors clustered by local market; no wild cluster bootstrap.'],
            markdown_table(header, [[r.get(k, '') for k in header] for r in table]))

    def ex5():
        table = safe(exhibit5(rows, inputs), 'table')
        files[prefix + 'exhibit5_cross_cycle.csv'] = table_text(table, metadata)
        range_note = ''
        if not mock:
            for r in leave_out_ranges(rows):
                label = 'state' if r['entry'] == 'leave_out_state' else 'supply region'
                range_note += (f' Leaving out one {label} at a time gives a range of '
                               f'[{fmt(r["range_lo_per_10_dollars"])}, {fmt(r["range_hi_per_10_dollars"])}] '
                               f'points per extra $10 a month over {r["n_drops"]} drops.')
        header = list(dict.fromkeys(k for r in table for k in r))
        files[prefix + 'exhibit5.md'] = document(
            'Exhibit 5: past elections and 2026', metadata,
            ['Past estimates for 2018–2024 (I_t·β̂_t) and the state-level β̂ for 2026, each with its intervals. '
             f'The 2026 entry is {H1_STATE_LABEL}. No test statistic.' + range_note],
            markdown_table(header, [[r.get(k, '') for k in header] for r in table]))

    def template():
        family['template'], text = result_template(rows, inputs, mock)
        files[prefix + 'result_template.md'] = document('Draft result statement (not registered)', metadata, [text])

    for name, make in [('exhibit2', ex2), ('exhibit4', ex4), ('exhibit5', ex5), ('exploratory_competitive_states', exploratory), ('template_status', template)]:
        try:
            make()
            family[name] = 'run'
        except Exception:
            family[name] = f'not_run: {name.split("_")[0]} unavailable'
    family['status'] = ('run' if any(family[k] == 'run' for k in ('exhibit2', 'exhibit4', 'exhibit5', 'exploratory_competitive_states', 'template_status'))
                        else 'not_run: no exhibit could be built')
    family['files'] = sorted(files)
    return family, files


def write_files(files: dict, directory) -> None:
    if not files:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text, encoding='utf-8')
