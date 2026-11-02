"""Outcome-free D22 treatment-level decision using registered price coverage."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import polars as pl
import yaml

from fb26.etl.substate_shocks import Rules
from fb26.models.frame2026 import build_assignment, registered_rows, level_config, require_final
from fb26.util import paths, runlog


def choose_level(counties: pl.DataFrame, races: pl.DataFrame, substate: pl.DataFrame, *,
                 forced_level: str | None = None, mock: bool = False,
                 thresholds: dict | None = None, rule: str | None = None) -> dict:
    if forced_level is not None and (not mock or forced_level != 'metro'):
        raise ValueError('a forced metro level is allowed only in MOCK')
    thresholds = level_config() if thresholds is None else thresholds
    if rule is None:
        from fb26.models.power import level_rule_default
        rule = level_rule_default()
    states = races.filter(pl.col('proposed_in_sample') & (pl.col('state_fips') != '02'))['state_fips'].unique().sort().to_list()
    universe = counties.filter(pl.col('state_fips').is_in(states)).select('geo_id', 'state_fips').unique()
    if not states or universe.height == 0:
        raise ValueError('not_run: empty included-race county universe')
    rows = registered_rows(substate, thresholds)
    covered = rows.filter(pl.col('coverage_ok') & pl.col('u_c').is_not_null() & pl.col('cluster_id').is_not_null())
    county_ids = covered.filter(pl.col('level') == 'county')['geo_id'].unique()
    coverage = universe.with_columns(pl.col('geo_id').is_in(county_ids.implode()).alias('covered'))
    by_state = coverage.group_by('state_fips').agg(pl.len().alias('counties'), pl.col('covered').sum().alias('covered'))
    stats = {r['state_fips']: {'counties': r['counties'], 'covered': r['covered'],
                              'share': r['covered'] / r['counties']} for r in by_state.sort('state_fips').to_dicts()}
    missing = sorted(set(states) - set(stats))
    if missing:
        raise ValueError('not_run: missing county universe for included states')
    share = coverage['covered'].sum() / universe.height
    metros = covered.filter((pl.col('level') == 'metro') & pl.col('geo_id').is_in(universe['geo_id'].implode()))
    metro_states = set(metros['state_fips'].to_list())
    d22_level = ('county_metro' if share >= thresholds['county_share'] and all(v['share'] >= thresholds['state_county_share'] for v in stats.values())
                 else 'metro' if set(states) <= metro_states else 'state')
    # per_market rule: use 'county' when at least one county is covered in every state
    # (markets without a covered county fall back to the state layer automatically);
    # the D22 computation is always reported as a diagnostic.
    county_states = set(covered.filter(pl.col('level') == 'county')['state_fips'].to_list())
    per_market_level = 'county' if set(states) <= county_states else d22_level
    level = per_market_level if rule == 'per_market' else d22_level
    return {'level': forced_level or level, 'rule_level': level, 'rule': rule, 'd22_level': d22_level,
            'forced': forced_level is not None,
            'label': 'MOCK — NOT 2026 RESULTS' if mock else '2026 treatment level',
            'states': states, 'n_counties': universe.height, 'n_counties_covered': coverage['covered'].sum(),
            'county_share': share, 'by_state': stats, 'metro_states': sorted(metro_states),
            'rules': asdict(Rules(min_days_pre=thresholds['min_days'], min_days_post=thresholds['min_days'])),
            'thresholds': thresholds}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=paths.config_dir() / 'analysis.yaml')
    args = parser.parse_args(argv)
    when = datetime.now(timezone.utc)
    cfg = yaml.safe_load(args.config.read_text())
    root = paths.repo_root()
    inputs = {'exposure': paths.processed_dir() / 'exposure_county.parquet',
              'shocks': paths.processed_dir() / 'shocks.parquet',
              'substate': paths.processed_dir() / 'substate_shocks.parquet',
              'races': paths.config_dir() / 'races_2026.csv'}
    shocks, substate = pl.read_parquet(inputs['shocks']), pl.read_parquet(inputs['substate'])
    require_final(shocks.filter(pl.col('year') == 2026), 'provisional state shocks')
    require_final(substate.filter((pl.col('cycle') == 2026) & (pl.col('measure') == 'w21')),
                  'provisional sub-state shocks')
    result = choose_level(pl.read_parquet(inputs['exposure']),
                          pl.read_csv(inputs['races'], schema_overrides={'state_fips': pl.Utf8}),
                          substate, thresholds=cfg['level'])
    assignment = build_assignment(shocks, substate, result['level'], thresholds=cfg['level'])
    st = assignment.state_table
    raw, dev = st['delta_p'].to_numpy(), st['dev'].to_numpy()
    total = float(np.mean((raw - raw.mean()) ** 2))
    within = float(np.mean(dev ** 2))
    result['t1'] = {'n_states_pool': len(assignment.states), 'n_clusters_pool': assignment.clusters.height,
                    'n_counties_layer2': assignment.counties.height, 'state_shock_variance': total,
                    'state_within_padd_variance': within,
                    'state_within_padd_share': within / total if total else None,
                    'state_between_padd_share': 1 - within / total if total else None,
                    'sample_weighted_diagnostics': 'not_run: outcome-free level decision has no baseline vote weights',
                    'serial_correlation': 'not_run: stage 1 treatment-level coverage artifact',
                    'pass_through_fit': 'not_run: stage 1 treatment-level coverage artifact'}
    result.update(decided_at_utc=when.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
                  input_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in inputs.values()})
    out = root / cfg.get('outputs', {}).get('level_file', 'reports/2026/treatment_level_2026.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    configs = [args.config, paths.config_dir() / 'windows.yaml', paths.config_dir() / 'gas_tax_holidays.csv', inputs['races'], out]
    chash = runlog.config_hash(configs)
    snapshot = {**result['input_hashes'], str(out.relative_to(root)): hashlib.sha256(out.read_bytes()).hexdigest()}
    runlog.append_run_log(run_id='level2026_' + uuid4().hex, utc_timestamp=result['decided_at_utc'],
                          git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash=chash,
                          spec_id='level2026', data_snapshot_ids=json.dumps(snapshot, sort_keys=True),
                          output_path=str(out.relative_to(root)), notes=json.dumps(
                              {'label': result['label'], 'level': result['level'], 'forced': False,
                               'n_units': assignment.clusters.height, 'status': 'run'}, sort_keys=True))
    print(f"D22 level: {result['level']}; {result['n_counties']} counties")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
