"""T8 stand-in outcomes. Numeric outcomes exist only in memory; disk copies are masked."""
import argparse
import warnings
from pathlib import Path

import polars as pl

from fb26.models import validate
from fb26.util import paths

LABEL = 'MOCK — NOT 2026 RESULTS'
PREFIX = 'MOCK-NOT-2026-RESULTS_'
SNAPSHOT_ID = PREFIX + 'stand-in-2022'
SNAPSHOT_UTC = '2026-11-03T00:00:00Z'


def window_label(meta: dict) -> str:
    """One-line label of a stand-in price window (string only, so the output mask keeps it)."""
    return (f"{meta['window']} stand-in window: {meta['post_start']}..{meta['post_end']} "
            f"against {meta['pre_start']}..{meta['pre_end']}")


def stand_in_prices(shocks: pl.DataFrame, substate: pl.DataFrame, hf: pl.DataFrame | None,
                    window: str, actual_min_days: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, str]:
    """Replace the 2026 w21 state and sub-state price shocks by a stand-in window (power_geo.WINDOWS).

    The current 2026 window may have no local unit with the registered minimum of days, so the main design
    (county markets, blocks) could not run. A stand-in window such as S1 (2026-09-13..10-03 against the
    2024 window) has the days. Only the w21 treatment rows are replaced; other measures and cycles stay.
    Outcomes are untouched: the stand-in is a price window, not a result. Never used outside MOCK.
    actual_min_days optionally overrides only the actual window's post-period day minimum.
    """
    from fb26.models import power_geo
    sh, sub, hf_w, meta = power_geo.window_inputs(window, actual_min_days)
    shocks = pl.concat([shocks.filter(pl.col('year') != 2026), sh], how='diagonal_relaxed')
    substate = pl.concat([substate.filter(~((pl.col('cycle') == 2026) & (pl.col('measure') == 'w21'))), sub],
                         how='diagonal_relaxed')
    return shocks, substate, (hf_w if hf is not None else None), window_label(meta)


def build(historical: pl.DataFrame, races: pl.DataFrame,
          snapshot_utc: str = SNAPSHOT_UTC) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Map the 2020→2022 swing to listed races, same office first.

    ``mock_dv`` is the outcome for the MOCK frame. Candidate rows retain the
    source votes, so the frame must not subtract the 2024 baseline from them.
    Literal 2026 controls and 2024 baseline weights are supplied by frame2026.
    """
    required = {'year', 'state_fips', 'geo_id', 'race_id', 'office', 'party', 'votes'}
    if not required.issubset(historical.columns):
        raise ValueError('not_run: historical input is missing required results columns')
    required_races = {'state_fips', 'state_po', 'race_id', 'office', 'proposed_in_sample'}
    if not required_races.issubset(races.columns):
        raise ValueError('not_run: race list is missing required columns')
    keep = races.filter(
        pl.col('proposed_in_sample').cast(pl.String).str.to_lowercase() == 'true'
    ).filter(pl.col('state_fips') != '02').sort('race_id')
    source = historical.filter(
        (pl.col('year') == 2022) & pl.col('office').is_in(['SENATE', 'GOVERNOR'])
        & pl.col('party').is_in(['D', 'R'])
    )
    if 'included_d3' in source.columns:
        source = source.filter(pl.col('included_d3').fill_null(False))
    base = validate._r2(historical.filter(
        (pl.col('year') == 2020) & (pl.col('office') == 'PRESIDENT')
    ), ['geo_id']).filter(pl.col('two_party') > 0).select(
        'geo_id', pl.col('r2').alias('r2_base'))
    pieces, audit = [], []
    for race in keep.iter_rows(named=True):
        candidates = source.filter(pl.col('state_fips') == race['state_fips'])
        choices = candidates.select('race_id', 'office').unique().sort('race_id')
        same = choices.filter(pl.col('office') == race['office'])
        options = same if same.height else choices
        record = {'race_id': race['race_id'], 'state_fips': race['state_fips'],
                  'mock_source_race_id': None, 'mapping': None, 'status': 'ok', 'label': LABEL}
        if options.is_empty():
            record['status'] = 'not_run: no_2022_statewide_race'
            audit.append(record)
            continue
        # Regular race first when the source has both regular and special elections.
        if 'special' in candidates.columns:
            regular = candidates.filter(~pl.col('special').fill_null(False))['race_id'].unique()
            preferred = options.filter(pl.col('race_id').is_in(regular.implode()))
            if preferred.height:
                options = preferred
        chosen = options['race_id'][0]
        record.update(mock_source_race_id=chosen,
                      mapping='same_office' if same.height else 'other_office')
        rows = candidates.filter(pl.col('race_id') == chosen)
        swing = (validate._r2(rows, ['geo_id']).filter(pl.col('two_party') > 0)
                 .join(base, on='geo_id', how='inner', validate='1:1')
                 .select('geo_id', (pl.col('r2') - pl.col('r2_base')).alias('mock_dv')))
        missing = rows.select('geo_id').unique().join(swing, on='geo_id', how='anti')
        for geo in missing['geo_id']:
            audit.append({**record, 'geo_id': geo,
                          'status': 'not_run: missing_2020_baseline_or_zero_two_party'})
        if swing.is_empty():
            record['status'] = 'not_run: no_counties_with_2020_baseline'
            audit.append(record)
            continue
        rows = rows.join(swing, on='geo_id', how='inner', validate='m:1').with_columns(
            pl.lit(2026).alias('year'), pl.lit(race['race_id']).alias('race_id'),
            pl.lit(race['office']).alias('office'), pl.lit(chosen).alias('mock_source_race_id'),
            pl.lit(SNAPSHOT_ID).alias('snapshot_id'), pl.lit(snapshot_utc).alias('snapshot_utc'),
            pl.lit(snapshot_utc).alias('retrieved_at_utc'),
            pl.lit(LABEL).alias('label'), pl.lit(True).alias('included_d3'),
        )
        pieces.append(rows)
        audit.append(record)
    empty_results = historical.head(0).with_columns(
        pl.lit(None, dtype=pl.Float64).alias('mock_dv'),
        pl.lit(None, dtype=pl.String).alias('mock_source_race_id'),
        pl.lit(SNAPSHOT_ID).alias('snapshot_id'), pl.lit(snapshot_utc).alias('snapshot_utc'),
        pl.lit(snapshot_utc).alias('retrieved_at_utc'), pl.lit(LABEL).alias('label'))
    complete = keep.select('state_po').unique().sort('state_po').with_columns(
        pl.lit(snapshot_utc).alias('snapshot_utc'), pl.lit(100.0).alias('pct_expected_counted'),
        pl.lit(LABEL).alias('source'), pl.lit(LABEL).alias('label'))
    audit_schema = {'race_id': pl.String, 'state_fips': pl.String,
                    'mock_source_race_id': pl.String, 'mapping': pl.String,
                    'status': pl.String, 'label': pl.String, 'geo_id': pl.String}
    return (pl.concat(pieces, how='diagonal_relaxed').sort('race_id', 'geo_id', 'party')
            if pieces else empty_results,
            complete,
            pl.DataFrame(audit, schema=audit_schema))


def write_inputs(results: pl.DataFrame, completeness: pl.DataFrame, audit: pl.DataFrame,
                 directory: Path | None = None) -> list[Path]:
    """Write visibly labelled artifacts after masking all outcome numeric fields."""
    directory = paths.interim_dir() if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sensitive = [name for name, dtype in results.schema.items()
                 if dtype.is_numeric() and name != 'year']
    sensitive += [name for name in ('votes', 'mock_dv') if name in results.columns]
    masked = results.with_columns([pl.lit('MASKED').alias(name) for name in sorted(set(sensitive))])
    outputs = []
    for name, table in [('results_2026_county', masked),
                        ('completeness_2026', completeness), ('mapping_2026', audit)]:
        target = directory / f'{PREFIX}{name}.parquet'
        table.with_columns(pl.lit(LABEL).alias('label')).write_parquet(target)
        outputs.append(target)
    return outputs


def restore_outcomes(results: pl.DataFrame, historical: pl.DataFrame) -> pl.DataFrame:
    """Restore masked stand-in inputs by source keys; computed outcomes stay in memory."""
    keys = ['geo_id', 'mock_source_race_id', 'party']
    if 'candidate' in results.columns:
        keys.append('candidate')
    source = historical.filter(pl.col('year') == 2022).rename({'race_id': 'mock_source_race_id'})
    numeric = [name for name, dtype in source.schema.items()
               if dtype.is_numeric() and name != 'year' and name in results.columns]
    restored = results.drop(*numeric, 'mock_dv').join(source.select(*keys, *numeric),
                                                    on=keys, how='left', validate='m:1')
    base = validate._r2(historical.filter((pl.col('year') == 2020) & (pl.col('office') == 'PRESIDENT')),
                       ['geo_id']).select('geo_id', pl.col('r2').alias('r2_base'))
    swing = (validate._r2(restored, ['geo_id', 'race_id']).join(base, on='geo_id', how='left')
             .select('geo_id', 'race_id', (pl.col('r2') - pl.col('r2_base')).alias('mock_dv')))
    return restored.join(swing, on=['geo_id', 'race_id'], how='left', validate='m:1')


def main(argv: list[str] | None = None) -> int:
    from fb26.models.run2026 import mock_failure
    parser = argparse.ArgumentParser(description=LABEL)
    parser.add_argument('--snapshot-utc', default=SNAPSHOT_UTC)
    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            historical = pl.read_parquet(paths.processed_dir() / 'results_county.parquet')
            races = pl.read_csv(paths.config_dir() / 'races_2026.csv',
                                schema_overrides={'state_fips': pl.String})
            write_inputs(*build(historical, races, args.snapshot_utc))
        except Exception as exc:
            mock_failure(exc)
            return 1
    print(LABEL)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
