"""House outcome construction for blind T5 power; no assignment is fitted here."""
import polars as pl

from fb26.etl.exposure import with_market_oil
from fb26.models import burden, validate
from fb26.util import paths


def outcomes(cycle: int, eligible: pl.DataFrame, house: pl.DataFrame | None = None,
             results: pl.DataFrame | None = None) -> tuple[pl.DataFrame, list[dict]]:
    """County × district parts, restricted to eligible counties; sequential filter audit.

    Stored unit_id is a source county identifier, not globally a county × district
    identifier. Group with race_id as well. The input already assigns fusion lines
    to their candidate's major party; sum by candidate before summing D and R.
    """
    if house is None:
        house = pl.read_parquet(paths.processed_dir() / 'house_county.parquet')
    if results is None:
        results = pl.read_parquet(paths.processed_dir() / 'results_county.parquet')
    d = (house.filter(pl.col('year') == cycle)
         .with_columns(pl.concat_str(pl.lit('HOUSE'), 'state_fips', 'district', 'year', separator='_').alias('race_id')))
    audit = []

    def record(stage: str) -> None:
        audit.append({'outcome': cycle, 'filter': stage,
                      'districts': d['race_id'].drop_nulls().n_unique(),
                      'units': d.select('race_id', 'unit_id').unique().height,
                      'states': d['state_fips'].n_unique()})

    record('Input year (observed districts only)')
    for label, keep in [
        ('Reference contested: exactly one D and one R nominee',
         pl.col('contested') & (pl.col('classification') == 'contested') & pl.col('district_in_reference')),
        ('District reconciliation pass', pl.col('reconciliation_status') == 'pass'),
        ('ACS match and assigned county', pl.col('acs_match') & pl.col('geo_id').is_not_null()),
        ('Exclude Alaska', (pl.col('state_fips') != '02') & ~pl.col('flags').fill_null('').str.contains('excluded_ak')),
        ('Eligible county universe', pl.col('geo_id').is_in(eligible['geo_id'].implode())),
        ('D/R candidate votes, no write-ins', pl.col('party').is_in(['D', 'R']) & ~pl.col('writein')),
    ]:
        d = d.filter(keep)
        record(label)
    keys = ['state_fips', 'geo_id', 'race_id', 'unit_id']
    d = (d.group_by(keys + ['cand', 'party'], maintain_order=True).agg(pl.col('votes').sum())
         .group_by(keys, maintain_order=True).agg(
             pl.col('votes').filter(pl.col('party') == 'R').sum().alias('r'),
             pl.col('votes').filter(pl.col('party') == 'D').sum().alias('d'))
         .with_columns((pl.col('r') + pl.col('d')).alias('two_party'))
         .filter(pl.col('two_party') > 0)
         .with_columns((100 * pl.col('r') / pl.col('two_party')).alias('r2')))
    record('Positive unit D+R votes')
    # Same presidential baseline and percentage-point scale as validate.outcomes.
    baseline = 2024 if cycle == 2026 else validate.BASELINE[cycle]
    base = (results.filter((pl.col('year') == baseline) &
                           (pl.col('office') == 'PRESIDENT') & pl.col('party').is_in(['D', 'R']))
            .group_by('geo_id', maintain_order=True).agg(
                pl.col('votes').sum().alias('v_c'),
                pl.col('votes').filter(pl.col('party') == 'R').sum().alias('r_base'))
            .with_columns((100 * pl.col('r_base') / pl.col('v_c')).alias('r2_base')))
    d = (d.join(base, on='geo_id', how='left', validate='m:1')
         .filter(pl.col('r2_base').is_not_null() & pl.col('r2_base').is_finite() & (pl.col('v_c') > 0))
         .with_columns((pl.col('r2') - pl.col('r2_base')).alias('dv'),
                       (pl.col('v_c') * pl.col('two_party') / pl.col('two_party').sum().over('geo_id')).alias('w'))
         .sort('race_id', 'geo_id', 'unit_id'))
    record('Usable baseline; final House outcomes')
    return d, audit


def stack(statewide: pl.DataFrame, house: pl.DataFrame) -> pl.DataFrame:
    """Each statewide race and the entire House block receive one county vote budget."""
    counts = statewide.group_by('geo_id', maintain_order=True).agg(pl.col('race_id').n_unique().alias('n_s'))
    h = house.join(counts, on='geo_id', how='left', validate='m:1').with_columns(pl.col('n_s').fill_null(0))
    s = (statewide.join(house.select('geo_id').unique().with_columns(pl.lit(1).alias('has_h')), on='geo_id', how='left')
         .join(counts, on='geo_id', how='left', validate='m:1')
         .with_columns((pl.col('w') * pl.col('n_s') / (pl.col('n_s') + pl.col('has_h').fill_null(0))).alias('w')))
    h = h.with_columns((pl.col('w') / (pl.col('n_s') + 1)).alias('w'))
    return pl.concat([s.select(statewide.columns), h.select(statewide.columns)]).sort('race_id', 'geo_id')


def frame(cycle: int) -> tuple[pl.DataFrame, list[dict]]:
    """All otherwise eligible House counties, including states without statewide races."""
    b = burden.build(cycle, validate.PRIMARY).filter(pl.col('b_tilde').is_not_null())
    if cycle == 2026:
        b = with_market_oil(b)
    out, audit = outcomes(cycle, b.select('geo_id'))
    baseline = 2024 if cycle == 2026 else validate.BASELINE[cycle]
    c1 = validate.pres_swing(baseline - 4, baseline).select(
        'geo_id', pl.col('swing').alias('c1_swing'))
    df = (out.join(b.select('geo_id', 'e_c', 'oil_gas_emp_share'), on='geo_id', how='left', validate='m:1')
          .join(c1, on='geo_id', how='left', validate='m:1')
          .with_columns(pl.col('oil_gas_emp_share').fill_null(0.0)))
    if df['c1_swing'].null_count():
        raise ValueError('C1 unavailable for House rows; do not silently change the control set or sample')
    return df, audit
