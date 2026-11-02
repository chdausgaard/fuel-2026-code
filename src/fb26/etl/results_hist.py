"""Historical Senate and governor county returns, 2002-2020 (placebo battery and same-office baselines).

Source: Amlani and Algara county-level returns (doi:10.7910/DVN/DGUMFI, CC0), the Senate and governor Rdata files in
`data/raw/county_returns_hist/`. Builds `data/processed/results_hist.parquet`: one row per county x race, with the
source's D and R votes (a candidate's total across ballot lines) and all votes cast. Nothing is dropped: the race rule
and the registered exclusions are columns (`one_d_one_r`, `usable`, `exclusion`); consumers filter on `usable`.

    python -m fb26.etl.results_hist        # build and write; appends logs/run_log.csv

Rules:
* Race rule: exactly one D and one R nominee on the November ballot. The source must name both and give both votes,
  no second D or R candidate with at least 1% of the statewide vote may be on the MEDSL general ballot (Senate; there
  is no second source for governors), and an independent the source codes as the Democrat (Sanders, King) fails.
* FIPS: Shannon SD (46113) is Oglala Lakota (46102); Bedford city VA (51515) is summed into Bedford County (51019).
* Exclusions, flagged and kept in the file: Connecticut (the source has only the eight legacy counties), Jackson
  County MO (the Kansas City board's votes are missing from every Missouri race), the Massachusetts special of 19
  January 2010 (not a November election), Alaska, rows with missing votes and counties outside the project universe.
* Fusion lines: D and R votes are candidate totals across ballot lines, as in the project's own results.
"""
import io
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from fb26.scrape import medsl
from fb26.util import paths, rawstore, runlog

PARSER_VERSION = 'results_hist-0.1.0'
SOURCE_DIR = 'county_returns_hist'
DATAFILES = {'senate': '5028534', 'governor': '5028535'}      # Dataverse file ids
YEARS = tuple(range(2002, 2022, 2))
SHANNON = ('46113', '46102')
BEDFORD = ('51515', '51019')
JACKSON_MO = ('MO', '29095')
INDEPENDENT_CODED_AS_DEM = ('Bernie Sanders', 'Angus King')
EXCLUDED_RACES = {'2010_MA_SEN_SPECIAL': 'special_election_held_2010-01-19_not_november'}
KEEP = ['election_id', 'election_year', 'fips', 'county_name', 'state', 'office', 'election_type', 'seat_class',
        'seat_status', 'democratic_raw_votes', 'dem_nominee', 'republican_raw_votes', 'rep_nominee',
        'raw_county_vote_totals']
VOTES = ['democratic_raw_votes', 'republican_raw_votes', 'raw_county_vote_totals']
RACE_KEY = ['race_id', 'election_year', 'state', 'office_o', 'seat_class', 'special', 'seat_status', 'dem_nominee',
            'rep_nominee']
OUT_COLUMNS = ['year', 'state_po', 'county_fips', 'office', 'race_id', 'dem_votes', 'rep_votes', 'total_votes',
               'flags', 'county_name', 'in_universe', 'state_fips', 'geo_id', 'special', 'seat_class', 'dem_nominee',
               'rep_nominee', 'one_d_one_r', 'usable', 'exclusion']
NO_NOMINEE = ('none', '', 'nan')


# ---------------------------------------------------------------------------------------------------- raw files

def raw_records() -> dict:
    """Newest raw record of each Rdata file."""
    out = {}
    for name, file_id in DATAFILES.items():
        found = sorted((paths.raw_dir() / SOURCE_DIR).glob(f'**/*datafile-{file_id}_*.rdata.gz'))
        if not found:
            raise FileNotFoundError(f'raw {name} Rdata file ({file_id}) is not in the raw store')
        out[name] = rawstore.record_from_meta(rawstore.meta_path_for(found[-1]))
    return out


def read_rdata(records: dict) -> dict[str, pl.DataFrame]:
    """The Senate and governor tables as exported by Rscript (all columns text; NA is null)."""
    rscript = shutil.which('Rscript')
    if rscript is None:
        raise RuntimeError('Rscript is required to read the Amlani and Algara Rdata files')
    tables = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, record in records.items():
            (work / f'{name}.rdata').write_bytes(rawstore.load_raw(record.path))
        code = ('for (f in c("senate","governor")) { e <- new.env(); n <- load(paste0(f,".rdata"), e); '
                'for (x in setdiff(n, ".Random.seed")) write.csv(get(x,e), paste0(f,".csv"), row.names=FALSE) }')
        subprocess.run([rscript, '-e', code], cwd=work, check=True, capture_output=True)
        for name in records:
            tables[name] = pl.read_csv(work / f'{name}.csv', infer_schema_length=0, null_values=['NA'])
    return tables


# ---------------------------------------------------------------------------------------------------- parsing

def _no_nominee(col: str) -> pl.Expr:
    return pl.col(col).is_null() | pl.col(col).str.strip_chars().str.to_lowercase().is_in(list(NO_NOMINEE))


def county_rows(tables: dict[str, pl.DataFrame], universe: set[str]) -> pl.DataFrame:
    """County x race rows of 2002-2020 with recodes and flags (race-level columns are added by `build`)."""
    sen = tables['senate'].select(KEEP)
    gov = tables['governor'].with_columns(pl.lit('').alias('seat_class')).select(KEEP)
    allrows = pl.concat([sen, gov])
    bedford_years = set(allrows.filter(pl.col('fips') == BEDFORD[0])['election_year'].cast(pl.Int32).to_list())
    df = (allrows.with_columns(pl.col('election_year').cast(pl.Int32), *[pl.col(c).cast(pl.Float64) for c in VOTES])
          .filter(pl.col('election_year').is_in(list(YEARS))))
    special = pl.col('election_type') == 'S'
    df = df.with_columns(special.alias('special'),
                         pl.when(pl.col('office') == 'GOV').then(pl.lit('GOV'))
                         .when(special).then(pl.lit('SEN_SPECIAL')).otherwise(pl.lit('SEN')).alias('office_o'))
    df = df.with_columns((pl.col('election_year').cast(pl.Utf8) + '_' + pl.col('state') + '_' + pl.col('office_o'))
                         .alias('race_id'))
    assert df.select(pl.struct('race_id', 'fips').is_duplicated().any()).item() is False, 'duplicate race x fips'
    df = df.with_columns(pl.when(pl.col('fips') == SHANNON[0]).then(pl.lit('shannon_recode')).otherwise(pl.lit(''))
                         .alias('flags'))
    df = df.with_columns(pl.col('fips').replace({SHANNON[0]: SHANNON[1], BEDFORD[0]: BEDFORD[1]}))
    key = ['election_id', 'election_year', 'race_id', 'fips', 'state', 'office', 'office_o', 'special', 'seat_class',
           'seat_status', 'dem_nominee', 'rep_nominee']
    # Bedford city and county become one row; a sum over missing votes only stays missing
    summed = [pl.when(pl.col(c).is_not_null().any()).then(pl.col(c).sum()).otherwise(None).alias(c) for c in VOTES]
    df = df.group_by(key, maintain_order=True).agg(*summed, pl.col('county_name').first(), pl.col('flags').first())
    bedford = pl.col('fips') == BEDFORD[1]
    df = df.with_columns(pl.when(bedford).then(pl.lit('')).otherwise(pl.col('flags')).alias('flags'))
    df = df.with_columns(pl.when(bedford & pl.col('election_year').is_in(sorted(bedford_years)))
                         .then(pl.lit('bedford_city_merged')).otherwise(pl.col('flags')).alias('flags'),
                         pl.col('fips').is_in(sorted(universe)).alias('in_universe'))
    ct = pl.col('state') == 'CT'
    df = df.with_columns(pl.when(ct).then(pl.lit('ct_legacy_county_not_in_universe')).otherwise(pl.col('flags'))
                         .alias('flags'),
                         pl.when(ct).then(False).otherwise(pl.col('in_universe')).alias('in_universe'))
    jackson = (pl.col('fips') == JACKSON_MO[1]) & (pl.col('state') == JACKSON_MO[0])
    df = df.with_columns(pl.when(jackson).then(pl.lit('kc_board_votes_missing')).otherwise(pl.col('flags')).alias('flags'))
    missing = pl.any_horizontal([pl.col(c).is_null() for c in VOTES])
    bad = pl.col('raw_county_vote_totals') < pl.col('democratic_raw_votes') + pl.col('republican_raw_votes')
    df = df.with_columns(pl.when(missing).then((pl.col('flags') + ';missing_votes').str.strip_chars(';'))
                         .otherwise(pl.col('flags')).alias('flags'))
    return df.with_columns(pl.when(bad.fill_null(False)).then((pl.col('flags') + ';total_lt_two_party').str.strip_chars(';'))
                           .otherwise(pl.col('flags')).alias('flags'))


def race_table(df: pl.DataFrame) -> pl.DataFrame:
    """One row per race: state sums of the county rows and the source's nominees."""
    r = (df.group_by(RACE_KEY, maintain_order=True)
         .agg(pl.col('democratic_raw_votes').sum().alias('dem_votes'), pl.col('republican_raw_votes').sum().alias('rep_votes'),
              pl.col('raw_county_vote_totals').sum().alias('total_votes'), pl.len().alias('n_counties')))
    assert r['race_id'].is_unique().all()
    r = r.with_columns((~_no_nominee('dem_nominee') & (pl.col('dem_votes') > 0)).alias('has_dem_nominee'),
                       (~_no_nominee('rep_nominee') & (pl.col('rep_votes') > 0)).alias('has_rep_nominee'))
    return (r.with_columns((pl.col('has_dem_nominee') & pl.col('has_rep_nominee')).alias('one_d_one_r_source'))
            .rename({'election_year': 'year', 'state': 'state_po', 'office_o': 'office', 'special': 'special_flag'}))


def medsl_candidate_counts() -> pl.DataFrame:
    """Senate: D- and R-labelled candidates with at least 1% of the statewide vote on the MEDSL general ballot.

    The 1% floor drops write-in-like listings and Wellstone's name on the Minnesota 2002 absentee ballots."""
    record = medsl.raw_record('senate_statewide')
    m = pl.read_csv(io.BytesIO(rawstore.load_raw(record.path)), infer_schema_length=0)
    m = (m.with_columns(pl.col('year').cast(pl.Int32), pl.col('candidatevotes').cast(pl.Float64), pl.col('totalvotes').cast(pl.Float64))
         .filter((pl.col('stage') == 'gen') & (pl.col('office') == 'US SENATE') & pl.col('year').is_in(list(YEARS))
                 & (pl.col('writein') != 'True')
                 & ~pl.col('candidate').fill_null('').str.contains('BLANK|SCATTERING|OVERVOTE|UNDERVOTE'))
         .with_columns(pl.when(pl.col('special') == 'True').then(pl.lit('SEN_SPECIAL')).otherwise(pl.lit('SEN')).alias('office')))
    m = m.with_columns((pl.col('candidatevotes') / pl.col('totalvotes').max().over('year', 'state_po', 'office')).alias('share'))
    cands = (m.group_by('year', 'state_po', 'office', 'candidate', 'party_simplified').agg(pl.col('share').sum())
             .filter(pl.col('share') >= .01))
    return (cands.group_by('year', 'state_po', 'office')
            .agg(pl.col('candidate').filter(pl.col('party_simplified') == 'DEMOCRAT').n_unique().alias('medsl_n_dem_cands'),
                 pl.col('candidate').filter(pl.col('party_simplified') == 'REPUBLICAN').n_unique().alias('medsl_n_rep_cands')))


def apply_race_rule(races: pl.DataFrame, counts: pl.DataFrame) -> pl.DataFrame:
    """Race rule and its notes. A race the MEDSL file does not list has no second-candidate evidence (governors)."""
    r = races.join(counts, on=['year', 'state_po', 'office'], how='left', validate='1:1')
    r = r.with_columns(pl.col('dem_nominee').is_in(list(INDEPENDENT_CODED_AS_DEM)).alias('ind_coded_as_dem'),
                       ((pl.col('medsl_n_dem_cands') > 1) | (pl.col('medsl_n_rep_cands') > 1)).fill_null(False)
                       .alias('jungle_or_multi_nominee'),
                       ((pl.col('dem_votes') + pl.col('rep_votes')) / pl.col('total_votes')).alias('two_party_share_of_total'))
    r = r.with_columns((pl.col('one_d_one_r_source') & ~pl.col('jungle_or_multi_nominee') & ~pl.col('ind_coded_as_dem'))
                       .alias('one_d_one_r'))
    notes = [(pl.col('race_id') == '2010_MA_SEN_SPECIAL', 'special_election_held_2010-01-19_not_november'),
             (pl.col('ind_coded_as_dem'), 'independent_coded_as_democrat_by_source'),
             (pl.col('jungle_or_multi_nominee'), 'multiple_D_or_R_candidates_on_medsl_general_ballot'),
             ((pl.col('state_po') == 'LA') & (pl.col('office') == 'SEN') & pl.col('year').is_in([2002, 2014]),
              'source_records_december_runoff_not_november_ballot'),
             (pl.col('state_po') == 'CT', 'ct_legacy_counties_not_in_universe'),
             (pl.col('state_po') == 'MO', 'jackson_county_excludes_kansas_city_board_votes'),
             (pl.col('race_id') == '2006_WV_SEN', 'medsl_statewide_for_this_race_is_wrong_source_matches_official'),
             (pl.col('two_party_share_of_total') < .85, 'strong_third_party_or_independent_two_party_share_lt_85pct')]
    text = pl.lit('')
    for cond, label in notes:
        text = pl.when(cond.fill_null(False)).then(pl.concat_str(text, pl.lit(';' + label))).otherwise(text)
    return r.with_columns(text.str.strip_chars(';').alias('notes'))


def build(tables: dict[str, pl.DataFrame] | None = None, counts: pl.DataFrame | None = None,
          universe: set[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(county x race rows, race table). Inputs default to the raw store and the project universe."""
    if tables is None:
        tables = read_rdata(raw_records())
    if counts is None:
        counts = medsl_candidate_counts()
    if universe is None:
        universe = set(pl.read_parquet(paths.processed_dir() / 'exposure_county.parquet')['geo_id'].to_list())
    df = county_rows(tables, universe)
    races = apply_race_rule(race_table(df), counts)
    rows = (df.drop('office').rename({'election_year': 'year', 'state': 'state_po', 'fips': 'county_fips', 'office_o': 'office',
                       'democratic_raw_votes': 'dem_votes', 'republican_raw_votes': 'rep_votes',
                       'raw_county_vote_totals': 'total_votes'})
            .join(races.select('race_id', 'one_d_one_r'), on='race_id', how='left', validate='m:1'))
    rows = rows.with_columns(pl.col('county_fips').alias('geo_id'), pl.col('county_fips').str.slice(0, 2).alias('state_fips'))
    jackson = (pl.col('state_po') == JACKSON_MO[0]) & (pl.col('county_fips') == JACKSON_MO[1])
    reasons = [(~pl.col('one_d_one_r'), 'race_rule'),
               (pl.col('race_id').is_in(list(EXCLUDED_RACES)), 'special_not_november'),
               (pl.col('state_po') == 'CT', 'ct_legacy_county'),
               (pl.col('state_po') == 'AK', 'alaska'),
               (jackson, 'jackson_mo_kc_votes_missing'),
               (~pl.col('in_universe'), 'not_in_universe'),
               (pl.col('dem_votes').is_null() | pl.col('rep_votes').is_null(), 'missing_votes')]
    exclusion = pl.lit('')
    for cond, label in reversed(reasons):                     # the first reason in the list wins
        exclusion = pl.when(cond).then(pl.lit(label)).otherwise(exclusion)
    rows = rows.with_columns(exclusion.alias('exclusion'))
    rows = rows.with_columns((pl.col('exclusion') == '').alias('usable'))
    rows = rows.select(OUT_COLUMNS).sort('year', 'state_po', 'office', 'race_id', 'county_fips')
    return rows, races.sort('race_id')


# ---------------------------------------------------------------------------------------------------- access

def load(usable_only: bool = True) -> pl.DataFrame:
    """County x race rows from the processed file; `usable` rows satisfy the race rule and every exclusion."""
    rows = pl.read_parquet(paths.processed_dir() / 'results_hist.parquet')
    return rows.filter(pl.col('usable')) if usable_only else rows


def two_party(rows: pl.DataFrame) -> pl.DataFrame:
    """Republican share of the two-party vote, in points, for rows with a positive two-party total."""
    return (rows.filter(pl.col('rep_votes') + pl.col('dem_votes') > 0)
            .with_columns((100 * pl.col('rep_votes') / (pl.col('rep_votes') + pl.col('dem_votes'))).alias('r2')))


# ---------------------------------------------------------------------------------------------------- CLI

def write_outputs(rows: pl.DataFrame, races: pl.DataFrame, records: dict) -> Path:
    out = paths.processed_dir() / 'results_hist.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(out, compression='zstd')
    snapshot = ';'.join(f'{name}:{record.sha256[:16]}' for name, record in records.items())
    snapshot += ';senate_statewide:' + medsl.raw_record('senate_statewide').sha256[:16]
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id='results_hist',
        data_snapshot_ids=snapshot, output_path='data/processed/results_hist.parquet',
        notes=(f'parser={PARSER_VERSION}; rows={rows.height}; races={races.height}; '
               f'races_passing_rule={int(races["one_d_one_r"].sum())}; usable_rows={int(rows["usable"].sum())}'))
    return out


def main() -> None:
    records = raw_records()
    rows, races = build(read_rdata(records))
    out = write_outputs(rows, races, records)
    print(f'{out.relative_to(paths.repo_root())}: {rows.height} county rows, {races.height} races, '
          f'{int(races["one_d_one_r"].sum())} pass the race rule, {int(rows["usable"].sum())} usable rows')
    cover = (races.group_by('year', 'office').agg(pl.len().alias('races'), pl.col('one_d_one_r').sum().alias('pass'))
             .sort('year', 'office'))
    print(cover)


if __name__ == '__main__':
    main()
