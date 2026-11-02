"""County-level election results on 2024-vintage county FIPS (WS5; sources S11, S12, S19).

Builds `data/processed/results_county.parquet` (long by candidate): the 2020 presidential baseline and the 2022 Senate
and governor races, plus the D3 race table, the reconciliation table and the unmatched-units list.

    python -m fb26.etl.results            # build all outputs from the raw store
    python -m fb26.etl.results build --no-reports   # the same, leaving the reports/validation tables untouched
    python -m fb26.etl.results diagnose   # 2020 county-file vs precinct-file cross-check, candidate-name audit

Rules (all in config/geography.yaml unless noted):
* Modes: within a precinct and candidate line, a TOTAL row wins; without one the mode rows are summed.
* Subtotal rows named like "COUNTY TOTAL" are dropped only when the other precincts of that county sum to them.
* Fusion lines: a candidate's votes are summed over party lines; the candidate's party is the major party (D or R)
  among the lines' party_detailed labels. Write-in rows are never nominees.
* Ballot-category "candidates" (blank, void, over/undervotes, ...) are tallied separately, never added to a candidate.
* Votes recorded as "*" (masked) are counted, not imputed; the county carries `masked_rows=n` in `flags`.
* Nothing is dropped silently: units outside the ACS 2024 county list are reported in results_unmatched.csv.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml

from fb26.etl import results_overrides
from fb26.scrape import gov_reference, medsl
from fb26.util import paths, rawstore, runlog

PARSER_VERSION = 'results-0.1.0'
BASE_YEAR, YEAR = 2020, 2022
PRES_YEARS = (2000, 2004, 2008, 2012, 2016, 2020, 2024)
# MEDSL's county file has Connecticut by legacy county in every year, and no town-level source is in the raw store for
# these years, so Connecticut has no rows (as in the Amlani and Algara Senate and governor file).
CT_LEGACY_YEARS = (2000, 2004, 2008)
OFFICE_CODE = {'US SENATE': 'SENATE', 'GOVERNOR': 'GOVERNOR', 'US PRESIDENT': 'PRESIDENT',
               'GOVERNOR/LIEUTENANT GOVERNOR': 'GOVERNOR'}  # the Iowa 2018 by-state file lists the ticket
OFFICE_TAG = {'SENATE': 'SEN', 'GOVERNOR': 'GOV', 'PRESIDENT': 'PRES'}
RAW_COLS = ['precinct', 'office', 'party_detailed', 'party_simplified', 'mode', 'votes', 'county_name', 'county_fips',
            'jurisdiction_name', 'jurisdiction_fips', 'candidate', 'district', 'stage', 'special', 'writein',
            'state_po', 'state_fips']
# Governor sources by state: the combined 2022 state-offices file, else the by-state release. TN has no governor race
# in either MEDSL 2022 release (checked in build()).
GOV_BYSTATE = {'AR': 'bystate_ar', 'AZ': 'bystate_az', 'MD': 'bystate_md', 'MI': 'bystate_mi'}
OUT_SCHEMA = {
    'year': pl.Int32, 'state_fips': pl.Utf8, 'state_po': pl.Utf8, 'geo_id': pl.Utf8, 'county_name': pl.Utf8,
    'office': pl.Utf8, 'race_id': pl.Utf8, 'special': pl.Boolean, 'included_d3': pl.Boolean, 'candidate': pl.Utf8,
    'party': pl.Utf8, 'party_labels': pl.Utf8, 'mode': pl.Utf8, 'votes': pl.Int64, 'source': pl.Utf8,
    'retrieved_at_utc': pl.Utf8, 'raw_sha256': pl.Utf8, 'certified': pl.Boolean, 'acs_match': pl.Boolean,
    'flags': pl.Utf8, 'parser_version': pl.Utf8}


def config() -> dict:
    return yaml.safe_load((paths.config_dir() / 'geography.yaml').read_text())


# ---------------------------------------------------------------------------------------------------- geography

def acs_counties(record: rawstore.RawRecord | None = None) -> pl.DataFrame:
    """ACS 2024 5-year county list: geo_id, state_fips, county_name."""
    record = record or medsl.raw_record('acs2024_counties')
    return acs_frame(json.loads(rawstore.load_raw(record.path)))


def acs_frame(rows: list) -> pl.DataFrame:
    return (pl.DataFrame(rows[1:], schema=rows[0], orient='row')
            .select((pl.col('state') + pl.col('county')).alias('geo_id'), pl.col('state').alias('state_fips'),
                    pl.col('NAME').alias('county_name')).sort('geo_id'))


def ct_town_regions(content: bytes) -> pl.DataFrame:
    """Town code (5 digits) -> planning-region county FIPS, from the Census 2022 CT cousub-tract relationship file."""
    df = pl.read_csv(io.BytesIO(content), separator='|', infer_schema_length=0)
    return (df.select(pl.col('GEOID_COUSUB_22').str.slice(5, 5).alias('town_code'),
                      pl.col('GEOID_COUSUB_22').str.slice(0, 5).alias('geo_id'))
            .filter(pl.col('town_code') != '00000').unique().sort('town_code'))


def _norm_fips(expr: pl.Expr) -> pl.Expr:
    """Raw FIPS-like code as digits only; blanks and non-numeric placeholders (e.g. 'NA') become null."""
    s = expr.cast(pl.Utf8).str.strip_chars().str.replace(r'\.0+$', '')
    return pl.when(s.is_null() | ~s.str.contains(r'^\d+$')).then(None).otherwise(s)


def assign_geo(df: pl.DataFrame, cfg: dict, ct_map: pl.DataFrame, acs_ids: set[str], year: int | None = None
               ) -> pl.DataFrame:
    """Adds geo_id (2024 FIPS or None), unit_kind and geo_flag. `df` needs state_po, county_fips (raw), county_name,
    jurisdiction_fips. unit_kind: county | ct_region | recoded | alaska | statewide | unassigned."""
    ak = cfg['alaska']['flag']
    df = df.with_columns(
        _norm_fips(pl.col('county_fips')).alias('_raw'),
        pl.col('county_name').fill_null('').str.to_uppercase().alias('_name'))
    # normalised raw code: county codes under 5 digits are zero-padded; MEDSL place codes (7 digits) are kept as is
    df = df.with_columns(pl.when(pl.col('_raw').str.len_chars() < 5).then(pl.col('_raw').str.zfill(5))
                         .otherwise(pl.col('_raw')).alias('_raw'))
    geo, kind, flag = pl.col('_raw'), pl.lit('county'), pl.lit('')
    for rule in cfg['unit_recodes']:
        if 'years' in rule and year not in rule['years']:
            continue
        hit = ((pl.col('state_po') == rule['state']) & pl.col('_raw').is_in(rule['source_fips'])
               & pl.col('_name').str.contains(rule['name']))
        geo = pl.when(hit).then(pl.lit(rule['to'])).otherwise(geo)
        kind = pl.when(hit).then(pl.lit('recoded')).otherwise(kind)
        flag = pl.when(hit).then(pl.lit(f"recode_{rule['source_fips'][0]}_to_{rule['to']}")).otherwise(flag)
    for rule in cfg['statewide_units']:
        hit = ((pl.col('state_po') == rule['state']) & (pl.col('_raw').is_in(rule['source_fips'])
               | pl.col('_raw').is_null()) & pl.col('_name').str.contains(rule['name_contains'].upper()))
        geo = pl.when(hit).then(None).otherwise(geo)
        kind = pl.when(hit).then(pl.lit('statewide')).otherwise(kind)
    df = df.with_columns(geo.alias('geo_id'), kind.alias('unit_kind'), flag.alias('geo_flag'))
    # Alaska: election districts; keep the district code as the unit id
    df = df.with_columns(
        pl.when(pl.col('state_po') == 'AK').then(pl.coalesce(_norm_fips(pl.col('jurisdiction_fips')), pl.col('geo_id')))
        .otherwise(pl.col('geo_id'))
        .alias('geo_id'),
        pl.when(pl.col('state_po') == 'AK').then(pl.lit('alaska')).otherwise(pl.col('unit_kind')).alias('unit_kind'),
        pl.when(pl.col('state_po') == 'AK').then(pl.lit(ak)).otherwise(pl.col('geo_flag')).alias('geo_flag'))
    # Connecticut: town -> planning region
    ct = df.filter(pl.col('state_po') == 'CT')
    if ct.height:
        town = pl.col('jurisdiction_fips').str.slice(-5)
        df = (df.with_columns(pl.when(pl.col('state_po') == 'CT').then(town).otherwise(None).alias('_town'))
              .join(ct_map.rename({'geo_id': '_region'}), left_on='_town', right_on='town_code', how='left')
              .with_columns(
                  pl.when(pl.col('state_po') == 'CT').then(pl.col('_region')).otherwise(pl.col('geo_id')).alias('geo_id'),
                  pl.when(pl.col('state_po') == 'CT').then(pl.lit('ct_region')).otherwise(pl.col('unit_kind'))
                  .alias('unit_kind'),
                  pl.when(pl.col('state_po') == 'CT').then(pl.lit('ct_towns_to_regions'))
                  .otherwise(pl.col('geo_flag')).alias('geo_flag'))
              .drop('_town', '_region'))
    df = df.with_columns(
        pl.when(pl.col('geo_id').is_null() & (pl.col('unit_kind') == 'county')).then(pl.lit('unassigned'))
        .otherwise(pl.col('unit_kind')).alias('unit_kind'))
    return (df.with_columns(pl.col('geo_id').is_in(sorted(acs_ids)).alias('acs_match'), pl.col('_raw').alias('src_fips'))
            .drop('_raw', '_name'))


# ---------------------------------------------------------------------------------------------------- reading

def state_fips() -> dict[str, str]:
    df = pl.read_csv(paths.config_dir() / 'states.csv', schema_overrides={'state_fips': pl.Utf8})
    return dict(zip(df['state_abbr'], df['state_fips']))


@contextlib.contextmanager
def _tempfile(content: bytes, suffix: str):
    with tempfile.TemporaryDirectory(dir=paths.interim_dir()) as tmp:
        path = Path(tmp) / f'src{suffix}'
        path.write_bytes(content)
        yield path


def _raw_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    cols = lf.collect_schema().names()
    return lf.select([pl.col(c) if c in cols else pl.lit(None, dtype=pl.Utf8).alias(c) for c in RAW_COLS])


def read_source(key: str, *, office: str | None = None, stage: str | None = 'GEN', state: str | None = None
                ) -> tuple[pl.DataFrame, rawstore.RawRecord]:
    """Precinct rows of one raw file (strings), filtered by office and stage. Large files go through a temp file."""
    record = medsl.raw_record(key)
    content = rawstore.load_raw(record.path)
    if content[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            content = z.read(z.namelist()[0])
    paths.interim_dir().mkdir(parents=True, exist_ok=True)
    with _tempfile(content, '.csv') as path:
        del content
        lf = _raw_cols(pl.scan_csv(path, infer_schema_length=0, encoding='utf8-lossy'))
        if office:
            lf = lf.filter(pl.col('office') == office)
        if stage:
            lf = lf.filter(pl.col('stage').fill_null('GEN') == stage)
        if state:
            lf = lf.filter(pl.col('state_po') == state)
        return lf.collect(), record


def read_county_file(key: str, *, office: str, year: int | None = None) -> tuple[pl.DataFrame, rawstore.RawRecord]:
    """MEDSL county-level files (2020 president; 2022 Senate) in the precinct-file layout: one 'precinct' per county."""
    record = medsl.raw_record(key)
    df = pl.read_csv(io.BytesIO(rawstore.load_raw(record.path)), infer_schema_length=0)
    if year is not None:
        df = df.filter(pl.col('year') == str(year))
    df = df.filter(pl.col('office') == office)
    fips = state_fips()
    party = pl.col('party') if 'party' in df.columns else pl.col('party_detailed')
    return (df.with_columns(
        pl.col('candidatevotes').alias('votes'), party.alias('party_detailed'),
        (pl.col('party_simplified') if 'party_simplified' in df.columns else party).alias('party_simplified'),
        _norm_fips(pl.col('county_fips')).alias('precinct'), pl.lit(None, dtype=pl.Utf8).alias('jurisdiction_fips'),
        pl.lit(None, dtype=pl.Utf8).alias('jurisdiction_name'), pl.lit(None, dtype=pl.Utf8).alias('district'),
        pl.col('state_po').replace_strict(fips, default=None).alias('state_fips'),
        (pl.col('special') if 'special' in df.columns else pl.lit('FALSE')).alias('special'),
        (pl.col('stage') if 'stage' in df.columns else pl.lit('GEN')).alias('stage'),
        (pl.col('writein') if 'writein' in df.columns else pl.lit('FALSE')).alias('writein'))
        .pipe(lambda d: _raw_cols(d.lazy()).collect())), record


def party_class(detailed: pl.Expr, simplified: pl.Expr) -> pl.Expr:
    """Row-level major-party label: D, R, DR (dual label such as REP/DEM) or other. party_detailed decides;
    party_simplified only fills gaps. Fusion labels such as DEM/PROG (Vermont) count as Democratic."""
    d, s = detailed.fill_null('').str.to_uppercase(), simplified.fill_null('').str.to_uppercase()
    # fusion labels such as 'WORKING FAMILIES / DEMOCRAT' (NY 2024) and 'DFL' (MN 2024) count as the major party
    is_d = d.str.contains(r'DEMOCRAT|^DFL$|^DEM/|^DEM$|^PROG/DEM|/DEM$')
    is_r = d.str.contains(r'REPUBLICAN|^REP/|^REP$|/REP$')
    return (pl.when(is_d & is_r).then(pl.lit('DR'))
            .when(is_d).then(pl.lit('D')).when(is_r).then(pl.lit('R'))
            .when((d == '') & (s == 'DEMOCRAT')).then(pl.lit('D'))
            .when((d == '') & (s == 'REPUBLICAN')).then(pl.lit('R'))
            .otherwise(pl.lit('other')))


def norm_name(expr: pl.Expr) -> pl.Expr:
    return (expr.fill_null('').str.to_uppercase().str.replace_all(r'[.,"\']', '')
            .str.replace_all(r'\s+', ' ').str.strip_chars())


def cand_key(expr: pl.Expr) -> pl.Expr:
    """Order-insensitive candidate identity ('SCHUMER, CHARLES E.' == 'CHARLES E SCHUMER')."""
    return norm_name(expr).str.split(' ').list.sort().list.join(' ')


SUFFIXES = {'JR', 'SR', 'II', 'III', 'IV'}


def display_name(raw: str | None) -> str:
    """'LAST, FIRST M. JR' -> 'FIRST M JR LAST'-style reordering is avoided: suffixes stay last."""
    if not raw:
        return ''
    raw = re.sub(r'\s+', ' ', raw.upper().replace('.', '')).strip()
    if ',' not in raw:
        return raw
    last, first = [x.strip() for x in raw.split(',', 1)]
    toks = first.split()
    suffix = [t for t in toks if t in SUFFIXES]
    return ' '.join([t for t in toks if t not in SUFFIXES] + [last] + suffix)


# ---------------------------------------------------------------------------------------------------- aggregation

def prepare_precinct_rows(raw: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, dict]:
    """Collapse modes and drop verified subtotal rows. Returns rows at precinct x candidate x party-line level and a
    diagnostics dict (masked rows, dropped subtotal rows, kept-but-unverified subtotal rows)."""
    df = raw.with_columns(
        pl.col('office').replace(OFFICE_CODE).alias('office'),
        cand_key(pl.col('candidate')).alias('cand'),
        pl.col('votes').str.strip_chars().cast(pl.Float64, strict=False).cast(pl.Int64).alias('v'),
        (pl.col('votes').is_not_null() & pl.col('votes').str.strip_chars().cast(pl.Float64, strict=False).is_null())
        .alias('masked'),
        pl.col('special').fill_null('FALSE').str.to_uppercase().eq('TRUE').alias('special_b'),
        pl.col('writein').fill_null('FALSE').str.to_uppercase().eq('TRUE').alias('writein_b'),
        pl.col('mode').fill_null('').str.to_uppercase().is_in(cfg['total_modes']).alias('is_total'),
        _norm_fips(pl.col('county_fips')).alias('county_fips'),
        pl.col('state_po').replace_strict(state_fips(), default=None).alias('state_fips'))
    key = ['state_po', 'county_fips', 'county_name', 'jurisdiction_fips', 'precinct', 'office', 'district',
           'special_b', 'cand', 'writein_b', 'party_detailed']
    coll = (df.group_by(key, maintain_order=False)
            .agg(pl.col('candidate').first().alias('candidate'), pl.col('party_simplified').first(),
                 pl.col('state_fips').first(), pl.col('masked').sum().alias('masked_rows'),
                 pl.when(pl.col('is_total').any()).then(pl.col('v').filter(pl.col('is_total')).sum())
                 .otherwise(pl.col('v').sum()).alias('v'),
                 pl.col('is_total').any().alias('had_total'), pl.col('is_total').not_().any().alias('had_components'))
            .sort(key, nulls_last=True))
    pat = cfg['subtotal_precinct_pattern']
    sub = coll.with_columns(pl.col('precinct').fill_null('').str.to_uppercase().str.strip_chars().str.contains(pat)
                            .alias('is_sub'))
    gkey = ['state_po', 'county_fips', 'office', 'special_b', 'cand', 'party_detailed']
    sums = sub.group_by(gkey).agg(pl.col('v').filter(pl.col('is_sub')).sum().alias('sub_v'),
                                  pl.col('v').filter(~pl.col('is_sub')).sum().alias('rest_v'),
                                  pl.col('is_sub').sum().alias('n_sub'))
    tol = cfg['subtotal_tolerance']
    verdict = (sums.filter(pl.col('n_sub') > 0)
               .with_columns(((pl.col('rest_v') > 0) & ((pl.col('sub_v') - pl.col('rest_v')).abs()
                                                         <= (pl.col('sub_v') * tol).clip(lower_bound=1))).alias('verified')))
    sub = sub.join(verdict.select(gkey + ['verified']), on=gkey, how='left')
    ok = pl.col('verified').fill_null(False)
    drop, unverified = sub.filter(pl.col('is_sub') & ok), sub.filter(pl.col('is_sub') & ~ok)
    out = sub.filter(~(pl.col('is_sub') & ok)).drop('is_sub', 'verified')
    diag = {'masked_rows': int(coll['masked_rows'].sum()), 'dropped_subtotal_rows': drop.height,
            'dropped_subtotal_votes': int(drop['v'].sum() or 0), 'kept_unverified_subtotal_rows': unverified.height,
            'rows_with_total_and_components': int((coll['had_total'] & coll['had_components']).sum())}
    return out, diag


def apply_party_overrides(rows: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Party for candidates the source leaves unlabelled (config `party_overrides`; each entry is documented there)."""
    ov = cfg.get('party_overrides') or []
    rows = rows.with_columns(pl.lit(False).alias('party_override'))
    for o in ov:
        hit = ((pl.col('state_po') == o['state']) & (pl.col('office') == o['office'])
               & (cand_key(pl.col('candidate')) == cand_key(pl.lit(o['candidate'])))
               & (pl.col('party_detailed').fill_null('') == o.get('from_label', ''))
               & ((pl.col('line_class') == 'other') | pl.lit('from_label' in o)))
        rows = rows.with_columns(pl.when(hit).then(pl.lit(o['party'])).otherwise(pl.col('line_class')).alias('line_class'),
                                 (pl.col('party_override') | hit).alias('party_override'))
    return rows


def aggregate_counties(rows: pl.DataFrame, cfg: dict, ct_map: pl.DataFrame, acs_ids: set[str],
                       writein_all: bool = False, year: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Precinct rows -> (county x candidate votes, non-candidate tallies, assigned-unit table). Votes sum over
    precincts, modes, party lines (fusion) and merged units."""
    rows = assign_geo(rows, cfg, ct_map, acs_ids, year)
    rows = rows.with_columns(norm_name(pl.col('candidate')).str.contains(cfg['non_candidate_pattern']).alias('non_candidate'),
                             party_class(pl.col('party_detailed'), pl.col('party_simplified')).alias('line_class'))
    rows = apply_party_overrides(rows, cfg)
    nc = (rows.filter(pl.col('non_candidate'))
          .group_by('state_po', 'office', 'special_b').agg(pl.col('v').sum().alias('non_candidate_votes')))
    cand = rows.filter(~pl.col('non_candidate'))
    cand = cand.with_columns(pl.when(pl.col('geo_id').is_null())
                             .then(pl.format('UNASSIGNED|{}|{}', pl.col('county_name').fill_null(''),
                                             pl.col('src_fips').fill_null('')))
                             .otherwise(pl.col('geo_id')).alias('unit_id'))
    out = (cand.group_by('state_po', 'state_fips', 'office', 'special_b', 'unit_id', 'cand')
           .agg(pl.col('geo_id').first(), pl.col('candidate').sort_by(['v', 'candidate'], descending=[True, False]).first().alias('candidate'),
                pl.col('v').sum().alias('votes'), pl.col('masked_rows').sum().alias('masked_rows'),
                pl.col('unit_kind').unique().sort().str.join('|').alias('unit_kind'),
                pl.col('acs_match').all().alias('acs_match'),
                pl.col('party_detailed').drop_nulls().unique().sort().str.join('|').alias('party_labels'),
                pl.col('line_class').unique().sort().str.join('|').alias('line_classes'),
                (pl.col('writein_b').all() if writein_all else pl.col('writein_b').any()).alias('writein'),
                pl.col('party_override').any().alias('party_override'),
                pl.col('geo_flag').filter(pl.col('geo_flag') != '').unique().sort().str.join('|').alias('geo_flag'),
                pl.col('county_name').first().alias('src_county_name'))
           .sort('state_po', 'office', 'special_b', 'unit_id', 'cand', nulls_last=True))
    return out, nc, out.filter(pl.col('geo_id').is_null())


def canonical_names(county: pl.DataFrame) -> pl.DataFrame:
    """Most-voted display name per (race, candidate key)."""
    return (county.sort(['votes', 'candidate'], descending=[True, False])
            .group_by('state_po', 'office', 'special_b', 'cand').agg(pl.col('candidate').first().alias('canon')))


def fill_gaps(base: pl.DataFrame, fill: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add (race, county) cells that are missing or empty in `base` from the county-level `fill` table. Returns the
    combined table (new rows flagged) and the table of filled cells. Candidate names follow `base` where the
    candidate key matches. Only ACS-matched counties are filled."""
    cell = ['state_po', 'office', 'special_b', 'geo_id']
    have = (base.filter(pl.col('geo_id').is_not_null()).group_by(cell).agg(pl.col('votes').sum().alias('v'))
            .filter(pl.col('v') > 0).select(cell))
    add = fill.filter(pl.col('acs_match')).join(have, on=cell, how='anti')
    cells = add.group_by(cell).agg(pl.col('votes').sum().alias('votes'), pl.col('cand').n_unique().alias('n_cand'))
    cells = cells.filter(pl.col('votes') > 0)
    add = add.join(cells.select(cell), on=cell, how='semi')
    names = canonical_names(base)
    add = (add.join(names, on=['state_po', 'office', 'special_b', 'cand'], how='left')
           .with_columns(pl.coalesce('canon', pl.col('candidate').map_elements(display_name, return_dtype=pl.Utf8))
                         .alias('candidate'),
                         pl.lit('filled_from_medsl_county_level').alias('fill_flag')).drop('canon'))
    base = base.join(cells.select(cell), on=cell, how='anti')  # drop the empty placeholder rows of the same cells
    return pl.concat([base, add], how='diagonal_relaxed'), cells


# ---------------------------------------------------------------------------------------------------- races

INDEP_RE = r'(?i)INDEPENDENT|NONPARTISAN|NON-PARTISAN|NO PARTY|UNAFFILIATED|NON-AFFILIATED|NO AFFILIATION'


def candidate_table(county: pl.DataFrame, writein_all: bool = False) -> pl.DataFrame:
    """Statewide candidate table per race: votes, labels and the candidate's party (D, R, DR, I, other)."""
    c = (county.group_by('state_po', 'office', 'special_b', 'cand')
         .agg(pl.col('candidate').sort_by(['votes', 'candidate'], descending=[True, False]).first(), pl.col('votes').sum(),
              (pl.col('writein').all() if writein_all else pl.col('writein').any()), pl.col('party_override').any(),
              pl.col('party_labels').str.split('|').explode().drop_nulls().unique().sort()
              .str.join('|').str.strip_chars('|').alias('labels'),
              pl.col('line_classes').str.split('|').explode().unique().sort().str.join('|').alias('classes')))
    has_d, has_r = pl.col('classes').str.contains('D'), pl.col('classes').str.contains('R')
    # A write-in row is never a nominee, even if the file gives it a major-party label.
    return c.with_columns(
        pl.when(pl.col('writein')).then(pl.lit('other'))
        .when(has_d & has_r).then(pl.lit('DR'))
        .when(has_d).then(pl.lit('D')).when(has_r).then(pl.lit('R'))
        .when(pl.col('labels').str.contains(INDEP_RE)).then(pl.lit('I'))
        .otherwise(pl.lit('other')).alias('party'))


def race_id(year: int, state_po: str, office: str, special: bool) -> str:
    return f'{year}_{state_po}_{OFFICE_TAG[office]}' + ('_SPECIAL' if special else '')


def race_id_expr(year: int) -> pl.Expr:
    tag = pl.col('office').replace(OFFICE_TAG)
    return pl.format('{}_{}_{}{}', pl.lit(year), pl.col('state_po'), tag,
                     pl.when(pl.col('special_b')).then(pl.lit('_SPECIAL')).otherwise(pl.lit('')))


MINOR_LABEL_SHARE = 0.005  # a second D/R-labelled candidate below this share of the race is a write-in-like listing


def classify_races(cands: pl.DataFrame, year: int, rcv_states: set[str]) -> pl.DataFrame:
    """D3 table: one row per statewide race. Included = exactly one Democratic and one Republican nominee.
    Nominee = the top candidate under each major label. Further candidates with a D/R label whose statewide share is
    below MINOR_LABEL_SHARE (certified write-in listings such as AZ and VT) are recorded, not counted; a second
    labelled candidate above it makes the race ineligible."""
    rows = []
    for (st, office, special), g in cands.group_by(['state_po', 'office', 'special_b'], maintain_order=True):
        total = int(g['votes'].sum())
        g = g.with_columns((pl.col('votes') / total).alias('share')) if total else g.with_columns(pl.lit(0.0).alias('share'))
        by = {p: g.filter(pl.col('party') == p).sort(['votes', 'candidate'], descending=[True, False]) for p in ['D', 'R', 'DR', 'I', 'other']}
        sig = lambda df: df.filter(pl.col('share') >= MINOR_LABEL_SHARE)  # noqa: E731
        minor = pl.concat([by['D'].tail(max(by['D'].height - 1, 0)), by['R'].tail(max(by['R'].height - 1, 0))])
        minor = minor.filter(pl.col('share') < MINOR_LABEL_SHARE)
        nd, nr = by['D'].height - minor.filter(pl.col('party') == 'D').height, by['R'].height - minor.filter(pl.col('party') == 'R').height
        ndr = sig(by['DR']).height
        others = pl.concat([by['I'], by['other'], by['DR']]).filter(~pl.col('writein')).sort(['votes', 'candidate'], descending=[True, False])
        top = others.head(1)
        top_share = float(top['votes'][0]) / total if total and top.height else 0.0
        edge, code, reason = [], '', ''
        if ndr:
            code, reason = 'dual_major_label', f'{ndr} candidate(s) carry both a D and an R party line'
        elif by['D'].height == 0:
            code, reason = 'no_democrat', 'no Democratic nominee on the ballot'
        elif by['R'].height == 0:
            code, reason = 'no_republican', 'no Republican nominee on the ballot'
        elif nd > 1 or nr > 1:
            code, reason = 'multiple_nominees', f'{nd} Democratic and {nr} Republican candidates above {MINOR_LABEL_SHARE:.1%}'
        if top_share >= 0.05:
            edge.append(f'strong_third_candidate:{top["candidate"][0]}={top_share:.1%}')
        if g.filter(pl.col('party').is_in(['D', 'R']) & pl.col('labels').str.contains(r'\|')).height:
            edge.append('fusion_lines')
        if minor.height:
            edge.append(f'minor_label_candidates_ignored={minor.height}')
        if g.filter(pl.col('party_override')).height:
            edge.append('party_from_config_override')
        if g.filter(pl.col('writein') & pl.col('classes').str.contains('[DR]')).height:
            edge.append('writein_with_major_label')
        if st in rcv_states or (st, office) in rcv_states:
            edge.append('ranked_choice_state_first_round_only')
        nom = lambda p: by[p].head(1) if code == '' else by[p]  # noqa: E731
        rows.append({
            'race_id': f'{year}_{st}_{OFFICE_TAG[office]}' + ('_SPECIAL' if special else ''), 'year': year,
            'state_po': st, 'state_fips': state_fips()[st], 'office': office, 'special': special,
            'included_d3': code == '', 'exclusion_code': code, 'reason': reason, 'n_d': nd, 'n_r': nr,
            'd_candidate': '|'.join(nom('D')['candidate'].to_list()), 'd_labels': ' ; '.join(nom('D')['labels'].to_list()),
            'r_candidate': '|'.join(nom('R')['candidate'].to_list()), 'r_labels': ' ; '.join(nom('R')['labels'].to_list()),
            'other_candidates': '; '.join(f"{r['candidate']} [{r['labels'] or 'no label'}] {r['votes']}"
                                          for r in others.head(6).iter_rows(named=True)),
            'n_other_candidates': others.height, 'top_other_share': round(top_share, 4),
            'statewide_candidate_votes': total, 'edge_flags': '|'.join(edge)})
    return pl.DataFrame(rows).sort('state_po', 'office', 'special')


# ---------------------------------------------------------------------------------------------------- references

def reference_totals(key: str, year: int, office: str) -> pl.DataFrame:
    """MEDSL statewide reference per race: D and R candidate votes (a candidate is D/R if any of its lines is) and
    the file's own total."""
    record = medsl.raw_record(key)
    df = pl.read_csv(io.BytesIO(rawstore.load_raw(record.path)), infer_schema_length=0)
    return (reference_from_frame(df, year, office)
            .with_columns(pl.lit(record.path.name).alias('ref_file'), pl.lit(record.sha256).alias('ref_sha256'),
                          pl.lit('medsl_statewide_file').alias('reference_source')))


def reference_from_frame(df: pl.DataFrame, year: int, office: str) -> pl.DataFrame:
    df = df.filter((pl.col('year') == str(year)) & (pl.col('office') == office))
    if 'stage' in df.columns:
        df = df.filter(pl.col('stage').fill_null('GEN').str.to_uppercase() == 'GEN')
    spec = pl.col('special').fill_null('False').str.to_uppercase().eq('TRUE') if 'special' in df.columns else pl.lit(False)
    wi = pl.col('writein').fill_null('False').str.to_uppercase().eq('TRUE')
    # Write-in-flagged candidates with a D/R label (e.g. Sanders in AZ 2016) are not nominees, but the top D and the top R
    # candidate always count: some reference rows flag real nominees as write-ins (DC 2020).
    df = df.with_columns(spec.alias('special_b'))
    df = df.with_columns(pl.col('candidatevotes').cast(pl.Float64).cast(pl.Int64).alias('v'),
                         party_class(pl.col('party_detailed'), pl.col('party_simplified')).alias('cls'),
                         wi.alias('wi'), cand_key(pl.col('candidate')).alias('cand'))
    per = (df.group_by('state_po', 'special_b', 'cand')
           .agg(pl.col('v').sum(), (pl.col('cls') == 'D').any().alias('is_d'), (pl.col('cls') == 'R').any().alias('is_r'),
                pl.col('wi').all().alias('wi'), pl.col('totalvotes').first().cast(pl.Float64).cast(pl.Int64).alias('tv')))
    return (per.group_by('state_po', 'special_b')
            .agg(pl.col('v').filter(pl.col('is_d') & (~pl.col('wi') | (pl.col('v') == pl.col('v').filter(pl.col('is_d')).max()))).sum().alias('ref_d'),
                 pl.col('v').filter(pl.col('is_r') & (~pl.col('wi') | (pl.col('v') == pl.col('v').filter(pl.col('is_r')).max()))).sum().alias('ref_r'),
                 pl.col('v').sum().alias('ref_candidate_votes'), pl.col('tv').first().alias('ref_totalvotes')))


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if b == 0:
        return 0.0 if a == 0 else float('inf')
    return (a - b) / b * 100


def reconcile(cands: pl.DataFrame, county: pl.DataFrame, nc: pl.DataFrame, refs: pl.DataFrame, year: int,
              tol: float, no_ref_reason: str, alt_refs: pl.DataFrame | None = None) -> pl.DataFrame:
    """County sums (assigned units) against the statewide reference, race by race. Pass needs both the D and the R
    sum within `tol`; the all-candidate total is reported separately (`total_ok`), because some MEDSL reference rows
    list only the two major candidates."""
    if 'fill_flag' not in county.columns:
        county = county.with_columns(pl.lit('').alias('fill_flag'))
    assigned = county.filter(pl.col('geo_id').is_not_null())
    mine = (cands.group_by('state_po', 'office', 'special_b')
            .agg(pl.col('votes').filter(pl.col('party') == 'D').sum().alias('state_d'),
                 pl.col('votes').filter(pl.col('party') == 'R').sum().alias('state_r'),
                 pl.col('votes').sum().alias('state_candidate_votes')))
    cty = (assigned.join(cands.select('state_po', 'office', 'special_b', 'cand', 'party'),
                         on=['state_po', 'office', 'special_b', 'cand'])
           .group_by('state_po', 'office', 'special_b')
           .agg(pl.col('votes').filter(pl.col('party') == 'D').sum().alias('county_d'),
                pl.col('votes').filter(pl.col('party') == 'R').sum().alias('county_r'),
                pl.col('votes').sum().alias('county_candidate_votes'),
                pl.col('geo_id').n_unique().alias('n_units'),
                pl.col('masked_rows').sum().alias('masked_rows'),
                (pl.col('fill_flag') != '').any().alias('has_fill')))
    unassigned = (county.filter(pl.col('geo_id').is_null())
                  .join(cands.select('state_po', 'office', 'special_b', 'cand', 'party'),
                        on=['state_po', 'office', 'special_b', 'cand'])
                  .group_by('state_po', 'office', 'special_b')
                  .agg(pl.col('votes').sum().alias('unassigned_votes'),
                       pl.col('votes').filter(pl.col('party') == 'D').sum().alias('unassigned_d'),
                       pl.col('votes').filter(pl.col('party') == 'R').sum().alias('unassigned_r')))
    out = (mine.join(cty, on=['state_po', 'office', 'special_b'], how='left')
           .join(unassigned, on=['state_po', 'office', 'special_b'], how='left')
           .join(nc, on=['state_po', 'office', 'special_b'], how='left')
           .join(refs, left_on=['state_po', 'special_b'], right_on=['state_po', 'special_b'], how='left')
           .with_columns(pl.col('unassigned_votes').fill_null(0), pl.col('unassigned_d').fill_null(0),
                         pl.col('unassigned_r').fill_null(0), pl.col('non_candidate_votes').fill_null(0),
                         race_id_expr(year).alias('race_id')))
    alt = {(r['state_po'], r['special_b']): r for r in alt_refs.iter_rows(named=True)} if alt_refs is not None else {}
    rows = []
    for r in out.sort('race_id').iter_rows(named=True):
        have_ref = r.get('ref_d') is not None and (r['ref_d'] or 0) + (r['ref_r'] or 0) > 0
        d_pct = _pct(r['county_d'], r['ref_d']) if have_ref else None
        r_pct = _pct(r['county_r'], r['ref_r']) if have_ref else None
        t_pct = _pct(r['county_candidate_votes'], r['ref_candidate_votes']) if have_ref else None
        thr = tol * 100
        if not have_ref:
            status = 'unreconciled'
            reason = no_ref_reason if r.get('ref_d') is None else 'reference row has no party labels'
        else:
            reason = ''
            if abs(d_pct) <= thr and abs(r_pct) <= thr:
                status = 'pass'
                if abs(t_pct) > thr:
                    reason = 'D and R pass; all-candidate total differs (reference may omit minor candidates)'
            else:
                d_all, r_all = _pct(r['county_d'] + r['unassigned_d'], r['ref_d']), _pct(r['county_r'] + r['unassigned_r'], r['ref_r'])
                if abs(d_all) <= thr and abs(r_all) <= thr:
                    status = 'pass_with_unassigned_units'
                    reason = (f"county sums miss the tolerance by {max(abs(d_pct), abs(r_pct)):.2f}%; counting the "
                              f"statewide units that cannot be assigned to a county ({r['unassigned_d']} D, "
                              f"{r['unassigned_r']} R) closes the gap")
                else:
                    status = 'fail'
        a = alt.get((r['state_po'], r['special_b']))
        alt_d = _pct(r['county_d'], a['ref_d']) if a else None
        alt_r = _pct(r['county_r'], a['ref_r']) if a else None
        if a and status in ('fail', 'unreconciled') and abs(alt_d) <= tol * 100 and abs(alt_r) <= tol * 100:
            reason = (f"{'primary reference differs from the county sums' if status == 'fail' else 'no primary reference'}"
                      f"; the alternative reference ({a['reference_source']}) agrees within tolerance")
            status = 'pass_alt_reference'
        rows.append({**{k: r[k] for k in ('race_id', 'state_po', 'office', 'special_b')},
                     'alt_reference_source': a['reference_source'] if a else None,
                     'alt_ref_d': a['ref_d'] if a else None, 'alt_ref_r': a['ref_r'] if a else None,
                     'alt_d_diff_pct': alt_d, 'alt_r_diff_pct': alt_r,
                     'year': year, 'status': status, 'reason': reason,
                     'ref_file': r.get('ref_file'), 'ref_sha256': r.get('ref_sha256'),
                     'reference_source': r.get('reference_source'), 'reference_reason': r.get('reference_reason'),
                     'ref_d': r.get('ref_d'), 'county_d': r['county_d'], 'd_diff_pct': d_pct,
                     'ref_r': r.get('ref_r'), 'county_r': r['county_r'], 'r_diff_pct': r_pct,
                     'ref_candidate_votes': r.get('ref_candidate_votes'),
                     'county_candidate_votes': r['county_candidate_votes'], 'total_diff_pct': t_pct,
                     'ref_totalvotes': r.get('ref_totalvotes'), 'unassigned_votes': r['unassigned_votes'],
                     'unassigned_d': r['unassigned_d'], 'unassigned_r': r['unassigned_r'],
                     'non_candidate_votes': r['non_candidate_votes'], 'masked_rows': r['masked_rows'],
                     'n_units': r['n_units'], 'county_level_fill': bool(r['has_fill'])})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------------------------------- build

SUSPECT_OTHER_SHARE = 0.25
RCV_STATES = {2022: {'AK'}}  # Alaska's first RCV general (2022); Maine's RCV covers federal races, none statewide in 2022
OUT_DIR_REPORTS = ('reports', 'validation')


def _provenance(df: pl.DataFrame, record: rawstore.RawRecord, source: str) -> pl.DataFrame:
    return df.with_columns(pl.lit(source).alias('source'),
                           pl.lit(rawstore.utc_iso(record.fetched_at_utc)).alias('retrieved_at_utc'),
                           pl.lit(record.sha256).alias('raw_sha256'), pl.lit('').alias('fill_flag'))


def _county_frame(raw: pl.DataFrame, record: rawstore.RawRecord, source: str, cfg, ct_map, acs_ids,
                  writein_all: bool = False, year: int | None = None):
    rows, diag = prepare_precinct_rows(raw, cfg)
    county, nc, _ = aggregate_counties(rows, cfg, ct_map, acs_ids, writein_all, year)
    return _provenance(county, record, source), nc, diag


def build_senate_2022(cfg, ct_map, acs_ids):
    raw, rec = read_source('senate_2022_precinct', office='US SENATE')
    base, nc, diag = _county_frame(raw, rec, 'medsl_precinct:senate_2022_precinct', cfg, ct_map, acs_ids)
    raw2, rec2 = read_county_file('senate_2022_county', office='US SENATE', year=2022)
    raw2 = raw2.filter(pl.col('stage').fill_null('GEN') == 'GEN')
    cl, _, diag2 = _county_frame(raw2, rec2, 'medsl_county_level:senate_2022_county', cfg, ct_map, acs_ids)
    combined, cells = fill_gaps(base, cl)
    return combined, nc, cells, {'precinct': diag, 'county_level': diag2}, cl


def build_governor_2022(cfg, ct_map, acs_ids):
    frames, ncs, diags = [], [], {}
    raw, rec = read_source('state_2022_precinct', office='GOVERNOR')
    raw_states = set(raw['state_po'].unique().to_list())
    assert not (raw_states & set(GOV_BYSTATE)), 'combined state file unexpectedly has by-state governor states'
    c, nc, d = _county_frame(raw, rec, 'medsl_precinct:state_2022_precinct', cfg, ct_map, acs_ids)
    frames.append(c), ncs.append(nc)
    diags['state_2022_precinct'] = d
    for st, key in sorted(GOV_BYSTATE.items()):
        raw, rec = read_source(key, office='GOVERNOR')
        c, nc, d = _county_frame(raw, rec, f'medsl_precinct:{key}', cfg, ct_map, acs_ids)
        frames.append(c), ncs.append(nc)
        diags[key] = d
    return pl.concat(frames, how='diagonal_relaxed'), pl.concat(ncs), diags, raw_states


def ct_town_names(content: bytes, aliases: dict, squeeze: bool = False) -> pl.DataFrame:
    """Town name (lower case, without ' town'; `squeeze` also drops spaces) -> 5-digit town code, from the Census CT
    relationship file."""
    df = pl.read_csv(io.BytesIO(content), separator='|', infer_schema_length=0)
    name = pl.col('NAMELSAD_COUSUB_22').str.replace(r' town$', '').str.to_lowercase()
    names = (df.select((name.str.replace_all(' ', '') if squeeze else name).alias('name'),
                       pl.col('GEOID_COUSUB_22').str.slice(5, 5).alias('town_code'))
             .filter(pl.col('town_code') != '00000').unique())
    keys = [k.lower().replace(' ', '') if squeeze else k.lower() for k in aliases]
    extra = pl.DataFrame({'name': keys, 'town_code': list(aliases.values())},
                         schema={'name': pl.Utf8, 'town_code': pl.Utf8})
    return pl.concat([names, extra]).unique()


def read_ct_2016(ct_names: pl.DataFrame) -> tuple[pl.DataFrame, rawstore.RawRecord]:
    """Connecticut 2016 president from the old-schema 2016 precinct file. Towns are identified by name only; the
    town code goes into jurisdiction_fips so that assign_geo maps it to its planning region."""
    record = medsl.raw_record('pres_2016_precinct')
    content = rawstore.load_raw(record.path)
    with _tempfile(content, '.csv') as path:
        del content
        df = (pl.scan_csv(path, infer_schema_length=0, encoding='utf8-lossy')
              .filter((pl.col('state_postal') == 'CT') & (pl.col('office') == 'US President')
                      & (pl.col('stage').str.to_lowercase() == 'gen')).collect())
    return ct_2016_rows(df, ct_names), record


def ct_2016_rows(df: pl.DataFrame, ct_names: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col('jurisdiction').str.to_lowercase().str.strip_chars().alias('name')).join(
        ct_names, on='name', how='left')
    assert df['town_code'].null_count() == 0, df.filter(pl.col('town_code').is_null())['name'].unique().to_list()
    return (df.select(
        'precinct', pl.lit('US PRESIDENT').alias('office'), pl.col('party').str.to_uppercase().alias('party_detailed'),
        pl.col('party').str.to_uppercase().alias('party_simplified'), pl.col('mode').str.to_uppercase().alias('mode'),
        'votes', 'county_name', 'county_fips', pl.col('jurisdiction').alias('jurisdiction_name'),
        (pl.lit('09000') + pl.col('town_code')).alias('jurisdiction_fips'), 'candidate', 'district',
        pl.lit('GEN').alias('stage'), pl.lit('FALSE').alias('special'),
        pl.col('writein').str.to_uppercase().alias('writein'), pl.lit('CT').alias('state_po'),
        pl.lit('09').alias('state_fips')))


SENATE_2018_NO_STATE_RACE = {'DC'}  # the DC "US SENATE" rows are the shadow senator, not a state race
GOV_BYSTATE_2018 = {'IA': 'bystate_2018_ia'}


def build_senate_2018(cfg, ct_map, acs_ids):
    raw, rec = read_source('senate_2018_precinct', office='US SENATE')
    raw = raw.filter(~pl.col('state_po').is_in(sorted(SENATE_2018_NO_STATE_RACE)))
    county, nc, diag = _county_frame(raw, rec, 'medsl_precinct:senate_2018_precinct', cfg, ct_map, acs_ids, True)
    return county, nc, diag


def build_governor_2018(cfg, ct_map, acs_ids):
    frames, ncs, diags = [], [], {}
    raw, rec = read_source('state_2018_precinct', office='GOVERNOR')
    in_file = set(raw['state_po'].unique().to_list())
    assert not (in_file & set(GOV_BYSTATE_2018)), 'combined 2018 state file unexpectedly has a by-state state'
    c, nc, d = _county_frame(raw, rec, 'medsl_precinct:state_2018_precinct', cfg, ct_map, acs_ids, True)
    frames.append(c), ncs.append(nc)
    diags['state_2018_precinct'] = d
    for st, key in sorted(GOV_BYSTATE_2018.items()):
        raw, rec = read_source(key, office='GOVERNOR/LIEUTENANT GOVERNOR' if st == 'IA' else 'GOVERNOR')
        c, nc, d = _county_frame(raw, rec, f'medsl_precinct:{key}', cfg, ct_map, acs_ids, True)
        frames.append(c), ncs.append(nc)
        diags[key] = d
    return pl.concat(frames, how='diagonal_relaxed'), pl.concat(ncs), diags


def read_ct_2012(ct_names_squeezed: pl.DataFrame) -> tuple[pl.DataFrame, rawstore.RawRecord]:
    """Connecticut 2012 president, town level, from the OpenElections transcription (no MEDSL 2012 precinct file exists).
    OpenElections writes some town names without spaces ('Beaconfalls'), so names are matched with spaces removed."""
    record = gov_reference.latest_raw(gov_reference.SOURCE, gov_reference.CT_2012_OPENELECTIONS_URL)
    df = pl.read_csv(io.BytesIO(rawstore.load_raw(record.path)), infer_schema_length=0).filter(pl.col('office') == 'President')
    df = df.with_columns(pl.col('town').str.to_lowercase().str.replace_all(' ', '').alias('name')).join(
        ct_names_squeezed, on='name', how='left')
    assert df['town_code'].null_count() == 0, df.filter(pl.col('town_code').is_null())['name'].unique().to_list()
    party = pl.col('party').fill_null('')
    return (df.select(
        'precinct', pl.lit('US PRESIDENT').alias('office'), party.alias('party_detailed'),
        party.alias('party_simplified'), pl.lit('TOTAL').alias('mode'), 'votes',
        pl.lit(None, dtype=pl.Utf8).alias('county_name'), pl.lit(None, dtype=pl.Utf8).alias('county_fips'),
        pl.col('town').alias('jurisdiction_name'), (pl.lit('09000') + pl.col('town_code')).alias('jurisdiction_fips'),
        'candidate', pl.lit(None, dtype=pl.Utf8).alias('district'), pl.lit('GEN').alias('stage'),
        pl.lit('FALSE').alias('special'), pl.lit('FALSE').alias('writein'), pl.lit('CT').alias('state_po'),
        pl.lit('09').alias('state_fips')), record)


def build_president(year, cfg, ct_map, acs_ids, ct_names=None, ct_names_squeezed=None):
    """Presidential county results for `year` (2000-2024). Connecticut comes from town-level precinct data
    because the county file has it by legacy county in every year; for CT_LEGACY_YEARS it is left out."""
    raw, rec = read_county_file('pres_county_2000_2024', office='US PRESIDENT', year=year)
    n_ct_legacy = raw.filter(pl.col('state_po') == 'CT').height
    base, nc, diag = _county_frame(raw.filter(pl.col('state_po') != 'CT'), rec,
                                   'medsl_county_level:pres_county_2000_2024', cfg, ct_map, acs_ids, year=year)
    if year in CT_LEGACY_YEARS:
        return base, nc, {'county_file': diag, 'ct_legacy_county_rows_dropped': n_ct_legacy}
    if year == 2020:
        raw_p, rec_p = read_source('pres_2020_precinct', office='US PRESIDENT')
        pr, nc_p, diag_p = _county_frame(raw_p, rec_p, 'medsl_precinct:pres_2020_precinct', cfg, ct_map, acs_ids)
        ct = pr.filter(pl.col('state_po') == 'CT')
        # Statewide units that cannot be assigned to a county (Maine UOCAVA; Rhode Island's federal-only precinct). The
        # county file has the Rhode Island unit already; the precinct file's version is added only for states where the
        # county file has no such unit, so nothing is counted twice.
        have_units = set(base.filter(pl.col('geo_id').is_null())['state_po'].unique().to_list())
        units = pr.filter(pl.col('geo_id').is_null() & (pl.col('state_po') != 'AK')
                          & ~pl.col('state_po').is_in(sorted(have_units)))
        return (pl.concat([base, ct, units], how='diagonal_relaxed'),
                pl.concat([nc, nc_p.filter(pl.col('state_po') == 'CT')]),
                {'county_file': diag, 'precinct_file': diag_p, 'ct_legacy_county_rows_replaced': n_ct_legacy})
    if year == 2016:
        raw_ct, rec_ct = read_ct_2016(ct_names)
        src = 'medsl_precinct:pres_2016_precinct'
    elif year == 2024:
        raw_ct, rec_ct = read_source('pres_2024_ct', office='US PRESIDENT', state='CT')
        src = 'medsl_precinct:pres_2024_ct'
    elif year == 2012:
        raw_ct, rec_ct = read_ct_2012(ct_names_squeezed)
        src = 'openelections_ct_2012:' + gov_reference.CT_2012_OPENELECTIONS_URL
    else:
        raise ValueError(year)
    ct, nc_ct, diag_ct = _county_frame(raw_ct, rec_ct, src, cfg, ct_map, acs_ids)
    if year == 2012:
        ct = ct.with_columns(pl.lit('ct_2012_openelections_transcription').alias('fill_flag'))
    parts = [base, ct]
    return (pl.concat(parts, how='diagonal_relaxed'), pl.concat([nc, nc_ct]),
            {'county_file': diag, 'ct_precinct': diag_ct, 'ct_legacy_county_rows_replaced': n_ct_legacy})


def build_president_2020(cfg, ct_map, acs_ids):
    return build_president(BASE_YEAR, cfg, ct_map, acs_ids)


def governor_references(year: int = 2022) -> pl.DataFrame:
    """Statewide governor totals transcribed in each state's Wikipedia article (fallback reference; see
    fb26.scrape.gov_reference). D/R = the candidate's block total (fusion lines already summed)."""
    parts = []
    for st in gov_reference.GOV_STATES:
        rec = gov_reference.latest_raw(gov_reference.SOURCE, gov_reference.wiki_url(st, year))
        rows = gov_reference.parse_general_table(rawstore.load_raw(rec.path))
        blocks = gov_reference.candidate_blocks(rows)
        parts.append({'state_po': st, 'special_b': False,
                      'ref_d': int(blocks.filter(pl.col('party_class') == 'D')['votes'].sum()),
                      'ref_r': int(blocks.filter(pl.col('party_class') == 'R')['votes'].sum()),
                      'ref_candidate_votes': int(blocks['votes'].sum()),
                      'ref_totalvotes': int(rows['table_total'][0]) if rows.height else None,
                      'ref_file': rec.url, 'ref_sha256': rec.sha256, 'reference_source': 'wikipedia',
                      'reference_reason': 'official certified canvass not fetched: 36 heterogeneous state sites, no '
                                          'per-state parser; Wikipedia general-election table transcribes the canvass'})
    for st, o in (load_cfg_official() if year == 2022 else {}).items():
        rec = gov_reference.latest_raw(gov_reference.SOURCE, o['url'])
        assert rec.sha256 == o['sha256'], f'{st} canvass file changed'
        for p in parts:
            if p['state_po'] == st:
                p.update(ref_d=o['d'], ref_r=o['r'], ref_candidate_votes=o['candidate_votes'], ref_file=o['url'],
                         ref_sha256=rec.sha256, reference_source='official_state_canvass',
                         reference_reason=f"{st} Secretary of State canvass, {o['page']}")
    return pl.DataFrame(parts)


def load_cfg_official() -> dict:
    return config().get('official_references') or {}


def apply_county_patches(county: pl.DataFrame, cfg: dict, year: int = 2022) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replace governor county cells that the MEDSL file lacks or has short, using the Wikipedia 'results by county'
    table of the same race (D, R, total). Config `county_patches`. The patched rows keep the D and R nominees' names,
    carry the remaining votes in one OTHER row and are flagged `patched_from_wikipedia_county_table`."""
    patches = [pt for pt in (cfg.get('county_patches') or []) if str(pt['race']).startswith(str(year))]
    if not patches:
        return county, pl.DataFrame()
    cands = candidate_table(county)
    keep, new, log = county, [], []
    OFFICES = {'SEN': 'SENATE', 'GOV': 'GOVERNOR', 'PRE': 'PRESIDENT'}
    expanded = []
    for pt in patches:
        if pt['wikipedia_county'] != '*':
            expanded.append(pt)
            continue
        # every county of the race: names matched to the ACS list of the state
        office = OFFICES[pt['race'].split('_')[2][:3]]
        rec = gov_reference.latest_raw(gov_reference.SOURCE, gov_reference.wiki_url(
            pt['state'], year, office.lower(), pt['race'].endswith('_SPECIAL')))
        tbl = gov_reference.parse_county_table(rawstore.load_raw(rec.path))
        key = lambda e: (e.str.replace(r'\s*\[.*$', '').str.replace(r',.*$', '')  # noqa: E731
                        .str.replace(r'\s+(County|Parish|Borough|Census Area)$', '').str.to_lowercase()
                        .str.replace_all(r'[^a-z]', ''))
        names = (acs_counties().filter(pl.col('state_fips') == state_fips()[pt['state']])
                 .with_columns(key(pl.col('county_name')).alias('k')).select('k', 'geo_id'))
        m = tbl.with_columns(key(pl.col('county')).alias('k'),
                             pl.col('county').str.replace(r'\s*\[.*$', '').str.strip_chars().alias('name')).join(names, on='k', how='left')
        assert m['geo_id'].null_count() == 0, m.filter(pl.col('geo_id').is_null())['county'].to_list()
        for r in m.iter_rows(named=True):
            expanded.append({**pt, 'geo_id': r['geo_id'], 'wikipedia_county': r['name']})
    for pt in expanded:
        st = pt['state']
        office = OFFICES[pt['race'].split('_')[2][:3]]
        special = pt['race'].endswith('_SPECIAL')
        rec = gov_reference.latest_raw(gov_reference.SOURCE, gov_reference.wiki_url(
            st, year, office.lower(), special))
        tbl = gov_reference.parse_county_table(rawstore.load_raw(rec.path))
        row = tbl.filter(pl.col('county').str.replace(r'\s*\[.*$', '').str.strip_chars() == pt['wikipedia_county'])
        assert row.height == 1, (st, pt['wikipedia_county'])
        row = row.row(0, named=True)
        c = cands.filter((pl.col('state_po') == st) & (pl.col('office') == office) & (pl.col('special_b') == special))
        nom = {p: c.filter(pl.col('party') == p).sort(['votes', 'candidate'], descending=[True, False]).row(0, named=True)
               for p in ('D', 'R')}
        cell = ((pl.col('state_po') == st) & (pl.col('geo_id') == pt['geo_id']) & (pl.col('office') == office)
                & (pl.col('special_b') == special))
        before = county.filter(cell)
        keep = keep.filter(~cell)
        base = {'state_po': st, 'state_fips': state_fips()[st], 'office': office, 'special_b': special,
                'unit_id': pt['geo_id'], 'geo_id': pt['geo_id'], 'masked_rows': 0, 'unit_kind': 'county',
                'acs_match': True, 'writein': False, 'party_override': False, 'geo_flag': '',
                'src_county_name': pt['wikipedia_county'], 'source': f'wikipedia_county_table:{rec.url}',
                'retrieved_at_utc': rawstore.utc_iso(rec.fetched_at_utc), 'raw_sha256': rec.sha256,
                'fill_flag': 'patched_from_wikipedia_county_table'}
        for p, v in (('D', row['d']), ('R', row['r'])):
            new.append({**base, 'cand': nom[p]['cand'], 'candidate': nom[p]['candidate'], 'votes': v,
                        'party_labels': nom[p]['labels'], 'line_classes': p})
        other = (row['total'] - row['d'] - row['r']) if row['total'] is not None else 0
        new.append({**base, 'cand': 'OTHER PATCHED COUNTY TOTAL', 'candidate': 'OTHER (county total minus D and R)',
                    'votes': other, 'party_labels': '', 'line_classes': 'other'})
        log.append({'race_id': pt['race'], 'geo_id': pt['geo_id'], 'medsl_votes_before': int(before['votes'].sum() or 0),
                    'patched_d': row['d'], 'patched_r': row['r'], 'patched_total': row['total']})
    return pl.concat([keep, pl.DataFrame(new, schema_overrides={'votes': pl.Int64})], how='diagonal_relaxed'), pl.DataFrame(log)


def build_tn_governor(acs: pl.DataFrame) -> pl.DataFrame:
    """Tennessee 2022 governor, county level, from the OpenElections transcription of the Secretary of State results
    (sos.tn.gov answers 403 to PoliteClient). Same frame layout as `_county_frame`."""
    rec = gov_reference.latest_raw(gov_reference.SOURCE, gov_reference.TN_OPENELECTIONS_URL)
    df = pl.read_csv(io.BytesIO(rawstore.load_raw(rec.path)), infer_schema_length=0).filter(pl.col('office') == 'Governor')
    names = (acs.filter(pl.col('state_fips') == '47')
             .with_columns(pl.col('county_name').str.replace(r' County, Tennessee$', '').str.to_lowercase().alias('k')))
    df = (df.with_columns(pl.col('county').str.to_lowercase().alias('k'), pl.col('votes').cast(pl.Int64).alias('votes_i'),
                          pl.col('candidate').str.to_uppercase().alias('candidate'),
                          pl.col('candidate').str.to_uppercase().str.starts_with('WRITE-IN').alias('writein'),
                          pl.col('party').fill_null('').str.to_uppercase().alias('plabel'))
          .join(names.select('k', 'geo_id', pl.col('county_name').alias('acs_name')), on='k', how='left'))
    assert df['geo_id'].null_count() == 0, 'unmatched Tennessee county names'
    df = df.with_columns(party_class(pl.col('plabel'), pl.lit('')).alias('line_class'),
                         cand_key(pl.col('candidate')).alias('cand'))
    out = (df.group_by('geo_id', 'cand', 'candidate', 'plabel', 'line_class', 'writein', 'acs_name')
           .agg(pl.col('votes_i').sum().alias('votes'))
           .sort('geo_id', 'cand'))
    return out.select(
        pl.lit('TN').alias('state_po'), pl.lit('47').alias('state_fips'), pl.lit('GOVERNOR').alias('office'),
        pl.lit(False).alias('special_b'), pl.col('geo_id').alias('unit_id'), 'geo_id', 'cand', 'candidate',
        pl.col('votes').cast(pl.Int64), pl.lit(0, dtype=pl.UInt32).alias('masked_rows'), pl.lit('county').alias('unit_kind'),
        pl.lit(True).alias('acs_match'), pl.col('plabel').alias('party_labels'), pl.col('line_class').alias('line_classes'),
        'writein', pl.lit(False).alias('party_override'), pl.lit('').alias('geo_flag'),
        pl.col('acs_name').alias('src_county_name'), pl.lit(f'openelections_tn_county:{rec.url}').alias('source'),
        pl.lit(rawstore.utc_iso(rec.fetched_at_utc)).alias('retrieved_at_utc'), pl.lit(rec.sha256).alias('raw_sha256'),
        pl.lit('').alias('fill_flag'))


def _flags(df: pl.DataFrame) -> pl.Expr:
    parts = [pl.when(pl.col('geo_flag').fill_null('') != '').then(pl.col('geo_flag')).otherwise(None),
             pl.when(pl.col('masked_rows') > 0).then(pl.format('masked_rows={}', pl.col('masked_rows'))).otherwise(None),
             pl.when(pl.col('fill_flag').fill_null('') != '').then(pl.col('fill_flag')).otherwise(None),
             pl.when(~pl.col('acs_match') & pl.col('geo_id').is_not_null() & (pl.col('state_po') != 'AK'))
             .then(pl.lit('not_in_acs_2024')).otherwise(None),
             pl.when(pl.col('geo_id').is_null()).then(pl.lit('unassigned_unit')).otherwise(None)]
    return pl.concat_str(parts, separator='|', ignore_nulls=True).alias('flags')


def to_output(county: pl.DataFrame, cands: pl.DataFrame, races: pl.DataFrame, year: int, acs: pl.DataFrame
              ) -> pl.DataFrame:
    """Long-by-candidate output rows for the units that carry a geo_id (statewide/unassigned units stay out)."""
    inc = races.select('race_id', 'included_d3')
    df = (county.filter(pl.col('geo_id').is_not_null())
          .join(cands.select('state_po', 'office', 'special_b', 'cand', 'party'), on=['state_po', 'office', 'special_b', 'cand'])
          .with_columns(race_id_expr(year).alias('race_id'))
          .join(inc, on='race_id', how='left')
          .join(acs.select('geo_id', pl.col('county_name').alias('acs_name')), on='geo_id', how='left'))
    return df.select(
        pl.lit(year).cast(pl.Int32).alias('year'), 'state_fips', 'state_po', 'geo_id',
        pl.coalesce(pl.col('acs_name'), pl.col('src_county_name'), pl.lit('')).alias('county_name'),
        pl.col('office'), 'race_id', pl.col('special_b').alias('special'), 'included_d3', 'candidate', 'party',
        pl.col('party_labels'), pl.lit('total').alias('mode'), pl.col('votes').cast(pl.Int64), 'source',
        'retrieved_at_utc', 'raw_sha256', pl.lit(None, dtype=pl.Boolean).alias('certified'), 'acs_match',
        _flags(county).alias('flags'), pl.lit(PARSER_VERSION).alias('parser_version')
    ).cast(OUT_SCHEMA).sort('state_po', 'race_id', 'geo_id', 'candidate')


def unmatched_table(county: pl.DataFrame, cands: pl.DataFrame, year: int, acs: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """(a) result units outside the ACS 2024 county list, (b) ACS counties with no votes in a race of that state."""
    rows = []
    keyed = county.with_columns(race_id_expr(year).alias('race_id'))
    out = (keyed.filter(~pl.col('acs_match') | pl.col('geo_id').is_null())
           .group_by('race_id', 'state_po', 'unit_id', 'unit_kind')
           .agg(pl.col('src_county_name').first().alias('unit_name'), pl.col('votes').sum().alias('votes'),
                pl.col('cand').n_unique().alias('n_candidates')))
    for r in out.iter_rows(named=True):
        if r['state_po'] == 'AK':
            continue
        rows.append({'kind': 'result_unit_not_in_acs', 'year': year, 'race_id': r['race_id'], 'state_po': r['state_po'],
                     'unit_id': r['unit_id'], 'unit_name': r['unit_name'], 'votes': r['votes'],
                     'reason': r['unit_kind'], 'action': 'kept out of the county file; votes counted in unassigned_votes'})
    ak = (keyed.filter(pl.col('state_po') == 'AK').group_by('race_id')
          .agg(pl.col('unit_id').n_unique().alias('n'), pl.col('votes').sum().alias('votes')))
    for r in ak.iter_rows(named=True):
        rows.append({'kind': 'alaska_districts', 'year': year, 'race_id': r['race_id'], 'state_po': 'AK',
                     'unit_id': f"{r['n']} election districts", 'unit_name': 'Alaska', 'votes': r['votes'],
                     'reason': 'results by election district, no borough crosswalk',
                     'action': 'kept in results_county.parquet, flagged excluded_ak'})
    have = (keyed.filter(pl.col('acs_match')).group_by('race_id', 'geo_id').agg(pl.col('votes').sum().alias('v')))
    races = (keyed.select('race_id', 'state_po').unique().filter(pl.col('state_po') != 'AK')
             .with_columns(pl.col('state_po').replace_strict(state_fips()).alias('state_fips')))
    miss = (races.join(acs.select('geo_id', 'state_fips', pl.col('county_name').alias('acs_name')), on='state_fips')
            .join(have, on=['race_id', 'geo_id'], how='left')
            .filter(pl.col('v').is_null() | (pl.col('v') == 0)))
    for r in miss.iter_rows(named=True):
        rows.append({'kind': 'acs_county_without_votes', 'year': year, 'race_id': r['race_id'],
                     'state_po': r['state_po'], 'unit_id': r['geo_id'], 'unit_name': r['acs_name'],
                     'votes': 0 if r['v'] is not None else None,
                     'reason': 'zero votes in source' if r['v'] == 0 else 'no result rows in any source used',
                     'action': 'absent from the county file for this race'})
    # (c) generic write-in / scattering lines with an implausible share of a county's two-party vote (e.g. a "WRITE-IN"
    # row that carries a ballot total). D and R are unaffected; the line is kept and listed.
    keyed2 = (keyed.filter(pl.col('geo_id').is_not_null())
              .join(cands.select('state_po', 'office', 'special_b', 'cand', 'party'),
                    on=['state_po', 'office', 'special_b', 'cand']))
    two = (keyed2.filter(pl.col('party').is_in(['D', 'R'])).group_by('race_id', 'geo_id')
           .agg(pl.col('votes').sum().alias('two_party')))
    sus = (keyed2.filter(~pl.col('party').is_in(['D', 'R'])).join(two, on=['race_id', 'geo_id'])
           .filter(pl.col('candidate').str.contains(r'(?i)WRITE|SCATTER|OTHER') & (pl.col('state_po') != 'AK')
                   & (pl.col('votes') > SUSPECT_OTHER_SHARE * pl.col('two_party')) & (pl.col('votes') > 500)))
    for r in sus.sort('race_id', 'geo_id').iter_rows(named=True):
        rows.append({'kind': 'suspect_other_line', 'year': year, 'race_id': r['race_id'], 'state_po': r['state_po'],
                     'unit_id': r['geo_id'], 'unit_name': f"{r['candidate']} ({r['src_county_name']})",
                     'votes': r['votes'], 'reason': f"non-D/R line holds {r['votes'] / r['two_party']:.0%} of the "
                                                    f"county's two-party votes",
                     'action': 'kept; D and R unaffected, other-vote totals for the county are unreliable'})
    schema = {'kind': pl.Utf8, 'year': pl.Int32, 'race_id': pl.Utf8, 'state_po': pl.Utf8, 'unit_id': pl.Utf8,
              'unit_name': pl.Utf8, 'votes': pl.Int64, 'reason': pl.Utf8, 'action': pl.Utf8}
    return pl.DataFrame(rows, schema=schema).sort('race_id', 'kind', 'unit_id')


# Short notes for races the PI should look at (the table's edge_flags are generated; these add the why).
EDGE_NOTES = {
    ('CA', 'SENATE', True): 'Same two candidates as the regular CA Senate race on the same ballot; the two races are '
                            'collinear. n_s counts both (weight V_c / 2 per race).',
    ('CA', 'SENATE', False): 'Collinear with the CA special Senate race (same candidates).',
    ('OK', 'SENATE', True): 'Second OK Senate race on the same ballot (Mullin v. Horn); regular race is Lankford v. Horn.',
    ('UT', 'SENATE', False): 'McMullin (independent) v. Lee; no Democratic nominee. D3 excludes; enters R16.',
    ('LA', 'SENATE', False): 'Open "jungle" contest on Nov 8 with several Democrats and Republicans; no runoff held.',
    ('OR', 'GOVERNOR', False): 'Independent Betsy Johnson took a large share; D v. R is formally satisfied.',
    ('GA', 'SENATE', False): 'Nov 8 round used (Warnock v. Walker v. Oliver); the Dec 6 runoff is excluded.',
    ('ID', 'GOVERNOR', False): 'Independent Ammon Bundy took a large share; D v. R is formally satisfied.',
    ('WY', 'GOVERNOR', False): 'MEDSL labels Theresa Livingston LIBERTARIAN; the Wyoming Secretary of State canvass '
                               '(2022 General Statewide Candidates Official Summary, p. 2) lists her as (D). Relabelled by '
                               'party_overrides; the race is D v. R. Reference totals are the official canvass.',
    ('TN', 'GOVERNOR', False): 'Absent from MEDSL. County rows come from the OpenElections transcription of the Tennessee '
                               'Secretary of State results (sos.tn.gov blocks automated requests); D and R agree with the '
                               'Wikipedia county table in 95 of 95 counties.',
    ('MI', 'GOVERNOR', False): 'Wayne County was about 31k D and 12k R votes short in MEDSL; patched from the Wikipedia county table.',
    ('AR', 'GOVERNOR', False): 'Phillips County missing in MEDSL; patched from the Wikipedia county table.',
    ('ID', 'GOVERNOR', False): 'Power County all zeros in MEDSL; patched from the Wikipedia county table. Independent Bundy 17%.',
    ('NM', 'GOVERNOR', False): 'MEDSL rows carry no party labels; the two nominees are coded by hand in '
                               'config/geography.yaml (party_overrides).',
    ('VT', 'GOVERNOR', False): 'Incumbent Scott (R) won about 71%; the Democratic nominee is labelled DEM/PROG. The 2020-to-2022 '
                               'swing (about +40 points) is a candidate effect, far outside every other race.',
    ('NV', 'GOVERNOR', False): 'Source marks some small precinct counts "*" (masked); not imputed, flagged per county.',
    ('NY', 'SENATE', False): 'Fusion lines (WFP, Conservative); candidate votes are summed across lines.',
    ('NY', 'GOVERNOR', False): 'Fusion lines (WFP, Conservative); candidate votes are summed across lines.',
    ('AK', 'SENATE', False): 'Ranked-choice; two Republicans (Murkowski, Tshibaka) plus Kelley (R) and Chesbro (D). Excluded '
                             'under D3 and under the Alaska rule; first-round counts only.',
    ('AK', 'GOVERNOR', False): 'Ranked-choice; two Republicans (Dunleavy, Pierce), Gara (D), Walker (I). Excluded under '
                               'D3 and under the Alaska rule; first-round counts only.',
}


def _read_gov_refs() -> pl.DataFrame:
    return pl.DataFrame(schema={'state_po': pl.Utf8, 'special_b': pl.Boolean, 'ref_d': pl.Int64, 'ref_r': pl.Int64,
                                'ref_candidate_votes': pl.Int64, 'ref_totalvotes': pl.Int64,
                                'ref_file': pl.Utf8, 'ref_sha256': pl.Utf8})


def senate_wikipedia_references(year: int, keys, cfg: dict) -> pl.DataFrame:
    """Wikipedia general-election table of each Senate race (alternative reference; MEDSL's statewide file has errors,
    e.g. PA and WV 2018). Races whose article table cannot be parsed into a D and an R total are skipped unless
    `manual_references` has them (transcribed from the stored page, SHA-256 checked)."""
    out = []
    manual = cfg.get('manual_references') or {}
    for st, special in keys:
        rid = race_id(year, st, 'SENATE', special)
        if rid in manual:
            m = manual[rid]
            rec = gov_reference.latest_raw(gov_reference.SOURCE, m['url'])
            assert rec.sha256 == m['sha256'], f'{rid} reference page changed'
            out.append({'state_po': st, 'special_b': special, 'ref_d': m['d'], 'ref_r': m['r'],
                        'ref_candidate_votes': m['candidate_votes'], 'ref_file': m['url'], 'ref_sha256': rec.sha256,
                        'reference_source': 'wikipedia'})
            continue
        try:
            rec = gov_reference.latest_raw(gov_reference.SOURCE, gov_reference.wiki_url(st, year, 'senate', special))
        except FileNotFoundError:
            continue
        rows = gov_reference.parse_general_table(rawstore.load_raw(rec.path))
        b = gov_reference.candidate_blocks(rows)
        d, r = (int(b.filter(pl.col('party_class') == x)['votes'].sum()) for x in ('D', 'R'))
        if d > 0 and r > 0:
            out.append({'state_po': st, 'special_b': special, 'ref_d': d, 'ref_r': r,
                        'ref_candidate_votes': int(b['votes'].sum()), 'ref_file': rec.url, 'ref_sha256': rec.sha256,
                        'reference_source': 'wikipedia'})
    return pl.DataFrame(out)


EDGE_NOTES_2018 = {
    ('MS', 'SENATE', True): 'Nov 6 all-party special: two Democrats and two Republicans (Hyde-Smith 41%, Espy 41%, McDaniel '
                            '16%, Bartee 1%). Excluded by D3 like a jungle contest; the Nov 27 runoff (Hyde-Smith v. Espy) '
                            'is not used. The MEDSL statewide file holds the runoff, so the reference is the Wikipedia '
                            'first-round table.',
    ('MS', 'SENATE', False): 'MEDSL labels Shawn O\'Hara REPUBLICAN; he ran as a Reform candidate (Wikipedia table). '
                             'Relabelled via party_overrides; without it the race looked like two Republicans.',
    ('MN', 'SENATE', True): 'Special election (Smith v. Housley), same ballot as the regular Klobuchar race.',
    ('MN', 'SENATE', False): 'Regular race (Klobuchar v. Newberger) on the same ballot as the special.',
    ('ME', 'SENATE', False): 'Independent King (D-caucusing) won 54% with the Democrat third; ranked-choice ballot, first round '
                             'only. D v. R is formally satisfied.',
    ('VT', 'SENATE', False): 'Sanders (I) v. Zupan (R); no Democrat on the ballot.',
    ('CA', 'SENATE', False): 'Top-two: Feinstein (D) v. de Leon (D); no Republican.',
    ('OR', 'GOVERNOR', False): 'MEDSL labels Patrick Starnes DEMOCRAT; he ran for the Independent Party of Oregon (Wikipedia '
                               'table). Relabelled via party_overrides.',
    ('VT', 'GOVERNOR', False): 'Incumbent Scott (R) won 55%; some precinct rows carry Scott as a write-in, handled by the '
                               'write-in rule (candidate is a write-in only if all rows are).',
    ('NY', 'GOVERNOR', False): 'Five counties missing in MEDSL, patched from Wikipedia county table; fusion lines summed.',
    ('NY', 'SENATE', False): 'Three counties missing in MEDSL, patched from Wikipedia county table; fusion lines summed.',
    ('IN', 'SENATE', False): 'Floyd County Democratic votes and Washington County missing/short in MEDSL; patched.',
    ('NJ', 'SENATE', False): 'Middlesex County about 27k D and 16k R votes short in MEDSL; patched.',
    ('PA', 'SENATE', False): 'MEDSL statewide reference is off (Casey 2,777,680 vs 2,792,437 official); county sums match '
                             'the Wikipedia table.',
    ('WV', 'SENATE', False): 'MEDSL statewide reference is off; county sums match the Wikipedia table.',
}


def build_2018(cfg, ct_map, acs, acs_ids):
    """2018 Senate (regular and special) and governor county results and their tables."""
    sen, nc_s, diag_s = build_senate_2018(cfg, ct_map, acs_ids)
    gov, nc_g, diag_g = build_governor_2018(cfg, ct_map, acs_ids)
    c = pl.concat([sen, gov], how='diagonal_relaxed')
    nc = pl.concat([nc_s, nc_g])
    c, patch_log = apply_county_patches(c, cfg, 2018)
    ref_sen = reference_totals('senate_statewide', 2018, 'US SENATE').with_columns(pl.lit('SENATE').alias('office'))
    keys = ref_sen.select('state_po', 'special_b').rows()
    alt = senate_wikipedia_references(2018, keys, cfg)
    races, rec, unm, cands = midterm_cycle(
        2018, c, nc, ref_sen, governor_references(2018), patch_log, pl.DataFrame(), cfg, acs, EDGE_NOTES_2018,
        {('ME', 'SENATE')}, gap_text='listed in the MEDSL statewide Senate file, but the 2018 Senate precinct file has no '
                                     'county rows for it', alt_sen=alt, writein_all=True)
    out = to_output(c, cands, races, 2018, acs)
    return {'races': races, 'rec': rec, 'unmatched': unm, 'out': out, 'patch_log': patch_log,
            'diag': {'senate': diag_s, 'governor': diag_g}}


def midterm_cycle(year, c, nc, ref_sen, ref_gov, patch_log, filled, cfg, acs, edge_notes, rcv_states, gap_text,
                  extra_gaps=(), alt_sen=None, writein_all=False):
    """Race table, reconciliation rows and unmatched list for one midterm year (Senate and governor). Returns
    (races, rec, unmatched, cands). `ref_sen` / `ref_gov` are reference tables (see reference_totals and
    governor_references)."""
    cands = candidate_table(c, writein_all)
    races = classify_races(cands, year, rcv_states)
    gaps = list(extra_gaps)
    seen = races.filter(pl.col('office') == 'SENATE').select('state_po', pl.col('special').alias('special_b'))
    for r in ref_sen.join(seen, on=['state_po', 'special_b'], how='anti').iter_rows(named=True):
        gaps.append({'race_id': race_id(year, r['state_po'], 'SENATE', r['special_b']), 'year': year,
                     'state_po': r['state_po'], 'state_fips': state_fips()[r['state_po']], 'office': 'SENATE',
                     'special': r['special_b'], 'included_d3': False, 'exclusion_code': 'no_source_data',
                     'reason': gap_text})
    if gaps:
        races = pl.concat([races, pl.DataFrame(gaps)], how='diagonal_relaxed')
    races = races.sort('state_po', 'office', 'special')
    tol = cfg['reconciliation_tolerance']
    rec_parts = []
    for office in ('SENATE', 'GOVERNOR'):
        refs = ref_sen if office == 'SENATE' else ref_gov
        reason = ('state is not in the MEDSL statewide Senate reference' if office == 'SENATE' else
                  'state is not in the governor reference table')
        rec_parts.append(reconcile(cands.filter(pl.col('office') == office), c.filter(pl.col('office') == office),
                                   nc.filter(pl.col('office') == office),
                                   refs.drop('office') if 'office' in refs.columns else refs, year, tol, reason,
                                   alt_sen if office == 'SENATE' else None))
    rec = pl.concat(rec_parts, how='diagonal_relaxed')
    sen_tot = (cands.filter((pl.col('office') == 'SENATE') & ~pl.col('special_b')).group_by('state_po')
               .agg(pl.col('votes').sum().alias('sen_votes')))
    gov_tot = (cands.filter(pl.col('office') == 'GOVERNOR').group_by('state_po')
               .agg(pl.col('votes').sum().alias('gov_votes')))
    cross = (gov_tot.join(sen_tot, on='state_po', how='left')
             .with_columns((pl.col('gov_votes') / pl.col('sen_votes')).round(4).alias('gov_over_senate_votes'))
             .select('state_po', 'gov_over_senate_votes'))
    rec = (rec.join(cross.with_columns(pl.lit('GOVERNOR').alias('office')), on=['state_po', 'office'], how='left')
           .rename({'special_b': 'special'}))
    if gaps:
        gap_rec = pl.DataFrame([{'race_id': g['race_id'], 'state_po': g['state_po'], 'office': g['office'],
                                 'special': g['special'], 'year': year, 'status': 'unreconciled',
                                 'reason': 'no source data for the race'} for g in gaps])
        rec = pl.concat([rec, gap_rec], how='diagonal_relaxed')

    unmatched = unmatched_table(c, cands, year, acs, cfg)
    miss_ct = (unmatched.filter(pl.col('kind') == 'acs_county_without_votes').group_by('race_id')
               .agg(pl.len().alias('n_acs_counties_without_votes')))
    n_exp = acs.group_by('state_fips').agg(pl.len().alias('n_acs_counties'))
    filled_ct = (filled.with_columns(race_id_expr(year).alias('race_id')).group_by('race_id')
                 .agg(pl.len().alias('n_cells_filled_from_county_level'))
                 if filled.height else pl.DataFrame(schema={'race_id': pl.Utf8, 'n_cells_filled_from_county_level': pl.UInt32}))
    patched_ct = (patch_log.group_by('race_id').agg(pl.len().alias('n_cells_patched_from_wikipedia'))
                  if patch_log.height else pl.DataFrame(schema={'race_id': pl.Utf8, 'n_cells_patched_from_wikipedia': pl.UInt32}))
    src = (c.with_columns(race_id_expr(year).alias('race_id')).group_by('race_id')
           .agg(pl.col('source').unique().sort().str.join(' + ').alias('source')))
    races = (races.join(n_exp, on='state_fips', how='left').join(miss_ct, on='race_id', how='left')
             .join(filled_ct, on='race_id', how='left').join(patched_ct, on='race_id', how='left')
             .join(src, on='race_id', how='left')
             .join(rec.select('race_id', pl.col('status').alias('recon_status'),
                              pl.col('reference_source').alias('recon_reference_source'), 'd_diff_pct', 'r_diff_pct'),
                   on='race_id', how='left')
             .with_columns(pl.col('n_acs_counties_without_votes').fill_null(0),
                           pl.col('n_cells_filled_from_county_level').fill_null(0),
                           pl.col('n_cells_patched_from_wikipedia').fill_null(0),
                           pl.struct('state_po', 'office', 'special').map_elements(
                               lambda r: edge_notes.get((r['state_po'], r['office'], r['special']), ''),
                               return_dtype=pl.Utf8).alias('pi_note')))
    races = races.with_columns(
        (pl.col('included_d3') & (pl.col('state_po') != 'AK') & (pl.col('recon_status') != 'fail')).alias('proposed_in_sample'),
        pl.when(~pl.col('included_d3')).then(pl.lit('excluded by D3: ') + pl.col('exclusion_code'))
        .when(pl.col('state_po') == 'AK').then(pl.lit('Alaska excluded'))
        .when(pl.col('recon_status') == 'fail').then(pl.lit('reconciliation failed; hold until resolved'))
        .when(pl.col('recon_status') == 'unreconciled').then(pl.lit('in, but no statewide reference to check totals'))
        .otherwise(pl.lit('in')).alias('proposed_reason'))
    return races, rec, unmatched, cands


def build(write: bool = True, reports: bool = True) -> dict:
    cfg = config()
    acs = acs_counties()
    acs_ids = set(acs['geo_id'])
    ct_rec = medsl.raw_record(cfg['connecticut']['region_file_key'])
    ct_map = ct_town_regions(rawstore.load_raw(ct_rec.path))
    assert ct_map.height == cfg['connecticut']['expected_towns'], ct_map.height
    assert ct_map['geo_id'].n_unique() == cfg['connecticut']['expected_regions']

    sen, nc_s, filled, diag_s, _ = build_senate_2022(cfg, ct_map, acs_ids)
    gov, nc_g, diag_g, gov_states = build_governor_2022(cfg, ct_map, acs_ids)
    gov = pl.concat([gov, build_tn_governor(acs)], how='diagonal_relaxed')
    gov, patch_log = apply_county_patches(gov, cfg)
    ct_names = ct_town_names(rawstore.load_raw(ct_rec.path), cfg['connecticut'].get('name_aliases') or {})
    ct_names_sq = ct_town_names(rawstore.load_raw(ct_rec.path), cfg['connecticut'].get('name_aliases_2012') or {}, True)
    pres_by_year = {}
    for y in PRES_YEARS:
        p_y, nc_y, diag_y = build_president(y, cfg, ct_map, acs_ids, ct_names, ct_names_sq)
        p_y, _ = apply_county_patches(p_y, cfg, y)
        p_y, _ = results_overrides.apply(p_y, y, acs)  # defective MEDSL state-years, config/results_overrides.csv
        pres_by_year[y] = (p_y, nc_y, diag_y)
    pres, nc_p, diag_p = pres_by_year[BASE_YEAR]
    c22 = pl.concat([sen, gov], how='diagonal_relaxed')
    nc22 = pl.concat([nc_s, nc_g])
    ref_sen = reference_totals('senate_statewide', YEAR, 'US SENATE').with_columns(pl.lit('SENATE').alias('office'))
    races, rec22, unm22, cands22 = midterm_cycle(
        YEAR, c22, nc22, ref_sen, governor_references(), patch_log, filled, cfg, acs, EDGE_NOTES, RCV_STATES[YEAR],
        gap_text='listed in the MEDSL statewide Senate file, but neither the 2022 Senate precinct file nor the '
                 'county-level Senate file has county rows for it')
    tol = cfg['reconciliation_tolerance']
    rec_parts = []
    cands_p = {}
    for y, (p_y, nc_y, _) in pres_by_year.items():
        cands_p[y] = candidate_table(p_y)
        rec_parts.append(reconcile(cands_p[y], p_y, nc_y, reference_totals('president_statewide', y, 'US PRESIDENT'), y,
                                   tol, 'state is not in the MEDSL statewide president reference',
                                   alt_refs=results_overrides.alt_references(y)).rename({'special_b': 'special'}))
    rec = pl.concat([rec22, *rec_parts], how='diagonal_relaxed').sort('year', 'race_id')
    notes = cfg.get('reconciliation_notes') or {}
    rec = rec.with_columns(pl.struct('race_id', 'status', 'reason').map_elements(
        lambda r: (r['reason'] + ' | ' if r['reason'] else '') + notes[r['race_id']]
        if r['race_id'] in notes and r['status'] in ('fail', 'pass_with_unassigned_units') else r['reason'],
        return_dtype=pl.Utf8).alias('reason'))
    why = results_overrides.reasons()
    rec = rec.with_columns(pl.struct('race_id', 'reason').map_elements(
        lambda r: (r['reason'] + ' | ' if r['reason'] else '') + 'county rows replaced from the Amlani and Algara county file: '
        + why[r['race_id']] if r['race_id'] in why else r['reason'], return_dtype=pl.Utf8).alias('reason'))
    r18 = build_2018(cfg, ct_map, acs, acs_ids)
    rec = pl.concat([rec, r18['rec']], how='diagonal_relaxed').sort('year', 'race_id')
    unmatched = pl.concat([unm22, r18['unmatched']] + [unmatched_table(p_y, cands_p[y], y, acs, cfg) for y, (p_y, _, _) in pres_by_year.items()])

    inc = races.select('race_id', 'included_d3')
    out22 = to_output(c22, cands22, races, YEAR, acs)
    pres_races = {y: classify_races(cands_p[y], y, set()) for y in PRES_YEARS}
    out_p = [to_output(p_y, cands_p[y], pres_races[y], y, acs) for y, (p_y, _, _) in pres_by_year.items()]
    results = pl.concat([*out_p, r18['out'], out22]).sort('year', 'state_po', 'race_id', 'geo_id', 'candidate')
    res = {'results': results, 'races': races, 'races_2018': r18['races'], 'reconciliation': rec, 'unmatched': unmatched, 'pres_races': pres_races[BASE_YEAR],
           'diag': {'senate': diag_s, 'governor': diag_g, 'president': diag_p}, 'filled_cells': filled, 'patch_log': patch_log,
           'gov_states_in_state_file': sorted(gov_states)}
    if write:
        write_outputs(res, reports)
    return res


def write_outputs(res: dict, reports: bool = True) -> None:
    """Writes results_county.parquet and, unless `reports` is False, the tables under reports/validation."""
    out = paths.processed_dir() / 'results_county.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    res['results'].write_parquet(out, compression='zstd')
    outputs = 'data/processed/results_county.parquet'
    if reports:
        vdir = paths.repo_root().joinpath(*OUT_DIR_REPORTS)
        vdir.mkdir(parents=True, exist_ok=True)
        res['races'].write_csv(vdir / 'races_2022.csv')
        res['races_2018'].write_csv(vdir / 'races_2018.csv')
        res['reconciliation'].write_csv(vdir / 'results_reconciliation.csv')
        res['unmatched'].write_csv(vdir / 'results_unmatched.csv')
        outputs += (';reports/validation/races_2022.csv;reports/validation/results_reconciliation.csv;'
                    'reports/validation/results_unmatched.csv')
    snapshot = ';'.join([f'{k}:{medsl.raw_record(k).sha256[:16]}' for k in
                         ['pres_county_2000_2024', 'pres_2020_precinct', 'senate_2022_precinct', 'senate_2022_county',
                          'state_2022_precinct', *GOV_BYSTATE.values(), 'senate_statewide', 'president_statewide',
                          'acs2024_counties', 'ct_cousub22_tract22']]
                        + [f'aa_pres:{results_overrides.county_pres_hist.SHA256[:16]}'])
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(),
        config_hash=runlog.config_hash([paths.config_dir() / 'geography.yaml', paths.config_dir() / 'states.csv']),
        spec_id='ws5_results', data_snapshot_ids=snapshot, output_path=outputs,
        notes=f'parser={PARSER_VERSION}; rows={res["results"].height}; pres_years={list(PRES_YEARS)}'
              + ('' if reports else '; reports/validation tables not rewritten'))


def _county_party_sums(df: pl.DataFrame, cands: pl.DataFrame) -> pl.DataFrame:
    j = df.filter(pl.col('geo_id').is_not_null()).join(cands.select('state_po', 'office', 'special_b', 'cand', 'party'),
                                                       on=['state_po', 'office', 'special_b', 'cand'])
    return (j.group_by('state_po', 'special_b', 'geo_id')
            .agg(pl.col('votes').filter(pl.col('party') == 'D').sum().alias('d'),
                 pl.col('votes').filter(pl.col('party') == 'R').sum().alias('r'), pl.col('votes').sum().alias('t')))


def diagnose(tol: float = 0.01, min_votes: int = 200) -> dict:
    """Independent county-level cross-checks that need no statewide reference.
    (1) 2022 Senate: precinct aggregation vs the MEDSL county-level Senate file, county by county.
    (2) 2020 president: county file vs the 2020 precinct file, county by county (all states).
    (3) 2022 governor: county two-party votes relative to the county's 2020 presidential two-party votes, against the
    state's median ratio."""
    cfg = config()
    acs_ids = set(acs_counties()['geo_id'])
    ct_map = ct_town_regions(rawstore.load_raw(medsl.raw_record(cfg['connecticut']['region_file_key']).path))
    out = {}
    # (1)
    sen, _, _, _, cl = build_senate_2022(cfg, ct_map, acs_ids)
    base = sen.filter(pl.col('fill_flag') == '')
    a, b = (_county_party_sums(x, candidate_table(x)) for x in (base, cl))
    j = (a.join(b, on=['state_po', 'special_b', 'geo_id'], how='full', coalesce=True, suffix='_cl')
         .with_columns(*[pl.col(c).fill_null(0) for c in ('d', 'r', 't', 'd_cl', 'r_cl', 't_cl')])
         .with_columns(((pl.col('t') - pl.col('t_cl')) / pl.col('t_cl').clip(lower_bound=1)).alias('rel_t'),
                       (pl.col('t') - pl.col('t_cl')).alias('abs_t')))
    out['senate_precinct_vs_county_level'] = j.filter((pl.col('rel_t').abs() > tol) & (pl.col('abs_t').abs() > min_votes)
                                                     ).sort('state_po', 'geo_id')
    out['senate_compared_counties'] = j.height
    # (2)
    raw, rec = read_county_file('pres_county_2000_2024', office='US PRESIDENT', year=2020)
    cf, _, _ = _county_frame(raw.filter(pl.col('state_po') != 'CT'), rec, 'county_file', cfg, ct_map, acs_ids)
    raw_p, rec_p = read_source('pres_2020_precinct', office='US PRESIDENT')
    pr, _, _ = _county_frame(raw_p, rec_p, 'precinct', cfg, ct_map, acs_ids)
    a, b = (_county_party_sums(x, candidate_table(x)) for x in (cf, pr.filter(pl.col('state_po') != 'CT')))
    j = (a.join(b, on=['state_po', 'special_b', 'geo_id'], how='full', coalesce=True, suffix='_pr')
         .with_columns(*[pl.col(c).fill_null(0) for c in ('d', 'r', 't', 'd_pr', 'r_pr', 't_pr')])
         .with_columns(((pl.col('t') - pl.col('t_pr')) / pl.col('t_pr').clip(lower_bound=1)).alias('rel_t'),
                       (pl.col('t') - pl.col('t_pr')).alias('abs_t')))
    out['pres_county_file_vs_precinct'] = j.filter((pl.col('rel_t').abs() > tol) & (pl.col('abs_t').abs() > min_votes)
                                                  ).sort('state_po', 'geo_id')
    out['pres_compared_counties'] = j.height
    nat = lambda df: df.select(pl.col('d').sum(), pl.col('r').sum()).row(0)  # noqa: E731
    out['pres_national_county_file'] = nat(a)
    out['pres_national_precinct_file'] = nat(b)
    # (3)
    gov, _, _, _ = build_governor_2022(cfg, ct_map, acs_ids)
    pres, _, _ = build_president_2020(cfg, ct_map, acs_ids)
    g = _county_party_sums(gov, candidate_table(gov)).select('state_po', 'geo_id', (pl.col('d') + pl.col('r')).alias('gov_t'))
    p = _county_party_sums(pres, candidate_table(pres)).select('state_po', 'geo_id',
                                                               (pl.col('d') + pl.col('r')).alias('pres_t'))
    r = (g.join(p, on=['state_po', 'geo_id']).filter(pl.col('pres_t') > 0)
         .with_columns((pl.col('gov_t') / pl.col('pres_t')).alias('ratio'))
         .with_columns((pl.col('ratio') / pl.col('ratio').median().over('state_po')).alias('rel_to_state')))
    out['gov_ratio_outliers'] = r.filter(((pl.col('rel_to_state') > 1.3) | (pl.col('rel_to_state') < 0.7))
                                         & (pl.col('pres_t') > 500)).sort('state_po', 'geo_id')
    out['gov_ratio_counties'] = r.height
    return out


def main(argv: list[str]) -> int:
    positional = [a for a in argv[1:] if not a.startswith('--')]
    step = positional[0] if positional else 'build'
    if step == 'diagnose':
        with pl.Config(tbl_rows=80, tbl_width_chars=200, tbl_cols=14):
            for k, v in diagnose().items():
                print('==', k)
                print(v)
        return 0
    if step == 'build':
        res = build(reports='--no-reports' not in argv)
        print(f"results rows {res['results'].height}; races {res['races'].height}; "
              f"reconciliation {res['reconciliation'].group_by('status').len().sort('status').rows()}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
