"""County presidential results replaced from a second source (placebo-battery repair; written 2026-10-23).

The MEDSL county presidential file is defective for 2004 Maine, Mississippi and New York and for 2008 New York
(`reports/validation/placebo_investigation_memo.md`): incomplete county rows in ME and MS, and fusion-line votes pooled
under OTHER or left out in NY. The replacement is the Amlani and Algara presidential county file (Harvard Dataverse,
doi:10.7910/DVN/DGUMFI, CC0), whose county sums equal the certified state totals for these races. The list of
state-years is `config/results_overrides.csv`; nothing outside it is touched.

For a listed state-year every MEDSL county row of the presidential race is dropped and replaced by three rows per
county: the D nominee, the R nominee (each a candidate total over ballot lines, under the MEDSL candidate name) and one
OTHER row (county total minus D and R). Non-candidate tallies (blank and scattering votes) stay as MEDSL has them.

If the MEDSL statewide reference is itself wrong for a listed race, `alt_reference = wikipedia` adds the Wikipedia
state table (the certified canvass as transcribed there) as the alternative reference of `results.reconcile`.
"""
from __future__ import annotations

import functools
import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from fb26.scrape import county_pres_hist, gov_reference
from fb26.util import paths, rawstore

OVERRIDES_FILE = 'results_overrides.csv'
OTHER_CAND = 'OTHER COUNTY TOTAL (AMLANI ALGARA)'
FILL_FLAG = 'replaced_from_amlani_algara_county_file'
AA_COLS = ['election_year', 'fips', 'county_name', 'state', 'office', 'election_type', 'democratic_raw_votes',
           'republican_raw_votes', 'raw_county_vote_totals']


def overrides() -> pl.DataFrame:
    return pl.read_csv(paths.config_dir() / OVERRIDES_FILE, schema_overrides={'year': pl.Int32, 'alt_reference': pl.Utf8},
                       null_values=[''])


def race_ids(year: int | None = None) -> list[str]:
    df = overrides() if year is None else overrides().filter(pl.col('year') == year)
    return [f"{r['year']}_{r['state_po']}_PRES" for r in df.iter_rows(named=True)]


def reasons() -> dict[str, str]:
    return {f"{r['year']}_{r['state_po']}_PRES": r['reason'] for r in overrides().iter_rows(named=True)}


@functools.lru_cache(maxsize=1)
def _read_aa() -> tuple[pl.DataFrame, rawstore.RawRecord]:
    """The presidential table of the Rdata file as exported by Rscript (text columns; NA is null)."""
    rscript = shutil.which('Rscript')
    if rscript is None:
        raise RuntimeError('Rscript is required to read the Amlani and Algara Rdata file')
    record = county_pres_hist.raw_record()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / 'pres.rdata').write_bytes(rawstore.load_raw(record.path))
        code = ('e <- new.env(); n <- load("pres.rdata", e); '
                'write.csv(get("pres_elections_release", e), "pres.csv", row.names=FALSE)')
        subprocess.run([rscript, '-e', code], cwd=work, check=True, capture_output=True)
        df = pl.read_csv(work / 'pres.csv', infer_schema_length=0, null_values=['NA'])
    return df.select(AA_COLS), record


def aa_county_votes(year: int, state_po: str) -> tuple[pl.DataFrame, rawstore.RawRecord]:
    """County rows (fips, name, D, R, total) of one general presidential election in the Amlani and Algara file."""
    df, record = _read_aa()
    df = (df.filter((pl.col('election_year') == str(year)) & (pl.col('state') == state_po) & (pl.col('office') == 'PRES')
                    & (pl.col('election_type') == 'G'))
          .select(pl.col('fips').str.zfill(5).alias('geo_id'), pl.col('county_name').alias('name'),
                  pl.col('democratic_raw_votes').cast(pl.Float64).cast(pl.Int64).alias('d'),
                  pl.col('republican_raw_votes').cast(pl.Float64).cast(pl.Int64).alias('r'),
                  pl.col('raw_county_vote_totals').cast(pl.Float64).cast(pl.Int64).alias('total')))
    assert df.height and df['geo_id'].n_unique() == df.height, (year, state_po)
    return df, record


# Codes in the Amlani and Algara file that differ from the project's county universe. The 2000 file sums Bedford city into
# Bedford County, so the earlier years must too. Clifton Forge city (51560) stays apart: the 2000 file leaves it out of Alleghany.
AA_FIPS = {'12025': '12086', '46113': '46102', '51515': '51019'}   # Dade -> Miami-Dade, Shannon -> Oglala Lakota, Bedford city


def aa_presidential(year: int) -> pl.DataFrame:
    """County two-party votes (geo_id, d, r) of a general presidential election in the Amlani and Algara file, all states,
    on the project's county codes. Counties with no two-party votes drop out."""
    df, _ = _read_aa()
    return (df.filter((pl.col('election_year') == str(year)) & (pl.col('office') == 'PRES') & (pl.col('election_type') == 'G'))
            .select(pl.col('fips').str.zfill(5).replace(AA_FIPS).alias('geo_id'), pl.col('democratic_raw_votes').cast(pl.Float64).alias('d'),
                    pl.col('republican_raw_votes').cast(pl.Float64).alias('r'))
            .group_by('geo_id').agg(pl.col('d').sum(), pl.col('r').sum())
            .filter((pl.col('d') + pl.col('r')) > 0))


def apply(county: pl.DataFrame, year: int, acs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replace the county rows of the listed presidential races of `year`. Returns the new frame and a log (one row per
    replaced race: MEDSL votes before, replacement votes). `acs` is the ACS 2024 county list (geo_id, state_fips)."""
    todo = overrides().filter(pl.col('year') == year)
    if todo.is_empty():
        return county, pl.DataFrame()
    from fb26.etl import results as R
    cands = R.candidate_table(county)
    fips = R.state_fips()
    keep, new, log = county, [], []
    for o in todo.iter_rows(named=True):
        st = o['state_po']
        votes, rec = aa_county_votes(year, st)
        expected = set(acs.filter(pl.col('state_fips') == fips[st])['geo_id'])
        assert set(votes['geo_id']) == expected, (year, st, sorted(set(votes['geo_id']) ^ expected))
        assert (votes['total'] >= votes['d'] + votes['r']).all(), (year, st)
        c = cands.filter((pl.col('state_po') == st) & (pl.col('office') == 'PRESIDENT') & ~pl.col('special_b'))
        nom = {}
        for p in ('D', 'R'):
            top = c.filter(pl.col('party') == p).sort(['votes', 'candidate'], descending=[True, False])
            assert top.height >= 1, (year, st, p)
            nom[p] = top.row(0, named=True)
        cell = (pl.col('state_po') == st) & (pl.col('office') == 'PRESIDENT') & ~pl.col('special_b')
        before = int(county.filter(cell)['votes'].sum() or 0)
        keep = keep.filter(~cell)
        source = f'amlani_algara_county_pres:doi:{county_pres_hist.DOI}:file {county_pres_hist.FILE_ID}'
        base = {'state_po': st, 'state_fips': fips[st], 'office': 'PRESIDENT', 'special_b': False, 'masked_rows': 0,
                'unit_kind': 'county', 'acs_match': True, 'writein': False, 'party_override': False, 'geo_flag': '',
                'source': source, 'retrieved_at_utc': rawstore.utc_iso(rec.fetched_at_utc), 'raw_sha256': rec.sha256,
                'fill_flag': FILL_FLAG}
        for r in votes.iter_rows(named=True):
            unit = {**base, 'unit_id': r['geo_id'], 'geo_id': r['geo_id'], 'src_county_name': r['name']}
            for p, v in (('D', r['d']), ('R', r['r'])):
                new.append({**unit, 'cand': nom[p]['cand'], 'candidate': nom[p]['candidate'], 'votes': v,
                            'party_labels': nom[p]['labels'], 'line_classes': p})
            new.append({**unit, 'cand': OTHER_CAND, 'candidate': 'OTHER (county total minus D and R)',
                        'votes': r['total'] - r['d'] - r['r'], 'party_labels': '', 'line_classes': 'other'})
        log.append({'race_id': f'{year}_{st}_PRES', 'n_counties': votes.height, 'medsl_candidate_votes_before': before,
                    'replacement_d': int(votes['d'].sum()), 'replacement_r': int(votes['r'].sum()),
                    'replacement_total': int(votes['total'].sum())})
    return (pl.concat([keep, pl.DataFrame(new, schema_overrides={'votes': pl.Int64})], how='diagonal_relaxed'),
            pl.DataFrame(log))


def alt_references(year: int) -> pl.DataFrame | None:
    """Alternative statewide references (D, R, all candidates) of the listed races that ask for one."""
    rows = []
    for o in overrides().filter((pl.col('year') == year) & pl.col('alt_reference').is_not_null()).iter_rows(named=True):
        assert o['alt_reference'] == 'wikipedia', o
        rec = gov_reference.latest_raw(gov_reference.SOURCE, gov_reference.wiki_url(o['state_po'], year, 'president'))
        t = gov_reference.parse_general_table(rawstore.load_raw(rec.path))
        assert t.height, (year, o['state_po'])
        rows.append({'state_po': o['state_po'], 'special_b': False,
                     'ref_d': int(t.filter(pl.col('party') == 'Democratic')['votes'].sum()),
                     'ref_r': int(t.filter(pl.col('party') == 'Republican')['votes'].sum()),
                     'ref_candidate_votes': int(t['votes'].sum()), 'ref_totalvotes': int(t['table_total'][0]),
                     'reference_source': f'wikipedia:{rec.url}'})
    return pl.DataFrame(rows) if rows else None
