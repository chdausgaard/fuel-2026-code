"""Statewide 2022 governor reference totals and Tennessee county results (WS5 follow-up; no MEDSL governor reference).

Two sources, both fetched through PoliteClient into data/raw/gov_reference/:
* Wikipedia article "2022 <State> gubernatorial election" (fallback reference for D/R/other totals; the article's
  general-election table transcribes the certified canvass). Every reference row carries reference_source = wikipedia.
* Official state pages where a parser exists (Tennessee Secretary of State county results; Wyoming canvass).

    python -m fb26.scrape.gov_reference            # fetch all Wikipedia pages
"""
from __future__ import annotations

import re
import sys

import polars as pl
from bs4 import BeautifulSoup

from fb26.scrape.eia import latest_raw
from fb26.util import paths, rawstore
from fb26.util.http import FetchError, PoliteClient

SOURCE = 'gov_reference'
PARSER_VERSION = 'govref-0.1.0'
CACHE_TTL_S = 30 * 24 * 3600
WY_CANVASS_URL = 'https://sos.wyo.gov/Elections/Docs/2022/Results/General/2022_General_Statewide_Candidates_Summary.pdf'
# Tennessee Secretary of State (sos.tn.gov, elections.tn.gov) answers 403 to every request; OpenElections' county file
# is a transcription of the same Secretary of State results (full candidate list by county).
TN_OPENELECTIONS_URL = ('https://raw.githubusercontent.com/openelections/openelections-data-tn/master/2022/'
                        '20221108__tn__general__county.csv')
CT_2012_OPENELECTIONS_URL = ('https://raw.githubusercontent.com/openelections/openelections-data-ct/master/2012/'
                             '20121106__ct__general__precinct.csv')
GOV_STATES = ('AL AK AZ AR CA CO CT FL GA HI ID IL IA KS ME MD MA MI MN NE NV NH NM NY OH OK OR PA RI SC SD TN TX VT WI WY').split()


def state_names() -> dict[str, str]:
    df = pl.read_csv(paths.config_dir() / 'states.csv')
    return dict(zip(df['state_abbr'], df['state_name']))


def wiki_url(state_po: str, year: int = 2022, office: str = 'governor', special: bool = False) -> str:
    """Wikipedia article of a statewide race. Governor: '<year> <State> gubernatorial election'; Senate: '<year> United
    States Senate election in <State>' (special: '... Senate special election in <State>')."""
    name = state_names()[state_po].replace(' ', '_')
    if office == 'governor':
        return f'https://en.wikipedia.org/wiki/{year}_{name}_gubernatorial_election'
    if office == 'president':
        return f'https://en.wikipedia.org/wiki/{year}_United_States_presidential_election_in_{name}'
    kind = 'special_election' if special else 'election'
    return f'https://en.wikipedia.org/wiki/{year}_United_States_Senate_{kind}_in_{name}'


def fetch_wikipedia(client: PoliteClient | None = None, states=GOV_STATES, year: int = 2022, office: str = 'governor',
                    special: bool = False) -> dict:
    client = client or PoliteClient()
    out = {}
    for st in states:
        try:
            out[st] = client.get(wiki_url(st, year, office, special), source=SOURCE, ext='html', cache_ttl_s=CACHE_TTL_S)
        except FetchError as exc:
            out[st] = exc
    return out


def _int(text: str) -> int | None:
    m = re.fullmatch(r'\s*([\d,]+)\s*', text or '')
    return int(m.group(1).replace(',', '')) if m else None


def parse_general_table(html: bytes) -> pl.DataFrame:
    """Candidate rows of the general-election results table: party, candidate, votes, plus the table's total row.
    The general table is the last wikitable with a 'Total votes' row that lists both a Democratic and a Republican
    candidate and is not a primary (primary tables list one party only)."""
    soup = BeautifulSoup(html, 'lxml')
    best = None
    for t in soup.find_all('table', class_='wikitable'):
        rows = []
        total = None
        for tr in t.find_all('tr'):
            cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
            if not cells:
                continue
            if cells[0].lower().startswith('total votes'):
                total = next((v for v in (_int(c) for c in cells[1:]) if v is not None), None)
                continue
            if cells[0].lower().startswith('turnout') and total is None:  # some articles have no 'Total votes' row
                total = next((v for v in (_int(c) for c in cells[1:]) if v is not None), None)
                continue
            nums = [(i, _int(c)) for i, c in enumerate(cells) if _int(c) is not None and i >= 2]
            if len(cells) >= 4 and nums and cells[-1].endswith('%') or (len(cells) >= 4 and nums and '%' in ''.join(cells[-2:])):
                i, votes = nums[0]
                rows.append({'party': cells[i - 2] if cells[i - 2] else cells[i - 3 if i >= 3 else 0],
                             'candidate': cells[i - 1], 'votes': votes})
        if total is None or not rows:
            continue
        parties = ' '.join(r['party'] for r in rows)
        if 'Democrat' in parties and 'Republican' in parties:
            best = (rows, total)  # keep the last qualifying table
    if best is None:
        return pl.DataFrame(schema={'party': pl.Utf8, 'candidate': pl.Utf8, 'votes': pl.Int64, 'table_total': pl.Int64})
    rows, total = best
    return pl.DataFrame(rows).with_columns(pl.col('votes').cast(pl.Int64), pl.lit(total).cast(pl.Int64).alias('table_total'))


def _class(parties: str) -> str:
    p = parties.lower()
    return 'D' if 'democrat' in p else 'R' if 'republican' in p else 'other'


def candidate_blocks(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per candidate. A fusion candidate appears as consecutive lines with the same candidate text, followed
    by a 'Total' row; the block's value is that Total row. Party class: D if any line is Democratic (incl. DFL, NPL),
    R if any is Republican."""
    recs = rows.to_dicts()
    out, i = [], 0
    while i < len(recs):
        r = recs[i]
        if r['party'].strip().lower() == 'total':
            i += 1
            continue
        j = i
        while j + 1 < len(recs) and recs[j + 1]['candidate'] == r['candidate'] and recs[j + 1]['party'].strip().lower() != 'total':
            j += 1
        nxt = recs[j + 1] if j + 1 < len(recs) else None
        if nxt is not None and nxt['party'].strip().lower() == 'total':
            out.append({'candidate': nxt['candidate'], 'votes': nxt['votes'], 'lines': j - i + 1,
                        'party_class': _class(' '.join(x['party'] for x in recs[i:j + 1]))})
            i = j + 2
        else:
            for x in recs[i:j + 1]:
                out.append({'candidate': x['candidate'], 'votes': x['votes'], 'lines': 1, 'party_class': _class(x['party'])})
            i = j + 1
    return pl.DataFrame(out, schema={'candidate': pl.Utf8, 'votes': pl.Int64, 'lines': pl.Int64, 'party_class': pl.Utf8})


def reference_table(states=GOV_STATES, year: int = 2022) -> pl.DataFrame:
    parts = []
    for st in states:
        rec = latest_raw(SOURCE, wiki_url(st, year))
        df = parse_general_table(rawstore.load_raw(rec.path)).with_columns(
            pl.lit(st).alias('state_po'), pl.lit('wikipedia').alias('reference_source'),
            pl.lit(rec.url).alias('reference_url'), pl.lit(rec.sha256).alias('ref_sha256'))
        parts.append(df)
    return pl.concat(parts)


def fetch_official(client: PoliteClient | None = None) -> dict:
    client = client or PoliteClient()
    return {'wy_canvass': client.get(WY_CANVASS_URL, source=SOURCE, ext='pdf', cache_ttl_s=CACHE_TTL_S),
            'tn_openelections': client.get(TN_OPENELECTIONS_URL, source=SOURCE, ext='csv', cache_ttl_s=CACHE_TTL_S),
            'ct_2012_openelections': client.get(CT_2012_OPENELECTIONS_URL, source=SOURCE, ext='csv', cache_ttl_s=CACHE_TTL_S)}


def main(argv: list[str]) -> int:
    for st, res in fetch_wikipedia().items():
        print(st, res if isinstance(res, Exception) else res.record.sha256[:12])
    for k, res in fetch_official().items():
        print(k, res.record.sha256[:12])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))


# ---------------------------------------------------------------------------------------------- county tables

COUNTY_WORDS = ('county', 'parish', 'borough', 'census area', 'city')


def parse_county_table(html: bytes) -> pl.DataFrame:
    """Wikipedia 'results by county' table: one row per county with Democratic, Republican and total votes.
    Layout: header row 1 has the candidates (colspan 2: % and #, order varies), row 2 the sub-headers."""
    soup = BeautifulSoup(html, 'lxml')
    for t in soup.find_all('table'):
        trs = t.find_all('tr')
        if len(trs) < 4:
            continue
        h0 = [(c.get_text(' ', strip=True), int(c.get('colspan', 1))) for c in trs[0].find_all(['th', 'td'])]
        if not h0 or 'county' not in h0[0][0].lower() and 'parish' not in h0[0][0].lower() and 'borough' not in h0[0][0].lower():
            continue
        h1 = [c.get_text(' ', strip=True) for c in trs[1].find_all(['th', 'td'])]
        # column positions: first cell is the name; the sub-header cells follow in order
        pos, k = {}, 1
        for name, span in h0[1:]:
            pos[name] = list(range(k, k + span))
            k += span
        d_cols = next((v for n, v in pos.items() if 'democrat' in n.lower() or 'dfl' in n.lower()), None)
        r_cols = next((v for n, v in pos.items() if 'republican' in n.lower()), None)
        if d_cols is None and r_cols is not None:
            # Democratic header mangled in the article (e.g. "Bob Menendez December"): take the candidate column just
            # before the Republican one, the usual order of these tables
            names = list(pos)
            k = next(i for i, n in enumerate(names) if 'republican' in n.lower())
            d_cols = pos[names[k - 1]] if k > 0 else None
        tot_col = next((v[-1] for n, v in pos.items() if n.lower().startswith('total')), None)
        if d_cols is None or r_cols is None:
            continue

        def vote_col(cols):
            labs = [h1[c - 1] if 0 <= c - 1 < len(h1) else '' for c in cols]
            for c, lab in zip(cols, labs):
                if lab.strip().lower() in ('#', 'votes'):
                    return c
            return cols[-1]
        dc, rc = vote_col(d_cols), vote_col(r_cols)
        out = []
        for tr in trs[2:]:
            cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
            if len(cells) <= max(dc, rc):
                continue
            name = cells[0]
            d, r = _int(cells[dc]), _int(cells[rc])
            tot = _int(cells[tot_col]) if tot_col is not None and tot_col < len(cells) else None
            if d is None or r is None or name.lower().startswith('total'):
                continue
            out.append({'county': name, 'd': d, 'r': r, 'total': tot})
        if out:
            return pl.DataFrame(out, schema={'county': pl.Utf8, 'd': pl.Int64, 'r': pl.Int64, 'total': pl.Int64})
    return pl.DataFrame(schema={'county': pl.Utf8, 'd': pl.Int64, 'r': pl.Int64, 'total': pl.Int64})
