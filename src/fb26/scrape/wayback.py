"""Internet Archive (Wayback) copies of AAA gas-price pages (WS1b; source S2).

AAA's own site is never requested. Everything goes through PoliteClient to web.archive.org
(CDX index + raw `id_` snapshots); parsing reads from the raw store and records PARSER_VERSION.

    python -m fb26.scrape.wayback inventory            # CDX inventory -> reports/validation/wayback_coverage.csv
    python -m fb26.scrape.wayback fetch  <plan>        # download one snapshot per page per day (see PLANS)
    python -m fb26.scrape.wayback fetch-range START END [lag] [dry]   # fill cells no source holds
    python -m fb26.scrape.wayback parse                # raw store -> data/interim/prices_state_daily_wayback.parquet
    python -m fb26.scrape.wayback fetch-metro YEAR [max] [dry]   # state pages whose lag rows reach the window (metro tables)
    python -m fb26.scrape.wayback parse-metro          # raw store -> data/interim/prices_metro_wayback.parquet
    python -m fb26.scrape.wayback report-metro         # -> reports/validation/metro_wayback_coverage.md
    python -m fb26.scrape.wayback overlap              # compare with the GitHub-archived AAA table

Dates come from the page's own "Price as of" text, never from the snapshot timestamp.
Lag fields (yesterday / week ago / month ago / year ago) on state pages are stored with
`derived_from_lag_field = True`; the lag arithmetic is in LAG_RULES.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import calendar
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

from bs4 import BeautifulSoup

from fb26.util import paths, rawstore
from fb26.util.http import PoliteClient

PARSER_VERSION = 'wayback-0.1.0'
METRO_PARSER_VERSION = 'wayback-metro-0.1.0'   # metro tables of the state pages; separate from PARSER_VERSION
SOURCE = 'wayback_aaa'      # value of the `source` column
SOURCE_METRO = 'wayback_aaa_metro'
RAW_SOURCE = 'wayback'      # raw-store directory
CDX = 'https://web.archive.org/cdx/search/cdx'
SNAP = 'https://web.archive.org/web/{ts}id_/{url}'
GAS_RANGE = (1.00, 8.00)
PRODUCTS = ('regular', 'midgrade', 'premium', 'diesel')
LAG_FIELDS = ('current', 'yesterday', 'week_ago', 'month_ago', 'year_ago')
COLUMNS = ['source', 'as_of_date', 'state_abbr', 'state_fips', 'product', 'price_usd', 'lag_field',
           'derived_from_lag_field', 'raw_file', 'raw_sha256', 'capture_ref', 'captured_at_utc',
           'parser_version', 'flags']


# ---------------------------------------------------------------- states

def load_states() -> dict[str, tuple[str, str]]:
    """lower-case name -> (abbr, fips); abbr -> (abbr, fips) also resolves."""
    out: dict[str, tuple[str, str]] = {}
    with (paths.config_dir() / 'states.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            out[row['state_name'].lower()] = (row['state_abbr'], row['state_fips'])
            out[row['state_abbr'].lower()] = (row['state_abbr'], row['state_fips'])
    out['washington dc'] = out['district of columbia']
    out['washington d.c.'] = out['district of columbia']
    return out


# ---------------------------------------------------------------- lag arithmetic

def shift_months(day: date, months: int) -> date:
    """Same calendar day `months` earlier (negative) or later; clipped to the month's last day."""
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


# lag_field -> function(page_date) giving the date the value refers to. "Same calendar date" for month_ago
# and year_ago was checked against directly observed days (audit S02, section 5).
LAG_RULES = {
    'current': lambda d: d,
    'yesterday': lambda d: d - timedelta(days=1),
    'week_ago': lambda d: d - timedelta(days=7),
    'month_ago': lambda d: shift_months(d, -1),
    'year_ago': lambda d: shift_months(d, -12),
}


# ---------------------------------------------------------------- parsing

class ParseError(ValueError):
    pass


@dataclass
class Parsed:
    layout: str
    page_date: date
    page_type: str                                   # state_averages | state_page
    state: str | None = None                         # abbr, state pages only
    national_regular: float | None = None
    # abbr -> lag_field -> product -> price (None = shown but empty)
    values: dict[str, dict[str, dict[str, float | None]]] = field(default_factory=dict)


PRICE_RE = re.compile(r'\$\s*(\d+\.\d+)')
ASOF_RE = re.compile(r'Prices?\s+(?:updated\s*:?\s*(?:as\s+of)?|as\s+of)\s*(?:<br\s*/?>)?\s*(\d{1,2})/(\d{1,2})/(\d{2,4})', re.I)


def _price(text: str) -> float | None:
    match = PRICE_RE.search(text)
    return float(match.group(1)) if match else None


def _asof_dates(html: str) -> list[date]:
    out = []
    for m, d, y in ASOF_RE.findall(html):
        y = int(y)
        out.append(date(y + 2000 if y < 100 else y, int(m), int(d)))
    return out


def _page_text(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, 'lxml')


def detect_layout(html: str, original_url: str) -> str:
    """Layout id of a snapshot. Unknown layouts raise so that nothing is parsed silently."""
    url = original_url.lower()
    if len(html) < 400 and re.search(r'http-equiv="refresh"', html, re.I):
        raise ParseError(f'redirect stub, not a price page ({original_url})')
    if 'state-gas-price-averages' in url and 'id="sortable"' in html and 'class="mid_grade"' in html:
        return 'gp_sga_v1'
    if re.search(r'[?&]state=[a-z]{2}$', url) and 'Current Avg.' in html and 'Year Ago Avg.' in html:
        return 'gp_state_v1'
    if re.fullmatch(r'https?://gasprices\.aaa\.com/?', url) and 'placestxt' in html:
        return 'gp_home_v1'
    if 'sbsavg' in url and 'Current State Averages' in html:
        return 'fgc_sbs_v1'
    if re.search(r'(?:/[a-z]{2}avg\.asp|display\.php\?lt=state&ls=[a-z]{2})$', url) and re.search(r'Current\s+Avg', html):
        return 'fg_avg_2008' if 'Unleaded Average' in html else 'fg_avg_v1'
    raise ParseError(f'unknown layout for {original_url}')


def parse_gp_sga_v1(html: str, states: dict) -> Parsed:
    """gasprices.aaa.com/state-gas-price-averages/, 2024-2026 markup: table#sortable, one row per state,
    cells class regular / mid_grade / premium / diesel; date in `Price as of mm/dd/yy` above the map."""
    dates = set(_asof_dates(html))
    if len(dates) != 1:
        raise ParseError(f'expected one "Price as of" date, found {sorted(dates)}')
    soup = _page_text(html)
    table = soup.find('table', id='sortable')
    parsed = Parsed('gp_sga_v1', dates.pop(), 'state_averages')
    national = soup.find('p', class_='numb')
    parsed.national_regular = _price(national.get_text()) if national else None
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        name = cells[0].get_text(' ', strip=True).lower()
        if name not in states:
            raise ParseError(f'unknown state name {name!r}')
        abbr = states[name][0]
        parsed.values[abbr] = {'current': {
            p: _price(cells[i].get_text()) for i, p in enumerate(PRODUCTS, start=1)}}
    return parsed


PLACES_RE = re.compile(r'placestxt\s*=\s*"([^"]+)"')


def parse_gp_home_v1(html: str, states: dict) -> Parsed:
    """gasprices.aaa.com/ (2016-2026): the national page. Besides the national table it embeds the map data as
    `placestxt = "AB,Name,$x.xxx,url,#colour;..."`: regular-gasoline average of all 51 areas. Date: `Price as of
    mm/dd/yy` (2018 on) or `Prices updated as of mm/dd/yy h:mm am ET` (2016). Regular only."""
    dates = set(_asof_dates(html))
    if len(dates) != 1:
        raise ParseError(f'expected one "as of" date, found {sorted(dates)}')
    match = PLACES_RE.search(html)
    if not match:
        raise ParseError('no placestxt map data')
    parsed = Parsed('gp_home_v1', dates.pop(), 'home')
    for entry in match.group(1).split(';'):
        parts = entry.split(',')
        if len(parts) < 3:
            continue
        abbr = parts[0].strip().upper()
        if abbr.lower() not in states:
            raise ParseError(f'unknown state code {abbr!r}')
        parsed.values[abbr] = {'current': {'regular': _price(parts[2])}}
    national = re.search(r'National Average\s*(?:</p>)?\s*<p class="numb">\s*\$\s*(\d+\.\d+)', html)
    parsed.national_regular = float(national.group(1)) if national else None
    return parsed


def parse_fgc_sbs_v1(html: str, states: dict) -> Parsed:
    """www.fuelgaugereport.com/sbsavg.asp (2001-2010) and sbsavg.html: `Current State Averages` table, one row per
    state (`<a href="AKavg.asp">Alaska </a>` then Regular/Mid/Premium/Diesel); date `Prices updated: m/d/yyyy h:mm:ss AM`."""
    dates = set(_asof_dates(html))
    if len(dates) != 1:
        raise ParseError(f'expected one "Prices updated" date, found {sorted(dates)}')
    soup = _page_text(html)
    parsed = Parsed('fgc_sbs_v1', dates.pop(), 'state_averages')
    for row in soup.find_all('tr'):
        tds = row.find_all('td', recursive=False)
        cells = [c.get_text(' ', strip=True) for c in tds]
        if len(cells) < 5 or tds[0].find('a', href=re.compile(r'[A-Za-z]{2}avg\.asp$')) is None:
            continue
        name = cells[0].lower()
        if name not in states:
            raise ParseError(f'unknown state name {name!r}')
        parsed.values[states[name][0]] = {'current': {p: _price(cells[i]) for i, p in enumerate(PRODUCTS, start=1)}}
    if len(parsed.values) < 50:
        raise ParseError(f'only {len(parsed.values)} states in the table')
    return parsed


def parse_fuelgauge_state(html: str, original_url: str, states: dict, layout: str) -> Parsed:
    """Fuel Gauge Report state pages: fuelgaugereport.com / opisnet.com `XXavg.asp` (2008-2013) and
    fuelgaugereport.aaa.com `import/display.php?lt=state&ls=XX` (2014-2015). Rows Current / Yesterday / [Week Ago] /
    Month Ago / Year Ago x Regular, Mid, Premium, Diesel (E85 columns, where present, are ignored). 2008 pages have no
    Week Ago row. The state is checked against the `<State> Unleaded Average` / `<State> Average Prices` heading."""
    match = re.search(r'(?:/([A-Za-z]{2})avg\.asp|ls=([A-Za-z]{2}))$', original_url)
    abbr = (match.group(1) or match.group(2)).upper()
    dates = set(_asof_dates(html))
    if len(dates) != 1:
        raise ParseError(f'expected one "Prices updated" date, found {sorted(dates)}')
    text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html)).lower()
    names = [n for n, (a, _) in states.items() if a == abbr and len(n) > 2]
    if not any(f'{n} average prices' in text or f'{n} unleaded average' in text for n in names):
        raise ParseError(f'page for {original_url} is not headed with {names}')
    values: dict[str, dict[str, float | None]] = {}
    for row in _page_text(html).find_all('tr'):
        cells = [re.sub(r'\s+', ' ', c.get_text(' ', strip=True)) for c in row.find_all('td', recursive=False)]
        lag = LAG_ROWS.get(cells[0].lower().strip()) if cells else None
        if lag and len(cells) >= 5 and lag not in values:
            values[lag] = {p: _price(cells[i]) for i, p in enumerate(PRODUCTS, start=1)}
    if 'current' not in values:
        raise ParseError('no Current Avg. row')
    parsed = Parsed(layout, dates.pop(), 'state_page', state=abbr)
    parsed.values[abbr] = values
    return parsed


LAG_ROWS = {'current avg.': 'current', 'yesterday avg.': 'yesterday', 'week ago avg.': 'week_ago',
            'month ago avg.': 'month_ago', 'year ago avg.': 'year_ago'}


def parse_gp_state_v1(html: str, original_url: str, states: dict) -> Parsed:
    """gasprices.aaa.com/?state=XX, 2024-2026 markup. The first `table.table-mob` after the
    `<h1 class="nati">` heading is the state table (rows Current/Yesterday/Week/Month/Year Ago Avg.);
    later copies of the same table belong to metros. State date: `Today's AAA <State> Avg. ... Price as of`."""
    match = re.search(r'state=([A-Za-z]{2})$', original_url)
    if not match:
        raise ParseError('state page URL without ?state=XX')
    abbr = match.group(1).upper()
    soup = _page_text(html)
    heading = soup.find('h1', class_='nati')
    # AAA answers ?state=ks (lower case) and unknown codes with the NATIONAL page; the heading names the area shown.
    shown = heading.find('span').get_text(' ', strip=True).lower()
    if shown not in states or states[shown][0] != abbr:
        raise ParseError(f'page for {original_url} shows {shown!r}, not {abbr}')
    table = heading.find_next('table', class_='table-mob')
    parsed = Parsed('gp_state_v1', _state_page_date(soup), 'state_page', state=abbr)
    values: dict[str, dict[str, float | None]] = {}
    for row in table.find('tbody').find_all('tr'):
        cells = [c.get_text(' ', strip=True) for c in row.find_all('td')]
        lag = LAG_ROWS.get(cells[0].lower())
        if lag is None:
            raise ParseError(f'unknown state-table row {cells[0]!r}')
        values[lag] = {p: _price(cells[i]) for i, p in enumerate(PRODUCTS, start=1)}
    if set(values) != set(LAG_FIELDS):
        raise ParseError(f'state table rows {sorted(values)}')
    parsed.values[abbr] = values
    national = soup.find('p', class_='numb')
    parsed.national_regular = _price(national.get_text()) if national else None
    return parsed


def _state_page_date(soup: BeautifulSoup) -> date:
    """The state's own `Price as of` date (second `mobi-average-price` block); falls back to the national one
    only if both agree."""
    blocks = soup.find_all('div', class_='mobi-average-price')
    dates = []
    for block in blocks:
        found = _asof_dates(block.get_text(' ', strip=True))
        if found:
            dates.append(found[0])
    if not dates:
        raise ParseError('no "Price as of" date')
    if len(set(dates)) != 1:
        raise ParseError(f'national and state dates differ: {dates}')
    return dates[0]


def parse(html: str, original_url: str, states: dict | None = None) -> Parsed:
    states = states or load_states()
    layout = detect_layout(html, original_url)
    if layout == 'gp_sga_v1':
        return parse_gp_sga_v1(html, states)
    if layout == 'gp_state_v1':
        return parse_gp_state_v1(html, original_url, states)
    if layout == 'gp_home_v1':
        return parse_gp_home_v1(html, states)
    if layout == 'fgc_sbs_v1':
        return parse_fgc_sbs_v1(html, states)
    if layout in ('fg_avg_2008', 'fg_avg_v1'):
        return parse_fuelgauge_state(html, original_url, states, layout)
    raise ParseError(layout)  # pragma: no cover


# ---------------------------------------------------------------- rows

def rows_from_parsed(parsed: Parsed, *, raw_file: str, raw_sha256: str, capture_ref: str,
                     states: dict | None = None) -> list[dict]:
    states = states or load_states()
    captured = datetime.strptime(capture_ref[:14], '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    out = []
    for abbr, by_lag in parsed.values.items():
        for lag, by_product in by_lag.items():
            as_of = LAG_RULES[lag](parsed.page_date)
            for product, price in by_product.items():
                flags = []
                if price is None:
                    flags.append('missing')
                elif not GAS_RANGE[0] <= price <= GAS_RANGE[1] and product != 'diesel':
                    flags.append('out_of_range')
                expected = (captured - timedelta(hours=8)).date()  # see Snap.page_day
                behind = (expected - parsed.page_date).days
                if behind > 0:
                    flags.append(f'stale_page_{behind}d')
                elif behind < 0:
                    flags.append(f'page_date_ahead_{-behind}d')
                out.append(dict(
                    source=SOURCE, as_of_date=as_of, state_abbr=abbr, state_fips=states[abbr.lower()][1],
                    product=product, price_usd=price, lag_field=lag, derived_from_lag_field=lag != 'current',
                    raw_file=raw_file, raw_sha256=raw_sha256, capture_ref=capture_ref, captured_at_utc=captured,
                    parser_version=PARSER_VERSION, flags=';'.join(flags)))
    return out


# ---------------------------------------------------------------- metro tables (state pages)

METRO_COLUMNS = ['source', 'as_of_date', 'state_abbr', 'state_fips', 'metro_raw', 'product', 'price_usd', 'lag_field',
                 'derived_from_lag_field', 'raw_file', 'raw_sha256', 'capture_ref', 'captured_at_utc',
                 'parser_version', 'flags']


@dataclass
class MetroParsed:
    page_date: date
    state: str
    # (metro name as printed, h3 data-cost) -> lag_field -> regular price (None = shown but empty)
    metros: list[tuple[str, float | None, dict[str, float | None]]] = field(default_factory=list)


def parse_gp_state_metros_v1(html: str, original_url: str, states: dict) -> MetroParsed:
    """Per-metro tables of a `gasprices.aaa.com/?state=XX` page (same markup as parse_gp_state_v1): after the state table,
    `<div class="accordion-prices metros-js">` holds `<h3 data-cost="X.XXXXXXX" data-title>Metro</h3>` and a `div` with a
    `table.table-mob` (rows Current/Yesterday/Week/Month/Year Ago Avg. x Regular/Mid/Premium/Diesel). Regular only.
    data-cost is the metro's current regular price (kept for a consistency flag). The state heading and page date are
    checked as in parse_gp_state_v1, which is left untouched."""
    match = re.search(r'state=([A-Za-z]{2})$', original_url)
    if not match:
        raise ParseError('state page URL without ?state=XX')
    abbr = match.group(1).upper()
    soup = _page_text(html)
    heading = soup.find('h1', class_='nati')
    shown = heading.find('span').get_text(' ', strip=True).lower()
    if shown not in states or states[shown][0] != abbr:
        raise ParseError(f'page for {original_url} shows {shown!r}, not {abbr}')
    parsed = MetroParsed(_state_page_date(soup), abbr)
    for h3 in soup.find_all('h3', attrs={'data-cost': True}):
        table = h3.find_next_sibling('div').find('table', class_='table-mob')
        heads = [th.get_text(' ', strip=True).lower() for th in table.find('thead').find_all('th')]
        if 'regular' not in heads:
            raise ParseError(f'metro table without Regular column ({h3.get_text(strip=True)!r})')
        col = heads.index('regular')
        values: dict[str, float | None] = {}
        for row in table.find('tbody').find_all('tr'):
            cells = [c.get_text(' ', strip=True) for c in row.find_all('td')]
            lag = LAG_ROWS.get(cells[0].lower())
            if lag is None:
                raise ParseError(f'unknown metro-table row {cells[0]!r}')
            values[lag] = _price(cells[col])
        if set(values) != set(LAG_FIELDS):
            raise ParseError(f'metro table rows {sorted(values)} ({h3.get_text(strip=True)!r})')
        try:
            cost = float(h3['data-cost'])
        except ValueError:
            cost = None
        parsed.metros.append((re.sub(r'\s+', ' ', h3.get_text(' ', strip=True)), cost, values))
    return parsed


def metro_rows(parsed: MetroParsed, *, raw_file: str, raw_sha256: str, capture_ref: str,
               states: dict | None = None) -> list[dict]:
    """One row per metro x lag field (regular gasoline); as_of_date by LAG_RULES from the page's own date."""
    states = states or load_states()
    captured = datetime.strptime(capture_ref[:14], '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    behind = ((captured - timedelta(hours=8)).date() - parsed.page_date).days  # see Snap.page_day
    names = [m[0] for m in parsed.metros]
    out = []
    for name, cost, by_lag in parsed.metros:
        for lag, price in by_lag.items():
            flags = []
            if price is None:
                flags.append('missing')
            elif not GAS_RANGE[0] <= price <= GAS_RANGE[1]:
                flags.append('out_of_range')
            if lag == 'current' and price is not None and cost is not None and abs(cost - price) > 0.0005:
                flags.append('cost_mismatch')
            if behind > 0:
                flags.append(f'stale_page_{behind}d')
            elif behind < 0:
                flags.append(f'page_date_ahead_{-behind}d')
            if names.count(name) > 1:
                flags.append('dup_name_on_page')
            out.append(dict(
                source=SOURCE_METRO, as_of_date=LAG_RULES[lag](parsed.page_date), state_abbr=parsed.state,
                state_fips=states[parsed.state.lower()][1], metro_raw=name, product='regular', price_usd=price,
                lag_field=lag, derived_from_lag_field=lag != 'current', raw_file=raw_file, raw_sha256=raw_sha256,
                capture_ref=capture_ref, captured_at_utc=captured, parser_version=METRO_PARSER_VERSION,
                flags=';'.join(flags)))
    return out


# ---------------------------------------------------------------- CDX inventory

# page type -> (CDX url, prefix?, extra `original` regex filter)
PAGE_TYPES = {
    'gp_sga': ('gasprices.aaa.com/state-gas-price-averages/', False, None),
    'gp_home': ('gasprices.aaa.com/', False, None),
    # upper-case codes only: ?state=ks returns the national page
    'gp_state': ('gasprices.aaa.com/?state=', True, r'.*state=[A-Z]{2}$'),
    'fg_home': ('fuelgaugereport.aaa.com/', False, None),
    'fg_state': ('fuelgaugereport.aaa.com/import/display.php', True, r'.*lt=state&ls=[A-Z]{2}$'),
    'fg_statepath': ('fuelgaugereport.aaa.com/states/', True, r'.*/states/[a-z%0-9-]+/$'),
    'op_state': ('fuelgaugereport.opisnet.com/', True, r'.*/[A-Z]{2}avg\.asp$'),
    # www.fuelgaugereport.com (AAA's first Fuel Gauge Report site, 2001-2014): state-by-state table and state pages
    'fgc_sbs': ('fuelgaugereport.com/sbsavg', True, r'.*sbsavg\.(asp|html)$'),
    'fgc_state': ('fuelgaugereport.com/', True, r'.*/[A-Z]{2}avg\.asp$'),
    'op_home': ('fuelgaugereport.opisnet.com/', False, None),
}
CDX_CACHE_S = 14 * 86400
STATE_PAGE_TYPES = ('gp_state', 'fg_state', 'op_state', 'fgc_state')


@dataclass(frozen=True)
class Snap:
    timestamp: str
    original: str
    digest: str
    length: int

    @property
    def day(self) -> date:
        return datetime.strptime(self.timestamp[:8], '%Y%m%d').date()

    @property
    def page_day(self) -> date:
        """Date the page is expected to show: AAA's date turns over between 07:07 and 11:11 UTC (seen on
        2024-11-04 and 2024-10-27 captures), so UTC time minus 8 h. Only used to choose captures; the stored
        date is always the one printed on the page."""
        return (datetime.strptime(self.timestamp, '%Y%m%d%H%M%S') - timedelta(hours=8)).date()

    @property
    def state(self) -> str | None:
        match = re.search(r'(?:state=|ls=)([A-Za-z]{2})$', self.original)
        if match:
            return match.group(1).upper()
        match = re.search(r'/([A-Z]{2})avg\.asp$', self.original)
        return match.group(1) if match else None


def cdx(client: PoliteClient, page_type: str, frm: date, to: date) -> list[Snap]:
    """All 200-status captures of a page type between two dates (inclusive), as listed by the CDX index."""
    url, prefix, regex = PAGE_TYPES[page_type]
    query = f'{CDX}?url={url}&output=json&from={frm:%Y%m%d}&to={to:%Y%m%d}&filter=statuscode:200'
    if prefix:
        query += '&matchType=prefix'
    if regex:
        query += '&filter=original:' + regex.replace('&', '%26').replace('=', '%3D')  # & would end the value
    query += '&fl=timestamp,original,digest,length'
    rows = json.loads(client.get(query, source=RAW_SOURCE, ext='json', cache_ttl_s=CDX_CACHE_S).content)[1:]
    snaps = [Snap(t, o, d, int(n)) for t, o, d, n in rows]
    if page_type in STATE_PAGE_TYPES:  # ?state=US is the national page under a state URL
        valid = {a.upper() for a in load_state_abbrs()}
        snaps = [s for s in snaps if s.state in valid]
    return snaps


def load_windows() -> dict[int, tuple[date, date]]:
    import yaml
    cfg = yaml.safe_load((paths.config_dir() / 'windows.yaml').read_text())['elections']
    return {int(y): (date.fromisoformat(str(v['start'])), date.fromisoformat(str(v['end']))) for y, v in cfg.items()}


def lag_snapshot_range(window: tuple[date, date], route: str) -> tuple[date, date]:
    """UTC dates of captures whose page (or lag field) points into the window. The end is one day later because a
    page dated D is live until about 08:00 UTC on D+1 (see Snap.page_day)."""
    shift = {'direct': 0, 'month_ago': 1, 'year_ago': 12}[route]
    return shift_months(window[0], shift), shift_months(window[1], shift) + timedelta(days=1)


def days_by_state(snaps: list[Snap]) -> dict[date, set[str | None]]:
    out: dict[date, set[str | None]] = {}
    for snap in snaps:
        out.setdefault(snap.page_day, set()).add(snap.state)
    return out


# first and last year in which a page type has any capture (from a whole-history CDX listing on 2026-10-17);
# keeps the slow prefix queries to years that can contain data
PAGE_TYPE_YEARS = {'gp_sga': (2017, 2026), 'gp_home': (2016, 2026), 'gp_state': (2019, 2026),
                   'fg_home': (2010, 2016), 'fg_state': (2014, 2015), 'fg_statepath': (2014, 2015),
                   'op_state': (2010, 2013), 'op_home': (2010, 2013), 'fgc_sbs': (2001, 2013),
                   'fgc_state': (2001, 2014)}
ROUTES = ('direct', 'month_ago', 'year_ago')


def inventory(client: PoliteClient, page_types=tuple(PAGE_TYPES)) -> dict[tuple[int, str, str], list[Snap]]:
    """(window year, page type, route) -> captures with snapshot dates in the route's range."""
    out = {}
    for year, window in load_windows().items():
        for ptype in page_types:
            first, last = PAGE_TYPE_YEARS[ptype]
            for route in ROUTES:
                frm, to = lag_snapshot_range(window, route)
                if to.year < first or frm.year > last:
                    out[(year, ptype, route)] = []
                    continue
                out[(year, ptype, route)] = cdx(client, ptype, frm, to)
    return out


def summarize(snaps: list[Snap]) -> dict:
    by_day = days_by_state(snaps)
    sizes = [len(v - {None}) or 1 for v in by_day.values()]
    return dict(snapshot_days=len(by_day), days_ge51=sum(s >= 51 for s in sizes), n_captures=len(snaps))


# ---------------------------------------------------------------- download

SNAP_RE = re.compile(r'^https://web\.archive\.org/web/(\d{14})id_/(.+)$')


def snapshot_url(snap: Snap) -> str:
    return SNAP.format(ts=snap.timestamp, url=snap.original)


def raw_snapshots() -> dict[str, Path]:
    """Snapshot URL (as stored) -> raw file path, for every snapshot already in the raw store."""
    out = {}
    for meta in (paths.raw_dir() / RAW_SOURCE).glob('**/*.meta.json'):
        info = json.loads(meta.read_text())
        if SNAP_RE.match(info['url']):
            out[info['url']] = Path(info['raw_path'])
    return out


def _log_failure(url: str, message: str) -> None:
    from fb26.util import runlog
    runlog.append_scrape_log(utc_timestamp=rawstore.utc_iso(datetime.now(timezone.utc)), source=RAW_SOURCE,
                             job='wayback.fetch', url=url, http_status='', raw_file='', raw_sha256='', n_bytes='',
                             rows_parsed='', status='fail', message=message[:200])


def failed_urls() -> set[str]:
    """Snapshot URLs that failed in an earlier run (scrape log), so they are not requested again."""
    from fb26.util.paths import logs_dir
    path = logs_dir() / 'scrape_log.csv'
    if not path.exists():
        return set()
    with path.open(newline='') as handle:
        return {r['url'] for r in csv.DictReader(handle) if r['job'] == 'wayback.fetch' and r['status'] == 'fail'}


def fetch_snapshots(client: PoliteClient, snaps: list[Snap]) -> dict[str, list[Snap]]:
    """Download each snapshot not yet in the raw store. Returns {'ok', 'cached', 'failed'}; a snapshot that failed
    in an earlier run counts as 'failed' without a request. A failure is logged and skipped; nothing is retried here
    beyond PoliteClient's own backoff."""
    have = raw_snapshots()
    bad = failed_urls()
    result: dict[str, list[Snap]] = {'ok': [], 'cached': [], 'failed': []}
    for snap in snaps:
        url = snapshot_url(snap)
        if url in have:
            result['cached'].append(snap)
            continue
        if url in bad:
            result['failed'].append(snap)
            continue
        try:
            client.get(url, source=RAW_SOURCE)
            result['ok'].append(snap)
        except Exception as exc:  # FetchError, RobotsDisallowed, connection errors
            _log_failure(url, f'{type(exc).__name__}: {exc}')
            result['failed'].append(snap)
    return result


def last_per_day(snaps: list[Snap]) -> list[Snap]:
    """One capture per (state, expected page date): the latest one (the page is then most likely to be current)."""
    best: dict[tuple, Snap] = {}
    for snap in snaps:
        key = (snap.state, snap.page_day)
        if key not in best or snap.timestamp > best[key].timestamp:
            best[key] = snap
    return sorted(best.values(), key=lambda s: (s.timestamp, s.original))


# ---------------------------------------------------------------- parse the raw store

KEY = ['source', 'as_of_date', 'state_abbr', 'product', 'lag_field']


def parse_raw_store(*, states: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Parse every stored snapshot. Returns (rows, report); `report` has one entry per snapshot with its layout,
    page date, number of rows, or the parse error. Unknown layouts are reported, never guessed."""
    states = states or load_states()
    rows: list[dict] = []
    report: list[dict] = []
    for meta_path in sorted((paths.raw_dir() / RAW_SOURCE).glob('**/*.meta.json')):
        meta = json.loads(meta_path.read_text())
        match = SNAP_RE.match(meta['url'])
        if not match:
            continue
        timestamp, original = match.groups()
        raw_path = Path(meta['raw_path'])
        entry = dict(capture_ref=timestamp, original=original, raw_file=str(raw_path), layout=None,
                     page_date=None, n_rows=0, error='')
        try:
            parsed = parse(rawstore.load_raw(raw_path).decode('utf-8', 'replace'), original, states)
            made = rows_from_parsed(parsed, raw_file=str(raw_path.relative_to(paths.repo_root())),
                                    raw_sha256=meta['sha256'], capture_ref=timestamp, states=states)
            entry.update(layout=parsed.layout, page_date=parsed.page_date, n_rows=len(made))
            rows.extend(made)
        except (ParseError, AttributeError, KeyError, IndexError) as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
        report.append(entry)
    return dedupe(rows), report


def dedupe(rows: list[dict], key: list[str] = KEY) -> list[dict]:
    """One row per key. Preference: not stale, then the latest capture. Disagreeing duplicates get flag `dup_conflict`."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in key), []).append(row)
    out = []
    for group in groups.values():
        group.sort(key=lambda r: ('stale' in r['flags'], -r['captured_at_utc'].timestamp()))
        best = dict(group[0])
        if len(group) > 1:
            conflict = any(r['price_usd'] != best['price_usd'] for r in group[1:])
            tag = f'dup_{len(group)}' + (';dup_conflict' if conflict else '')
            best['flags'] = ';'.join(x for x in (best['flags'], tag) if x)
        out.append(best)
    return sorted(out, key=lambda r: (r['as_of_date'], r['state_abbr'], r.get('metro_raw', ''), r['product'], r['lag_field']))


KEY_METRO = ['source', 'as_of_date', 'state_abbr', 'metro_raw', 'product', 'lag_field']


def parse_raw_store_metro(*, states: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Metro rows of every stored `?state=XX` snapshot (gp_state_v1 layout). Returns (rows, report) like parse_raw_store."""
    states = states or load_states()
    rows: list[dict] = []
    report: list[dict] = []
    for meta_path in sorted((paths.raw_dir() / RAW_SOURCE).glob('**/*.meta.json')):
        meta = json.loads(meta_path.read_text())
        match = SNAP_RE.match(meta['url'])
        if not match or not re.search(r'gasprices\.aaa\.com/\?state=[A-Z]{2}$', match.group(2)):
            continue
        timestamp, original = match.groups()
        raw_path = Path(meta['raw_path'])
        entry = dict(capture_ref=timestamp, original=original, raw_file=str(raw_path), page_date=None, n_metros=0, error='')
        try:
            parsed = parse_gp_state_metros_v1(rawstore.load_raw(raw_path).decode('utf-8', 'replace'), original, states)
            rows.extend(metro_rows(parsed, raw_file=str(raw_path.relative_to(paths.repo_root())),
                                   raw_sha256=meta['sha256'], capture_ref=timestamp, states=states))
            entry.update(page_date=parsed.page_date, n_metros=len(parsed.metros))
        except (ParseError, AttributeError, KeyError, IndexError) as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
        report.append(entry)
    return dedupe(rows, KEY_METRO), report


def write_metro_parquet(rows: list[dict], path: Path | None = None) -> Path:
    import polars as pl
    path = path or paths.interim_dir() / 'prices_metro_wayback.parquet'
    schema = {**_schema(), 'metro_raw': pl.String}
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema).select(METRO_COLUMNS).write_parquet(path)
    return path


def write_parquet(rows: list[dict], path: Path | None = None) -> Path:
    import polars as pl
    path = path or paths.interim_dir() / 'prices_state_daily_wayback.parquet'
    schema = {'source': pl.String, 'as_of_date': pl.Date, 'state_abbr': pl.String, 'state_fips': pl.String,
              'product': pl.String, 'price_usd': pl.Float64, 'lag_field': pl.String,
              'derived_from_lag_field': pl.Boolean, 'raw_file': pl.String, 'raw_sha256': pl.String,
              'capture_ref': pl.String, 'captured_at_utc': pl.Datetime('us', 'UTC'),
              'parser_version': pl.String, 'flags': pl.String}
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema).write_parquet(path)
    return path


# ---------------------------------------------------------------- comparison with the GitHub-archived AAA tables

def evenly_spaced(snaps: list[Snap], n: int) -> list[Snap]:
    """n captures spread over the list (first, last, and evenly in between), one per day."""
    days = last_per_day(snaps)
    if len(days) <= n:
        return days
    return [days[round(i * (len(days) - 1) / (n - 1))] for i in range(n)]


def compare_tables(wayback_rows: list[dict], other, *, product: str = 'regular') -> dict:
    """Compare Wayback `current` prices with another price_state_daily table (polars) on shared
    (date, state) pairs. Returns counts, exact-match share, max absolute difference, and the worst pairs."""
    import polars as pl
    mine = (pl.DataFrame(wayback_rows).filter((pl.col('lag_field') == 'current') & (pl.col('product') == product))
            .select('as_of_date', 'state_abbr', pl.col('price_usd').alias('p_wb'), 'capture_ref'))
    theirs = (other.filter((pl.col('lag_field') == 'current') & (pl.col('product') == product))
              .group_by('as_of_date', 'state_abbr').agg(pl.col('price_usd').first().alias('p_gh')))
    joined = mine.join(theirs, on=['as_of_date', 'state_abbr'], how='inner').drop_nulls(['p_wb', 'p_gh'])
    joined = joined.with_columns((pl.col('p_wb') - pl.col('p_gh')).abs().round(4).alias('absdiff'))
    n = joined.height
    return dict(n_pairs=n, n_dates=joined['as_of_date'].n_unique(),
                exact_share=float((joined['absdiff'] < 0.00051).sum() / n) if n else None,
                max_abs_diff=float(joined['absdiff'].max()) if n else None,
                worst=joined.sort('absdiff', descending=True).head(5), table=joined)


# ---------------------------------------------------------------- planning and coverage

def window_days(window: tuple[date, date]) -> list[date]:
    n = (window[1] - window[0]).days + 1
    return [window[0] + timedelta(days=i) for i in range(n)]


def full_days(df, window: tuple[date, date], *, derived: bool | None, product: str = 'regular') -> set[date]:
    """Window days with a regular-gasoline price for all 51 states. derived=False: direct `current` rows only;
    derived=True: lag-field rows only; None: both."""
    import polars as pl
    sel = df.filter((pl.col('product') == product) & pl.col('price_usd').is_not_null()
                    & pl.col('as_of_date').is_between(*window))
    if derived is not None:
        sel = sel.filter(pl.col('derived_from_lag_field') == derived)
    counts = sel.group_by('as_of_date').agg(pl.col('state_abbr').n_unique().alias('n'))
    return {d for d, n in counts.iter_rows() if n >= 51}


def plan_direct(snaps: list[Snap]) -> list[Snap]:
    return last_per_day(snaps)


def lag_cells(snap: Snap, window: tuple[date, date]) -> set[tuple[date, str]]:
    """(date, state) cells inside the window that a state-page capture can supply through any of its five rows."""
    cells = set()
    for rule in LAG_RULES.values():
        day = rule(snap.page_day)
        if window[0] <= day <= window[1]:
            cells.add((day, snap.state))
    return cells


def plan_lag(candidates: list[Snap], window: tuple[date, date], missing: set[tuple[date, str]],
             max_requests: int | None = None) -> list[Snap]:
    """Greedy cover of the missing (date, state) cells by state-page captures. One capture per (state, page date)
    is a candidate (the latest of the day); each supplies up to five dates for its state through its lag rows.
    Ties go to the capture whose cells are closest to the page date (short lags are revised less)."""
    best = {(s.state, s.page_day): s for s in last_per_day(candidates)}
    cover = {key: lag_cells(snap, window) & missing for key, snap in best.items()}
    left = set(missing)
    chosen: list[Snap] = []
    while left and cover and (max_requests is None or len(chosen) < max_requests):
        key = max(cover, key=lambda k: (len(cover[k] & left), -sum((k[1] - d).days for d, _ in cover[k] & left)))
        gain = cover[key] & left
        if not gain:
            break
        chosen.append(best[key])
        left -= gain
    return sorted(chosen, key=lambda s: (s.timestamp, s.original))


# state-page families usable for the lag route, by window year (what existed around that time)
LAG_PAGE_TYPES = {2008: ('fgc_state',), 2012: ('fgc_state', 'op_state'), 2014: ('fg_state', 'op_state'),
                  2016: ('gp_state', 'fg_state'), 2018: ('gp_state',), 2020: ('gp_state',), 2022: ('gp_state',),
                  2024: ('gp_state',), 2026: ('gp_state',)}


def cdx_lag_candidates(client: PoliteClient, window: tuple[date, date], ptypes=('gp_state',)) -> list[Snap]:
    """Captures whose current/yesterday/week-ago rows (window start to end + 8 d), month-ago rows or
    year-ago rows can fall inside the window."""
    out = {}
    ranges = [(window[0], window[1] + timedelta(days=9)), lag_snapshot_range(window, 'month_ago'),
              lag_snapshot_range(window, 'year_ago')]
    for ptype in ptypes:
        first, last = PAGE_TYPE_YEARS[ptype]
        for frm, to in ranges:
            if to.year < first or frm.year > last:
                continue
            for snap in cdx(client, ptype, frm, to):
                out[(snap.timestamp, snap.original)] = snap
    return list(out.values())


# ---------------------------------------------------------------- other public scrapers (git bundles in the raw store)

def repo_state_daily_rows(git_dir: Path, *, source: str, path_prefix: str, raw_file: str, raw_sha256: str,
                          parser_version: str = PARSER_VERSION, states: dict | None = None) -> list[dict]:
    """Rows from a scraper repo whose history holds daily CSVs `State-Abbreviation,Regular,Mid-Grade,Premium,Diesel,Date`
    under `path_prefix` (lykmapipo/US-Gas-Prices layout). Read through `git --git-dir ... log/cat-file` only;
    the repo's code is never run or checked out. A blob is read once; capture_ref is the first commit that
    introduced it and captured_at_utc that commit's author time (UTC)."""
    import io
    import subprocess
    states = states or load_states()
    log = subprocess.check_output(['git', '--git-dir', str(git_dir), 'log', '--raw', '-r', '--no-renames',
                                   '--format=C %H %aI', '--', path_prefix], text=True)
    first_seen = _first_commit_per_blob(log)
    rows: list[dict] = []
    proc = subprocess.Popen(['git', '--git-dir', str(git_dir), 'cat-file', '--batch'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for blob, (commit, when) in first_seen.items():
            proc.stdin.write((blob + '\n').encode())
            proc.stdin.flush()
            size = int(proc.stdout.readline().split()[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)
            reader = list(csv.DictReader(io.StringIO(data.decode('utf-8', 'replace'))))
            if not reader or 'State-Abbreviation' not in reader[0]:
                continue  # early files of this repo used another header and no abbreviations
            captured = datetime.fromisoformat(when).astimezone(timezone.utc)
            for rec in reader:
                abbr = rec['State-Abbreviation'].upper()
                for product, column in zip(PRODUCTS, ('Regular', 'Mid-Grade', 'Premium', 'Diesel')):
                    value = rec[column].strip()
                    price = float(value) if value else None
                    flags = [] if price is not None else ['missing']
                    if price is not None and product != 'diesel' and not GAS_RANGE[0] <= price <= GAS_RANGE[1]:
                        flags.append('out_of_range')
                    rows.append(dict(
                        source=source, as_of_date=date.fromisoformat(rec['Date']), state_abbr=abbr,
                        state_fips=states[abbr.lower()][1], product=product, price_usd=price, lag_field='current',
                        derived_from_lag_field=False, raw_file=raw_file, raw_sha256=raw_sha256, capture_ref=commit,
                        captured_at_utc=captured, parser_version=parser_version, flags=';'.join(flags)))
    finally:
        proc.stdin.close()
        proc.wait()
    return rows


def _first_commit_per_blob(log: str) -> dict[str, tuple[str, str]]:
    """From `git log --raw --format='C <sha> <iso>'` (newest first): blob -> (oldest commit introducing it, time)."""
    first: dict[str, tuple[str, str]] = {}
    commit = when = ''
    for line in log.splitlines():
        if line.startswith('C '):
            _, commit, when = line.split()
        elif line.startswith(':'):
            blob = line.split('\t')[0].split()[3]
            if blob.strip('0'):
                first[blob] = (commit, when)  # later (older) commits overwrite earlier (newer) ones
    return first


def repo_national_rows(git_dir: Path, *, source: str, path_prefix: str, raw_file: str, raw_sha256: str,
                       hours_utc: tuple[int, int] = (11, 14), last_date: date | None = None,
                       parser_version: str = PARSER_VERSION, states: dict | None = None) -> list[dict]:
    """Rows from nathanqthai/gas_prices: one CSV per scrape, `prices/national/<UTC time>.csv`, columns
    `abbr,name,$regular,url`. The files carry no AAA page date. The scrape time is used instead, and only for
    scrapes at 11:00-13:59 UTC (the daily run): their UTC date equals AAA's printed date in 99.5% of 18,156
    state-days compared with lykmapipo/US-Gas-Prices (audit S02). All rows get flag `date_from_scrape_time`."""
    import io
    import subprocess
    states = states or load_states()
    listing = subprocess.check_output(['git', '--git-dir', str(git_dir), 'ls-tree', '-r', 'HEAD', '--', path_prefix],
                                      text=True)
    rows: list[dict] = []
    proc = subprocess.Popen(['git', '--git-dir', str(git_dir), 'cat-file', '--batch'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for line in listing.splitlines():
            meta, name = line.split('\t')
            blob = meta.split()[2]
            match = re.search(r'(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)', name)
            if not match:
                continue
            captured = datetime.fromisoformat(f'{match.group(1)}T{match.group(2)}+00:00')
            if not hours_utc[0] <= captured.hour < hours_utc[1]:
                continue
            if last_date is not None and captured.date() > last_date:
                continue  # the repo's README: AAA until 2026-03-27, GasBuddy afterwards
            proc.stdin.write((blob + '\n').encode())
            proc.stdin.flush()
            size = int(proc.stdout.readline().split()[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)
            for rec in csv.reader(io.StringIO(data.decode('utf-8', 'replace'))):
                if len(rec) < 3 or rec[0].lower() not in states:
                    continue
                price = float(rec[2].replace('$', '')) if rec[2].strip() else None
                flags = ['date_from_scrape_time'] + (['missing'] if price is None else [])
                if price is not None and not GAS_RANGE[0] <= price <= GAS_RANGE[1]:
                    flags.append('out_of_range')
                rows.append(dict(
                    source=source, as_of_date=captured.date(), state_abbr=rec[0].upper(),
                    state_fips=states[rec[0].lower()][1], product='regular', price_usd=price, lag_field='current',
                    derived_from_lag_field=False, raw_file=raw_file, raw_sha256=raw_sha256, capture_ref=blob,
                    captured_at_utc=captured, parser_version=parser_version, flags=';'.join(flags)))
    finally:
        proc.stdin.close()
        proc.wait()
    return rows


# ---------------------------------------------------------------- metro report (coverage, agreement, sparse days)

LYK_FULL_WINDOW = (date(2024, 10, 15), date(2024, 11, 4))   # the only complete 21-day election window in the daily panel
# complete later 21-day windows of the daily panel, used as stand-in "target" cycles for the shock version
LYK_TARGETS = {'Jul 2026': (date(2026, 6, 30), date(2026, 7, 20)), 'Sep 2026': (date(2026, 9, 1), date(2026, 9, 21))}
MATCH_TOL = 0.0005
N_RANDOM_DRAWS = 200
SEED = 20261022


def _keyed(df, col: str = 'metro_raw'):
    """Add `k`, the normalized metro name used in src/fb26/etl/metro_prices.py, and drop (state, k) keys that are
    ambiguous within a state."""
    import polars as pl
    from fb26.etl.metro_prices import normalize
    names = df.select(col).unique().with_columns(pl.col(col).map_elements(normalize, return_dtype=pl.String).alias('k'))
    return df.join(names, on=col, how='left')


def _share(x, n):
    return f'{100 * x / n:.1f}%' if n else 'n/a'


def _md(rows: list[dict], cols: list[str] | None = None) -> str:
    cols = cols or list(rows[0])
    fmt = lambda v: f'{v:.4f}' if isinstance(v, float) else '' if v is None else str(v)
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
                     + ['| ' + ' | '.join(fmt(r[c]) for c in cols) + ' |' for r in rows]) + '\n'


def metro_window_coverage(wb, window: tuple[date, date], units: list[str]) -> dict:
    """Per-unit counts of metros and observed days in a window: any route, direct (`current`), and per lag field."""
    import polars as pl
    sel = wb.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*window))
    days = {'any': sel}
    days.update({lag: sel.filter(pl.col('lag_field') == lag) for lag in LAG_FIELDS})
    out = {'metros': {u: 0 for u in units}}
    for u, n in sel.group_by('state_abbr').agg(pl.col('metro_raw').n_unique()).iter_rows():
        out['metros'][u] = n
    for name, df in days.items():
        out[name] = {u: 0 for u in units}
        for u, n in df.group_by('state_abbr').agg(pl.col('as_of_date').n_unique()).iter_rows():
            out[name][u] = n
    return out


def _median(xs):
    return statistics.median(xs) if xs else None


def _fmean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def sparse_positions(wb, window: tuple[date, date], units: list[str]) -> dict[str, list[int]]:
    """State -> day offsets (0 = window start) on which the archive holds at least one metro price, any route."""
    import polars as pl
    sel = wb.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*window))
    out: dict[str, set[int]] = {u: set() for u in units}
    for u, d in sel.select('state_abbr', 'as_of_date').unique().iter_rows():
        out[u].add((d - window[0]).days)
    return {u: sorted(v) for u, v in out.items()}


def _window_matrix(lyk, window: tuple[date, date]):
    """(state_fips, metro_raw) -> list of 21 daily prices from the daily panel; only metros with every day."""
    import polars as pl
    n = (window[1] - window[0]).days + 1
    sel = lyk.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*window))
    out = {}
    for (fips, name), g in sel.group_by('state_fips', 'metro_raw'):
        by_day = dict(zip(g['as_of_date'].to_list(), g['price_usd'].to_list()))
        if len(by_day) == n:
            out[(fips, name)] = [by_day[window[0] + timedelta(days=i)] for i in range(n)]
    return out


def _dev(means: dict, groups: dict) -> dict:
    """Metro mean minus the mean over its state's metros."""
    by_state: dict[str, list] = {}
    for key in means:
        by_state.setdefault(groups[key], []).append(key)
    out = {}
    for keys in by_state.values():
        if len(keys) < 2:
            continue
        m = sum(means[k] for k in keys) / len(keys)
        out.update({k: means[k] - m for k in keys})
    return out


def _stats(truth: dict, est: dict, groups: dict) -> dict:
    """Pooled and per-state agreement of within-state deviations. reliability = var(truth) / (var(truth) + var(error))."""
    keys = [k for k in truth if k in est]
    t, e = [truth[k] for k in keys], [est[k] - truth[k] for k in keys]
    s = [est[k] for k in keys]
    var = lambda x: statistics.pvariance(x) if len(x) > 1 else float('nan')
    corr = statistics.correlation(t, s) if len(keys) > 2 else float('nan')
    per_state = []
    by_state: dict[str, list] = {}
    for k in keys:
        by_state.setdefault(groups[k], []).append(k)
    for ks in by_state.values():
        if len(ks) >= 4:
            tt, ss = [truth[k] for k in ks], [est[k] for k in ks]
            if statistics.pstdev(tt) > 0 and statistics.pstdev(ss) > 0:
                per_state.append(statistics.correlation(tt, ss))
    return dict(n_metros=len(keys), n_states=len(by_state), sd_truth=math.sqrt(var(t)), sd_error=math.sqrt(var(e)),
                corr=corr, reliability=var(t) / (var(t) + var(e)), states_ge4=len(per_state),
                median_state_corr=_median(per_state), share_state_corr_lt_05=(sum(c < 0.5 for c in per_state) / len(per_state)
                                                                          if per_state else None))


def volatile_window(lyk, min_metros: int = 470) -> tuple[date, date, float]:
    """The complete 21-day window of the daily panel with the largest spread (SD) of the daily mean price: a stress test
    for the sparse-day check, since the election window of 2024 was calm."""
    import polars as pl
    daily = (lyk.filter(pl.col('price_usd').is_not_null()).group_by('as_of_date')
             .agg(pl.col('price_usd').mean().alias('m'), pl.len().alias('n')))
    cnt, mean = dict(zip(daily['as_of_date'], daily['n'])), dict(zip(daily['as_of_date'], daily['m']))
    best = None
    for start in sorted(cnt):
        days = [start + timedelta(i) for i in range(21)]
        if all(cnt.get(d, 0) >= min_metros for d in days):
            sd = statistics.pstdev(mean[d] for d in days)
            if best is None or sd > best[2]:
                best = (start, days[-1], sd)
    return best


def sparse_reliability(lyk, patterns: dict[str, dict[str, list[int]]], states_df) -> tuple[list[dict], list[dict], dict]:
    """Within-state metro deviations of the 21-day window mean, computed from the days the archive covers in 2020 /
    2022 only, against the full window (daily 2024 panel; also the most volatile complete window as a stress test).
    Also the shock version (target window minus baseline window) for a 2022-type (both sides sparse) and a 2024-type
    (baseline sparse, target full) cycle."""
    fips_of = dict(zip(states_df['state_abbr'], states_df['state_fips']))
    abbr_of = {v: k for k, v in fips_of.items()}
    vol = volatile_window(lyk)
    truth_windows = {'2024 election window': LYK_FULL_WINDOW, f'most volatile complete window ({vol[0]} to {vol[1]})': vol[:2]}
    base = _window_matrix(lyk, LYK_FULL_WINDOW)
    targets = {name: _window_matrix(lyk, win) for name, win in LYK_TARGETS.items()}
    keys = [k for k in base if k[0] in abbr_of]
    groups = {k: k[0] for k in keys}
    full = {k: sum(base[k]) / 21 for k in keys}
    rng = random.Random(SEED)
    level_rows, shock_rows = [], []
    info = {'n_metros_daily_2024': len(keys), 'volatile': vol}
    for wname, win in truth_windows.items():
        mat = _window_matrix(lyk, win)
        wkeys = [k for k in keys if k in mat]
        wfull = {k: sum(mat[k]) / 21 for k in wkeys}
        full_dev = _dev(wfull, groups)
        for name, pat in patterns.items():
            usable = {s for s, pos in pat.items() if pos and fips_of[s] in {k[0] for k in wkeys}}
            ks = [k for k in wkeys if abbr_of[k[0]] in usable]
            sparse = {k: sum(mat[k][i] for i in pat[abbr_of[k[0]]]) / len(pat[abbr_of[k[0]]]) for k in ks}
            info[name] = dict(states_with_days=len(usable), states_without=len(pat) - len(usable))
            dev = _dev(sparse, groups)
            truth = {k: full_dev[k] for k in dev}
            level_rows.append(dict(truth_window=wname, pattern=name, design='archive days', **_stats(truth, dev, groups)))
            # reference: the same number of days per state, drawn at random
            draws = []
            for _ in range(N_RANDOM_DRAWS):
                pos = {s: sorted(rng.sample(range(21), len(pat[s]))) for s in usable}
                r = {k: sum(mat[k][i] for i in pos[abbr_of[k[0]]]) / len(pos[abbr_of[k[0]]]) for k in ks}
                draws.append(_stats(truth, _dev(r, groups), groups))
            level_rows.append(dict(truth_window=wname, pattern=name, design=f'random days, same count ({N_RANDOM_DRAWS} draws, mean)',
                                   **{c: statistics.fmean(d[c] for d in draws) for c in ('n_metros', 'n_states', 'sd_truth', 'sd_error',
                                                                                        'corr', 'reliability', 'states_ge4')},
                                   median_state_corr=_fmean(d['median_state_corr'] for d in draws),
                                   share_state_corr_lt_05=_fmean(d['share_state_corr_lt_05'] for d in draws)))
    # shock: target window (full or sparse with the 2022 pattern) minus baseline window (sparse with the 2020 pattern)
    p20, p22 = patterns['2020 days'], patterns['2022 days']
    ok = {s for s in p20 if p20[s] and p22[s]}
    for tname, target in targets.items():
        ks = [k for k in keys if k in target and abbr_of[k[0]] in ok]
        tfull = {k: sum(target[k]) / 21 for k in ks}
        gap = statistics.fmean(tfull[k] - full[k] for k in ks) if ks else float('nan')
        truth_shock = _dev({k: tfull[k] - full[k] for k in ks}, groups)
        for label, tpat in (('2024-type: baseline sparse (2020 days), target full', None),
                            ('2022-type: baseline sparse (2020 days), target sparse (2022 days)', p22)):
            est = {}
            for k in ks:
                s = abbr_of[k[0]]
                b = sum(base[k][i] for i in p20[s]) / len(p20[s])
                t = tfull[k] if tpat is None else sum(target[k][i] for i in tpat[s]) / len(tpat[s])
                est[k] = t - b
            shock_rows.append(dict(target=tname, cycle=label, mean_change=gap, **_stats(truth_shock, _dev(est, groups), groups)))
    info['shock_states'] = len(ok)
    return level_rows, shock_rows, info


class _NoNetwork:
    """Session stand-in: any request raises, so cached CDX answers are used or the report fails loudly."""
    headers: dict = {}

    def get(self, *args, **kwargs):
        raise RuntimeError('network disabled in report')


def _fetch_accounting(windows: dict) -> str:
    """Snapshots in the raw store by capture year, those fetched for this source, failures, and per window the (date, unit)
    cells reachable if every capture listed by the cached CDX inventories were fetched."""
    import polars as pl
    rows, _ = parse_raw_store_metro()
    have = {(r['as_of_date'], r['state_abbr']) for r in rows if r['price_usd'] is not None}
    states_with = {r['state_abbr'] for r in rows}
    today = datetime.now(timezone.utc).date().isoformat()
    metas = []
    for meta_path in (paths.raw_dir() / RAW_SOURCE).glob('**/*.meta.json'):
        meta = json.loads(meta_path.read_text())
        m = SNAP_RE.match(meta['url'])
        if m and re.search(r'gasprices\.aaa\.com/\?state=[A-Z]{2}$', m.group(2)):
            metas.append((m.group(1)[:4], meta['fetched_at_utc'][:10]))
    by_year = Counter(y for y, _ in metas)
    fetched_today = sum(d == today for _, d in metas)
    failed = sorted({u for u in failed_urls() if re.search(r'\?state=[A-Z]{2}$', u)})
    out = [f'State-page snapshots in the raw store: {len(metas)} ({", ".join(f"{y}: {n}" for y, n in sorted(by_year.items()))} by capture year); '
           f'{fetched_today} of them fetched on {today}. Snapshot URLs that failed (all "Redirect requires an explicitly audited destination URL": '
           f'the archive answers with a redirect to another capture; not followed) and are skipped in planning: {len(failed)}.\n']
    client = PoliteClient(session=_NoNetwork())
    ceiling = []
    for year in (2020, 2022):
        win = windows[year]
        cands = [c for c in cdx_lag_candidates(client, win, ('gp_state',)) if snapshot_url(c) not in set(failed)]
        best = {(c.state, c.page_day): c for c in last_per_day(cands)}
        reach = set()
        for c in best.values():
            reach |= {(d, c.state) for d, _ in lag_cells(c, win)}
        cells = {(d, st) for d in window_days(win) for st in states_with}
        ceiling.append(dict(window=year, listed_captures=len(cands), listed_page_days=len(best), cells_possible=len(cells),
                            cells_held=len(cells & have), cells_held_or_listed=len(cells & (have | reach)),
                            ceiling_share=len(cells & (have | reach)) / len(cells), held_share=len(cells & have) / len(cells)))
    out.append('Cells = (reference date, unit) in the window. "Listed" = reachable through any capture in the CDX inventory '
               '(`current`, `yesterday`, `week_ago`, `month_ago`, `year_ago` of captures dated to the window, a week after it, a month '
               'after it and a year after it; page dates as estimated from the capture time).\n')
    out.append(_md(ceiling))
    return '\n'.join(out)


def _daily_mean(lyk, window: tuple[date, date]) -> list[float]:
    import polars as pl
    sel = lyk.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*window))
    return sel.group_by('as_of_date').agg(pl.col('price_usd').mean()).sort('as_of_date')['price_usd'].to_list()


METRO_LIMITATIONS = """## 5. Limitations

- **Coverage is set by the archive, not by effort.** After fetching every listed capture that adds cells, the 2020 window holds well under a fifth of its unit-days and the ceiling implied by the CDX inventory is only slightly higher (section 1, last table). The 21-day mean for most 2020 units is a mean of a handful of days, some of them reached only through year-ago columns of captures taken in Oct to Nov 2021.
- **Clumping.** Observed days are not spread evenly: many cells come from a few days (the last columns of the route table in section 1 give the largest single day's share of the cells and the number of days with 40 or more units). Clumped days are fine for the within-state deviations tested here when prices are calm, and the check against random days of the same count shows the archive's pattern is not materially worse, but a pattern concentrated at one end of the window would matter more in a window with a trend.
- **The reliability checks are proxies.** Truth is metro prices in 2024 to 2026, not 2020 or 2022: no complete daily metro series exists for those windows. Within-state differences between metros are dominated by persistent price-level differences, which any subset of days recovers, so level reliability is high by construction. The shock version is closer to the design quantity but its price change (2024 to 2026) differs in size and geography from 2020 to 2022 or 2020 to 2024. Whether metros move differently in a given 2020 to 2022 episode cannot be tested here.
- **Metro values are AAA's own averages, restated.** About a fifth of year-ago and month-ago values differ from the originally published daily value by $0.001 (the maximum difference is $0.001). Page dates are taken from the page text, not from the capture time; the flag counts at the top show how often the printed date is a day behind (`stale_page_1d`) or ahead (`page_date_ahead_1d`) of the capture time. Dates assigned through lag columns inherit any error in the page date.
- **Metro identity.** Names are AAA's raw strings. Two spellings differ between the 2020 pages and later pages (one in WV, one in MA); a name key of (state, normalized name) is ambiguous for one metro in each source. Some metros are split by state ("(XX only)") or carry "Entire Metro Average" variants; no county crosswalk was built or tested in this phase, and the metro-to-county step of the D22 route is not evaluated here.
- **Only regular gasoline, one value per page.** Pages show one set of metro tables; metros that AAA adds or removes between years are visible only through the name comparison in section 1.
- **Not tested.** Whether the archived 2020 window prices would shift the primary level under D22; effects on estimates; any outcome data (none were loaded).
- **Data handling (memo D19).** The prices come from Internet Archive copies of AAA pages. The parquet is an interim file and is not redistributed; this report holds aggregates only.
"""


def metro_report() -> Path:
    """Write reports/validation/metro_wayback_coverage.md from data/interim/prices_metro_wayback.parquet and the daily
    metro panel. Aggregates only: no AAA price appears in the report (memo D19)."""
    import polars as pl
    wb = pl.read_parquet(paths.interim_dir() / 'prices_metro_wayback.parquet')
    lyk = pl.read_parquet(paths.interim_dir() / 'prices_metro_daily.parquet')
    states_df = pl.read_csv(paths.config_dir() / 'states.csv', schema_overrides={'state_fips': pl.Utf8})
    units = states_df['state_abbr'].to_list()
    windows = load_windows()
    wb, lyk = _keyed(wb), _keyed(lyk)
    L: list[str] = [f'<!-- role: result snapshot | as-of: {date.today().isoformat()} | written by an agent; regenerated by '
                    '`python -m fb26.scrape.wayback report-metro`; superseded by the next run -->', '',
                    '# Archived AAA metro tables: coverage and reliability (phase 1)', '']

    # --- inputs
    pages = wb.select('raw_sha256').n_unique()
    L.append(f'Inputs: `prices_metro_wayback.parquet` ({wb.height:,} rows, {pages} state-page snapshots, '
             f'{wb["as_of_date"].min()} to {wb["as_of_date"].max()}); `prices_metro_daily.parquet` ({lyk.height:,} rows, '
             f'{lyk["as_of_date"].min()} to {lyk["as_of_date"].max()}). Regular gasoline. Units: the {len(units)} rows of '
             f'`config/states.csv` (50 states and DC).\n')
    L.append('Row flags in the Wayback metro table: ' + ', '.join(f'`{f}` {n}' for f, n in sorted(
        wb.select(pl.col('flags').str.split(';').explode().alias('f')).filter(pl.col('f') != '')['f'].value_counts().iter_rows())) + '.\n')

    # --- 1. coverage
    L.append('## 1. Coverage by election window\n')
    cov_rows, cov_by_year = [], {}
    for year in (2020, 2022, 2024):
        c = metro_window_coverage(wb, windows[year], units)
        cov_by_year[year] = c
        ms = [v for v in c['metros'].values() if v]
        row = dict(window=year, units_with_any_day=sum(v > 0 for v in c['any'].values()),
                   metros_total=wb.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*windows[year]))
                   .select('state_abbr', 'k').unique().height,
                   median_metros_per_unit=_median(ms), median_days_any=_median(list(c['any'].values())),
                   units_ge10=sum(v >= 10 for v in c['any'].values()), units_ge15=sum(v >= 15 for v in c['any'].values()),
                   units_21=sum(v == 21 for v in c['any'].values()))
        for lag in LAG_FIELDS:
            row[f'med_days_{lag}'] = _median(list(c[lag].values()))
        cov_rows.append(row)
    L.append('Wayback metro table, window = [ED-21, ED-1]. "Days" = distinct reference dates with at least one metro price '
             'in the unit, over all 21 window days; medians run over all 51 units, units with no day count as 0. '
             '2024 here is only what the stored archive pages give (not the daily panel).\n')
    L.append(_md(cov_rows) + '\n')
    lyk_w = lyk.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*windows[2024]))
    per_unit = lyk_w.group_by('state_fips').agg(pl.col('as_of_date').n_unique().alias('d'), pl.col('metro_raw').n_unique().alias('m'))
    L.append(f'Reference, daily panel (`prices_metro_daily`), 2024 window: {per_unit.height} units, {lyk_w.select("state_fips", "metro_raw").unique().height} '
             f'metros, median {per_unit["m"].median():.0f} metros per unit, median {per_unit["d"].median():.0f} of 21 days.\n')
    L.append('Days per unit, distribution (any route), number of units at each count:\n')
    dist = {y: sorted(Counter(cov_by_year[y]['any'].values()).items()) for y in cov_by_year}
    L.append(_md([dict(window=y, days_to_units=', '.join(f'{d}:{n}' for d, n in v)) for y, v in dist.items()]) + '\n')
    # snapshots per route
    L.append('Observed (date, unit) cells per window by route (a cell can be reached by several routes):\n')
    route_rows = []
    for year in (2020, 2022, 2024):
        sel = wb.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*windows[year]))
        row = dict(window=year, cells_any=sel.select('state_abbr', 'as_of_date').unique().height)
        for lag in LAG_FIELDS:
            row[f'cells_{lag}'] = sel.filter(pl.col('lag_field') == lag).select('state_abbr', 'as_of_date').unique().height
        only_lag = (sel.group_by('state_abbr', 'as_of_date').agg((pl.col('lag_field') != 'current').all().alias('lag_only'))
                    .filter(pl.col('lag_only')).height)
        row['cells_lag_only'] = only_lag
        per_day = sel.group_by('as_of_date').agg(pl.col('state_abbr').n_unique().alias('n'))
        row.update(window_days_any=per_day.height, window_days_ge40_units=int((per_day['n'] >= 40).sum()),
                   top_day_units=int(per_day['n'].max()), top_day_share_of_cells=float(per_day['n'].max() / per_day['n'].sum()))
        row['cells_possible'] = 51 * 21
        route_rows.append(row)
    L.append(_md(route_rows) + '\n')

    # --- 1b. stability of the metro set
    L.append('### Metro set stability\n')
    sets = {}
    for year in (2020, 2022, 2024):
        s = wb.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*windows[year])).select('state_abbr', 'k').unique()
        sets[f'wb{year}'] = set(s.iter_rows())
    sets['daily2024'] = set(lyk.with_columns(pl.col('state_fips').replace_strict(dict(zip(states_df['state_fips'], units)), default=None).alias('state_abbr'))
                            .select('state_abbr', 'k').unique().iter_rows())
    st_rows = []
    for a, b in (('wb2020', 'wb2022'), ('wb2022', 'wb2024'), ('wb2020', 'wb2024'), ('wb2020', 'daily2024'), ('wb2022', 'daily2024')):
        ua = {s for s, _ in sets[a]}
        ub = {s for s, _ in sets[b]}
        common = ua & ub
        A = {x for x in sets[a] if x[0] in common}
        B = {x for x in sets[b] if x[0] in common}
        per = [len({k for s2, k in A if s2 == s} & {k for s2, k in B if s2 == s}) / max(1, len({k for s2, k in A if s2 == s} | {k for s2, k in B if s2 == s}))
               for s in common]
        st_rows.append(dict(a=a, b=b, units_in_both=len(common), metros_a=len(A), metros_b=len(B), metros_both=len(A & B),
                            jaccard=len(A & B) / len(A | B) if A | B else None,
                            units_identical=sum(p == 1 for p in per), only_a=len(A - B), only_b=len(B - A)))
    L.append('Metro keys (state, normalized name) seen in each window, restricted to units observed in both windows being '
             'compared. `daily2024` is the full daily panel.\n')
    L.append(_md(st_rows) + '\n')
    for a, b in (('wb2020', 'daily2024'), ('wb2022', 'daily2024')):
        ua = {s for s, _ in sets[a]}
        diff_a = sorted(f'{s}:{k}' for s, k in sets[a] - sets[b] if s in {x for x, _ in sets[b]})
        diff_b = sorted(f'{s}:{k}' for s, k in sets[b] - sets[a] if s in ua)
        L.append(f'- {a} vs {b}, in {a} only ({len(diff_a)}): {", ".join(diff_a[:40]) or "none"}{" ..." if len(diff_a) > 40 else ""}\n')
        L.append(f'- {a} vs {b}, in {b} only ({len(diff_b)}): {", ".join(diff_b[:40]) or "none"}{" ..." if len(diff_b) > 40 else ""}\n')

    # --- 2. lag semantics
    L.append('\n## 2. Lag semantics of the metro tables\n')
    L.append('For each lag field, its value (reference date = page date shifted by `LAG_RULES`) is compared with the same metro\'s '
             '`current` value on the reference date shifted by k days; k = 0 is the rule. Within the Wayback metro table, then '
             'against the daily panel. "match" = within $0.0005.\n')
    cur = wb.filter((pl.col('lag_field') == 'current') & pl.col('price_usd').is_not_null()).select('state_abbr', 'k', 'as_of_date', pl.col('price_usd').alias('p_cur'))
    lyk_cur = lyk.filter(pl.col('price_usd').is_not_null()).with_columns(
        pl.col('state_fips').replace_strict(dict(zip(states_df['state_fips'], units)), default=None).alias('state_abbr')
    ).group_by('state_abbr', 'k', 'as_of_date').agg(pl.col('price_usd').first().alias('p_cur'), pl.len().alias('n_dup'))
    sem_rows = []
    for ref_name, ref in (('wayback current', cur), ('daily panel', lyk_cur.filter(pl.col('n_dup') == 1).drop('n_dup'))):
        for lag in LAG_FIELDS[1:]:
            lagged = wb.filter((pl.col('lag_field') == lag) & pl.col('price_usd').is_not_null())
            for k in (-2, -1, 0, 1, 2):
                j = lagged.with_columns((pl.col('as_of_date') + pl.duration(days=k)).alias('ref_date')).join(
                    ref.rename({'as_of_date': 'ref_date'}), on=['state_abbr', 'k', 'ref_date'], how='inner')
                n = j.height
                d = (j['price_usd'] - j['p_cur']).abs() if n else None
                sem_rows.append(dict(reference=ref_name, lag_field=lag, shift_days=k, pairs=n,
                                     match_share=float((d < MATCH_TOL).mean()) if n else None,
                                     mean_abs_diff=float(d.mean()) if n else None))
    L.append(_md(sem_rows) + '\n')

    # --- 3. agreement with the daily panel
    L.append('## 3. Agreement with the daily metro panel (`prices_metro_daily`)\n')
    names_wb = wb.select('state_abbr', 'metro_raw', 'k').unique()
    names_ly = lyk.with_columns(pl.col('state_fips').replace_strict(dict(zip(states_df['state_fips'], units)), default=None).alias('state_abbr')
                                ).select('state_abbr', 'metro_raw', 'k').unique()
    dup_wb = names_wb.group_by('state_abbr', 'k').len().filter(pl.col('len') > 1).select('state_abbr', 'k')
    dup_ly = names_ly.group_by('state_abbr', 'k').len().filter(pl.col('len') > 1).select('state_abbr', 'k')
    both = names_wb.select('state_abbr', 'k').unique().join(names_ly.select('state_abbr', 'k').unique(), on=['state_abbr', 'k'], how='inner')
    only_wb = names_wb.join(names_ly, on=['state_abbr', 'k'], how='anti')
    only_ly = names_ly.join(names_wb, on=['state_abbr', 'k'], how='anti')
    # restrict the unmatched-name lists to states and dates where the other source could have the metro
    n_wb = names_wb.select('state_abbr', 'k').unique().height
    L.append(f'Name matching: within state, on the normalized name of `metro_prices.normalize` (case, accents, "Saint"/"St", '
             f'"(XX only)" suffixes removed). Wayback metro keys {n_wb}; matched to the daily panel {both.height} '
             f'({_share(both.height, n_wb)}); Wayback-only {only_wb.height}; daily-panel-only {only_ly.height}; keys ambiguous '
             f'within a state (dropped from the comparison): Wayback {dup_wb.height}, daily panel {dup_ly.height}.\n')
    L.append('Wayback-only names (raw, with state): ' + (', '.join(sorted(f'{r["metro_raw"]} ({r["state_abbr"]})' for r in only_wb.iter_rows(named=True))[:60]) or 'none') + '.\n')
    L.append('Daily-panel-only names (raw, with state; only states that appear in the Wayback table): ' +
             (', '.join(sorted(f'{r["metro_raw"]} ({r["state_abbr"]})' for r in only_ly.filter(pl.col('state_abbr').is_in(wb['state_abbr'].unique().implode())).iter_rows(named=True))[:80]) or 'none') + '.\n')
    ly_ok = lyk.with_columns(pl.col('state_fips').replace_strict(dict(zip(states_df['state_fips'], units)), default=None).alias('state_abbr')
                             ).filter(pl.col('price_usd').is_not_null()).join(dup_ly.with_columns(pl.lit(True).alias('dup')), on=['state_abbr', 'k'], how='left'
                                                                              ).filter(pl.col('dup').is_null()).select(
        'state_abbr', 'k', 'as_of_date', pl.col('price_usd').alias('p_ly'))
    wb_ok = wb.filter(pl.col('price_usd').is_not_null()).join(dup_wb.with_columns(pl.lit(True).alias('dup')), on=['state_abbr', 'k'], how='left').filter(pl.col('dup').is_null())
    j = wb_ok.join(ly_ok, on=['state_abbr', 'k', 'as_of_date'], how='inner').with_columns((pl.col('price_usd') - pl.col('p_ly')).abs().alias('ad'))
    ag_rows = []
    for lag in (*LAG_FIELDS, 'all'):
        jj = j if lag == 'all' else j.filter(pl.col('lag_field') == lag)
        n = jj.height
        ag_rows.append(dict(lag_field=lag, pairs=n, dates=jj['as_of_date'].n_unique(), metros=jj.select('state_abbr', 'k').unique().height,
                            exact_share=float((jj['ad'] < MATCH_TOL).mean()) if n else None,
                            within_1c=float((jj['ad'] < 0.0105).mean()) if n else None,
                            mean_abs_diff=float(jj['ad'].mean()) if n else None, max_abs_diff=float(jj['ad'].max()) if n else None))
    L.append('Same metro, same reference date, Wayback value (via the lag field) against the daily panel:\n')
    L.append(_md(ag_rows) + '\n')
    wb_dates = wb_ok['as_of_date'].unique()
    L.append(f'Panel overlap: Wayback reference dates inside the daily panel\'s range: {wb_dates.is_between(lyk["as_of_date"].min(), lyk["as_of_date"].max()).sum()} of {wb_dates.len()} '
             f'distinct dates; pairs found for {j["as_of_date"].n_unique()} of them (the panel is patchy after 2024-12).\n')

    # --- 3b. consistency inside the archive, in the two historical windows
    L.append('### Consistency inside the archive (2020 and 2022 windows)\n')
    L.append('Metro-days in the window that more than one stored value reaches (different captures or lag fields); spread = max minus min '
             'of the values for the same metro and reference date.\n')
    inner = []
    for year in (2020, 2022):
        sel = wb.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*windows[year]))
        g = (sel.group_by('state_abbr', 'metro_raw', 'as_of_date').agg((pl.col('price_usd').max() - pl.col('price_usd').min()).alias('sp'),
                                                           pl.len().alias('n'), pl.col('lag_field').n_unique().alias('routes'))
             .filter(pl.col('n') > 1))
        inner.append(dict(window=year, metro_days_total=sel.select('state_abbr', 'metro_raw', 'as_of_date').unique().height, metro_days_multi=g.height,
                          units=g['state_abbr'].n_unique(), spread_zero=float((g['sp'] < MATCH_TOL).mean()) if g.height else None,
                          spread_le_0_0015=float((g['sp'] < 0.0015).mean()) if g.height else None,
                          mean_spread=float(g['sp'].mean()) if g.height else None, max_spread=float(g['sp'].max()) if g.height else None))
    L.append(_md(inner) + '\n')

    # --- accounting
    L.append('### Fetch accounting and the ceiling of what the archive lists\n')
    L.append(_fetch_accounting(windows) + '\n')

    # --- 4. sparse-day reliability
    L.append('## 4. Sparse-day reliability\n')
    patterns = {f'{y} days': sparse_positions(wb, windows[y], units) for y in (2020, 2022)}
    level_rows, shock_rows, info = sparse_reliability(lyk.with_columns(pl.col('state_fips').alias('state_fips')), patterns, states_df)
    L.append(f'Truth: the complete daily 2024 window {LYK_FULL_WINDOW[0]} to {LYK_FULL_WINDOW[1]} ({info["n_metros_daily_2024"]} metros with all 21 days), '
             f'plus, as a stress test, the most volatile complete 21-day window of the daily panel ({info["volatile"][0]} to {info["volatile"][1]}; '
             f'SD of the daily mean {info["volatile"][2]:.3f} against '
             f'{statistics.pstdev(_daily_mean(lyk, LYK_FULL_WINDOW)):.3f} in the election window). '
             f'For each state, the day offsets (0 = window start) on which the Wayback metro table has any price in the 2020 (2022) '
             f'window are the "archive days"; each metro\'s window mean is recomputed from those days of the truth window only. '
             f'Within-state deviation = metro mean minus the unweighted mean over the state\'s metros; states with one metro carry no '
             f'deviation and drop out. reliability = var(full deviation) / (var(full deviation) + var(sparse minus full)), the same form as '
             f'`thin_windows.py` (signal SD squared over signal plus error). States with no archive day in the window cannot be evaluated: ' +
             ', '.join(f'{k}: {v["states_with_days"]} states evaluated, {v["states_without"]} without a day' for k, v in info.items() if k.endswith('days')) + '.\n')
    L.append('Deviations from `thin_windows.py`: (1) the unit is a metro within its state (that script works on states within PADD groups); '
             '(2) the day pattern is state-specific and taken from the archive, where `thin_windows.py` uses one pattern for all states; '
             '(3) the truth is one daily window (2024) of metro prices, not three; (4) the random-days rows draw, per state, as many days as the archive has '
             '(`thin_windows.py` draws fixed k = 2 and 5); (5) states with no archive day drop out here, rather than being filled.\n')
    L.append('Level (window mean):\n')
    L.append(_md(level_rows) + '\n')
    L.append('Shock version, supplementary: target = a complete 21-day window of the daily panel (' +
             '; '.join(f'{k}: {v[0]} to {v[1]}' for k, v in LYK_TARGETS.items()) + '), baseline = the 2024 election window; '
             f'shock = target mean minus baseline mean (`mean_change` is its average over the metros used); truth uses all 21 days on both sides; '
             f'{info["shock_states"]} states have days in both patterns. Within-state deviation of the shock, as above. The 2022-type row '
             'applies the 2020 pattern to the baseline and the 2022 pattern to the target.\n')
    L.append(_md(shock_rows) + '\n')

    # --- limitations
    L.append(METRO_LIMITATIONS)

    # --- summary, from the tables above
    def pick(rows, **kw):
        return next(r for r in rows if all(r[k] == v for k, v in kw.items()))
    summary = ['## Summary\n']
    for r in cov_rows:
        cells = next(x for x in route_rows if x['window'] == r['window'])
        summary.append(f"- {r['window']} window: {r['units_with_any_day']} of 51 units have at least one metro day, median {r['median_days_any']} of 21 days per unit, "
                       f"{r['units_ge10']} units with 10 or more days; {cells['cells_any']} of {cells['cells_possible']} unit-days "
                       f"({100 * cells['cells_any'] / cells['cells_possible']:.0f}%), {cells['cells_lag_only']} of them only through lag columns.")
    sem0 = {(r['reference'], r['lag_field']): r for r in sem_rows if r['shift_days'] == 0}
    sem_other = [r['match_share'] for r in sem_rows if r['shift_days'] != 0 and r['reference'] == 'daily panel' and r['match_share'] is not None]
    summary.append('- Lag semantics: the metro tables follow the state-row rule. Against the daily panel the rule (shift 0) matches ' +
                   ', '.join(f"{lag} {100 * sem0[('daily panel', lag)]['match_share']:.0f}%" for lag in LAG_FIELDS[1:]) +
                   f' of metro-days; the best of the shifted readings (-2 to +2 days) reaches {100 * max(sem_other):.0f}%.')
    allr = pick(ag_rows, lag_field='all')
    summary.append(f"- Agreement with the daily panel on {allr['pairs']:,} metro-days ({allr['dates']} dates, {allr['metros']} metros): exact {100 * allr['exact_share']:.1f}%, "
                   f"mean absolute difference ${allr['mean_abs_diff']:.4f}, maximum ${allr['max_abs_diff']:.3f}; name match {both.height} of {n_wb} Wayback metro keys.")
    for tw in dict.fromkeys(r['truth_window'] for r in level_rows):
        for pat in ('2020 days', '2022 days'):
            r = pick(level_rows, truth_window=tw, pattern=pat, design='archive days')
            summary.append(f"- Sparse days, {pat}, truth = {tw}: within-state deviation of the window mean, correlation {r['corr']:.3f}, reliability {r['reliability']:.3f} "
                           f"(SD of error ${r['sd_error']:.3f} against SD of signal ${r['sd_truth']:.3f}; {r['n_states']} states, {r['n_metros']} metros).")
    for r in shock_rows:
        summary.append(f"- Sparse days, shock to {r['target']} (mean change ${r['mean_change']:.2f}), {r['cycle']}: correlation {r['corr']:.3f}, reliability {r['reliability']:.3f}.")
    L.insert(2, '\n'.join(summary) + '\n')

    out = paths.repo_root() / 'reports' / 'validation' / 'metro_wayback_coverage.md'
    out.write_text('\n'.join(L))
    import uuid
    from fb26.util import runlog
    runlog.append_run_log(run_id=uuid.uuid4().hex[:12], utc_timestamp=rawstore.utc_iso(datetime.now(timezone.utc)),
                          git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(),
                          config_hash=runlog.config_hash([paths.config_dir() / 'windows.yaml', paths.config_dir() / 'states.csv']),
                          spec_id='diag_metro_wayback', data_snapshot_ids='prices_metro_wayback.parquet;prices_metro_daily.parquet',
                          output_path='reports/validation/metro_wayback_coverage.md',
                          notes=f'seed={SEED}; draws={N_RANDOM_DRAWS}; {METRO_PARSER_VERSION}; treatment data only')
    return out


# ---------------------------------------------------------------- command line

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _wayback_df():
    import polars as pl
    rows, _ = parse_raw_store()
    return pl.DataFrame(rows, schema=_schema()) if rows else None


def _schema():
    import polars as pl
    return {'source': pl.String, 'as_of_date': pl.Date, 'state_abbr': pl.String, 'state_fips': pl.String,
            'product': pl.String, 'price_usd': pl.Float64, 'lag_field': pl.String,
            'derived_from_lag_field': pl.Boolean, 'raw_file': pl.String, 'raw_sha256': pl.String,
            'capture_ref': pl.String, 'captured_at_utc': pl.Datetime('us', 'UTC'),
            'parser_version': pl.String, 'flags': pl.String}


def cmd_inventory() -> None:
    client = PoliteClient()
    inv = inventory(client)
    path = write_coverage(inv)
    _log(f'wrote {path}')


LAYOUT_PAGE_TYPE = {'gp_sga_v1': 'gp_sga', 'gp_home_v1': 'gp_home', 'gp_state_v1': 'gp_state', 'fgc_sbs_v1': 'fgc_sbs'}


def page_type_of(layout: str, original: str) -> str:
    if layout in LAYOUT_PAGE_TYPE:
        return LAYOUT_PAGE_TYPE[layout]
    host = original.split('//', 1)[-1].lower()
    return 'fg_state' if 'aaa.com' in host else 'op_state' if 'opisnet' in host else 'fgc_state'


def write_coverage(inv: dict) -> Path:
    """reports/validation/wayback_coverage.csv: window x page type x route from the CDX index, joined with what was
    downloaded and parsed (raw store) and with the third-party GitHub scrapers for comparison."""
    import polars as pl
    rows, report = all_rows()
    df = pl.DataFrame(rows, schema=_schema())
    wb = df.filter(pl.col('source') == SOURCE)
    # page type per Wayback row, through the capture timestamp in the parse report
    ptype_of_capture = {}
    for r in report:
        if r['layout']:
            ptype_of_capture.setdefault(r['capture_ref'], set()).add(page_type_of(r['layout'], r['original']))
    wb = wb.with_columns(pl.col('capture_ref').map_elements(lambda c: sorted(ptype_of_capture.get(c, {'?'}))[0],
                                                              return_dtype=pl.String).alias('ptype'))
    windows = load_windows()
    out = paths.repo_root() / 'reports' / 'validation' / 'wayback_coverage.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['window', 'page_type', 'route', 'snapshot_range_from', 'snapshot_range_to', 'cdx_captures',
              'snapshot_days', 'snapshot_days_ge51_states', 'fetched_parsed_snapshots', 'days_all51_direct',
              'lag_cells_regular', 'lag_days_all51', 'usable_days_all51', 'cells_regular_of_1071', 'window_days',
              'flag', 'notes']
    national_only = {'fg_home', 'op_home'}
    with out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (year, ptype, route), snaps in inv.items():
            if not snaps and route != 'direct':
                continue  # nothing archived in a lag-route range: the direct row already shows the page type is absent
            if ptype in national_only:
                notes = 'national average only; no state prices in the static HTML'
            else:
                notes = ''
            window = windows[year]
            frm, to = lag_snapshot_range(window, route)
            by_day = days_by_state(snaps)
            per_state_page = ptype in STATE_PAGE_TYPES
            ge51 = sum(len(v - {None}) >= 51 for v in by_day.values()) if per_state_page else len(by_day)
            ours = wb.filter(pl.col('ptype') == ptype)
            fetched = ours.filter(pl.col('capture_ref').str.slice(0, 8).is_between(pl.lit(f'{frm:%Y%m%d}'), pl.lit(f'{to:%Y%m%d}')))
            fetched_n = fetched['capture_ref'].n_unique() if fetched.height else 0
            direct = lagged = cells = ''
            if route == 'direct' and ptype in ('gp_sga', 'gp_home', 'gp_state', 'fgc_sbs'):
                direct = len(full_days(ours.filter(pl.col('lag_field') == 'current'), window, derived=False))
            if per_state_page:
                lag_rows = ours.filter(pl.col('derived_from_lag_field') & (pl.col('product') == 'regular')
                                       & pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_between(*window))
                if route != 'direct':
                    fld = lag_rows.filter(pl.col('lag_field') == route)
                    cells = fld.select('as_of_date', 'state_abbr').unique().height
                    lagged = len(full_days(fld, window, derived=True))
            writer.writerow(dict(
                window=year, page_type=ptype, route=route, snapshot_range_from=frm, snapshot_range_to=to,
                cdx_captures=len(snaps), snapshot_days=len(by_day), snapshot_days_ge51_states=ge51,
                fetched_parsed_snapshots=fetched_n, days_all51_direct=direct, lag_cells_regular=cells,
                lag_days_all51=lagged, window_days=(window[1] - window[0]).days + 1, notes=notes))
        for year, window in windows.items():  # combined Wayback result and the repos, per window
            mine = wb.filter(pl.col('as_of_date').is_between(*window))
            direct_days = full_days(wb.filter(pl.col('lag_field') == 'current'), window, derived=False)
            usable = full_days(wb, window, derived=None)
            cells = (mine.filter((pl.col('product') == 'regular') & pl.col('price_usd').is_not_null())
                     .select('as_of_date', 'state_abbr').unique().height)
            n_days = (window[1] - window[0]).days + 1
            flag = ('FALLBACK_EIA: fewer than 3 usable days' if len(usable) < 3 else
                    'FALLBACK_EIA: fewer than 15 usable days' if len(usable) < 15 else '')
            writer.writerow(dict(window=year, page_type='ALL_WAYBACK', route='combined', days_all51_direct=len(direct_days),
                                 lag_days_all51=len(usable - direct_days), usable_days_all51=len(usable),
                                 cells_regular_of_1071=cells, window_days=n_days, flag=flag,
                                 notes='51 states, regular gasoline, wayback_aaa rows only'))
            for tag in ('lykmapipo_us_gas_prices', 'nathanqthai_gas_prices'):
                got = df.filter(pl.col('source') == tag)
                days = full_days(got, window, derived=False)
                writer.writerow(dict(window=year, page_type=f'repo:{tag}', route='direct', days_all51_direct=len(days),
                                     usable_days_all51=len(days), window_days=n_days,
                                     notes='third-party GitHub scraper (not Wayback)'))
    return out


def cmd_fetch_direct(year: int, ptype: str, missing_only: bool = False) -> None:
    """One capture per expected page date. With missing_only, only page dates for which the parsed Wayback data
    do not yet hold all 51 states (regular gasoline)."""
    client = PoliteClient()
    window = load_windows()[year]
    snaps = plan_direct(cdx(client, ptype, window[0], window[1] + timedelta(days=1)))
    snaps = [s for s in snaps if window[0] <= s.page_day <= window[1]]
    if missing_only:
        df = _wayback_df()
        done = full_days(df, window, derived=None) if df is not None else set()
        snaps = [s for s in snaps if s.page_day not in done]
    _log(f'{ptype} {year}: {len(snaps)} captures to fetch')
    res = fetch_snapshots(client, snaps)
    _log({k: len(v) for k, v in res.items()})


def cmd_fetch_lag(year: int, max_requests: int | None) -> None:
    client = PoliteClient()
    window = load_windows()[year]
    candidates = cdx_lag_candidates(client, window, LAG_PAGE_TYPES[year])
    df = _wayback_df()
    cells = {(d, s) for d in window_days(window) for s in load_state_abbrs()}
    if df is not None:
        import polars as pl
        got = df.filter((pl.col('product') == 'regular') & pl.col('price_usd').is_not_null()
                        & pl.col('as_of_date').is_between(*window)).select('as_of_date', 'state_abbr').unique()
        cells -= set(got.iter_rows())
    snaps = plan_lag(candidates, window, cells, max_requests)
    _log(f'{year}: {len(cells)} (date, state) cells missing; {len(snaps)} state-page captures planned')
    for i, snap in enumerate(snaps, 1):
        res = fetch_snapshots(client, [snap])
        if i % 25 == 0 or i == len(snaps):
            _log(f'{i}/{len(snaps)} {snap.timestamp} {snap.state} {dict((k, len(v)) for k, v in res.items())}')


def safe_page_day(snap: Snap) -> date | None:
    """Page date of a capture when it is not in doubt: AAA's date turns over between about 07:00 and 12:00 UTC
    (S02 audit, section 8), so captures from 12:00 UTC on show the same UTC date, captures before 07:00 show the
    previous one, and captures in between (page date unknown) give None."""
    hour = int(snap.timestamp[8:10])
    if hour >= 12:
        return snap.day
    return snap.day - timedelta(days=1) if hour < 7 else None


def covered_cells(start: date, end: date) -> set[tuple[date, str]]:
    """(date, state) cells with a regular-gasoline price in the parsed Wayback data or the GitHub-scraper table."""
    import polars as pl
    frames = []
    wb = _wayback_df()
    if wb is not None:
        frames.append(wb)
    github = paths.interim_dir() / 'prices_state_daily_aaagas.parquet'
    if github.exists():
        frames.append(pl.read_parquet(github))
    cells: set[tuple[date, str]] = set()
    for df in frames:
        got = (df.filter((pl.col('product') == 'regular') & pl.col('price_usd').is_not_null()
                         & pl.col('as_of_date').is_between(start, end)).select('as_of_date', 'state_abbr').unique())
        cells |= set(got.iter_rows())
    return cells


def cmd_fetch_range(start: date, end: date, *, lag: bool = False, dry_run: bool = False) -> None:
    """Fill the (date, state) cells in [start, end] that neither the parsed Wayback data nor the GitHub scrapers hold.
    Step 1: one state-averages (else home-page) capture per page date that has any missing cell. Step 2 (lag=True):
    a greedy cover of what is still missing with state-page captures through their lag rows. dry_run lists the plan."""
    client = PoliteClient()
    states = load_state_abbrs()
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    have = covered_cells(start, end)
    missing = {(d, s) for d in days for s in states if (d, s) not in have}
    _log(f'{start}..{end}: {len(missing)} (date, state) cells missing on {len({d for d, _ in missing})} days')
    if lag:
        snaps = plan_lag(cdx_lag_candidates(client, (start, end), ('gp_state',)), (start, end), missing)
        _log(f'{len(snaps)} state-page captures planned')
    else:
        need = {d for d, _ in missing}
        chosen: dict[date, Snap] = {}
        for ptype in ('gp_sga', 'gp_home'):
            by_day: dict[date, list[Snap]] = {}
            for snap in cdx(client, ptype, start - timedelta(days=1), end + timedelta(days=1)):
                day = safe_page_day(snap)
                if day in need and day not in chosen:
                    by_day.setdefault(day, []).append(snap)
            for day, cands in by_day.items():
                # afternoon/evening UTC captures first (well after the page date turns over), then the latest
                chosen[day] = max(cands, key=lambda sn: (sn.timestamp[8:10] >= '12', sn.timestamp))
        snaps = sorted(chosen.values(), key=lambda sn: sn.timestamp)
        _log(f'{len(snaps)} page captures planned; days with no listed capture: '
             f'{sorted(need - set(chosen))}')
    if dry_run:
        return
    res = fetch_snapshots(client, snaps)
    _log({k: len(v) for k, v in res.items()})
    for snap in res['failed']:
        _log(f'  failed {snap.timestamp} {snap.original}')


def load_state_abbrs() -> list[str]:
    with (paths.config_dir() / 'states.csv').open(newline='') as handle:
        return [row['state_abbr'] for row in csv.DictReader(handle)]


OTHER_REPOS = {  # raw-store source -> (source tag, reader name)
    'lykmapipo_US-Gas-Prices_repo': ('lykmapipo_us_gas_prices', 'state_daily'),
    'nathanqthai_gas_prices_repo': ('nathanqthai_gas_prices', 'national'),
}


def repo_rows(raw_source: str, tmp: Path, *, states: dict | None = None) -> list[dict]:
    """Rows from the newest stored bundle of a scraper repo. The bundle is unpacked into `tmp` (outside the
    project) and read with `git --git-dir`; nothing in it is executed."""
    import gzip
    import subprocess
    metas = sorted((paths.raw_dir() / raw_source).glob('**/*.meta.json'))
    if not metas:
        return []
    meta = json.loads(metas[-1].read_text())
    raw_path = Path(meta['raw_path'])
    bundle = tmp / f'{raw_source}.bundle'
    bundle.write_bytes(gzip.decompress(raw_path.read_bytes()))
    git_dir = tmp / f'{raw_source}.git'
    subprocess.run(['git', 'clone', '--bare', '-q', str(bundle), str(git_dir)], check=True, capture_output=True)
    tag, kind = OTHER_REPOS[raw_source]
    rel = str(raw_path.relative_to(paths.repo_root()))
    if kind == 'state_daily':
        return repo_state_daily_rows(git_dir, source=tag, path_prefix='data/state-daily-averages', raw_file=rel,
                                     raw_sha256=meta['sha256'], states=states)
    return repo_national_rows(git_dir, source=tag, path_prefix='prices/national', raw_file=rel,
                              raw_sha256=meta['sha256'], last_date=date(2026, 3, 27), states=states)


def all_rows() -> tuple[list[dict], list[dict]]:
    import tempfile
    rows, report = parse_raw_store()
    states = load_states()
    with tempfile.TemporaryDirectory(prefix='fb26-repos-') as tmp:
        for raw_source in OTHER_REPOS:
            made = repo_rows(raw_source, Path(tmp), states=states)
            _log(f'{raw_source}: {len(made)} rows')
            rows.extend(made)
    return rows, report


def cmd_parse() -> None:
    """Writes wayback_aaa rows only. The GitHub scrapers (lykmapipo, nathanqthai, ...) are mirrored and parsed by
    ws1-repo into prices_state_daily_aaagas.parquet; repo_rows() here is for read-only checks and coverage."""
    rows, report = parse_raw_store()
    failed = [r for r in report if r['error']]
    _log(f'{len(report)} snapshots, {len(failed)} unparsed, {len(rows)} rows')
    for r in failed[:10]:
        _log(f"  unparsed {r['capture_ref']} {r['original']}: {r['error']}")
    _log(str(write_parquet(rows)))


def cmd_fetch_metro(year: int, max_requests: int | None, dry_run: bool = False) -> None:
    """State-page captures that give metro prices for dates in the window of `year` (direct, week-ago, month-ago and
    year-ago routes), by greedy cover of the (date, state) cells that no stored page supplies. States whose stored pages
    never show a metro table are left out. Snapshots already in the raw store are skipped."""
    client = PoliteClient()
    window = load_windows()[year]
    rows, _ = parse_raw_store_metro()
    have = {(r['as_of_date'], r['state_abbr']) for r in rows if r['price_usd'] is not None}
    with_metros = {r['state_abbr'] for r in rows}
    cells = {(d, s) for d in window_days(window) for s in with_metros} - have
    bad = failed_urls()  # e.g. captures that redirect: planning must fall back to another capture of that day
    snaps = plan_lag([s for s in cdx_lag_candidates(client, window, ('gp_state',)) if snapshot_url(s) not in bad],
                     window, cells, max_requests)
    new = [s for s in snaps if snapshot_url(s) not in raw_snapshots()]
    _log(f'{year}: {len(cells)} (date, state) cells without metro prices; {len(snaps)} captures planned, {len(new)} not stored')
    if dry_run:
        return
    for i, snap in enumerate(new, 1):
        res = fetch_snapshots(client, [snap])
        if res['failed'] or i % 25 == 0 or i == len(new):
            _log(f'{i}/{len(new)} {snap.timestamp} {snap.state} {dict((k, len(v)) for k, v in res.items())}')


def cmd_parse_metro() -> None:
    rows, report = parse_raw_store_metro()
    failed = [r for r in report if r['error']]
    _log(f'{len(report)} state pages, {len(failed)} unparsed, {len(rows)} rows')
    for r in failed[:10]:
        _log(f"  unparsed {r['capture_ref']} {r['original']}: {r['error']}")
    _log(str(write_metro_parquet(rows)))


def cmd_overlap(n: int) -> None:
    """Fetch n state-averages captures after 2024-11-16 (evenly spread) and compare their 51 regular prices with
    the GitHub tables already in data/interim/prices_state_daily_aaagas.parquet."""
    import polars as pl
    client = PoliteClient()
    snaps = cdx(client, 'gp_sga', date(2024, 11, 16), date(2026, 10, 17))
    pick = evenly_spaced(snaps, n)
    _log(f'{len(snaps)} captures in range; fetching {len(pick)}')
    fetch_snapshots(client, pick)
    rows, _ = parse_raw_store()
    mine = [r for r in rows if r['capture_ref'] in {s.timestamp for s in pick}]
    github = pl.read_parquet(paths.interim_dir() / 'prices_state_daily_aaagas.parquet')
    for product in PRODUCTS:
        res = compare_tables(mine, github, product=product)
        _log(f"{product}: {res['n_pairs']} pairs on {res['n_dates']} dates, exact share {res['exact_share']}, "
             f"max abs diff {res['max_abs_diff']}")
    res = compare_tables(mine, github, product='regular')
    print(res['table'].group_by('as_of_date').agg(pl.len(), (pl.col('absdiff') < 0.00051).mean().alias('exact'),
                                                  pl.col('absdiff').max().alias('max')).sort('as_of_date'))
    print(res['worst'])


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else ''
    if cmd == 'inventory':
        cmd_inventory()
    elif cmd == 'fetch-direct':
        cmd_fetch_direct(int(argv[2]), argv[3] if len(argv) > 3 else 'gp_sga', 'missing' in argv[4:])
    elif cmd == 'fetch-lag':
        cmd_fetch_lag(int(argv[2]), int(argv[3]) if len(argv) > 3 else None)
    elif cmd == 'fetch-range':
        cmd_fetch_range(date.fromisoformat(argv[2]), date.fromisoformat(argv[3]),
                        lag='lag' in argv[4:], dry_run='dry' in argv[4:])
    elif cmd == 'parse':
        cmd_parse()
    elif cmd == 'fetch-metro':
        cmd_fetch_metro(int(argv[2]), int(argv[3]) if len(argv) > 3 and argv[3].isdigit() else None, 'dry' in argv[3:])
    elif cmd == 'parse-metro':
        cmd_parse_metro()
    elif cmd == 'report-metro':
        _log(str(metro_report()))
    elif cmd == 'overlap':
        cmd_overlap(int(argv[2]) if len(argv) > 2 else 12)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv)
