"""County fuel-cost exposure E_c, Oil_c and the C2 controls (WS3; memo §5.2, §5.4, §6.2; DECISIONS 2026-10-19).

    E_c = 100 x G_c / Y_c,  G_c = VMT_hh / MPG / 12,  Y_c = median household income / 12

Two household-VMT measures (long by `g_method`):
  commute_vehicles (primary):  commute_VMT_hh + vehicles_hh x M_nc
  vehicles (R19):              vehicles_hh x M_veh

Commute VMT: (T_alone + T_carpool / occ) / 60 x speed x 2 x days / households, T = one-way aggregate minutes. B08136
is null for about 57% of counties, so T comes from B08134 bin midpoints (open bin fitted to B08136 where it exists).
`days` is solved so commute VMT is the NHTS share of household VMT nationally (config/exposure.yaml).

    python -m fb26.etl.exposure            # build parquet + summary csv + run log
    python -m fb26.etl.exposure checks     # printed sanity checks
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import io
import math
import sys
import uuid
import zipfile

from lxml import etree
import numpy as np
import polars as pl
import yaml

from fb26.scrape import acs, cbp
from fb26.scrape.eia import latest_raw
from fb26.util import paths, rawstore, runlog

CODE_VERSION = 'exposure-0.1.0'
PRIMARY, ALTERNATIVE = 'commute_vehicles', 'vehicles'
VM1_URL = 'https://www.fhwa.dot.gov/policyinformation/statistics/2024/xls/vm1.xlsx'
OUT_NAME = 'exposure_county.parquet'
SUMMARY_NAME = 'exposure_summary.csv'
CONTROLS = ['ba_share', 'hisp_share', 'nh_black_share', 'log_pop_density', 'median_age', 'log_median_income']
OUTPUT_COLUMNS = ['geo_id', 'state_fips', 'acs_vintage', 'households', 'vehicles_hh', 'commute_vmt_hh', 'vmt_hh',
                  'g_gal_hh_month', 'g_method', 'median_hh_income', 'e_c', *CONTROLS, 'oil_gas_emp_share',
                  'oil_gas_flags', 'reliability_flag', 'flags', 'median_hh_income_cv', 'commute_vmt_hh_b08136',
                  'oil_gas_state_unpub_estab']


def load_config() -> dict:
    return yaml.safe_load((paths.config_dir() / 'exposure.yaml').read_text())


# ------------------------------------------------------------------------------------------------ FHWA VM-1

def parse_vm1(content: bytes, year: int = 2024) -> dict:
    """Light-duty column of Table VM-1 for `year`: VMT (million miles), registrations, miles per vehicle, MPG."""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        shared = [''.join(si.itertext()) for si in etree.fromstring(zf.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        sheet = etree.fromstring(zf.read('xl/worksheets/sheet1.xml'))
    found = {}
    for row in sheet.iter('{%s}row' % ns['m']):
        cells = {}
        for c in row.findall('m:c', ns):
            v = c.find('m:v', ns)
            if v is not None:
                cells[''.join(ch for ch in c.get('r') if ch.isalpha())] = shared[int(v.text)] if c.get('t') == 's' else v.text
        if cells.get('A', '').strip() != str(year):
            continue
        label = cells.get('B', '').strip()
        for key, test in (('vmt_million', label.startswith('Total Rural and Urban')),
                          ('registrations', label == 'Number of motor vehicles'),
                          ('miles_per_vehicle', label == 'Average miles traveled'),
                          ('fuel_thousand_gal', label == 'Fuel consumed'),
                          ('mpg', label == 'Average miles traveled per')):
            if test:
                found[key] = float(cells['I'])  # column I = All Light Duty Vehicles
    missing = {'vmt_million', 'registrations', 'miles_per_vehicle', 'fuel_thousand_gal', 'mpg'} - set(found)
    if missing:
        raise ValueError(f'VM-1 rows not found for {year}: {sorted(missing)}')
    return found


def vm1_constants(cfg: dict, content: bytes | None = None) -> dict:
    """VM-1 constants with M_veh = light-duty VMT / registrations; guards against the values recorded in config."""
    if content is None:
        content = rawstore.load_raw(latest_raw('fhwa', VM1_URL).path)
    vm1 = parse_vm1(content)
    out = {'m_veh': vm1['vmt_million'] * 1e6 / vm1['registrations'], 'mpg': vm1['mpg'],
           'ld_fuel_thousand_gal': vm1['fuel_thousand_gal'], 'ld_vmt_million': vm1['vmt_million']}
    exp = cfg['fhwa_vm1']['expected']
    for key, got in (('light_duty_vmt_million_miles', vm1['vmt_million']), ('light_duty_registrations', vm1['registrations']),
                     ('mpg', vm1['mpg'])):
        if not math.isclose(got, exp[key], rel_tol=1e-6):
            raise ValueError(f'VM-1 {key} = {got} differs from config expected {exp[key]}')
    if not math.isclose(out['m_veh'], vm1['miles_per_vehicle'], rel_tol=1e-6):
        raise ValueError('VM-1 VMT / registrations does not match the sheet average miles per vehicle')
    return out


# --------------------------------------------------------------------------------------- commute minutes

def _bins(df: pl.DataFrame, kind: str) -> np.ndarray:
    return np.column_stack([df[c].fill_null(0).to_numpy() for c in acs.BIN_COLUMNS[kind]]).astype(float)


def fit_open_bin(df: pl.DataFrame, closed_mids: list[float]) -> tuple[float, dict]:
    """Least-squares value of the 60+ bin so that bin minutes reproduce B08136 car minutes (alone + carpool) over
    counties where B08136 is published. Returns the value and fit diagnostics."""
    ok = df.filter(pl.col('B08136_003E').is_not_null() & pl.col('B08136_004E').is_not_null())
    n = _bins(ok, 'alone') + _bins(ok, 'carpool')
    y = (ok['B08136_003E'] + ok['B08136_004E']).to_numpy()
    base = n[:, :8] @ np.array(closed_mids)
    m60 = float(n[:, 8] @ (y - base) / (n[:, 8] @ n[:, 8]))
    pred = base + n[:, 8] * m60
    nz = y > 0
    return m60, {'n_counties': int(ok.height), 'aggregate_error': float(pred.sum() / y.sum() - 1),
                 'county_mape': float(np.mean(np.abs(pred[nz] / y[nz] - 1))),
                 'corr': float(np.corrcoef(pred, y)[0, 1])}


def carpool_occupancy(df: pl.DataFrame, midpoints: dict) -> float:
    """Workers per carpool vehicle. B08301_005..009 count workers by carpool size k, so a size-k group is n_k / k
    vehicles; the divisor for carpool minutes is workers / vehicles (not the worker-weighted mean size)."""
    n = {k: float(df[k].sum()) for k in midpoints}
    return sum(n.values()) / sum(n[k] / midpoints[k] for k in midpoints)


def add_minutes(df: pl.DataFrame, mids: list[float]) -> pl.DataFrame:
    m = np.array(mids)
    return df.with_columns(pl.Series('t_alone', _bins(df, 'alone') @ m), pl.Series('t_carpool', _bins(df, 'carpool') @ m))


# ---------------------------------------------------------------------------------------------- CBP -> Oil_c

def oil_gas_table(cells: pl.DataFrame, state_cells: pl.DataFrame, codes=cbp.OIL_GAS_CODES) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One row per CBP county: oil and gas employment, total employment, flags. An absent cell counts as 0
    employment (as published). The county file omits some cells, and the state file shows how many establishments
    that hides, so each county also carries its state's unpublished oil and gas establishments (an exact count
    summed over the three codes) for the caller to bound the zero.

    Returns (county table, per state-and-code accounting of unpublished establishments)."""
    county_sum = (cells.with_columns(pl.col('geo_id').str.slice(0, 2).alias('state_fips'))
                  .group_by('state_fips', 'naics').agg(pl.col('estab').sum().alias('estab_county_sum')))
    acct = (state_cells.select(pl.col('geo_id').alias('state_fips'), 'naics', pl.col('estab').alias('estab_state'),
                               pl.col('emp').alias('emp_state'))
            .join(county_sum, on=['state_fips', 'naics'], how='left')
            .with_columns(pl.col('estab_county_sum').fill_null(0.0))
            .with_columns((pl.col('estab_state') - pl.col('estab_county_sum')).alias('estab_unpublished'))
            .sort('state_fips', 'naics'))
    unpub_state = (acct.filter(pl.col('naics').is_in(list(codes)))
                   .group_by('state_fips').agg(pl.col('estab_unpublished').clip(lower_bound=0).sum()
                                               .alias('oil_gas_state_unpub_estab')))
    wide = None
    for code in (cbp.TOTAL_CODE, *codes):
        part = (cells.filter(pl.col('naics') == code)
                .select('geo_id', pl.col('emp').alias(f'emp_{code}'), pl.col('emp_n_f').alias(f'nf_{code}'),
                        pl.col('emp_f').alias(f'ef_{code}')))
        wide = part if wide is None else wide.join(part, on='geo_id', how='full', coalesce=True)
    # counties in any oil/gas cell but with no total row are kept (total null)
    wide = wide.with_columns(pl.col('geo_id').str.slice(0, 2).alias('state_fips'))
    flags = []
    for row in wide.iter_rows(named=True):
        toks = []
        for code in codes:
            if row[f'emp_{code}'] is not None:
                toks.append(f"{code}:{row[f'nf_{code}'] or ''}" + (f"|emp_f={row[f'ef_{code}']}" if row[f'ef_{code}'] else ''))
        if row[f'emp_{cbp.TOTAL_CODE}'] is None:
            toks.append('total_emp_missing')
        flags.append(';'.join(toks))
    wide = wide.with_columns(pl.Series('oil_gas_flags', flags, dtype=pl.Utf8))
    oil = sum((pl.col(f'emp_{c}').fill_null(0.0) for c in codes), pl.lit(0.0))
    wide = wide.with_columns(oil.alias('oil_gas_emp'))
    wide = wide.with_columns(
        pl.when(pl.col(f'emp_{cbp.TOTAL_CODE}') > 0).then(pl.col('oil_gas_emp') / pl.col(f'emp_{cbp.TOTAL_CODE}'))
        .otherwise(None).alias('oil_gas_emp_share'))
    wide = wide.join(unpub_state, on='state_fips', how='left').with_columns(
        pl.col('oil_gas_state_unpub_estab').fill_null(0.0))
    return wide.select('geo_id', 'oil_gas_emp', pl.col(f'emp_{cbp.TOTAL_CODE}').alias('cbp_total_emp'),
                       'oil_gas_emp_share', 'oil_gas_flags', 'oil_gas_state_unpub_estab').sort('geo_id'), acct


def market_oil_gas_table(markets: pl.DataFrame, oil: pl.DataFrame) -> pl.DataFrame:
    """Employment-weighted CBP share over every county in each local market, including unpriced counties."""
    members = markets.select('geo_id', 'cluster_id').unique()
    if members['geo_id'].n_unique() != members.height or members['cluster_id'].null_count():
        raise ValueError('county market mapping missing or ambiguous')
    totals = (members.join(oil.select('geo_id', 'oil_gas_emp', 'cbp_total_emp'), on='geo_id', how='left', validate='1:1')
              .group_by('cluster_id').agg(pl.col('oil_gas_emp').fill_null(0.0).sum(),
                                         pl.col('cbp_total_emp').fill_null(0.0).sum())
              .with_columns(pl.when(pl.col('cbp_total_emp') > 0)
                            .then(pl.col('oil_gas_emp') / pl.col('cbp_total_emp'))
                            .otherwise(0.0).alias('oil_gas_market_emp_share')))
    return members.join(totals.select('cluster_id', 'oil_gas_market_emp_share'), on='cluster_id', validate='m:1')


@lru_cache(maxsize=1)
def analysis_market_oil() -> pl.DataFrame:
    """2026 control from stored CBP cells and the full geography, independent of price coverage."""
    from fb26.etl import substate_shocks as ss
    from fb26.etl import metro_prices as mp
    universe = (pl.read_parquet(paths.processed_dir() / OUT_NAME)
                .select(pl.col('geo_id').alias('county_fips'), 'state_fips').unique())
    states = pl.read_csv(paths.config_dir() / 'states.csv', schema_overrides={'state_fips': pl.String, 'padd_group': pl.String})
    geography = ss.build_geography(universe, ss.read_cbsa_types(rawstore.load_raw(mp.cbsa_path())),
                                  ss.read_cz(rawstore.load_raw(ss.latest_raw('commuting_zones', 'csv.gz'))), states)
    cells, state_cells, _ = cbp.parse(load_config()['cbp']['year'])
    return market_oil_gas_table(geography.rename({'county_fips': 'geo_id'}), oil_gas_table(cells, state_cells)[0])


def with_market_oil(df: pl.DataFrame) -> pl.DataFrame:
    """Use the market share in the shared estimator's oil-control slot, on 2026 paths only."""
    if 'oil_gas_market_emp_share' not in df.columns:
        df = df.join(analysis_market_oil().select('geo_id', 'oil_gas_market_emp_share'),
                     on='geo_id', how='left', validate='m:1')
    if df['oil_gas_market_emp_share'].null_count():
        raise ValueError('missing local-market oil and gas control')
    return df.with_columns(pl.col('oil_gas_market_emp_share').alias('oil_gas_emp_share'))


# ---------------------------------------------------------------------------------------------- the build

def _tokens(pairs: list[tuple[pl.Expr, str]]) -> pl.Expr:
    return pl.concat_str([pl.when(e).then(pl.lit(t)).otherwise(None) for e, t in pairs], separator=';',
                         ignore_nulls=True)


def build(county: pl.DataFrame, land: pl.DataFrame, cbp_cells: pl.DataFrame, cbp_state: pl.DataFrame, cfg: dict,
          vm1: dict, *, nhts_year: int | None = None, commute_scale: float = 1.0,
          m60_override: float | None = None) -> tuple[pl.DataFrame, dict]:
    """Exposure long by g_method plus a dict of constants and diagnostics. Puerto Rico is dropped with a count."""
    nhts_year = nhts_year or cfg['nhts']['primary_year']
    nhts = cfg['nhts']['years'][nhts_year]
    info = {'nhts_year': nhts_year, 'n_puerto_rico_dropped': int(county.filter(pl.col('state_fips') == '72').height)}
    c = county.filter(pl.col('state_fips') != '72')
    bins_cfg = cfg['derived_constants']['commute_minute_bins']
    closed = bins_cfg['midpoints_closed_bins']
    m60, fit = fit_open_bin(c, closed)
    if m60_override is not None:  # sensitivity only: an uncalibrated value for the 60+ bin
        m60 = m60_override
    mids = [*closed, m60]
    occ = carpool_occupancy(c, cfg['derived_constants']['occ']['midpoints'])
    c = add_minutes(c, mids)

    hh_nat = float(c['B11001_001E'].sum())
    veh_hh_nat = float(c['B25046_001E'].sum()) / hh_nat
    t_eff_nat = float((c['t_alone'] + c['t_carpool'] / occ).sum())
    speed = nhts['speed_mph']
    # Commute share of household VMT from the NHTS ratio (per-household commute / all-purpose VMT). Total household
    # VMT is held at M_veh x vehicles per household, so both measures have the same national mean; `commute_scale`
    # (default 1) multiplies the share for the hybrid-work sensitivity.
    commute_share = nhts['commute_vmt_per_household'] / nhts['total_vmt_per_household'] * commute_scale
    target_commute_vmt_hh = commute_share * vm1['m_veh'] * veh_hh_nat
    days = target_commute_vmt_hh * hh_nat / (2 * speed / 60 * t_eff_nat)
    m_nc = vm1['m_veh'] * (1 - commute_share)
    info.update(m60=m60, m60_fit=fit, occ=occ, days=days, speed_mph=speed, m_nc=m_nc, m_veh=vm1['m_veh'],
                mpg=vm1['mpg'], commute_share=commute_share, households_national=hh_nat,
                vehicles_per_household_national=veh_hh_nat, target_commute_vmt_hh=target_commute_vmt_hh)

    hh, veh = pl.col('B11001_001E'), pl.col('B25046_001E')
    commute_miles = lambda t_alone, t_carpool: (t_alone + t_carpool / occ) / 60 * speed * 2 * days / hh
    wfh_nat = float(c['B08301_021E'].sum() / c['B08301_001E'].sum())
    info['wfh_share_national'] = wfh_nat

    base = (c.join(land, on='geo_id', how='left')
            .join(oil_gas_table(cbp_cells, cbp_state)[0], on='geo_id', how='left')
            .with_columns(
                pl.when(hh > 0).then(veh / hh).otherwise(None).alias('vehicles_hh'),
                pl.when(hh > 0).then(commute_miles(pl.col('t_alone'), pl.col('t_carpool'))).otherwise(None)
                .alias('commute_vmt_hh_calc'),
                pl.when((hh > 0) & pl.col('B08136_003E').is_not_null() & pl.col('B08136_004E').is_not_null())
                .then(commute_miles(pl.col('B08136_003E'), pl.col('B08136_004E'))).otherwise(None)
                .alias('commute_vmt_hh_b08136'),
                pl.col('B19013_001E').alias('median_hh_income'),
                (pl.col('B19013_001M') / 1.645 / pl.col('B19013_001E')).alias('median_hh_income_cv'),
                hh.alias('households'), pl.lit(acs.VINTAGE).alias('acs_vintage'),
                pl.when(pl.col('B15003_001E') > 0)
                .then((pl.col('B15003_022E') + pl.col('B15003_023E') + pl.col('B15003_024E') + pl.col('B15003_025E'))
                      / pl.col('B15003_001E')).otherwise(None).alias('ba_share'),
                pl.when(pl.col('B03002_001E') > 0).then(pl.col('B03002_012E') / pl.col('B03002_001E'))
                .otherwise(None).alias('hisp_share'),
                pl.when(pl.col('B03002_001E') > 0).then(pl.col('B03002_004E') / pl.col('B03002_001E'))
                .otherwise(None).alias('nh_black_share'),
                pl.when((pl.col('B01003_001E') > 0) & (pl.col('land_sqmi') > 0))
                .then((pl.col('B01003_001E') / pl.col('land_sqmi')).log()).otherwise(None).alias('log_pop_density'),
                pl.col('B01002_001E').alias('median_age'),
                pl.when(pl.col('B19013_001E') > 0).then(pl.col('B19013_001E').log()).otherwise(None)
                .alias('log_median_income'),
            ))
    base = base.with_columns(
        _tokens([(pl.col('state_fips') == '02', 'alaska_excluded_from_sample'),
                 (pl.col('state_fips') == '09', 'ct_planning_region'),
                 (pl.col('B08136_003E').is_null(), 'b08136_null'),
                 (pl.col('land_sqmi').is_null(), 'no_land_area'),
                 (pl.col('cbp_total_emp').is_null(), 'no_cbp_county_row'),
                 (pl.col('B08301_003E') + pl.col('B08301_004E') == 0, 'no_car_commuters'),
                 ((_alone_sum() - pl.col('B08301_003E')).abs() > 0.5, 'bins_alone_mismatch')]).alias('flags'),
        _tokens([(pl.col('B19013_001E').is_null(), 'income_missing'), (veh.is_null(), 'vehicles_missing'),
                 (~(hh > 0), 'no_households')]).alias('rel_tokens'))
    base = base.with_columns(
        pl.when(pl.col('rel_tokens') == '').then(pl.lit('ok')).otherwise(pl.col('rel_tokens')).alias('reliability_flag'),
        pl.col('oil_gas_flags').fill_null('no_cbp_county_row'))

    def frame(method: str) -> pl.DataFrame:
        if method == PRIMARY:
            commute = pl.col('commute_vmt_hh_calc')
            vmt = commute + pl.col('vehicles_hh') * m_nc
            extra = pl.col('commute_vmt_hh_b08136')
        else:
            commute, vmt, extra = pl.lit(None, dtype=pl.Float64), pl.col('vehicles_hh') * vm1['m_veh'], pl.lit(None, dtype=pl.Float64)
        return (base.with_columns(commute.alias('commute_vmt_hh'), vmt.alias('vmt_hh'), pl.lit(method).alias('g_method'),
                                  extra.alias('commute_vmt_hh_b08136'))
                .with_columns((pl.col('vmt_hh') / vm1['mpg'] / 12).alias('g_gal_hh_month'))
                .with_columns(pl.when(pl.col('median_hh_income') > 0)
                              .then(100 * pl.col('g_gal_hh_month') / (pl.col('median_hh_income') / 12))
                              .otherwise(None).alias('e_c')))
    out = pl.concat([frame(PRIMARY), frame(ALTERNATIVE)]).select(OUTPUT_COLUMNS).cast(
        {'households': pl.Float64}).sort('g_method', 'geo_id')
    return out, info


def _alone_sum() -> pl.Expr:
    return pl.sum_horizontal([pl.col(x) for x in acs.BIN_COLUMNS['alone']])


# ------------------------------------------------------------------------------------------ summary and checks

def _corr(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float | None:
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    if rank:
        x, y = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(x, y)[0, 1])


def wide_e(df: pl.DataFrame) -> pl.DataFrame:
    return (df.select('geo_id', 'state_fips', 'g_method', 'e_c').pivot(on='g_method', index=['geo_id', 'state_fips'],
                                                                       values='e_c')
            .rename({PRIMARY: 'e_primary', ALTERNATIVE: 'e_alt'}).sort('geo_id'))


def summary_table(df: pl.DataFrame) -> pl.DataFrame:
    """E_c distribution by state and method, plus the correlation between methods (pooled; within state)."""
    w = wide_e(df.filter(pl.col('state_fips') != '02')).drop_nulls(['e_primary', 'e_alt'])  # Alaska is outside the sample
    rows = []
    groups = sorted(((k[0], g) for k, g in w.group_by('state_fips')), key=lambda t: t[0]) + [('ALL', w)]
    for state, g in groups:
        row = {'state_fips': state, 'scope': 'pooled' if state == 'ALL' else 'state', 'n_counties': g.height}
        for col, label in (('e_primary', PRIMARY), ('e_alt', ALTERNATIVE)):
            v = g[col]
            q = v.quantile
            row.update({f'{label}_mean': v.mean(), f'{label}_sd': v.std() if g.height > 1 else None, f'{label}_min': v.min(),
                        f'{label}_p10': q(0.10, 'linear'), f'{label}_p50': q(0.50, 'linear'),
                        f'{label}_p90': q(0.90, 'linear'), f'{label}_max': v.max()})
        row['corr_pearson'] = _corr(g['e_primary'].to_numpy(), g['e_alt'].to_numpy())
        row['corr_spearman'] = _corr(g['e_primary'].to_numpy(), g['e_alt'].to_numpy(), rank=True)
        rows.append(row)
    demeaned = w.with_columns((pl.col('e_primary') - pl.col('e_primary').mean().over('state_fips')).alias('p'),
                              (pl.col('e_alt') - pl.col('e_alt').mean().over('state_fips')).alias('a'))
    rows.append({'state_fips': 'ALL', 'scope': 'within_state_demeaned', 'n_counties': w.height,
                 'corr_pearson': _corr(demeaned['p'].to_numpy(), demeaned['a'].to_numpy()),
                 'corr_spearman': _corr(demeaned['p'].to_numpy(), demeaned['a'].to_numpy(), rank=True)})
    return pl.DataFrame(rows, infer_schema_length=None)


def sanity_checks(county, land, cells, state_cells, cfg, vm1, df, info) -> None:
    """Printed diagnostics for the WS3 return: national gallons, ranges, correlations, sensitivity to the NHTS year."""
    pd = pl.Config(tbl_rows=60, tbl_cols=20, float_precision=4)
    with pd:
        gal = df.group_by('g_method').agg((pl.col('households') * pl.col('g_gal_hh_month') * 12).sum().alias('gal_per_year'),
                                          (pl.col('households') * pl.col('vmt_hh')).sum().alias('vmt'),
                                          pl.col('households').sum().alias('hh'))
        gal = gal.with_columns((pl.col('vmt') / pl.col('hh')).alias('vmt_per_hh'),
                               (pl.col('gal_per_year') / 1e9).alias('gal_bn'),
                               (pl.col('gal_per_year') / vm1['ld_fuel_thousand_gal'] / 1e3).alias('share_of_fhwa_ld_fuel'))
        print('national implied gasoline (includes AK, excludes PR):'); print(gal.drop('gal_per_year'))
        print('FHWA VM-1 2024 light-duty fuel (billion gal):', vm1['ld_fuel_thousand_gal'] / 1e6)
        print('constants:', {k: round(v, 4) for k, v in info.items() if isinstance(v, float)})
        print('B08134 fit vs B08136:', info['m60_fit'])
        for m in (PRIMARY, ALTERNATIVE):
            e = df.filter((pl.col('g_method') == m) & (pl.col('state_fips') != '02'))['e_c'].drop_nulls()
            print(m, 'E_c n', e.len(), 'min/p1/p10/p50/p90/p99/max',
                  [round(e.quantile(q), 4) for q in (0, .01, .1, .5, .9, .99, 1)], 'sd', round(e.std(), 4))
        summ = summary_table(df)
        print(summ.filter(pl.col('state_fips') == 'ALL').select('scope', 'n_counties', 'corr_pearson', 'corr_spearman'))
        st = summ.filter(pl.col('scope') == 'state').drop_nulls('corr_pearson')
        print('within-state corr across states: min/median/max', st['corr_pearson'].min(), st['corr_pearson'].median(),
              st['corr_pearson'].max(), 'n states', st.height)
        # sensitivity of the composite to the weight of the commute part
        base = wide_e(df.filter(pl.col('state_fips') != '02'))
        for label, kw in (('NHTS 2022 constants', {'nhts_year': 2022}),
                          ('hybrid-era commute weight x 261/339 (2022 vs 2017 trips per worker, 2022 Summary Table 7-1)',
                           {'commute_scale': 261 / 339}),
                          ('commute weight x 0.5', {'commute_scale': 0.5}), ('commute weight x 2', {'commute_scale': 2.0})):
            alt, ainfo = build(county, land, cells, state_cells, cfg, vm1, **kw)
            b = wide_e(alt.filter(pl.col('state_fips') != '02'))
            j = base.join(b.select('geo_id', pl.col('e_primary').alias('e_alt2')), on='geo_id').drop_nulls(['e_primary', 'e_alt2'])
            jd = j.with_columns((pl.col('e_primary') - pl.col('e_primary').mean().over('state_fips')).alias('x'),
                                (pl.col('e_alt2') - pl.col('e_alt2').mean().over('state_fips')).alias('y'))
            print(f'{label}: share {ainfo["commute_share"]:.4f} days {ainfo["days"]:.1f} m_nc {ainfo["m_nc"]:.0f} | corr with primary: pooled '
                  f'{_corr(j["e_primary"].to_numpy(), j["e_alt2"].to_numpy()):.5f}, within-state '
                  f'{_corr(jd["x"].to_numpy(), jd["y"].to_numpy()):.5f}')
        # exact B08136 commute (1,371 counties) vs bin-based, in the composite
        ex = df.filter((pl.col('g_method') == PRIMARY) & pl.col('commute_vmt_hh_b08136').is_not_null())
        print('commute VMT/hh bins vs B08136 (n=%d): corr' % ex.height,
              _corr(ex['commute_vmt_hh'].to_numpy(), ex['commute_vmt_hh_b08136'].to_numpy()),
              'sum ratio', ex['commute_vmt_hh'].sum() / ex['commute_vmt_hh_b08136'].sum())
        # coverage
        acs_ids, cbp_ids = set(county.filter(pl.col('state_fips') != '72')['geo_id']), set(cells.filter(pl.col('naics') == '00')['geo_id'])
        print('ACS counties', len(acs_ids), '| in CBP total file', len(acs_ids & cbp_ids), '| ACS not in CBP', sorted(acs_ids - cbp_ids),
              '| CBP not in ACS (non-PR)', sorted(x for x in cbp_ids - acs_ids if not x.startswith('72')))
        oil = df.filter(pl.col('g_method') == PRIMARY)
        print('Oil_c: counties with share > 0', oil.filter(pl.col('oil_gas_emp_share') > 0).height,
              'null share', oil['oil_gas_emp_share'].null_count(), 'max', oil['oil_gas_emp_share'].max(),
              'counties with a noise-J cell', oil.select(pl.col('oil_gas_flags').str.contains('J').sum()).item())
        top = (oil.filter(pl.col('oil_gas_state_unpub_estab') > 0).group_by('state_fips')
               .agg(pl.col('oil_gas_state_unpub_estab').first(), pl.col('oil_gas_emp_share').gt(0).sum().alias('counties_with_oil_cells'))
               .sort('oil_gas_state_unpub_estab', descending=True).head(12))
        print('states with most unpublished oil/gas establishments:'); print(top)
        _, acct = oil_gas_table(cells, state_cells)
        print(acct.group_by('naics').agg(pl.col('estab_state').sum(), pl.col('estab_county_sum').sum(), pl.col('estab_unpublished').sum(),
                                         (pl.col('estab_unpublished') > 0).sum().alias('states_with_unpublished'),
                                         (pl.col('estab_unpublished') < 0).sum().alias('states_negative')).sort('naics'))
        print('flags:', df.filter(pl.col('g_method') == PRIMARY).select(pl.col('flags').str.split(';').explode().alias('f'))
              .group_by('f').len().sort('f'))
        print('reliability:', df.filter(pl.col('g_method') == PRIMARY).group_by('reliability_flag').len())


def calibration_diagnostics(county, land, cells, state_cells, cfg: dict, vm1: dict, df: pl.DataFrame) -> None:
    """Printed: do the counties where B08136 is published (the only ones the 60+ fit uses) resemble the rest, is the
    fitted value stable across subsets, and how much does the single fitted parameter matter."""
    closed = cfg['derived_constants']['commute_minute_bins']['midpoints_closed_bins']
    c = county.filter(pl.col('state_fips') != '72').join(land, on='geo_id', how='left')
    c = c.with_columns(
        pl.col('B08136_003E').is_not_null().alias('has'), (pl.col('B01003_001E') / pl.col('land_sqmi')).alias('dens'),
        (pl.col('B08134_030E') + pl.col('B08134_040E')).alias('n60'),
        (sum(pl.col(x) for x in acs.BIN_COLUMNS['alone'] + acs.BIN_COLUMNS['carpool'])).alias('ncar'))
    c = c.with_columns((pl.col('n60') / pl.col('ncar')).alias('share60'),
                       ((pl.col('B08301_003E') + pl.col('B08301_004E')) / pl.col('B08301_001E')).alias('car_share'),
                       (pl.col('B08301_021E') / pl.col('B08301_001E')).alias('wfh_share'),
                       (pl.col('B25046_001E') / pl.col('B11001_001E')).alias('veh_hh'))
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=5):
        print(c.group_by('has').agg(
            pl.len().alias('n'), pl.col('B11001_001E').sum().alias('households'),
            pl.col('B01003_001E').median().alias('pop_median'), pl.col('B01003_001E').mean().alias('pop_mean'),
            pl.col('dens').median().alias('dens_median'), pl.col('B19013_001E').median().alias('income_median'),
            pl.col('share60').median().alias('share60_median'), pl.col('car_share').median().alias('car_share_med'),
            pl.col('wfh_share').median().alias('wfh_med'), pl.col('veh_hh').median().alias('veh_hh_med')).sort('has'))
        c = c.with_columns(pl.when(pl.col('B01003_001E') < 20000).then(pl.lit('1 <20k')).when(pl.col('B01003_001E') < 65000)
                           .then(pl.lit('2 20-65k')).when(pl.col('B01003_001E') < 250000).then(pl.lit('3 65-250k'))
                           .otherwise(pl.lit('4 250k+')).alias('size'),
                           pl.when(pl.col('dens') < 25).then(pl.lit('1 <25/sqmi')).when(pl.col('dens') < 100)
                           .then(pl.lit('2 25-100')).when(pl.col('dens') < 1000).then(pl.lit('3 100-1000'))
                           .otherwise(pl.lit('4 1000+')).alias('dens_band'))
        for col in ('size', 'dens_band'):
            print(c.group_by(col).agg(pl.len().alias('n'), pl.col('has').mean().alias('share_with_b08136')).sort(col))
        st = c.group_by('state_fips').agg(pl.col('has').mean().alias('p'), pl.len().alias('n'))
        print('state share with B08136: min/median/max', st['p'].min(), st['p'].median(), st['p'].max(),
              '| states with none:', st.filter(pl.col('p') == 0)['state_fips'].to_list(),
              '| with all:', st.filter(pl.col('p') == 1)['state_fips'].to_list())
        ok = c.filter(pl.col('has'))
        rows = [('all with B08136', ok)]
        for col in ('size', 'dens_band'):
            rows += [(f'{col} {k[0]}', g) for k, g in sorted(ok.group_by(col), key=lambda t: t[0])]
        half = (pl.col('geo_id').str.slice(2).cast(pl.Int32) // 2) % 2  # county codes are odd; this splits them in two
        rows += [('split A (county code // 2 even)', ok.filter(half == 0)), ('split B', ok.filter(half == 1))]
        out = []
        for label, g in rows:
            if g.height >= 5:
                m, f = fit_open_bin(g, closed)
                out.append({'subset': label, 'n': g.height, 'm60': m, 'agg_err': f['aggregate_error'], 'mape': f['county_mape']})
        print('fitted 60+ value by subset:'); print(pl.DataFrame(out))
        # one parameter: uncalibrated 60+ conventions and their error against B08136
        n = _bins(ok, 'alone') + _bins(ok, 'carpool')
        y = (ok['B08136_003E'] + ok['B08136_004E']).to_numpy()
        base_m = n[:, :8] @ np.array(closed)
        full = county.filter(pl.col('state_fips') != '72')
        m_fit, _ = fit_open_bin(full, closed)
        e_fit = None
        out = []
        for label, m60 in (('fitted', m_fit), ('60 (lower bound)', 60.0), ('70', 70.0), ('75', 75.0)):
            pred = base_m + n[:, 8] * m60
            t = add_minutes(full, [*closed, m60])
            hh = full['B11001_001E']
            v = ((t['t_alone'] + t['t_carpool'] / 2.25) / hh).to_numpy()  # relative scale only; level is calibrated away
            if e_fit is None:
                e_fit = v
            ok_v = ~np.isnan(v) & ~np.isnan(e_fit)
            out.append({'60+ value': label, 'm60': m60, 'agg_err_vs_B08136': pred.sum() / y.sum() - 1,
                        'county_mape': float(np.mean(np.abs(pred / y - 1))),
                        'corr_with_fitted_minutes_per_hh_all_counties': float(np.corrcoef(v[ok_v], e_fit[ok_v])[0, 1])})
        print('uncalibrated 60+ conventions:'); print(pl.DataFrame(out))
        base = wide_e(df.filter(pl.col('state_fips') != '02'))
        out = []
        for label, m60 in (('60+ = 60', 60.0), ('60+ = 70', 70.0), ('60+ = 75', 75.0)):
            alt, ainfo = build(county, land, cells, state_cells, cfg, vm1, m60_override=m60)
            b = wide_e(alt.filter(pl.col('state_fips') != '02'))
            j = base.join(b.select('geo_id', pl.col('e_primary').alias('e2')), on='geo_id').drop_nulls(['e_primary', 'e2'])
            jd = j.with_columns((pl.col('e_primary') - pl.col('e_primary').mean().over('state_fips')).alias('x'),
                                (pl.col('e2') - pl.col('e2').mean().over('state_fips')).alias('y'))
            out.append({'version': label, 'days': ainfo['days'], 'corr_E_pooled': _corr(j['e_primary'].to_numpy(), j['e2'].to_numpy()),
                        'corr_E_within_state': _corr(jd['x'].to_numpy(), jd['y'].to_numpy()),
                        'max_abs_rel_diff_E': float(((j['e2'] / j['e_primary'] - 1).abs()).max())})
        print('E_c (commute_vehicles) under uncalibrated 60+ values vs the fitted one:'); print(pl.DataFrame(out))


def main(argv: list[str]) -> int:
    step = argv[1] if len(argv) > 1 else 'build'
    cfg = load_config()
    county, land, acs_info = acs.parse(cfg['acs']['vintage'])
    cells, state_cells, cbp_info = cbp.parse(cfg['cbp']['year'])
    vm1_rec = latest_raw('fhwa', VM1_URL)
    vm1 = vm1_constants(cfg, rawstore.load_raw(vm1_rec.path))
    df, info = build(county, land, cells, state_cells, cfg, vm1)
    if step == 'checks':
        sanity_checks(county, land, cells, state_cells, cfg, vm1, df, info)
        calibration_diagnostics(county, land, cells, state_cells, cfg, vm1, df)
        return 0
    out = paths.processed_dir() / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression='zstd', statistics=False)
    summ = summary_table(df)
    vdir = paths.repo_root() / 'reports' / 'validation'
    vdir.mkdir(parents=True, exist_ok=True)
    summ.write_csv(vdir / SUMMARY_NAME, float_precision=6)
    snapshots = ';'.join(f'{k}:{v[:12]}' for k, v in {**acs_info, **cbp_info, 'vm1': vm1_rec.sha256}.items()
                         if k.endswith('sha256') or k == 'vm1')
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(),
        config_hash=runlog.config_hash([paths.config_dir() / 'exposure.yaml']), spec_id='ws3_exposure',
        data_snapshot_ids=snapshots, output_path=f'data/processed/{OUT_NAME};reports/validation/{SUMMARY_NAME}',
        notes=f"{CODE_VERSION}; rows={df.height}; nhts_year={info['nhts_year']}; days={info['days']:.3f}; "
              f"occ={info['occ']:.4f}; m60={info['m60']:.3f}; m_nc={info['m_nc']:.1f}; m_veh={info['m_veh']:.1f}; "
              f"mpg={info['mpg']:.4f}; speed={info['speed_mph']}")
    print(f'wrote {out} ({df.height} rows) and {SUMMARY_NAME} ({summ.height} rows)')
    print({k: v for k, v in info.items() if k != 'm60_fit'}, info['m60_fit'])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
