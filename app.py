import datetime
import os
import time
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

API_KEY = os.environ['NASS_API_KEY']

# --- Data configuration -----------------------------------------------------

# Commodity label -> NASS commodity_desc + optional class_desc filter.
# Wheat is split by class: CONDITION is reported separately for winter and
# spring wheat, and filtering server-side also keeps each request under the
# API's 50,000-record cap.
COMMODITIES = {
    'Cotton': ('COTTON', None),
    'Corn': ('CORN', None),
    'Soybeans': ('SOYBEANS', None),
    'Winter Wheat': ('WHEAT', 'WINTER'),
    'Spring Wheat': ('WHEAT', 'SPRING, (EXCL DURUM)'),
}

# The state the history panel opens on — the leading producer of each crop.
DEFAULT_STATE = {
    'Cotton': 'TX', 'Corn': 'IA', 'Soybeans': 'IL',
    'Winter Wheat': 'KS', 'Spring Wheat': 'ND',
}

# Display label -> (column, higher-is-better, axis range).
METRICS = {
    'Good + Excellent (%)': ('GE', True, (0, 100)),
    'Poor + Very Poor (%)': ('PVP', False, (0, 100)),
    'Condition Index': ('INDEX', True, (100, 500)),
    'Progress Index (%)': ('PROGRESS', True, (0, 100)),
}

CONDITIONS = ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'VERY POOR']

# Each crop's condition-reporting season, as the (first, last) calendar week
# NASS publishes weekly conditions for it. Seasonality differs enough that one
# global cutoff cannot serve them all: spring wheat is done by week 35 while
# cotton runs to week 47, and winter wheat's season *wraps the turn of the
# year* — autumn establishment (weeks 39-52) belongs to the crop harvested the
# following summer (weeks 1-33). A window whose start is after its end is read
# as wrapping.
#
# Anything outside the window is a hangover from the neighbouring crop year —
# NASS carries, for instance, a stray week-1 cotton report each January that is
# really the tail of the previous harvest.
SEASON = {
    'Cotton': (12, 50),
    'Corn': (8, 48),
    'Soybeans': (12, 50),
    'Spring Wheat': (12, 40),
    'Winter Wheat': (36, 35),
}


def season_shape(label):
    """(first week, last week, wraps the year end?, weeks in the season)."""
    start, end = SEASON[label]
    wraps = start > end
    length = (end - start) % 52 + 1 if wraps else end - start + 1
    return start, end, wraps, length


def season_week(cal_week, start, wraps):
    """Position within the crop's own season, 1-based, so seasons align."""
    return (cal_week - start) % 52 + 1 if wraps else cal_week - start + 1


def calendar_week(week, start):
    """Inverse of season_week — used to label the axis in familiar weeks."""
    return (start + week - 2) % 52 + 1


def season_span(label, year):
    """How a season is named: a wrapping season spans two calendar years."""
    _, _, wraps, _ = season_shape(label)
    return f'{year - 1}/{str(year)[-2:]}' if wraps else str(year)


def in_season(cal_week, label):
    """Mask of the calendar weeks that belong to this crop's season."""
    start, end, wraps, _ = season_shape(label)
    if wraps:
        return (cal_week >= start) | (cal_week <= end)
    return (cal_week >= start) & (cal_week <= end)


# Every request pulls the full window once and is cached, so changing metric
# or state costs no further API calls.
SEASONS = 11
CACHE_TTL = 3600

# --- Palette (dark) ---------------------------------------------------------
# Dark-mode steps of the same validated system: categorical slots 1 and 2 for
# the series, its dark chart chrome for everything else.
SURFACE = '#1a1a19'
PAGE = '#0d0d0d'
INK = '#ffffff'
INK_2 = '#c3c2b7'
MUTED = '#898781'
GRID = '#2c2c2a'
AXIS = '#383835'
LAND = '#262625'        # states with no reported condition
SERIES_1 = '#3987e5'    # current season
SERIES_2 = '#d95926'    # prior season
HISTORY = '#5598e7'     # older seasons, faded behind the current one
GOOD = '#0ca30c'
BAD = '#d03b3b'

# Condition ramp: red (poor) -> amber -> green (good), the convention the
# trade reads. Every step is dark enough for the white state labels to hold,
# and because the number is printed on each state, hue is never the only
# channel — which is what keeps a red/green ramp usable for CVD readers.
LEVEL_SCALE = [
    [0.00, '#7a1616'], [0.20, '#a52a2a'], [0.40, '#b8502a'],
    [0.55, '#a87a12'], [0.72, '#6f8f2a'], [0.88, '#3d8b3d'],
    [1.00, '#2e9b45'],
]
# Signed week-on-week change: deterioration red, improvement green, through
# the neutral dark gray at zero.
CHANGE_SCALE = [
    [0.00, '#7a1616'], [0.22, '#b8352b'], [0.44, '#383835'],
    [0.56, '#383835'], [0.78, '#3d8b3d'], [1.00, '#2e9b45'],
]


def flip(scale):
    """Mirror a colour scale, for metrics where a high number is bad."""
    return [[round(1 - pos, 2), colour] for pos, colour in reversed(scale)]

# Approximate label anchors for on-map state annotations.
CENTROIDS = {
    'AL': (32.8, -86.8), 'AZ': (34.3, -111.7), 'AR': (34.9, -92.4),
    'CA': (37.2, -119.4), 'CO': (39.0, -105.5), 'CT': (41.6, -72.7),
    'DE': (39.0, -75.5), 'FL': (28.6, -82.4), 'GA': (32.6, -83.4),
    'ID': (44.4, -114.6), 'IL': (40.0, -89.2), 'IN': (39.9, -86.3),
    'IA': (42.1, -93.5), 'KS': (38.5, -98.4), 'KY': (37.5, -85.3),
    'LA': (31.1, -92.0), 'ME': (45.4, -69.2), 'MD': (39.0, -76.8),
    'MA': (42.3, -71.8), 'MI': (44.3, -85.4), 'MN': (46.3, -94.3),
    'MS': (32.7, -89.7), 'MO': (38.4, -92.5), 'MT': (47.0, -109.6),
    'NE': (41.5, -99.8), 'NV': (39.3, -116.6), 'NH': (43.7, -71.6),
    'NJ': (40.2, -74.7), 'NM': (34.4, -106.1), 'NY': (42.9, -75.5),
    'NC': (35.5, -79.4), 'ND': (47.4, -100.5), 'OH': (40.3, -82.8),
    'OK': (35.6, -97.5), 'OR': (43.9, -120.6), 'PA': (40.9, -77.8),
    'RI': (41.7, -71.6), 'SC': (33.9, -80.9), 'SD': (44.4, -100.2),
    'TN': (35.8, -86.4), 'TX': (31.5, -99.3), 'UT': (39.3, -111.7),
    'VT': (44.1, -72.7), 'VA': (37.5, -78.9), 'WA': (47.4, -120.4),
    'WV': (38.6, -80.6), 'WI': (44.6, -89.7), 'WY': (43.0, -107.6),
}


# --- Data layer -------------------------------------------------------------

_cache = {}

# What identifies one reported state-week, and so what the two statistics are
# joined on.
KEYS = ['state_alpha', 'season', 'week', 'cal_week', 'week_ending']


NATIONAL = 'US'


def _request(commodity, klass, first_year, last_year,
             stat='CONDITION', level='STATE'):
    """One NASS request for a year range, split in half if it is too large."""
    url = (
        'https://quickstats.nass.usda.gov/api/api_GET/?'
        f'key={API_KEY}&commodity_desc={commodity}'
        f'&statisticcat_desc={quote(stat)}&agg_level_desc={level}'
        f'&year__GE={first_year}&year__LE={last_year}&format=csv'
    )
    if klass:
        url += f'&class_desc={quote(klass)}'
    try:
        raw = pd.read_csv(url)
    except Exception:
        # The API caps a response at 50k records; halve the window and retry.
        if last_year <= first_year:
            raise
        mid = (first_year + last_year) // 2
        return pd.concat(
            [_request(commodity, klass, first_year, mid, stat, level),
             _request(commodity, klass, mid + 1, last_year, stat, level)],
            ignore_index=True,
        )
    if level == 'NATIONAL':
        # National rows carry no state_alpha; give them one so they travel
        # through the same pipeline as a state.
        raw['state_alpha'] = NATIONAL
    return raw


def _stamp(df, label):
    """Stamp raw rows with the season (crop year) and season-relative week.

    NASS's own `year` is the calendar year, which is not the crop year for a
    crop that overwinters — and NASS is inconsistent about it: most autumn
    winter-wheat rows carry the following crop year, but a handful carry the
    year just harvested. Deriving the season from `week_ending` gives one rule
    that holds for every crop and both statistics.
    """
    start, _, wraps, _ = season_shape(label)
    df['cal_week'] = pd.to_numeric(df['end_code'], errors='coerce').clip(1, 52)
    ending = pd.to_datetime(df['week_ending'], errors='coerce')
    df = df[df['cal_week'].notna() & ending.notna()]
    if df.empty:
        return df

    df = df[in_season(df['cal_week'], label)]
    if df.empty:
        return df

    # The season a week belongs to: its own calendar year, except that for a
    # wrapping crop everything from the season's start week onward belongs to
    # the season that ends the *next* summer.
    df['season'] = ending[df.index].dt.year + (
        (df['cal_week'] >= start).astype(int) if wraps else 0
    )
    df['week'] = season_week(df['cal_week'], start, wraps)
    return df


def _trim(out):
    """Keep the newest SEASONS seasons that have real coverage.

    NASS occasionally files a crop's last harvest week under the *next* crop
    year, which _stamp() then hands back to the season it belongs to — leaving
    a two- or three-row phantom season just outside the requested window.
    """
    if out.empty:
        return out
    weeks = out.groupby('season')['week'].nunique()
    keep = set(weeks[weeks >= 0.25 * weeks.median()].index[-SEASONS:])
    return out[out['season'].isin(keep)]


def _tidy(raw, klass, label):
    """Pivot the "PCT <condition>" rows into INDEX/GE/PVP per state-week."""
    df = raw[raw['unit_desc'].str.startswith('PCT')].copy()
    if klass:
        df = df[df['class_desc'] == klass]
    if df.empty:
        return df

    df['cond'] = df['unit_desc'].str.replace('PCT ', '', regex=False)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    df = _stamp(df, label)
    if df.empty:
        return df

    wide = df.pivot_table(
        index=KEYS, columns='cond', values='Value', aggfunc='first',
    )
    for c in CONDITIONS:
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide.fillna(0).astype(float)

    wide['INDEX'] = (
        5 * wide['EXCELLENT'] + 4 * wide['GOOD'] + 3 * wide['FAIR']
        + 2 * wide['POOR'] + 1 * wide['VERY POOR']
    )
    wide['GE'] = wide['GOOD'] + wide['EXCELLENT']
    wide['PVP'] = wide['POOR'] + wide['VERY POOR']

    out = _trim(wide.reset_index())
    return out.sort_values(['state_alpha', 'season', 'week'])


def _tidy_progress(raw, klass, label):
    """Collapse the weekly stage percentages into a single progress index.

    The index is the sum of every stage percentage over 100 × the number of
    stages the crop reports — i.e. the mean of the stages — so it runs from 0
    before planting to 100 once the last stage is complete, and is comparable
    across seasons and states.

    The one trap is that NASS stops publishing a stage once it reaches 100%:
    cotton's PCT PLANTED simply disappears from the file in July. A stage that
    has dropped out is carried forward at its last reading rather than read as
    zero; a stage that has not started yet is zero.
    """
    df = raw[raw['unit_desc'].str.startswith('PCT')].copy()
    if klass:
        df = df[df['class_desc'] == klass]
    if df.empty:
        return pd.DataFrame()

    df['stage'] = df['unit_desc'].str.replace('PCT ', '', regex=False)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    df = _stamp(df, label)
    if df.empty:
        return pd.DataFrame()

    wide = df.pivot_table(
        index=KEYS, columns='stage', values='Value', aggfunc='first',
    ).sort_index()
    if wide.empty or not len(wide.columns):
        return pd.DataFrame()

    filled = wide.groupby(level=['state_alpha', 'season']).ffill().fillna(0.0)
    out = filled.mean(axis=1).rename('PROGRESS').reset_index()
    return _trim(out).sort_values(['state_alpha', 'season', 'week'])


def _merge(cond, prog):
    """One row per state-week carrying whichever statistics reported that week.

    Outer, because the two do not cover the same weeks: planting is under way
    weeks before the first condition report, and either statistic can be
    missing entirely for a crop. Consumers drop the rows their own column is
    absent from.
    """
    if prog.empty:
        return cond
    if cond.empty:
        return prog
    return cond.merge(prog, on=KEYS, how='outer').sort_values(
        ['state_alpha', 'season', 'week'])


def load(label):
    """Return every state-week for a commodity over the last SEASONS years.

    Four NASS requests — condition and progress, each at state and national
    level — joined into one frame. The national total arrives as its own
    reported "state", US, rather than being rebuilt from the states: NASS
    weights it by acreage, which we do not have. Cached for an hour, so the
    only thing that costs an API call is picking a commodity you have not
    looked at yet.
    """
    commodity, klass = COMMODITIES[label]
    hit = _cache.get(label)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    this_year = datetime.date.today().year
    first_year = this_year - SEASONS + 1

    def pull(stat, tidy):
        frames = []
        for level in ('STATE', 'NATIONAL'):
            try:
                raw = _request(commodity, klass, first_year, this_year,
                               stat, level)
                frames.append(tidy(raw, klass, label))
            except Exception:
                continue
        frames = [f for f in frames if not f.empty]
        return (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame())

    cond = pull('CONDITION', _tidy)
    prog = pull('PROGRESS', _tidy_progress)

    df = _merge(cond, prog)
    _cache[label] = (time.time(), df)
    return df


def reported(df, *cols):
    """Rows that actually carry every one of these columns.

    A state-week can hold a progress reading and no condition reading (or the
    reverse), and a crop can be missing one statistic outright.
    """
    if df.empty or any(c not in df.columns for c in cols):
        return pd.DataFrame()
    return df.dropna(subset=list(cols))


def snapshot(df, col):
    """Latest week per state in the latest season, with its week-on-week move.

    Ordered by season week, not calendar week: for a crop that overwinters the
    two disagree, and sorting on the calendar would report last autumn's
    reading as the most recent one.
    """
    df = reported(df, col)
    if df.empty:
        return pd.DataFrame()
    # The national total is a region you can chart, not a shape on the map.
    df = df[df['state_alpha'] != NATIONAL]
    if df.empty:
        return pd.DataFrame()
    season = df[df['season'] == df['season'].max()]
    rows = []
    for state, g in season.groupby('state_alpha'):
        g = g.sort_values('week')
        last = g.iloc[-1]
        # Only a genuinely consecutive report is a week-on-week move. Winter
        # wheat goes quiet from December to February, so the reading either
        # side of that gap is not a weekly change.
        prev = g.iloc[-2] if len(g) > 1 else None
        step = None if prev is None else int(last['week'] - prev['week'])
        rows.append({
            'state': state,
            'value': last[col],
            'delta': last[col] - prev[col] if step == 1 else None,
            'week': int(last['cal_week']),
            'week_ending': last['week_ending'],
        })
    return pd.DataFrame(rows)


def regions(df):
    """Everything the history panel can chart, national total first."""
    if df.empty:
        return []
    reporting = set(df['state_alpha'])
    return ([NATIONAL] if NATIONAL in reporting else []) + sorted(
        reporting - {NATIONAL})


def default_region(label, df):
    """The national total, falling back to the crop's leading producer."""
    if df.empty:
        return None
    reporting = set(df['state_alpha'])
    for preferred in (NATIONAL, DEFAULT_STATE.get(label)):
        if preferred in reporting:
            return preferred
    return df['state_alpha'].value_counts().idxmax()


# --- Figures ----------------------------------------------------------------

def _blank(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=13, color=INK_2))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=460,
        margin=dict(l=16, r=16, t=16, b=16),
    )
    return fig


def _arrow(delta):
    if delta is None or pd.isna(delta):
        return '', ''
    if abs(delta) < 0.05:
        return '–', ''
    return ('▲', f'{delta:.0f}') if delta > 0 else ('▼', f'{abs(delta):.0f}')


def map_figure(label, metric_label, mode, selected):
    col, higher_better, span = METRICS[metric_label]
    snap = snapshot(load(label), col)
    if snap.empty:
        return _blank(f'No {label.lower()} {metric_label.lower()} '
                      'reported yet.')

    if mode == 'change':
        snap = snap.dropna(subset=['delta'])
        if snap.empty:
            return _blank('Only one week reported so far this season — '
                          'no change to show.')
        z = snap['delta']
        limit = max(4.0, float(z.abs().max()))
        zmin, zmax = -limit, limit
        scale = CHANGE_SCALE if higher_better else flip(CHANGE_SCALE)
        bar_title = 'week<br>change'
    else:
        z = snap['value']
        zmin, zmax = span
        scale = LEVEL_SCALE if higher_better else flip(LEVEL_SCALE)
        bar_title = metric_label.split(' (')[0].replace(' + ', '+<br>')

    # The freshest reading on the map, by date — with a wrapping season a high
    # calendar week can be the oldest thing on it, not the newest.
    newest = snap.loc[pd.to_datetime(snap['week_ending']).idxmax()]
    week, ending = int(newest['week']), newest['week_ending']

    fig = go.Figure(go.Choropleth(
        locations=snap['state'], locationmode='USA-states', z=z,
        zmin=zmin, zmax=zmax, colorscale=scale,
        marker_line_color=SURFACE, marker_line_width=1.2,
        customdata=snap[['value', 'delta', 'week', 'week_ending']],
        hovertemplate=(
            '<b>%{location}</b><br>'
            f'{metric_label}: ' '%{customdata[0]:.0f}<br>'
            'vs prior week: %{customdata[1]:+.1f}<br>'
            'week %{customdata[2]}, ending %{customdata[3]}'
            '<extra></extra>'
        ),
        colorbar=dict(
            title=dict(text=bar_title, font=dict(size=11, color=INK_2)),
            thickness=10, len=0.6, x=0.98,
            tickfont=dict(size=10, color=MUTED), outlinewidth=0,
        ),
    ))

    # Two label layers per state: the level above the centroid, the
    # week-on-week move below it, so neither has to share a line.
    known = snap[snap['state'].isin(CENTROIDS)]
    lat = [CENTROIDS[s][0] for s in known['state']]
    lon = [CENTROIDS[s][1] for s in known['state']]
    moves = [''.join(_arrow(d)) for d in known['delta']]

    fig.add_trace(go.Scattergeo(
        lat=lat, lon=lon, mode='text', hoverinfo='skip', showlegend=False,
        text=[f'{s} <b>{v:.0f}</b>'
              for s, v in zip(known['state'], known['value'])],
        textposition='top center', textfont=dict(size=12, color=INK),
    ))
    fig.add_trace(go.Scattergeo(
        lat=lat, lon=lon, mode='text', hoverinfo='skip', showlegend=False,
        text=moves, textposition='bottom center',
        textfont=dict(size=11, color=INK_2),
    ))

    # Ring the selected state rather than recolouring it.
    if selected and selected in set(snap['state']):
        fig.add_trace(go.Choropleth(
            locations=[selected], locationmode='USA-states', z=[0],
            showscale=False, hoverinfo='skip',
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
            marker_line_color=INK, marker_line_width=2,
        ))

    fig.update_geos(
        scope='usa', bgcolor=SURFACE, lakecolor=SURFACE,
        landcolor=LAND, subunitcolor=SURFACE, coastlinecolor=AXIS,
    )
    fig.update_layout(
        title=dict(
            text=f'{label} — {metric_label} by state'
                 f'<br><span style="font-size:12px;color:{INK_2}">'
                 f'week {week}, ending {ending} · labels show the level and '
                 'the change from the prior week · click a state</span>',
            font=dict(size=17, color=INK), x=0.02, y=0.97,
        ),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=560,
        margin=dict(l=8, r=8, t=76, b=8), dragmode=False,
        hoverlabel=dict(bgcolor=PAGE, bordercolor=AXIS,
                        font=dict(color=INK, size=12)),
    )
    return fig


def history_figure(label, metric_label, state, xaxis='progress'):
    col, _, span = METRICS[metric_label]
    df = load(label)
    if df.empty or not state:
        return _blank('Click a state on the map.')

    # Against the crop's own development rather than the calendar: on the
    # progress axis a late-planted season is compared with earlier ones at the
    # same stage of the crop, not on the same date.
    by_progress = xaxis == 'progress' and col != 'PROGRESS'
    xcol = 'PROGRESS' if by_progress else 'week'

    df = reported(df[df['state_alpha'] == state], col, xcol)
    if df.empty:
        missing = 'progress' if by_progress else metric_label.lower()
        return _blank(f'No {label.lower()} {missing} data for {state}.')

    start, _, _, _ = season_shape(label)
    current = int(df['season'].max())
    prior_years = sorted(y for y in df['season'].unique() if y < current)

    fig = go.Figure()

    # Every prior season, faded, oldest first so recent ones sit on top.
    for i, year in enumerate(prior_years):
        g = df[df['season'] == year].sort_values('week')
        newest = bool(year == current - 1)
        fig.add_trace(go.Scatter(
            x=g[xcol], y=g[col], mode='lines',
            line=dict(color=SERIES_2 if newest else HISTORY,
                      width=2 if newest else 1.4),
            opacity=0.8 if newest else 0.32,
            name=season_span(label, int(year)) if newest else 'earlier seasons',
            legendgroup=None if newest else 'prior',
            showlegend=newest or i == 0,
            hovertemplate='%{y:.0f}<extra>'
                          f'{season_span(label, int(year))}' '</extra>',
        ))

    if prior_years:
        # Averaged at the same season week in every year, then placed at the
        # average progress those weeks had reached.
        past = df[df['season'].isin(prior_years)].groupby('week')
        mean = past[col].mean()
        mean_x = past[xcol].mean() if by_progress else mean.index
        fig.add_trace(go.Scatter(
            x=mean_x, y=mean.values, mode='lines',
            line=dict(color=INK_2, width=1.8, dash='dash'),
            name=f'{len(prior_years)}-season average',
            hovertemplate='avg %{y:.0f}<extra></extra>',
        ))

    now = df[df['season'] == current].sort_values('week')
    fig.add_trace(go.Scatter(
        x=now[xcol], y=now[col], mode='lines+markers',
        line=dict(color=SERIES_1, width=3), marker=dict(size=5),
        name=season_span(label, current), customdata=now['week_ending'],
        hovertemplate='%{y:.0f}<br>ending %{customdata}'
                      '<extra>' f'{season_span(label, current)}' '</extra>',
    ))

    if by_progress:
        x_conf = dict(
            title=dict(text='crop progress index (%)',
                       font=dict(size=11, color=INK_2)),
            range=[0, 100], dtick=20,
        )
        against = (f'against the previous {len(prior_years)} seasons, '
                   'at the same stage of the crop')
    else:
        # Plotted on the crop's own season week so seasons overlay, but ticked
        # in the calendar weeks the trade reads. Only the weeks this crop
        # actually reports get axis room, instead of a full year of it.
        present = df['week']
        first, last = int(present.min()), int(present.max())
        ticks = list(range(first, last + 1, max(2, round((last - first) / 8))))
        x_conf = dict(
            title=dict(text='week of year', font=dict(size=11, color=INK_2)),
            range=[first - 0.5, last + 0.5],
            tickmode='array', tickvals=ticks,
            ticktext=[str(calendar_week(w, start)) for w in ticks],
        )
        against = f'against the previous {len(prior_years)} seasons'

    fig.update_layout(
        title=dict(
            text=f'{state} — {metric_label}'
                 f'<br><span style="font-size:12px;color:{INK_2}">'
                 f'{season_span(label, current)} {against}</span>',
            font=dict(size=17, color=INK), x=0.02, y=0.96,
        ),
        xaxis=dict(
            gridcolor=GRID, linecolor=AXIS, zeroline=False,
            tickfont=dict(size=10, color=MUTED), **x_conf,
        ),
        yaxis=dict(
            range=list(span), gridcolor=GRID, linecolor=AXIS, zeroline=False,
            tickfont=dict(size=10, color=MUTED),
        ),
        legend=dict(orientation='h', y=-0.18, x=0,
                    font=dict(size=11, color=INK_2)),
        hovermode='closest',
        hoverlabel=dict(bgcolor=PAGE, bordercolor=AXIS,
                        font=dict(color=INK, size=12)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=560,
        margin=dict(l=48, r=16, t=76, b=64),
    )
    return fig


# --- Stat tiles -------------------------------------------------------------

TILE = {
    'flex': '1 1 0', 'padding': '12px 14px', 'background': PAGE,
    'border': f'1px solid {GRID}', 'borderRadius': '8px',
}
TILE_LABEL = {'fontSize': '11px', 'color': MUTED, 'margin': '0 0 4px'}
TILE_VALUE = {'fontSize': '22px', 'fontWeight': '600', 'color': INK,
              'margin': '0'}


def tile(label, value, color=INK):
    return html.Div([
        html.P(label, style=TILE_LABEL),
        html.P(value, style={**TILE_VALUE, 'color': color}),
    ], style=TILE)


def tiles(label, metric_label, state):
    col, higher_better, _ = METRICS[metric_label]
    df = load(label)
    if df.empty or not state:
        return []
    df = reported(df[df['state_alpha'] == state], col)
    if df.empty:
        return []

    current = int(df['season'].max())
    now = df[df['season'] == current].sort_values('week')
    last = now.iloc[-1]
    week = int(last['week'])
    value = last[col]

    # Consecutive reports only — see snapshot(); the winter gap is not a week.
    step = int(week - now['week'].iloc[-2]) if len(now) > 1 else None
    wow = value - now[col].iloc[-2] if step == 1 else None
    # Compared at the same point in each season, which for a wrapping crop is
    # the season week rather than the calendar week.
    prior = df[(df['season'] < current) & (df['week'] == week)]
    versus = value - prior[col].mean() if not prior.empty else None
    n_prior = prior['season'].nunique()

    def signed(delta):
        if delta is None or pd.isna(delta):
            return 'n/a', INK_2
        if abs(delta) < 0.05:
            return '– 0.0', INK_2
        good = (delta > 0) == higher_better
        arrow = '▲' if delta > 0 else '▼'
        return f'{arrow} {abs(delta):.1f}', GOOD if good else BAD

    wow_text, wow_color = signed(wow)
    vs_text, vs_color = signed(versus)
    return [
        tile(f'{state} · week {int(last["cal_week"])}, '
             f'{season_span(label, current)}', f'{value:.0f}'),
        tile('vs prior week', wow_text, wow_color),
        tile(f'vs {n_prior}-season avg, same week', vs_text, vs_color),
    ]


# --- Layout -----------------------------------------------------------------

app = Dash(__name__, title='Crop Conditions')

CARD = {
    'background': SURFACE, 'border': f'1px solid {GRID}',
    'borderRadius': '10px', 'padding': '8px',
}
FIELD = {'fontSize': '11px', 'color': MUTED, 'margin': '0 0 5px'}

app.layout = html.Div([
    html.Div([
        html.H1('Crop conditions', style={
            'fontSize': '24px', 'fontWeight': '600', 'color': INK,
            'margin': '0 0 4px'}),
        html.P(
            'USDA/NASS weekly crop progress. Condition Index = '
            '(5 × Excellent) + (4 × Good) + (3 × Fair) + (2 × Poor) '
            '+ (1 × Very Poor). Progress Index = the sum of every stage '
            'percentage ÷ (100 × the number of stages), 0 = before planting, '
            'and 100 = fully harvested.',
            style={'fontSize': '13px', 'color': INK_2, 'margin': '0'}),
    ], style={'margin': '0 0 16px'}),

    html.Div([
        html.Div([
            html.P('Commodity', style=FIELD),
            dcc.Dropdown(list(COMMODITIES), 'Cotton', id='commodity',
                         clearable=False),
        ], style={'flex': '1 1 180px'}),
        html.Div([
            html.P('Region', style=FIELD),
            dcc.Dropdown([NATIONAL], NATIONAL, id='region', clearable=False),
        ], style={'flex': '1 1 130px'}),
        html.Div([
            html.P('Metric', style=FIELD),
            dcc.Dropdown(list(METRICS), 'Good + Excellent (%)', id='metric',
                         clearable=False),
        ], style={'flex': '1 1 200px'}),
        html.Div([
            html.P('Map colour', style=FIELD),
            dcc.RadioItems(
                [{'label': ' level', 'value': 'level'},
                 {'label': ' change from prior week', 'value': 'change'}],
                'level', id='mode', inline=True,
                style={'fontSize': '13px'},
                inputStyle={'marginRight': '5px', 'accentColor': SERIES_1},
                labelStyle={'marginRight': '16px', 'color': INK,
                            'cursor': 'pointer'},
            ),
        ], style={'flex': '1 1 260px'}),
        html.Div([
            html.P('History x-axis', style=FIELD),
            dcc.RadioItems(
                [{'label': ' progress index', 'value': 'progress'},
                 {'label': ' week of year', 'value': 'week'}],
                'progress', id='xaxis', inline=True,
                style={'fontSize': '13px'},
                inputStyle={'marginRight': '5px', 'accentColor': SERIES_1},
                labelStyle={'marginRight': '16px', 'color': INK,
                            'cursor': 'pointer'},
            ),
        ], style={'flex': '1 1 240px'}),
    ], style={'display': 'flex', 'gap': '18px', 'flexWrap': 'wrap',
              'alignItems': 'flex-end', 'margin': '0 0 14px'}),

    html.Div([
        html.Div(dcc.Graph(id='map', config={'displayModeBar': False}),
                 style={**CARD, 'flex': '1 1 560px'}),
        html.Div([
            html.Div(id='tiles', style={'display': 'flex', 'gap': '8px',
                                        'margin': '0 0 8px'}),
            dcc.Graph(id='history', config={'displayModeBar': False}),
        ], style={**CARD, 'flex': '1 1 460px'}),
    ], style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap'}),
], style={'background': PAGE, 'minHeight': '100vh', 'padding': '24px',
          'boxSizing': 'border-box'})


# --- Callbacks --------------------------------------------------------------

@app.callback(
    Output('region', 'options'),
    Output('region', 'value'),
    Input('commodity', 'value'),
    Input('map', 'clickData'),
    State('region', 'value'),
)
def choose_region(commodity, click, current):
    """The Region dropdown is the single selection; the map writes into it."""
    df = load(commodity)
    options = regions(df)
    if ctx.triggered_id == 'map' and click:
        clicked = click['points'][0].get('location')
        if clicked in options:
            return options, clicked
    # Commodity changed (or first load): keep the region if it still reports.
    if current in options:
        return options, current
    return options, default_region(commodity, df)


@app.callback(
    Output('map', 'figure'),
    Input('commodity', 'value'),
    Input('metric', 'value'),
    Input('mode', 'value'),
    Input('region', 'value'),
)
def draw_map(commodity, metric, mode, region):
    return map_figure(commodity, metric, mode, region)


@app.callback(
    Output('history', 'figure'),
    Output('tiles', 'children'),
    Input('commodity', 'value'),
    Input('metric', 'value'),
    Input('region', 'value'),
    Input('xaxis', 'value'),
)
def draw_history(commodity, metric, region, xaxis):
    return (history_figure(commodity, metric, region, xaxis),
            tiles(commodity, metric, region))


if __name__ == '__main__':
    app.run(debug=True)
