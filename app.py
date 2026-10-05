import os
import time
from datetime import date
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

API_KEY = os.environ['NASS_API_KEY']

# --- Settings ---------------------------------------------------------------

# Label -> (NASS commodity, NASS class, (first, last) calendar week reported).
# Wheat is split by class because NASS reports winter and spring separately.
# Winter wheat's season wraps the new year: autumn planting (week 36 on)
# belongs to the crop harvested the following summer.
CROPS = {
    'Cotton': ('COTTON', None, (12, 50)),
    'Corn': ('CORN', None, (8, 48)),
    'Soybeans': ('SOYBEANS', None, (12, 50)),
    'Winter Wheat': ('WHEAT', 'WINTER', (36, 35)),
    'Spring Wheat': ('WHEAT', 'SPRING, (EXCL DURUM)', (12, 40)),
}

# Label -> (column, higher is better, axis range).
METRICS = {
    'Good + Excellent (%)': ('GE', True, (0, 100)),
    'Poor + Very Poor (%)': ('PVP', False, (0, 100)),
    'Condition Index': ('INDEX', True, (100, 500)),
    'Progress Index (%)': ('PROGRESS', True, (0, 100)),
}

SEASONS = 11          # seasons of history to show
CACHE_TTL = 3600      # seconds to keep a crop's data in memory
NATIONAL = 'US'
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# One row per state-week.
KEYS = ['state_alpha', 'season', 'week', 'cal_week', 'week_ending']

# --- Palette ----------------------------------------------------------------

SURFACE = '#1a1a19'
HOVER = '#0d0d0d'
INK = '#ffffff'
INK_2 = '#c3c2b7'
MUTED = '#898781'
GRID = '#2c2c2a'
AXIS = '#383835'
LAND = '#262625'        # states with nothing reported
SERIES_1 = '#3987e5'    # current season
SERIES_2 = '#d95926'    # last season
HISTORY = '#5598e7'     # older seasons
GOOD = '#0ca30c'
BAD = '#d03b3b'
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Red (bad) to green (good). Every step is dark enough for white labels.
LEVEL_SCALE = [
    [0.00, '#7a1616'], [0.20, '#a52a2a'], [0.40, '#b8502a'],
    [0.55, '#a87a12'], [0.72, '#6f8f2a'], [0.88, '#3d8b3d'],
    [1.00, '#2e9b45'],
]
CHANGE_SCALE = [
    [0.00, '#7a1616'], [0.22, '#b8352b'], [0.44, '#383835'],
    [0.56, '#383835'], [0.78, '#3d8b3d'], [1.00, '#2e9b45'],
]


def flip(scale):
    """Reverse a colour scale, for metrics where high is bad."""
    return [[round(1 - pos, 2), colour] for pos, colour in reversed(scale)]


# Where to put each state's label on the map.
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


def season_name(label, year):
    """'2026', or '2025/26' for a season that spans the new year."""
    start, end = CROPS[label][2]
    return f'{year - 1}/{str(year)[-2:]}' if start > end else str(year)


# --- Data -------------------------------------------------------------------

def fetch(commodity, klass, stat, level, first, last):
    """One Quick Stats query. The API caps a response at 50k rows, so a range
    that is too big is split in two and retried."""
    url = ('https://quickstats.nass.usda.gov/api/api_GET/?'
           f'key={API_KEY}&commodity_desc={commodity}'
           f'&statisticcat_desc={stat}&agg_level_desc={level}'
           f'&year__GE={first}&year__LE={last}&format=csv')
    if klass:
        url += f'&class_desc={quote(klass)}'
    try:
        raw = pd.read_csv(url)
    except Exception:
        if first >= last:
            raise
        mid = (first + last) // 2
        return pd.concat([fetch(commodity, klass, stat, level, first, mid),
                          fetch(commodity, klass, stat, level, mid + 1, last)],
                         ignore_index=True)
    if level == 'NATIONAL':
        raw['state_alpha'] = NATIONAL
    return raw


def tidy(raw, label, stat):
    """Turn raw NASS rows into one row per state-week.

    Condition becomes INDEX, GE and PVP. Progress becomes one column per stage,
    and is reduced to a single index later in load().
    """
    _, klass, (start, end) = CROPS[label]
    # PCT DEFOLIATED was only ever reported by California, and only to 2021.
    df = raw[raw['unit_desc'].str.startswith('PCT')
             & (raw['unit_desc'] != 'PCT DEFOLIATED')]
    if klass:
        df = df[df['class_desc'] == klass]
    df = df.assign(
        Value=pd.to_numeric(df['Value'], errors='coerce'),
        cal_week=pd.to_numeric(df['end_code'], errors='coerce').clip(1, 52),
        ending=pd.to_datetime(df['week_ending'], errors='coerce'),
    ).dropna(subset=['cal_week', 'ending'])

    # Drop weeks outside the crop's season (e.g. a stray January cotton report
    # that is really the tail of last year's harvest).
    wraps = start > end
    if wraps:
        df = df[(df['cal_week'] >= start) | (df['cal_week'] <= end)]
    else:
        df = df[df['cal_week'].between(start, end)]
    if df.empty:
        return pd.DataFrame()

    # The season is worked out from the date, not NASS's `year`, which is
    # inconsistent for winter wheat. Autumn weeks of a wrapping crop belong to
    # the next year's season. `week` counts from the season start so that
    # seasons line up.
    df['season'] = df['ending'].dt.year + ((df['cal_week'] >= start) & wraps)
    df['week'] = (df['cal_week'] - start) % 52 + 1

    wide = df.pivot_table(index=KEYS, columns='unit_desc', values='Value',
                          aggfunc='first')
    if stat == 'CONDITION':
        pct = wide.reindex(columns=['PCT EXCELLENT', 'PCT GOOD', 'PCT FAIR',
                                    'PCT POOR', 'PCT VERY POOR']).fillna(0)
        ex, good, fair, poor, very_poor = (pct[c] for c in pct)
        out = pd.DataFrame({
            'INDEX': 5 * ex + 4 * good + 3 * fair + 2 * poor + very_poor,
            'GE': ex + good,
            'PVP': poor + very_poor,
        })
    else:
        # NASS stops publishing a stage once it hits 100%, so carry the last
        # reading forward. A stage reported later in the season is 0 until
        # then; one the state never reported that season stays empty.
        season = wide.groupby(level=['state_alpha', 'season'])
        reports = wide.notna().groupby(level=['state_alpha', 'season']) \
                              .transform('any')
        out = season.ffill().fillna(0).where(reports)
        out.columns.name = None
    out = out.reset_index()

    # NASS sometimes files a crop's last harvest week under the next year,
    # which leaves a phantom season of two or three rows. Drop those, and keep
    # only the newest SEASONS seasons.
    weeks = out.groupby('season')['week'].nunique()
    keep = weeks[weeks >= 0.25 * weeks.median()].index[-SEASONS:]
    return out[out['season'].isin(keep)]


_cache = {}


def load(label):
    """Every state-week of condition and progress for a crop, plus the US.

    Finished seasons are downloaded once and saved under data/. The current
    season is always fetched fresh: NASS caches each query, and a cached
    multi-year query can lag a week behind. The US figures come from NASS
    directly because they are acreage-weighted, which we can't redo.
    """
    if label in _cache and time.time() - _cache[label][0] < CACHE_TTL:
        return _cache[label][1]

    commodity, klass, (start, end) = CROPS[label]
    this_year = date.today().year
    first_year = this_year - SEASONS + 1

    def get(stat, first, last):
        parts = []
        for level in ('STATE', 'NATIONAL'):
            try:
                raw = fetch(commodity, klass, stat, level, first, last)
            except Exception:
                continue    # a crop can lack a statistic altogether
            parts.append(tidy(raw, label, stat))
        parts = [p for p in parts if len(p)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    data = {}
    for stat in ('CONDITION', 'PROGRESS'):
        # The window and season weeks are in the file name, so a new year or
        # a changed season starts a fresh download.
        kind = 'progress-stages' if stat == 'PROGRESS' else 'condition'
        slug = label.lower().replace(' ', '-')
        path = os.path.join(HISTORY_DIR, f'{slug}-{kind}-{first_year}-'
                                         f'{this_year - 1}-w{start}-{end}.csv')
        if os.path.exists(path):
            history = pd.read_csv(path)
        else:
            history = get(stat, first_year, this_year - 1)
            if len(history):
                # NASS ignores year__LE, so the current year comes back too.
                # Cut it by date, not season: winter wheat planted last autumn
                # is this year's season but belongs in the history file.
                history = history[pd.to_datetime(history['week_ending'])
                                  .dt.year < this_year]
                os.makedirs(HISTORY_DIR, exist_ok=True)
                history.to_csv(path, index=False)
        parts = [f for f in (history, get(stat, this_year, this_year)) if len(f)]
        data[stat] = (pd.concat(parts, ignore_index=True)
                      .drop_duplicates(subset=KEYS, keep='last')
                      if parts else pd.DataFrame())

    # Progress index: the average of a state's stage percentages, 0 before
    # planting and 100 once the last stage is done. Each state only counts the
    # stages it reported in every finished season, so the index is built the
    # same way each year. Counting stages a state doesn't report as 0 would
    # cap most cotton states at 83%, because only a few report PCT EMERGED.
    prog = data['PROGRESS']
    if len(prog):
        stages = [c for c in prog.columns if c.startswith('PCT ')]
        seen = prog.groupby(['state_alpha', 'season'])[stages].agg(
            lambda s: s.notna().any())
        finished = seen[seen.index.get_level_values('season')
                        < prog['season'].max()]
        keep = seen.groupby(level='state_alpha').all()   # no finished season yet
        keep.update(finished.groupby(level='state_alpha').all())
        mask = keep.reindex(prog['state_alpha']).to_numpy(dtype=bool)
        counts = mask.sum(axis=1)
        values = prog[stages].fillna(0).to_numpy()
        prog = prog[KEYS].assign(
            PROGRESS=(values * mask).sum(axis=1) / counts.clip(min=1))
        prog = prog[counts > 0]

    # Outer join: planting starts weeks before the first condition report.
    cond = data['CONDITION']
    if len(cond) and len(prog):
        df = cond.merge(prog, on=KEYS, how='outer')
    else:
        df = cond if len(cond) else prog
    df = df.reindex(columns=KEYS + ['INDEX', 'GE', 'PVP', 'PROGRESS'])
    df = df.sort_values(['state_alpha', 'season', 'week'])
    _cache[label] = (time.time(), df)
    return df


def snapshot(df, col):
    """Each state's latest reading this season, and its change on the week."""
    df = df[df['state_alpha'] != NATIONAL].dropna(subset=[col])
    df = df[df['season'] == df['season'].max()].sort_values('week')
    rows = []
    for state, g in df.groupby('state_alpha'):
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        # Only a report from the week before is a weekly change; winter wheat
        # goes quiet from December to February.
        weekly = prev is not None and last['week'] - prev['week'] == 1
        rows.append({
            'state': state,
            'value': last[col],
            'delta': last[col] - prev[col] if weekly else None,
            'week': int(last['cal_week']),
            'week_ending': last['week_ending'],
        })
    return pd.DataFrame(rows)


# --- Figures ----------------------------------------------------------------

def blank(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=13, color=INK_2))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=520,
        margin=dict(l=16, r=16, t=16, b=16), font=dict(family=FONT),
    )
    return fig


def map_figure(label, metric_label, mode, selected):
    col, higher_better, span = METRICS[metric_label]
    snap = snapshot(load(label), col)
    if snap.empty:
        return blank(f'No {label.lower()} {metric_label.lower()} '
                     'reported yet.')

    if mode == 'change':
        snap = snap.dropna(subset=['delta'])
        if snap.empty:
            return blank('Only one week reported so far this season — '
                         'no change to show.')
        z = snap['delta']
        limit = max(4.0, float(z.abs().max()))
        zmin, zmax = -limit, limit
        scale = CHANGE_SCALE if higher_better else flip(CHANGE_SCALE)
    else:
        z = snap['value']
        zmin, zmax = span
        scale = LEVEL_SCALE if higher_better else flip(LEVEL_SCALE)

    # Title the map with the newest report, by date (for winter wheat a high
    # calendar week can be the oldest).
    newest = snap.loc[pd.to_datetime(snap['week_ending']).idxmax()]

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
            thickness=10, len=0.6, x=0.98,
            tickfont=dict(size=10, color=MUTED), outlinewidth=0,
        ),
    ))

    # The number on each state.
    known = snap[snap['state'].isin(CENTROIDS)]
    fig.add_trace(go.Scattergeo(
        lat=[CENTROIDS[s][0] for s in known['state']],
        lon=[CENTROIDS[s][1] for s in known['state']],
        mode='text', hoverinfo='skip', showlegend=False,
        text=([f'{v:+.0f}' for v in known['delta']] if mode == 'change'
              else [f'{v:.0f}' for v in known['value']]),
        textfont=dict(size=12, color=INK),
    ))

    # Outline the selected state.
    if selected in set(snap['state']):
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
            text=f'{metric_label}, week {int(newest["week"])} '
                 f'ending {newest["week_ending"]}',
            font=dict(size=15, color=INK_2), x=0.01, y=0.97,
        ),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=520,
        margin=dict(l=0, r=0, t=44, b=0), dragmode=False,
        font=dict(family=FONT),
        hoverlabel=dict(bgcolor=HOVER, bordercolor=AXIS,
                        font=dict(color=INK, size=12)),
    )
    return fig


def history_figure(label, metric_label, state, xaxis):
    col, _, span = METRICS[metric_label]
    if not state:
        return blank('Click a state on the map.')

    # On the progress axis, a late season is compared with earlier ones at the
    # same stage of the crop rather than on the same date.
    by_progress = xaxis == 'progress' and col != 'PROGRESS'
    xcol = 'PROGRESS' if by_progress else 'week'

    df = load(label)
    df = df[df['state_alpha'] == state].dropna(subset=[col, xcol])
    if df.empty:
        missing = 'progress' if by_progress else metric_label.lower()
        return blank(f'No {label.lower()} {missing} data for {state}.')

    current = int(df['season'].max())
    prior_years = sorted(y for y in df['season'].unique() if y < current)

    fig = go.Figure()

    # Earlier seasons, faded, with last season highlighted.
    for i, year in enumerate(prior_years):
        g = df[df['season'] == year].sort_values('week')
        last_season = bool(year == current - 1)
        fig.add_trace(go.Scatter(
            x=g[xcol], y=g[col], mode='lines',
            line=dict(color=SERIES_2 if last_season else HISTORY,
                      width=2 if last_season else 1.4),
            opacity=0.8 if last_season else 0.32,
            name=(season_name(label, int(year)) if last_season
                  else 'earlier seasons'),
            legendgroup=None if last_season else 'prior',
            showlegend=last_season or i == 0,
            hovertemplate='%{y:.0f}<extra>'
                          f'{season_name(label, int(year))}' '</extra>',
        ))

    if prior_years:
        # Average by season week, plotted at that week's average progress.
        past = df[df['season'].isin(prior_years)].groupby('week')
        mean = past[col].mean()
        fig.add_trace(go.Scatter(
            x=past[xcol].mean() if by_progress else mean.index, y=mean.values,
            mode='lines', line=dict(color=INK_2, width=1.8, dash='dash'),
            name=f'{len(prior_years)}-season average',
            hovertemplate='avg %{y:.0f}<extra></extra>',
        ))

    now = df[df['season'] == current].sort_values('week')
    fig.add_trace(go.Scatter(
        x=now[xcol], y=now[col], mode='lines+markers',
        line=dict(color=SERIES_1, width=3), marker=dict(size=5),
        name=season_name(label, current), customdata=now['week_ending'],
        hovertemplate='%{y:.0f}<br>ending %{customdata}'
                      '<extra>' f'{season_name(label, current)}' '</extra>',
    ))

    if by_progress:
        x_conf = dict(
            title=dict(text='crop progress index (%)',
                       font=dict(size=11, color=INK_2)),
            range=[0, 100], dtick=20,
        )
    else:
        # Plotted by season week so seasons overlay, labelled with the
        # calendar week everyone reads.
        start = CROPS[label][2][0]
        first, last = int(df['week'].min()), int(df['week'].max())
        ticks = list(range(first, last + 1, max(2, round((last - first) / 8))))
        x_conf = dict(
            title=dict(text='week of year', font=dict(size=11, color=INK_2)),
            range=[first - 0.5, last + 0.5],
            tickmode='array', tickvals=ticks,
            ticktext=[str((start + w - 2) % 52 + 1) for w in ticks],
        )

    fig.update_layout(
        title=dict(text=f'{state}, {metric_label.lower()}',
                   font=dict(size=15, color=INK_2), x=0.01, y=0.97),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                   tickfont=dict(size=10, color=MUTED), **x_conf),
        yaxis=dict(range=list(span), gridcolor=GRID, linecolor=AXIS,
                   zeroline=False, tickfont=dict(size=10, color=MUTED)),
        legend=dict(orientation='h', y=-0.16, x=0,
                    font=dict(size=11, color=INK_2)),
        hovermode='closest',
        hoverlabel=dict(bgcolor=HOVER, bordercolor=AXIS,
                        font=dict(color=INK, size=12)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=520,
        margin=dict(l=36, r=8, t=44, b=56), font=dict(family=FONT),
    )
    return fig


# --- Stat tiles -------------------------------------------------------------

TILE_VALUE = {'fontSize': '28px', 'fontWeight': '500', 'color': INK,
              'margin': '0', 'lineHeight': '1.1',
              'fontVariantNumeric': 'tabular-nums'}
TILE_LABEL = {'fontSize': '11px', 'color': MUTED, 'margin': '3px 0 0'}


def tiles(label, metric_label, state):
    col, higher_better, _ = METRICS[metric_label]
    df = load(label)
    df = df[df['state_alpha'] == state].dropna(subset=[col])
    if df.empty:
        return []

    current = int(df['season'].max())
    now = df[df['season'] == current].sort_values('week')
    last = now.iloc[-1]
    week, value = last['week'], last[col]

    # Weekly change only from the week before (see snapshot). The average is
    # taken at the same season week in earlier years.
    weekly = len(now) > 1 and week - now['week'].iloc[-2] == 1
    wow = value - now[col].iloc[-2] if weekly else None
    prior = df[(df['season'] < current) & (df['week'] == week)]
    versus = value - prior[col].mean() if len(prior) else None

    def tile(caption, text, color=INK):
        return html.Div([html.P(text, style={**TILE_VALUE, 'color': color}),
                         html.P(caption, style=TILE_LABEL)],
                        style={'flex': '1 1 0'})

    def change(caption, delta):
        if delta is None or pd.isna(delta):
            return tile(caption, 'n/a', INK_2)
        if abs(delta) < 0.05:
            return tile(caption, '– 0.0', INK_2)
        arrow = '▲' if delta > 0 else '▼'
        good = (delta > 0) == higher_better
        return tile(caption, f'{arrow} {abs(delta):.1f}', GOOD if good else BAD)

    return [
        tile(f'{state}, {season_name(label, current)}', f'{value:.0f}'),
        change('on the week', wow),
        change(f'vs {prior["season"].nunique()}-season average', versus),
    ]


# --- Layout -----------------------------------------------------------------

app = Dash(__name__, title='Crop Conditions')

FIELD = {'fontSize': '11px', 'color': MUTED, 'margin': '0 0 5px'}
RADIO = dict(
    inline=True, style={'fontSize': '13px'},
    inputStyle={'marginRight': '5px', 'accentColor': SERIES_1},
    labelStyle={'marginRight': '16px', 'color': INK, 'cursor': 'pointer'},
)

app.layout = html.Div([
    html.H1('Crop conditions', style={
        'fontSize': '20px', 'fontWeight': '600', 'color': INK,
        'margin': '0 0 18px'}),

    html.Div([
        html.Div([
            html.P('Commodity', style=FIELD),
            dcc.Dropdown(list(CROPS), 'Cotton', id='commodity',
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
                'level', id='mode', **RADIO),
        ], style={'flex': '1 1 260px'}),
        html.Div([
            html.P('History x-axis', style=FIELD),
            dcc.RadioItems(
                [{'label': ' progress index', 'value': 'progress'},
                 {'label': ' week of year', 'value': 'week'}],
                'progress', id='xaxis', **RADIO),
        ], style={'flex': '1 1 240px'}),
    ], style={'display': 'flex', 'gap': '18px', 'flexWrap': 'wrap',
              'alignItems': 'flex-end', 'margin': '0 0 14px'}),

    html.Div([
        html.Div(dcc.Graph(id='map', config={'displayModeBar': False}),
                 style={'flex': '1 1 560px'}),
        html.Div([
            html.Div(id='tiles', style={'display': 'flex', 'gap': '20px',
                                        'margin': '0 0 14px'}),
            dcc.Graph(id='history', config={'displayModeBar': False}),
        ], style={'flex': '1 1 460px'}),
    ], style={'display': 'flex', 'gap': '32px', 'flexWrap': 'wrap'}),
], style={'background': SURFACE, 'minHeight': '100vh', 'padding': '28px 32px',
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
    """The Region dropdown holds the selection; clicking the map sets it."""
    reporting = set(load(commodity)['state_alpha'])
    options = ([NATIONAL] if NATIONAL in reporting else []) \
        + sorted(reporting - {NATIONAL})
    clicked = click['points'][0].get('location') if click else None
    if ctx.triggered_id == 'map' and clicked in options:
        return options, clicked
    if current in options:
        return options, current
    return options, options[0] if options else None


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
