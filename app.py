"""Crop conditions explorer (Dash).

A US choropleth of the latest weekly USDA/NASS crop condition by state, with
the week-over-week change labelled on each state. Click a state to see its
season history against the previous seasons.

Run: python app.py
"""

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
}

CONDITIONS = ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'VERY POOR']

# Weeks 1-10 are hangovers from the prior crop year (cotton, for instance,
# only gets going around week 20) and distort the season shape.
FIRST_WEEK = 11

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


def _request(commodity, klass, first_year, last_year):
    """One NASS request for a year range, split in half if it is too large."""
    url = (
        'https://quickstats.nass.usda.gov/api/api_GET/?'
        f'key={API_KEY}&commodity_desc={commodity}'
        f'&statisticcat_desc=CONDITION&agg_level_desc=STATE'
        f'&year__GE={first_year}&year__LE={last_year}&format=csv'
    )
    if klass:
        url += f'&class_desc={quote(klass)}'
    try:
        return pd.read_csv(url)
    except Exception:
        # The API caps a response at 50k records; halve the window and retry.
        if last_year <= first_year:
            raise
        mid = (first_year + last_year) // 2
        return pd.concat(
            [_request(commodity, klass, first_year, mid),
             _request(commodity, klass, mid + 1, last_year)],
            ignore_index=True,
        )


def _tidy(raw, klass):
    """Pivot the "PCT <condition>" rows into INDEX/GE/PVP per state-week."""
    df = raw[raw['unit_desc'].str.startswith('PCT')].copy()
    if klass:
        df = df[df['class_desc'] == klass]
    if df.empty:
        return df

    df['cond'] = df['unit_desc'].str.replace('PCT ', '', regex=False)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    wide = df.pivot_table(
        index=['state_alpha', 'year', 'end_code', 'week_ending'],
        columns='cond', values='Value', aggfunc='first',
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

    out = wide.reset_index()
    out['end_code'] = pd.to_numeric(out['end_code'], errors='coerce')
    return out[out['end_code'] >= FIRST_WEEK].sort_values(
        ['state_alpha', 'year', 'end_code']
    )


def load(label):
    """Return every state-week for a commodity over the last SEASONS years.

    Cached for an hour, so the only thing that costs an API call is picking a
    commodity you have not looked at yet.
    """
    commodity, klass = COMMODITIES[label]
    hit = _cache.get(label)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    this_year = datetime.date.today().year
    try:
        raw = _request(commodity, klass, this_year - SEASONS + 1, this_year)
        df = _tidy(raw, klass)
    except Exception:
        df = pd.DataFrame()
    _cache[label] = (time.time(), df)
    return df


def snapshot(df, col):
    """Latest week per state in the latest season, with its week-on-week move."""
    if df.empty:
        return pd.DataFrame()
    season = df[df['year'] == df['year'].max()]
    rows = []
    for state, g in season.groupby('state_alpha'):
        g = g.sort_values('end_code')
        prev = g[col].iloc[-2] if len(g) > 1 else None
        rows.append({
            'state': state,
            'value': g[col].iloc[-1],
            'delta': None if prev is None else g[col].iloc[-1] - prev,
            'week': int(g['end_code'].iloc[-1]),
            'week_ending': g['week_ending'].iloc[-1],
        })
    return pd.DataFrame(rows)


def default_state(label, df):
    """Leading producer if it reports, else whichever state reports most."""
    if df.empty:
        return None
    preferred = DEFAULT_STATE.get(label)
    if preferred in set(df['state_alpha']):
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
        return _blank(f'No {label.lower()} condition data reported yet.')

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

    week = int(snap['week'].max())
    ending = snap.loc[snap['week'].idxmax(), 'week_ending']

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


def history_figure(label, metric_label, state):
    col, _, span = METRICS[metric_label]
    df = load(label)
    if df.empty or not state:
        return _blank('Click a state on the map.')

    df = df[df['state_alpha'] == state]
    if df.empty:
        return _blank(f'No {label.lower()} condition data for {state}.')

    current = int(df['year'].max())
    prior_years = sorted(y for y in df['year'].unique() if y < current)

    fig = go.Figure()

    # Every prior season, faded, oldest first so recent ones sit on top.
    for i, year in enumerate(prior_years):
        g = df[df['year'] == year]
        newest = bool(year == current - 1)
        fig.add_trace(go.Scatter(
            x=g['end_code'], y=g[col], mode='lines',
            line=dict(color=SERIES_2 if newest else HISTORY,
                      width=2 if newest else 1.4),
            opacity=0.8 if newest else 0.32,
            name=str(int(year)) if newest else 'earlier seasons',
            legendgroup=None if newest else 'prior',
            showlegend=newest or i == 0,
            hovertemplate='%{y:.0f}<extra>' f'{int(year)}' '</extra>',
        ))

    if prior_years:
        mean = df[df['year'].isin(prior_years)].groupby('end_code')[col].mean()
        fig.add_trace(go.Scatter(
            x=mean.index, y=mean.values, mode='lines',
            line=dict(color=INK_2, width=1.8, dash='dash'),
            name=f'{len(prior_years)}-season average',
            hovertemplate='avg %{y:.0f}<extra></extra>',
        ))

    now = df[df['year'] == current]
    fig.add_trace(go.Scatter(
        x=now['end_code'], y=now[col], mode='lines+markers',
        line=dict(color=SERIES_1, width=3), marker=dict(size=5),
        name=str(current), customdata=now['week_ending'],
        hovertemplate='%{y:.0f}<br>ending %{customdata}'
                      '<extra>' f'{current}' '</extra>',
    ))

    fig.update_layout(
        title=dict(
            text=f'{state} — {metric_label}'
                 f'<br><span style="font-size:12px;color:{INK_2}">'
                 f'{current} against the previous {len(prior_years)} seasons'
                 '</span>',
            font=dict(size=17, color=INK), x=0.02, y=0.96,
        ),
        xaxis=dict(
            title=dict(text='week of year', font=dict(size=11, color=INK_2)),
            gridcolor=GRID, linecolor=AXIS, zeroline=False,
            tickfont=dict(size=10, color=MUTED),
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
    df = df[df['state_alpha'] == state]
    if df.empty:
        return []

    current = int(df['year'].max())
    now = df[df['year'] == current].sort_values('end_code')
    week = int(now['end_code'].iloc[-1])
    value = now[col].iloc[-1]

    wow = value - now[col].iloc[-2] if len(now) > 1 else None
    prior = df[(df['year'] < current) & (df['end_code'] == week)]
    versus = value - prior[col].mean() if not prior.empty else None
    n_prior = prior['year'].nunique()

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
        tile(f'{state} · week {week}, {current}', f'{value:.0f}'),
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
    dcc.Store(id='selected'),

    html.Div([
        html.H1('Crop conditions', style={
            'fontSize': '24px', 'fontWeight': '600', 'color': INK,
            'margin': '0 0 4px'}),
        html.P(
            'USDA/NASS weekly crop progress. Condition Index = '
            '(5 × Excellent) + (4 × Good) + (3 × Fair) + (2 × Poor) '
            '+ (1 × Very Poor). Weeks 1–10 are excluded as pre-season.',
            style={'fontSize': '13px', 'color': INK_2, 'margin': '0'}),
    ], style={'margin': '0 0 16px'}),

    html.Div([
        html.Div([
            html.P('Commodity', style=FIELD),
            dcc.Dropdown(list(COMMODITIES), 'Cotton', id='commodity',
                         clearable=False),
        ], style={'flex': '1 1 180px'}),
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
    Output('selected', 'data'),
    Input('map', 'clickData'),
    Input('commodity', 'value'),
    State('selected', 'data'),
)
def choose_state(click, commodity, current):
    df = load(commodity)
    reported = set(df['state_alpha']) if not df.empty else set()
    if ctx.triggered_id == 'map' and click:
        clicked = click['points'][0].get('location')
        if clicked in reported:
            return clicked
    # Commodity changed (or first load): keep the state if it still reports.
    if current in reported:
        return current
    return default_state(commodity, df)


@app.callback(
    Output('map', 'figure'),
    Input('commodity', 'value'),
    Input('metric', 'value'),
    Input('mode', 'value'),
    Input('selected', 'data'),
)
def draw_map(commodity, metric, mode, selected):
    return map_figure(commodity, metric, mode, selected)


@app.callback(
    Output('history', 'figure'),
    Output('tiles', 'children'),
    Input('commodity', 'value'),
    Input('metric', 'value'),
    Input('selected', 'data'),
)
def draw_history(commodity, metric, selected):
    return (history_figure(commodity, metric, selected),
            tiles(commodity, metric, selected))


if __name__ == '__main__':
    app.run(debug=True)
