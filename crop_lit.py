import datetime
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Crop Conditions', layout='wide')

api_key = os.environ['NASS']

# Commodity label -> NASS commodity_desc + optional class_desc filter.
# Wheat is split by class because CONDITION data is reported separately for
# winter and spring wheat (they cannot be pivoted together).
COMMODITIES = {
    'Corn': ('CORN', None),
    'Soybeans': ('SOYBEANS', None),
    'Winter Wheat': ('WHEAT', 'WINTER'),
    'Spring Wheat': ('WHEAT', 'SPRING, (EXCL DURUM)'),
    'Cotton': ('COTTON', None),
}

# Display label -> column produced in load().
METRICS = {
    'Condition Index': 'INDEX',
    'Good + Excellent (%)': 'GE',
    'Poor + Very Poor (%)': 'PVP',
}

# US plus the major producing states across corn/soy/wheat/cotton belts.
STATES = [
    'US', 'IA', 'IL', 'IN', 'OH', 'NE', 'MN', 'SD', 'ND', 'KS', 'MO', 'WI',
    'MI', 'KY', 'CO', 'MT', 'WA', 'ID', 'OR', 'TX', 'OK', 'AR', 'MS', 'LA',
    'GA', 'AL', 'NC', 'SC', 'VA', 'TN', 'FL', 'CA', 'AZ', 'NM',
]

CONDITIONS = ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'VERY POOR']


@st.cache_data(ttl=3600, show_spinner=False)
def load(commodity, klass, state):
    """Return a tidy frame of weekly conditions with INDEX/GE/PVP columns."""
    url = (
        'https://quickstats.nass.usda.gov/api/api_GET/?'
        f'key={api_key}&commodity_desc={commodity}'
        f'&statisticcat_desc=CONDITION&state_alpha={state}&format=csv'
    )
    df = pd.read_csv(url)

    # Keep only the "PCT <condition>" rows (e.g. "PCT GOOD").
    df = df[df['unit_desc'].str.startswith('PCT')].copy()
    if klass:
        df = df[df['class_desc'] == klass]
    if df.empty:
        return None

    df['cond'] = df['unit_desc'].str.replace('PCT ', '', regex=False)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    wide = df.pivot_table(
        index=['year', 'end_code'], columns='cond', values='Value', aggfunc='first'
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
    return wide.reset_index()


@st.cache_data(ttl=3600, show_spinner=False)
def load_states(commodity, klass):
    """Return latest-week Good + Excellent per state for a commodity.

    Fetches every state in one API call (no state_alpha filter) rather than
    one call per state, then keeps the most recent week for each state.
    """
    this_year = datetime.date.today().year
    url = (
        'https://quickstats.nass.usda.gov/api/api_GET/?'
        f'key={api_key}&commodity_desc={commodity}'
        f'&statisticcat_desc=CONDITION&agg_level_desc=STATE'
        f'&year__GE={this_year - 1}&format=csv'
    )
    df = pd.read_csv(url)

    df = df[df['unit_desc'].str.startswith('PCT')].copy()
    if klass:
        df = df[df['class_desc'] == klass]
    if df.empty:
        return None

    df['cond'] = df['unit_desc'].str.replace('PCT ', '', regex=False)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    wide = df.pivot_table(
        index=['state_alpha', 'year', 'end_code'],
        columns='cond', values='Value', aggfunc='first',
    )
    for c in CONDITIONS:
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide.fillna(0).astype(float)
    wide['GE'] = wide['GOOD'] + wide['EXCELLENT']
    wide = wide.reset_index()

    # Keep only the most recent week available for each state.
    latest_year = wide['year'].max()
    wide = wide[wide['year'] == latest_year]
    idx = wide.groupby('state_alpha')['end_code'].idxmax()
    latest = wide.loc[idx, ['state_alpha', 'end_code', 'GE']].copy()
    latest['year'] = latest_year
    return latest


def state_map(label, commodity, klass):
    """Render a US choropleth of Good + Excellent (%) by state."""
    try:
        df = load_states(commodity, klass)
    except Exception:
        df = None
    if df is None or df.empty:
        st.info(f'No {label} condition data available yet this season.')
        return

    week = int(df['end_code'].max())
    year = int(df['year'].iloc[0])
    fig = px.choropleth(
        df,
        locations='state_alpha',
        locationmode='USA-states',
        color='GE',
        scope='usa',
        color_continuous_scale='RdYlGn',
        range_color=(0, 100),
        labels={'GE': 'Good + Excellent (%)', 'state_alpha': 'State'},
        title=f'{label.upper()} — Good + Excellent (%) by state '
              f'(week {week}, {year})',
    )
    fig.update_coloraxes(colorbar_title='G+E %')
    st.plotly_chart(fig, use_container_width=True)


def chart(label, commodity, klass, state, metric_key, metric_label):
    try:
        df = load(commodity, klass, state)
    except Exception:
        df = None
    if df is None or df.empty:
        st.info(f'No {label} condition data available for {state}.')
        return

    # Weeks (end_code) on the x-axis, one line per year (last 11 years).
    series = df.pivot(index='end_code', columns='year', values=metric_key).iloc[:, -11:]
    fig = px.line(
        series,
        title=f'{label.upper()} — {metric_label} ({state})',
        labels={'end_code': 'week', 'value': metric_label, 'year': 'year'},
    )
    fig['data'][-1]['line']['width'] = 7  # emphasise the most recent year
    st.plotly_chart(fig, use_container_width=True)


# --- Sidebar controls -------------------------------------------------------
st.sidebar.header('Crop Conditions')

selected_commodities = st.sidebar.multiselect(
    'Commodities', list(COMMODITIES), default=['Cotton'],
)
selected_states = st.sidebar.multiselect(
    'Regions', STATES, default=['US'],
)
metric_label = st.sidebar.radio('Metric', list(METRICS))
metric_key = METRICS[metric_label]

run_btn = st.sidebar.button('Run')

# --- Main -------------------------------------------------------------------
st.caption(
    'Condition Index = (5 * Excellent) + (4 * Good) + (3 * Fair) '
    '+ (2 * Poor) + (1 * Very Poor).'
)
st.caption(
    'Good + Excellent and Poor + Very Poor are the summed percentages for '
    'those categories.'
)
st.caption('Select commodities and regions, choose a metric, and click "Run".')

# --- Overview map (renders on open) -----------------------------------------
st.subheader('Latest crop condition by state')
map_commodity = st.selectbox(
    'Map commodity', list(COMMODITIES), key='map_commodity',
)
map_comm, map_klass = COMMODITIES[map_commodity]
state_map(map_commodity, map_comm, map_klass)

st.divider()

if run_btn:
    if not selected_commodities or not selected_states:
        st.warning('Select at least one commodity and one region.')
    for label in selected_commodities:
        commodity, klass = COMMODITIES[label]
        for state in selected_states:
            chart(label, commodity, klass, state, metric_key, metric_label)
