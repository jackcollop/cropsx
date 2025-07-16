import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.caption('Crop conditions index is calculated as follows: (5 * Excellent) + (4 * Good) + (3 * Fair) + (2 * Poor) + (1 * Very Poor).')
st.caption('Select a region, or multiple regions, and click "Run"')

api_key = ${{secrets.NASS}}
state = 'US'

def cc(state,type):
    url = f'https://quickstats.nass.usda.gov/api/api_GET/?key={api_key}&commodity_desc=COTTON&statisticcat_desc=CONDITION&state_alpha={state}&format=csv'
    exc = pd.read_csv(url).set_index(['year','end_code'])
    exc = exc.pivot(columns='short_desc', values='Value')
    exc.replace(np.nan, 0, inplace=True)
    exc.columns = ['EXCELLENT','FAIR','GOOD','POOR','VERY_POOR']
    exc['INDEX'] = 5*exc['EXCELLENT'].astype(float) + 4*exc['GOOD'].astype(float) + 3*exc['FAIR'].astype(float) + 2*exc['POOR'].astype(float) + exc['VERY_POOR'].astype(float)
    exc['PVP'] = exc['POOR'].astype(float) + exc['VERY_POOR'].astype(float)
    exc['GE'] = exc['GOOD'].astype(float) + exc['EXCELLENT'].astype(float)
    week = exc.reset_index().end_code.iloc[-1]
    fig = px.line(exc.reset_index().pivot(index='end_code', columns='year', values=type).iloc[:,-10:], range_x=(20,50), title=f'CROP CONDITIONS INDEX ({state})', labels={'end_code':'week'})
    fig['data'][-1]['line']['width']=7
    return st.plotly_chart(fig)


def execute():
    if us: cc('US','INDEX')
    if tx: cc('TX','INDEX')
    if ga: cc('GA','INDEX')
    if ar: cc('AR','INDEX')
    if al: cc('AL','INDEX')
    if mo: cc('MO','INDEX')
    if ms: cc('MS','INDEX')
    if ok: cc('OK','INDEX')
    if nm: cc('NM','INDEX')
    if az: cc('AZ','INDEX')
    if ca: cc('CA','INDEX')
    if sc: cc('SC','INDEX')
    if fl: cc('FL','INDEX')
    if nc: cc('NC','INDEX')
    if la: cc('LA','INDEX')
    if va: cc('VA','INDEX')
    if tn: cc('TN','INDEX')




# Sidebar configurations
us = st.sidebar.checkbox('USA', key='a')
tx = st.sidebar.checkbox('TX', key='b')
ga = st.sidebar.checkbox('GA', key='c')
ar = st.sidebar.checkbox('AR', key='d')
al = st.sidebar.checkbox('AL', key='e')
mo = st.sidebar.checkbox('MO', key='f')
ms = st.sidebar.checkbox('MS', key='g')
ok = st.sidebar.checkbox('OK', key='h')
nm = st.sidebar.checkbox('NM', key='i')
az = st.sidebar.checkbox('AZ', key='j')
ca = st.sidebar.checkbox('CA', key='k')
sc = st.sidebar.checkbox('SC', key='l')
fl = st.sidebar.checkbox('FL', key='m')
nc = st.sidebar.checkbox('NC', key='n')
la = st.sidebar.checkbox('LA', key='q')
va = st.sidebar.checkbox('VA', key='s')
tn = st.sidebar.checkbox('TN', key='t')



run_btn = st.sidebar.button('Run', on_click=execute, key='z')
