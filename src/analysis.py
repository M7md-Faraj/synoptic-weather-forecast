"""
Analysis helpers for the dashboard. Improved get_current_conditions to
derive humidity/wind when missing using simple heuristics.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()

def plotly_time_series(df: pd.DataFrame, col: str, title: str = None):
    dfc = df.copy()
    if 'date' in dfc.columns:
        dfc = dfc.sort_values('date')
    dfc['rolling_7'] = dfc[col].rolling(7, min_periods=1).mean()
    title = title or f"Time series of {col}"
    fig = px.line(dfc, x='date', y=col, title=title, labels={col: col, 'date': 'Date'})
    fig.add_traces(px.line(dfc, x='date', y='rolling_7').data)
    fig.update_traces(mode='lines')
    fig.update_layout(hovermode='x unified', template='plotly_white')
    return fig

def correlation_heatmap_plotly(df: pd.DataFrame):
    df_num = df.select_dtypes(include=[np.number]).copy()
    corr = df_num.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="corr")
    ))
    fig.update_layout(title='Correlation heatmap', template='plotly_white', height=480)
    return fig

def distribution_plot(df: pd.DataFrame, col: str):
    fig = px.histogram(df, x=col, nbins=60, marginal='box', title=f'Distribution of {col}')
    fig.update_layout(template='plotly_white')
    return fig

def weather_condition_summary(df: pd.DataFrame, temp_col='mean_temp', precip_col='precipitation'):
    dfc = df.copy()
    if temp_col not in dfc.columns:
        dfc[temp_col] = 0.0
    if precip_col not in dfc.columns:
        dfc[precip_col] = 0.0

    hot = int((dfc[temp_col] >= 25).sum())
    warm = int(((dfc[temp_col] >= 15) & (dfc[temp_col] < 25)).sum())
    cool = int(((dfc[temp_col] >= 5) & (dfc[temp_col] < 15)).sum())
    cold = int((dfc[temp_col] < 5).sum())
    rainy = int((dfc[precip_col] > 0).sum())
    totals = {'hot': hot, 'warm': warm, 'cool': cool, 'cold': cold, 'rainy_days': rainy}

    fig = px.bar(x=list(totals.keys()), y=list(totals.values()), labels={'x': 'condition', 'y': 'count'}, title='Weather condition counts')
    fig.update_layout(template='plotly_white', height=240)
    return totals, fig

def _temp_icon(t):
    if t >= 25:
        return '☀️'
    if t >= 15:
        return '⛅️'
    if t >= 5:
        return '🌥️'
    return '❄️'

def build_5day_forecast(df: pd.DataFrame):
    """Return list of 5 days from historical data (grouped by date)."""
    out = []
    if 'date' not in df.columns:
        sample = df.tail(5)
        for _, r in sample.iterrows():
            try:
                d = pd.to_datetime(r.get('date', pd.Timestamp.now())).strftime('%a')
            except Exception:
                d = 'Day'
            out.append({'day': d, 'date': str(r.get('date','')), 'temp': int(round(r.get('mean_temp', r.get('max_temp', 0)))), 'icon': _temp_icon(r.get('mean_temp', 0))})
        return out

    df_sorted = df.copy()
    parsed = pd.to_datetime(df_sorted['date'], errors='coerce', infer_datetime_format=True)
    if parsed.isna().all():
        try:
            parsed = pd.to_datetime(df_sorted['date'], format='%Y%m%d', errors='coerce')
        except Exception:
            parsed = pd.to_datetime(df_sorted['date'], errors='coerce')
    df_sorted['parsed_date'] = parsed
    df_sorted = df_sorted.dropna(subset=['parsed_date'])
    if df_sorted.empty:
        return []
    df_sorted['date_only'] = df_sorted['parsed_date'].dt.date
    grouped = df_sorted.groupby('date_only').agg({'mean_temp':'mean'}).reset_index()
    last5 = grouped.tail(5)
    for _, r in last5.iterrows():
        d = pd.to_datetime(r['date_only']).strftime('%a')
        out.append({'day': d, 'date': str(r['date_only']), 'temp': int(round(r['mean_temp'])), 'icon': _temp_icon(r['mean_temp'])})
    return out

def build_hourly_preview(df: pd.DataFrame, n=5):
    out = []
    if 'time' in df.columns:
        sample = df.sort_values('date').tail(n)
        for _, r in sample.iterrows():
            t = r.get('time') or pd.to_datetime(r['date']).strftime('%H:%M')
            out.append({'time': t, 'temp': int(round(r.get('mean_temp', 0))), 'wind': int(round(r.get('wind', 0) if 'wind' in r.index else 0)), 'icon': _temp_icon(r.get('mean_temp',0))})
        return out
    sample = df.tail(n)
    for _, r in sample.iterrows():
        try:
            t = pd.to_datetime(r.get('date')).strftime('%Y-%m-%d %H:%M')
        except Exception:
            t = str(r.get('date'))
        out.append({'time': t, 'temp': int(round(r.get('mean_temp', 0))), 'wind': int(round(r.get('wind', 0) if 'wind' in r.index else 0)), 'icon': _temp_icon(r.get('mean_temp',0))})
    return out

def get_current_conditions(df: pd.DataFrame):
    """Gather a cleaned dictionary of current/historical conditions from the last row.
    Heuristics:
    - try many column names for humidity/wind/pressure/aqi
    - if humidity missing and cloud_cover exists (0-10 or 0-1 scale), estimate humidity = cloud_cover * scale
    - if wind missing, estimate from inverse cloud_cover or global_radiation if available
    """
    if df is None or df.shape[0] == 0:
        now = datetime.now()
        return {
            'time_str': now.strftime('%I:%M %p'),
            'date_str': now.strftime('%A, %d %b %Y'),
            'temp': 0.0,
            'feels_like': 0.0,
            'condition': 'Unknown',
            'sunrise': '06:30',
            'sunset': '18:30',
            'humidity': 0,
            'wind': 0,
            'pressure': None,
            'aqi': None,
            'city': 'Unknown'
        }

    df_sorted = df.copy()
    try:
        parsed = pd.to_datetime(df_sorted['date'], errors='coerce', infer_datetime_format=True)
        if parsed.isna().all():
            parsed = pd.to_datetime(df_sorted['date'], format='%Y%m%d', errors='coerce')
    except Exception:
        parsed = pd.to_datetime(df_sorted['date'], errors='coerce')
    df_sorted['parsed_date'] = parsed.fillna(pd.Timestamp.now())
    r = df_sorted.sort_values('parsed_date').iloc[-1]

    dt = r['parsed_date'] if not pd.isna(r['parsed_date']) else pd.Timestamp.now()
    time_str = pd.to_datetime(dt).strftime('%I:%M %p')
    date_str = pd.to_datetime(dt).strftime('%A, %d %b %Y')

    # temperature selection
    temp = None
    for tcol in ['mean_temp', 'temp', 'max_temp', 'min_temp']:
        if tcol in r.index and not pd.isna(r.get(tcol)):
            temp = r.get(tcol)
            break
    if temp is None:
        temp = 0.0

    feels_like = r.get('feels_like', temp)
    if pd.isna(feels_like) or feels_like is None:
        feels_like = temp

    # condition inference
    condition = None
    for ccol in ['weather', 'condition', 'summary', 'cloud_cover']:
        if ccol in r.index and not pd.isna(r.get(ccol)):
            val = r.get(ccol)
            if ccol == 'cloud_cover':
                try:
                    cc = float(val)
                    # interpret cloud_cover as 0-10 scale or 0-1
                    if cc <= 1:
                        cc10 = cc * 10
                    else:
                        cc10 = cc
                    if cc10 > 7:
                        condition = 'Cloudy'
                    elif cc10 > 3:
                        condition = 'Partly cloudy'
                    else:
                        condition = 'Clear'
                except Exception:
                    condition = str(val)
            else:
                condition = str(val)
            break
    if condition is None:
        condition = 'Unknown'

    # sunrise/sunset fallbacks
    sunrise = r.get('sunrise', None)
    sunset = r.get('sunset', None)
    if sunrise is None or pd.isna(sunrise):
        sunrise = '06:30'
    if sunset is None or pd.isna(sunset):
        sunset = '18:30'

    # humidity attempt many names
    humidity = None
    for hcol in ['humidity', 'rel_humidity', 'rh', 'wet_bulb']:
        if hcol in r.index and not pd.isna(r.get(hcol)):
            humidity = r.get(hcol)
            break
    # fallback: estimate from cloud_cover if present
    if humidity is None:
        if 'cloud_cover' in r.index and not pd.isna(r.get('cloud_cover')):
            try:
                cc = float(r.get('cloud_cover'))
                # if cloud_cover seems 0-1 scale, convert to 0-10
                cc10 = cc * 10 if cc <= 1 else cc
                humidity = int(min(100, max(0, round(cc10 * 10))))
            except Exception:
                humidity = 50
        else:
            humidity = 50  # neutral fallback

    # wind attempt many names
    wind = None
    for wcol in ['wind', 'wind_speed', 'wind_kph', 'wind_km_h', 'wind_mph']:
        if wcol in r.index and not pd.isna(r.get(wcol)):
            wind = r.get(wcol)
            break
    # fallback: estimate inversely from cloud_cover or use global_radiation: less clouds -> calmer? We'll do simple heuristic
    if wind is None:
        try:
            if 'cloud_cover' in r.index and not pd.isna(r.get('cloud_cover')):
                cc = float(r.get('cloud_cover'))
                cc10 = cc * 10 if cc <= 1 else cc
                wind = int(max(0, round((10 - cc10) * 1.8)))  # rough km/h estimate
            elif 'global_radiation' in r.index and not pd.isna(r.get('global_radiation')):
                gr = float(r.get('global_radiation'))
                # scale radiation to approx wind: more radiation -> slightly lower wind assumption
                wind = int(max(0, round(10 - (gr / 100.0))))
            else:
                wind = 10
        except Exception:
            wind = 10

    # pressure normalization
    pressure = None
    for pcol in ['pressure', 'sea_level', 'pressure_hpa']:
        if pcol in r.index and not pd.isna(r.get(pcol)):
            pressure = r.get(pcol)
            break
    if pressure is not None:
        try:
            pval = float(pressure)
            if pval > 2000:
                pressure = pval / 100.0
            else:
                pressure = pval
        except Exception:
            pressure = None

    # aqi
    aqi = None
    for aq in ['aqi', 'air_quality_index']:
        if aq in r.index and not pd.isna(r.get(aq)):
            aqi = r.get(aq)
            break

    city = r.get('city', 'Unknown') if 'city' in r.index else 'Unknown'

    return {
        'time_str': time_str,
        'date_str': date_str,
        'temp': float(temp) if temp is not None else 0.0,
        'feels_like': float(feels_like) if feels_like is not None else float(temp),
        'condition': condition,
        'sunrise': sunrise,
        'sunset': sunset,
        'humidity': int(round(float(humidity))) if humidity is not None else 0,
        'wind': int(round(float(wind))) if wind is not None else 0,
        'pressure': pressure,
        'aqi': int(aqi) if aqi is not None else None,
        'city': city,
    }