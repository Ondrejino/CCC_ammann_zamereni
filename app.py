import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io

# --- 1. NASTAVENÍ ---
st.set_page_config(page_title="CCC Heatmapa & Analýza", layout="wide")
st.title("Ammann CCC: Plošná analýza tuhosti")

# --- 2. FUNKCE ---
@st.cache_data
def nacti_data(file_bytes):
    sample = file_bytes[:10000].decode("utf-8", errors="ignore")
    lines = sample.splitlines()
    h_idx = 0
    for i, line in enumerate(lines):
        if "latitude" in line.lower() or "time" in line.lower():
            h_idx = i
            break
    header = lines[h_idx]
    sep = ';' if header.count(';') > header.count(',') else ','
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, skiprows=h_idx, on_bad_lines='skip', dtype=str)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    return df

def dej_obdelnik(lon, lat, azimut, sirka, delka, geod):
    fc_lon, fc_lat, _ = geod.fwd(lon, lat, azimut, delka/2)
    bc_lon, bc_lat, _ = geod.fwd(lon, lat, (azimut+180)%360, delka/2)
    fr_lon, fr_lat, _ = geod.fwd(fc_lon, fc_lat, (azimut+90)%360, sirka/2)
    fl_lon, fl_lat, _ = geod.fwd(fc_lon, fc_lat, (azimut-90)%360, sirka/2)
    bl_lon, bl_lat, _ = geod.fwd(bc_lon, bc_lat, (azimut-90)%360, sirka/2)
    br_lon, br_lat, _ = geod.fwd(bc_lon, bc_lat, (azimut+90)%360, sirka/2)
    return [fr_lon, fl_lon, bl_lon, br_lon, fr_lon], [fr_lat, fl_lat, bl_lat, br_lat, fr_lat]

# --- 3. SIDEBAR ---
with st.sidebar:
    up_file = st.file_uploader("Nahrát CSV", type=['csv'])
    if up_file:
        df_raw = nacti_data(up_file.getvalue())
        c_lat = st.selectbox("Latitude", df_raw.columns, index=0)
        c_lon = st.selectbox("Longitude", df_raw.columns, index=1)
        c_stiff = st.selectbox("Tuhost (Kb)", df_raw.columns, index=2)
        c_dir = st.selectbox("Směr jízdy", df_raw.columns, index=3)
        c_time = st.selectbox("Čas", df_raw.columns, index=4)
        
        st.divider()
        t_lat = st.number_input("Cíl Lat", value=49.2793, format="%.6f")
        t_lon = st.number_input("Cíl Lon", value=17.0212, format="%.6f")
        rad_m = st.slider("Radius (m)", 0.5, 5.0, 1.5)
        offset_m = st.number_input("Offset anténa->buben (m)", value=2.0)
        
        show_heatmap = st.checkbox("Zobrazit plošnou Heatmapu", value=True)
        decim_h = st.slider("Decimace Heatmapy", 1, 20, 5)

# --- 4. VÝPOČTY ---
if up_file:
    df = df_raw.copy()
    for c in [c_lat, c_lon, c_stiff]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
    df['parsed_time'] = pd.to_datetime(df[c_time].astype(str).str.split(' GMT').str[0], errors='coerce')
    df = df.dropna(subset=[c_lat, c_lon, c_stiff]).sort_values('parsed_time').reset_index(drop=True)

    geod = Geod(ellps="WGS84")
    cos_c = 1 / np.cos(np.radians(t_lat))

    # Směr a korekce
    lons, lats = df[c_lon].values, df[c_lat].values
    f_az = np.zeros(len(df))
    # Počítáme azimut s krokem pro stabilitu
    az, _, _ = geod.inv(lons[:-3], lats[:-3], lons[3:], lats[3:])
    f_az[:-3] = az
    f_az[-3:] = az[-1] if len(az)>0 else 0
    
    # Předpokládáme, že "1" je vpřed, jinak otočíme azimut
    heading = np.where(df[c_dir].astype(str) == "1", f_az, (f_az + 180) % 360)
    c_lons, c_lats, _ = geod.fwd(lons, lats, heading, np.full(len(lons), offset_m))
    df['corr_lon'], df['corr_lat'] = c_lons, c_lats

    # --- A. HEATMAPA CELÉ OBLASTI ---
    if show_heatmap:
        st.subheader("1. Plošná mapa tuhosti (Heatmapa)")
        fig_h = go.Figure()
        
        # Celá dráha barevně dle Kb
        df_h = df.iloc[::decim_h]
        fig_h.add_trace(go.Scatter(
            x=df_h['corr_lon'], y=df_h['corr_lat'], mode='markers',
            marker=dict(size=6, color=df_h[c_stiff], colorscale='Jet', showscale=True, 
                        colorbar=dict(title="Kb [MN/m]")),
            name="Naměřená tuhost", hovertext=df_h[c_stiff]
        ))
        
        # Vyznačení zkušebního bodu
        fig_h.add_trace(go.Scatter(x=[t_lon], y=[t_lat], mode='markers',
                                   marker=dict(size=15, symbol='x', color='black'), name="Zkušební bod"))
        
        # Kruh radiusu
        theta = np.linspace(0, 2*np.pi, 100)
        r_lon = rad_m / (111320 * np.cos(np.radians(t_lat)))
        r_lat = rad_m / 111320
        fig_h.add_trace(go.Scatter(x=t_lon + r_lon * np.cos(theta), y=t_lat + r_lat * np.sin(theta),
                                   mode='lines', line=dict(color='black', dash='dot'), name="Radius"))

        fig_h.update_layout(yaxis=dict(scaleanchor="x", scaleratio=cos_c), height=600, dragmode='pan')
        st.plotly_chart(fig_h, use_container_width=True)

    # --- B. DETAILNÍ ANALÝZA BODU ---
    _, _, dists = geod.inv(df['corr_lon'].values, df['corr_lat'].values, np.full(len(df), t_lon), np.full(len(df), t_lat))
    df['dist'] = dists
    in_c = df[df['dist'] <= rad_m].copy()

    if not in_c.empty:
        # Detekce pojezdů (časová mezera + změna směru)
        in_c['pass_id'] = ( (in_c['parsed_time'].diff().dt.total_seconds() > 10) | (in_c[c_dir] != in_c[c_dir].shift()) ).cumsum() + 1
        
        st.divider()
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("2. Detail pojezdů v místě")
            fig_d = go.Figure()
            pal = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
            
            for p_id, g in in_c.groupby('pass_id'):
                color = pal[int(p_id-1) % len(pal)]
                # Model stroje (střední bod pojezdu)
                mid = len(g)//2
                m_lon, m_lat = g[c_lon].iloc[mid], g[c_lat].iloc[mid]
                r_lon, r_lat = g['corr_lon'].iloc[mid], g['corr_lat'].iloc[mid]
                h = geod.inv(m_lon, m_lat, r_lon, r_lat)[0]
                
                b_lon, b_lat = dej_obdelnik(r_lon, r_lat, h, 2.1, 1.0, geod)
                fig_d.add_trace(go.Scatter(x=b_lon, y=b_lat, mode='lines', fill='toself', 
                                           fillcolor=color, opacity=0.4, line=dict(color=color), showlegend=False))
                
                fig_d.add_trace(go.Scatter(x=g['corr_lon'], y=g['corr_lat'], mode='markers+lines',
                                           marker=dict(color=color), name=f"Pojezd {int(p_id)}"))

            fig_d.update_layout(yaxis=dict(scaleanchor="x", scaleratio=cos_c), height=500)
            st.plotly_chart(fig_d, use_container_width=True)

        with col2:
            st.subheader("3. Nárůst tuhosti (Kb)")
            stats = in_c.groupby('pass_id')[c_stiff].agg(['mean', 'std']).reset_index()
            
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(x=stats['pass_id'], y=stats['mean'], mode='markers+lines',
                                       error_y=dict(type='data', array=stats['std'], visible=True),
                                       marker=dict(size=12, color='royalblue'), line=dict(width=4)))
            
            fig_g.update_layout(xaxis_title="Číslo pojezdu", yaxis_title="Kb [MN/m]", height=500)
            st.plotly_chart(fig_g, use_container_width=True)
            
            st.dataframe(stats.rename(columns={'pass_id': 'Pojezd', 'mean': 'Průměr Kb', 'std': 'Odchylka'}), hide_index=True)
    else:
        st.warning("V daném radiusu nebyla nalezena žádná data. Zkontroluj souřadnice cíle nebo zvětši radius.")
