import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="Analýza pojezdů Ammann", layout="wide")
st.title("CCC Analýza: Válec Ammann")
st.caption("Interaktivní mapa s reálným geometrickým modelem stroje a auto-zoomem.")

# --- 2. POMOCNÉ FUNKCE S CACHINGEM A GEOMETRIÍ ---
@st.cache_data(show_spinner="Načítám data...")
def nacti_surova_data(uploaded_file):
    """Načte CSV a automaticky detekuje oddělovač (sep=None)."""
    file_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = file_content.splitlines()
    
    header_idx = 0
    for i, line in enumerate(lines[:100]):
        if "latitude" in line.lower() or "time" in line.lower():
            header_idx = i
            break
            
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=None, engine='python', skiprows=header_idx, on_bad_lines='skip', dtype=str)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    return df

def najdi_vychozi_sloupec(columns, klicove_slovo):
    for col in columns:
        if klicove_slovo in col.lower():
            return col
    return columns[0] if len(columns) > 0 else None

def dej_obdelnik(lon, lat, azimut, sirka, delka, geod):
    """Vypočítá souřadnice 4 rohů obdélníku pro reálný model stroje."""
    # Střed přední hrany
    fc_lon, fc_lat, _ = geod.fwd(lon, lat, azimut, delka/2)
    # Střed zadní hrany
    bc_lon, bc_lat, _ = geod.fwd(lon, lat, (azimut+180)%360, delka/2)
    
    # Rohy (posun kolmo na azimut)
    fr_lon, fr_lat, _ = geod.fwd(fc_lon, fc_lat, (azimut+90)%360, sirka/2)
    fl_lon, fl_lat, _ = geod.fwd(fc_lon, fc_lat, (azimut-90)%360, sirka/2)
    bl_lon, bl_lat, _ = geod.fwd(bc_lon, bc_lat, (azimut-90)%360, sirka/2)
    br_lon, br_lat, _ = geod.fwd(bc_lon, bc_lat, (azimut+90)%360, sirka/2)
    
    # Vrátí uzavřený polygon
    return [fr_lon, fl_lon, bl_lon, br_lon, fr_lon], [fr_lat, fl_lat, bl_lat, br_lat, fr_lat]

# --- 3. BOČNÍ PANEL ---
with st.sidebar:
    st.header("1. Nahrání dat")
    uploaded_file = st.file_uploader("Vložte CSV z válce", type=['csv'])
    
    if uploaded_file is not None:
        df_raw = nacti_surova_data(uploaded_file)
        
        st.header("2. Ověření sloupců")
        col_time = st.selectbox("Sloupec s ČASEM", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'time')))
        col_lat = st.selectbox("Sloupec LATITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'latitude')))
        col_lon = st.selectbox("Sloupec LONGITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'longitude')))
        col_stiff = st.selectbox("Sloupec TUHOSTI (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'stiff')))
        col_dir = st.selectbox("Sloupec SMĚRU", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'direction')))
        
        st.header("3. Parametry zkoušky")
        target_lat = st.number_input("Cílová šířka", value=49.2793, format="%.6f")
        target_lon = st.number_input("Cílová délka", value=17.0212, format="%.6f")
        radius_m = st.slider("Poloměr místa (m)", 0.1, 5.0, 1.0, 0.1)
        
        st.header("4. Korekce a zobrazení")
        offset_m = st.number_input("Posun anténa -> běhoun (m)", value=2.0, step=0.1)
        drum_width_m = st.number_input("Šířka běhounu (m)", value=2.1, step=0.1)
        forward_val = st.text_input("Hodnota jízdy VPŘED", value="1")
        time_gap_s = st.slider("Mezera mezi pojezdy (s)", 5, 60, 15, 5)
        decimation = st.slider("Decimace mapy (každý X-tý bod)", 1, 50, 10, 1)
        show_machine_model = st.checkbox("Zobrazit model stroje (obdélníky)", value=True)

# --- 4. HLAVNÍ VÝPOČETNÍ LOGIKA ---
if uploaded_file is not None:
    df = df_raw.copy()
    
    # Převod textových sloupců na čísla
    df[col_lat] = pd.to_numeric(df[col_lat].astype(str).str.replace(',', '.'), errors='coerce')
    df[col_lon] = pd.to_numeric(df[col_lon].astype(str).str.replace(',', '.'), errors='coerce')
    df[col_stiff] = pd.to_numeric(df[col_stiff].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Oprava času: Split dle GMT a konverze
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.split(' GMT').str[0], errors='coerce')
    
    # Odstranění nevalidních řádků
    df = df.dropna(subset=[col_lat, col_lon, col_stiff, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if len(df) < 2:
        st.error("Málo platných dat.")
    else:
        # Geometrická korekce
        geod = Geod(ellps="WGS84")
        lons, lats = df[col_lon].values, df[col_lat].values
        
        # Výpočet azimutu (Course Over Ground)
        fwd_az, _, _ = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
        fwd_az = np.append(fwd_az, fwd_az[-1])
        
        # Heading a posun
        is_forward = (df[col_dir].astype(str) == str(forward_val)).values
        machine_heading = np.where(is_forward, fwd_az, (fwd_az + 180) % 360)
        new_lons, new_lats, _ = geod.fwd(lons, lats, machine_heading, np.full(len(lons), offset_m))
        df['corr_lon'], df['corr_lat'] = new_lons, new_lats
        
        # Filtrace v kruhu
        target_lon_array = np.full(len(df), target_lon)
        target_lat_array = np.full(len(df), target_lat)
        _, _, dists = geod.inv(df['corr_lon'].values, df['corr_lat'].values, target_lon_array, target_lat_array)
        df['distance'] = dists
        in_circle = df[df['distance'] <= radius_m].copy()
        
        if not in_circle.empty:
            # Detekce pojezdů
            in_circle['pass_id'] = (in_circle['parsed_time'].diff().dt.total_seconds() > time_gap_s).cumsum() + 1
            summary = in_circle.groupby('pass_id')[col_stiff].agg(['mean', 'count']).reset_index()
            summary.columns = ['Pojezd', 'Kb [-]', 'Body']

            # MAPA PLOTLY
            fig_map = go.Figure()
            
            # Trajektorie (mimo zkušební místo)
            df_dec = df.iloc[::decimation]
            fig_map.add_trace(go.Scatter(x=df_dec['corr_lon'], y=df_dec['corr_lat'], mode='markers', 
                                         marker=dict(size=4, color='#E5E7EB', opacity=0.4), name='Okolní data'))
            
            # Pevně definovaná paleta barev (už neklekne)
            color_pal = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
            color_idx = 0
            
            # Pojezdy v kruhu a vizualizace REÁLNÉHO stroje
            for p_id, g in in_circle.groupby('pass_id'):
                p_color = color_pal[color_idx % len(color_pal)]
                
                # Zobrazení geometrického modelu pro body v kruhu
                if show_machine_model:
                    # Uděláme cca 5 vizualizací na pojezd, ať se ty obdélníky neslijí do jedné kaše
                    step = max(1, len(g)//5) 
                    for i in range(0, len(g), step):
                        l_lon, l_lat = g[col_lon].iloc[i], g[col_lat].iloc[i] # Kabina (anténa)
                        r_lon, r_lat = g['corr_lon'].iloc[i], g['corr_lat'].iloc[i] # Běhoun (střed)
                        
                        # Opraveno malé geod.inv - azimut od antény k běhounu
                        head = geod.inv(l_lon, l_lat, r_lon, r_lat)[0]
                        
                        # 1. BUBEN (Běhoun) - 2.1m šířka, 1.0m délka. Získáme rohy přes pomocnou funkci.
                        b_lons, b_lats = dej_obdelnik(r_lon, r_lat, head, drum_width_m, 1.0, geod)
                        fig_map.add_trace(go.Scatter(
                            x=b_lons, y=b_lats, mode='lines', fill='toself',
                            fillcolor=p_color, opacity=0.3, line=dict(color=p_color, width=2), showlegend=False
                        ))
                        
                        # 2. KABINA - menší obdélník (cca 1.5x1.5m), světle šedý, umístěný tam, kde je anténa
                        k_lons, k_lats = dej_obdelnik(l_lon, l_lat, head, 1.5, 1.5, geod)
                        fig_map.add_trace(go.Scatter(
                            x=k_lons, y=k_lats, mode='lines', fill='toself',
                            fillcolor='gray', opacity=0.15, line=dict(color='gray', width=1), showlegend=False
                        ))
                        
                        # 3. SPOJNICE (Osa stroje od kabiny k běhounu)
                        fig_map.add_trace(go.Scatter(
                            x=[l_lon, r_lon], y=[l_lat, r_lat], mode='lines', 
                            line=dict(color=p_color, width=2, dash='dot'), opacity=0.7, showlegend=False
                        ))
                
                # Samotné body posunuté do středu bubnu
                fig_map.add_trace(go.Scatter(x=g['corr_lon'], y=g['corr_lat'], mode='markers+lines', 
                                             marker=dict(size=12, color=p_color), line=dict(color=p_color), name=f'Pojezd {int(p_id)}'))
                color_idx += 1

            # Kruh a střed
            theta = np.linspace(0, 2*np.pi, 100)
            m_lat, m_lon = 111320, 111320 * np.cos(np.radians(target_lat))
            fig_map.add_trace(go.Scatter(x=target_lon + (radius_m/m_lon) * np.cos(theta), y=target_lat + (radius_m/m_lat) * np.sin(theta),
                                         mode='lines', line=dict(color='black', dash='dash'), name='Zkušební místo'))
            
            # --- AUTO-ZOOM NA VÝSLEDKY ---
            min_lat_r, max_lat_r = target_lat - (radius_m / m_lat), target_lat + (radius_m / m_lat)
            min_lon_r, max_lon_r = target_lon - (radius_m / m_lon), target_lon + (radius_m / m_lon)
            
            points_lons, points_lats = in_circle['corr_lon'].values, in_circle['corr_lat'].values
            min_lat_p, max_lat_p = points_lats.min(), points_lats.max()
            min_lon_p, max_lon_p = points_lons.min(), points_lons.max()
            
            overall_min_lat, overall_max_lat = min(min_lat_r, min_lat_p), max(max_lat_r, max_lat_p)
            overall_min_lon, overall_max_lon = min(min_lon_r, min_lon_p), max(max_lon_r, max_lon_p)
            
            range_lat, range_lon = overall_max_lat - overall_min_lat, overall_max_lon - overall_min_lon
            buffer = 0.15 # Malá rezerva okolo, aby se nelepil okraj kruhu na zeď okna
            
            fig_map.update_layout(
                yaxis=dict(scaleanchor="x", scaleratio=1, range=[overall_min_lat - range_lat * buffer, overall_max_lat + range_lat * buffer]),
                xaxis=dict(range=[overall_min_lon - range_lon * buffer, overall_max_lon + range_lon * buffer]),
                height=700, title="Interaktivní mapa (Auto-zoom na výsledky)", dragmode="pan"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Graf nárůstu tuhosti a Excel Export
            st.subheader("Nárůst tuhosti CCC")
            st.line_chart(summary.set_index('Pojezd')['Kb [-]'])
            st.dataframe(summary, hide_index=True)
            
            csv = summary.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
            st.download_button("📥 Stáhnout CSV", csv, "Vysledky.csv", "text/csv")
        else:
            st.warning("V kruhu nejsou žádná data. Zkuste zvětšit poloměr zkušebního místa.")
else:
    st.info("👋 Připraveno na analýzu! Nahrajte v levém panelu soubor.")
