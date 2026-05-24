import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod

# --- 1. NASTAVENÍ APLIKACE (Optimalizace pro mobilní zobrazení) ---
st.set_page_config(page_title="Analýza pojezdů Ammann", layout="wide")
st.title("🚜 CCC Analýza: Válec Ammann")
st.caption("Optimalizováno pro Streamlit Cloud a mobilní zařízení na stavbě.")

# --- 2. POMOCNÉ FUNKCE S CACHINGEM (Zabraňuje padání aplikace) ---
@st.cache_data(show_spinner="Načítám a parsuji těžká data z CSV...")
def nacti_surova_data(uploaded_file):
    """
    Načte CSV pouze jednou. Při změně posuvníků (radius, offset) 
    se data nečtou znovu z disku, což drasticky zrychluje aplikaci.
    """
    file_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = file_content.splitlines()
    
    header_idx = 0
    for i, line in enumerate(lines[:100]):  # Hledáme hlavičku jen v prvních 100 řádcích
        if "latitude" in line.lower() or "time" in line.lower():
            header_idx = i
            break
            
    uploaded_file.seek(0)
    # Načteme vše jako text (str), abychom předešli chybám při parsování
    df = pd.read_csv(uploaded_file, sep=';', skiprows=header_idx, on_bad_lines='skip', dtype=str)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    return df

def najdi_vychozi_sloupec(columns, klicove_slovo):
    """Pomocná funkce pro automatickou předvolbu sloupců v UI"""
    for col in columns:
        if klicove_slovo in col.lower():
            return col
    return columns[0] if len(columns) > 0 else None

# --- 3. BOČNÍ PANEL (Všechna nastavení na jednom místě) ---
with st.sidebar:
    st.header("1. Nahrání dat")
    uploaded_file = st.file_uploader("Vložte CSV z válce", type=['csv'])
    
    if uploaded_file is not None:
        # Načteme data do cache
        df_raw = nacti_surova_data(uploaded_file)
        
        st.header("2. Ověření sloupců")
        st.info("Zkontrolujte, zda systém správně spároval sloupce z CSV.")
        
        # Chytrý kompromis: Systém sloupce odhadne, ale uživatel je může na mobilu změnit
        col_time = st.selectbox("Sloupec s ČASEM", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'time')))
        col_lat = st.selectbox("Sloupec LATITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'latitude')))
        col_lon = st.selectbox("Sloupec LONGITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'longitude')))
        col_stiff = st.selectbox("Sloupec TUHOSTI (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'stiff')))
        col_dir = st.selectbox("Sloupec SMĚRU", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'direction')))
        
        st.header("3. Parametry zkoušky CCC")
        target_lat = st.number_input("Cílová šířka (Latitude)", value=49.2793, format="%.6f")
        target_lon = st.number_input("Cílová délka (Longitude)", value=17.0212, format="%.6f")
        radius_m = st.slider("Poloměr zkušebního místa (m)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        
        st.header("4. Pokročilé korekce")
        offset_m = st.number_input("Posun anténa -> střed běhounu (m)", value=2.0, step=0.1)
        forward_val = st.text_input("Hodnota pro jízdu VPŘED", value="1")
        
        # Uživatelské nastavení časové mezery pro detekci nového pojezdu
        time_gap_s = st.slider("Detekce nového pojezdu: mezera (s)", min_value=5, max_value=60, value=15, step=5)
        
        # Mobilní optimalizace: Vykreslit každý X-tý bod mimo kruh
        decimation = st.slider("Mobilní optimalizace grafu (vykreslit každý X-tý bod trajektorie)", min_value=1, max_value=50, value=10, step=1)

# --- 4. HLAVNÍ VÝPOČETNÍ LOGIKA ---
if uploaded_file is not None:
    # Vytvoříme pracovní kopii dat z cache
    df = df_raw.copy()
    
    # Převod textových sloupců na čísla
    df[col_lat] = pd.to_numeric(df[col_lat].astype(str).str.replace(',', '.'), errors='coerce')
    df[col_lon] = pd.to_numeric(df[col_lon].astype(str).str.replace(',', '.'), errors='coerce')
    df[col_stiff] = pd.to_numeric(df[col_stiff].astype(str).str.replace(',', '.'), errors='coerce')
    
    # REÁLNÝ ČAS (Oprava bodu 2): Konverze Ammann formátu času
    # "Tue Jun 11 2024 08:56:06 GMT+0200" -> ořízneme na prvních 24 znaků a převedeme na datetime
    df['parsed_time'] = pd.to_datetime(df[col_time].str.slice(0, 24), format='%a %b %d %Y %H:%M:%S', errors='coerce')
    
    # Odstranění nevalidních řádků
    df = df.dropna(subset=[col_lat, col_lon, col_stiff, 'parsed_time'])
    
    if len(df) < 2:
        st.error("Soubor neobsahuje dostatek platných dat pro analýzu.")
    else:
        # Seřadíme data chronologicky pro správný výpočet azimutu
        df = df.sort_values('parsed_time').reset_index(drop=True)
        
        # --- KOREKCE GEOMETRIE ANTÉNY ---
        geod = Geod(ellps="WGS84")
        lons = df[col_lon].values
        lats = df[col_lat].values
        
        # Výpočet azimutu mezi sousedními body GPS logu
        fwd_az, _, _ = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
        fwd_az = np.append(fwd_az, fwd_az[-1])  # Duplikace posledního pro zachování délky pole
        
        # Směr stroje (Oprava zpátečky): Pokud válec couvá, otočíme vektor o 180°
        is_forward = (df[col_dir].astype(str) == str(forward_val)).values
        machine_heading = np.where(is_forward, fwd_az, (fwd_az + 180) % 360)
        
        # Výpočet reálné pozice běhounu válce (posun z pozice antény)
        offset_array = np.full(len(lons), offset_m)
        new_lons, new_lats, _ = geod.fwd(lons, lats, machine_heading, offset_array)
        df['corr_lon'] = new_lons
        df['corr_lat'] = new_lats
        
        # --- FILTRACE BODŮ V ZKUŠEBNÍM MÍSTĚ (KRUHU) ---
        target_lon_array = np.full(len(df), target_lon)
        target_lat_array = np.full(len(df), target_lat)
        
        _, _, distances = geod.inv(df['corr_lon'].values, df['corr_lat'].values, target_lon_array, target_lat_array)
        df['distance'] = distances
        
        # Výběr bodů, které spadají do našeho poloměru R
        in_circle = df[df['distance'] <= radius_m].copy()
        
        if in_circle.empty:
            st.warning(f"V zadaném poloměru R = {radius_m} m nebyly nalezeny žádné průjezdy. Zkontrolujte souřadnice nebo zvětšete poloměr.")
        else:
            # --- VĚDECKÁ IDENTIFIKACE POJEZDŮ PŘES ČASOVÝ ROZDÍL (Oprava bodu 2) ---
            # Pokud je časová mezera mezi body v kruhu větší než uživatelem zvolený limit (např. 15s), 
            # znamená to, že válec z kruhu odjel, otočil se a vrátil se jako nový pojezd.
            in_circle['time_diff'] = in_circle['parsed_time'].diff().dt.total_seconds()
            in_circle['pass_id'] = (in_circle['time_diff'] > time_gap_s).cumsum() + 1
            
            # Výpočet průměrné tuhosti (Kb) pro každý jednotlivý pojezd
            summary = in_circle.groupby('pass_id')[col_stiff].agg(['mean', 'count']).reset_index()
            summary.columns = ['Číslo pojezdu', 'Průměrná Tuhost Kb [-]', 'Počet bodů v kruhu']
            
            # --- VIZUALIZACE GRAPHICS (Mobilní optimalizace - Oprava bodu 4) ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # KRESLENÍ TRAJEKTORIE: Pro mobil decimujeme data (vykreslíme např. každý 10. bod)
            # Tím pádem graf vykreslíme bleskově a mobilní prohlížeč nezkolabuje.
            df_decimated = df.iloc[::decimation]
            ax1.scatter(df_decimated['corr_lon'], df_decimated['corr_lat'], c='#E0E0E0', s=3, alpha=0.5, label=f'Celá trajektorie (každý {decimation}. bod)')
            
            # Body uvnitř kruhu vykreslíme kompletní a barevně rozlišené podle pojezdů
            for p_id, group_data in in_circle.groupby('pass_id'):
                ax1.scatter(group_data['corr_lon'], group_data['corr_lat'], s=25, label=f'Pojezd {int(p_id)}')
                
            # Vykreslení hranice zkušebního kruhu
            theta = np.linspace(0, 2*np.pi, 100)
            m_lat = 111320 
            m_lon = 111320 * np.cos(np.radians(target_lat))
            circle_lon = target_lon + (radius_m/m_lon) * np.cos(theta)
            circle_lat = target_lat + (radius_m/m_lat) * np.sin(theta)
            
            ax1.plot(circle_lon, circle_lat, 'k--', linewidth=1.5, label='Zkušební kruh')
            ax1.plot(target_lon, target_lat, 'kx', markersize=10, markeredgewidth=2)
            
            # Dynamický zoom kolem zkušebního místa (okno cca 30x30 metrů)
            zoom_deg_lat = 15 / m_lat
            zoom_deg_lon = 15 / m_lon
            ax1.set_xlim([target_lon - zoom_deg_lon, target_lon + zoom_deg_lon])
            ax1.set_ylim([target_lat - zoom_deg_lat, target_lat + zoom_deg_lat])
            
            ax1.set_title(f'Mapa pojezdů (R = {radius_m}m, Posun antény = {offset_m}m)')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_aspect('equal')
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.legend(loc='upper right', fontsize='small')
            
            # Graf nárůstu tuhosti (CCC výstup pro diplomku)
            ax2.plot(summary['Číslo pojezdu'], summary['Průměrná Tuhost Kb [-]'], '-o', 
                     linewidth=2, markersize=8, color='#1f77b4')
            ax2.set_title('Křivka hutnění (Nárůst tuhosti CCC)')
            ax2.set_xlabel('Číslo pojezdu (Chronologicky)')
            ax2.set_ylabel('Průměrná hodnota Kb [-]')
            ax2.set_xticks(summary['Číslo pojezdu'])
            ax2.grid(True, linestyle=':', alpha=0.6)
            
            # Vykreslení do Streamlitu
            st.pyplot(fig)
            
            # --- DATUMLOVÝ EXPORT VÝSLEDKŮ ---
            st.subheader("Výsledná data hutnění")
            col_table, col_btn = st.columns([2, 1])
            with col_table:
                st.dataframe(summary, hide_index=True)
            with col_btn:
                st.write("")
                csv_export = summary.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
                st.download_button(
                    label="📥 Stáhnout výsledky (CSV pro Excel)",
                    data=csv_export,
                    file_name='CCC_Vysledky_Hutneni.csv',
                    mime='text/csv',
                    use_container_width=True
                )
else:
    st.info("👋 Vítejte! V bočním panelu nahrajte CSV soubor vyexportovaný z válce Ammann.")
