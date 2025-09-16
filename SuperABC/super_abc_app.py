"""
Súper ABC & Perfiles - App Interactiva
=====================================

Versión mejorada:
- ABC por contribución (el usuario define cortes A/B por criterio)
- Combinación de dos criterios en AA..CC
- Resumen extendido (incluye % ventas por categoría)
- Perfiles solicitados (líneas por orden, cubicaje por orden, días, tabla cruzada)
- CSV export con nombres sanitizados (sin caracteres especiales)
- Generación opcional de informe PDF (requiere reportlab + matplotlib)
"""

import io
import unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import statsmodels.api as sm
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


# Para generar PDF/plots
try:
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
    from reportlab.lib.styles import getSampleStyleSheet
    PDF_LIBS_AVAILABLE = True
except Exception:
    PDF_LIBS_AVAILABLE = False

# -------------------------------
# Utilidades
# -------------------------------

def sanitize_filename(s: str) -> str:
    # elimina acentos y caracteres especiales, deja ascii y guiones bajos
    s = str(s)
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = s.replace(' ', '_')
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return ''.join(c for c in s if c in allowed)

def sanitize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [unicodedata.normalize('NFKD', str(c)).encode('ascii','ignore').decode('ascii') for c in df.columns]
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df

def minmax_normalize(s: pd.Series) -> pd.Series:
    s = s.fillna(0)
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / rng

@st.cache_data(show_spinner=False)
def read_excel_bytes(file_bytes: bytes, sheet_name=None):
    """
    Lee un archivo Excel desde bytes.
    
    Args:
        file_bytes: Bytes del archivo Excel
        sheet_name: Nombre de la hoja a leer (None para todas las hojas)
    
    Returns:
        DataFrame si sheet_name está especificado, dict si no
    """
    try:
        if sheet_name:
            # Leer hoja específica
            return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine='openpyxl')
        else:
            # Leer todas las hojas
            return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine='openpyxl')
    except Exception as e:
        # Si falla con openpyxl, intentar con xlrd para archivos .xls
        try:
            return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine='xlrd')
        except:
            raise e

# ABC por contribución acumulada

def safe_col(df: pd.DataFrame, name: str, alt_names=None):
    """Busca una columna tolerando espacios, mayúsculas/minúsculas o nombres alternativos."""
    if alt_names is None:
        alt_names = []
    # Diccionario para búsqueda insensible a mayúsculas/minúsculas y espacios
    alt = {c.strip().lower(): c for c in df.columns}
    key = name.strip().lower()
    # Revisar nombre principal
    if key in alt:
        return df[alt[key]]
    # Revisar nombres alternativos
    for alt_name in alt_names:
        k = alt_name.strip().lower()
        if k in alt:
            return df[alt[k]]
    raise KeyError(f"No se encontró la columna requerida: {name}")


def cycle_count_freq(zone: str) -> str:
    return {'Oro':'Semanal/Mensual','Plata':'Mensual/Trimestral','Bronce':'Trimestral/Semestral'}.get(zone, 'Trimestral')


def abc_by_contribution(series: pd.Series, A_cut: float, B_cut: float) -> pd.Series:
    df_tmp = series.rename('metric').to_frame()
    df_tmp = df_tmp.sort_values('metric', ascending=False)
    total = df_tmp['metric'].sum()
    if total <= 0:
        return pd.Series('C', index=series.index)
    df_tmp['cum_contrib'] = df_tmp['metric'].cumsum() / total
    df_tmp['cls'] = np.where(df_tmp['cum_contrib'] <= A_cut, 'A', np.where(df_tmp['cum_contrib'] <= B_cut, 'B', 'C'))
    return df_tmp['cls'].reindex(series.index)

def generate_super_abc_combinations(by_item: pd.DataFrame, criterios_seleccionados: list, cortes_abc: dict, criterios_map: dict) -> pd.DataFrame:
    """
    Genera todas las combinaciones posibles de clasificaciones ABC para múltiples criterios.
    
    Args:
        by_item: DataFrame con métricas por artículo
        criterios_seleccionados: Lista de criterios seleccionados
        cortes_abc: Diccionario con los cortes A y B para cada criterio
        criterios_map: Mapeo de nombres de criterios a columnas del DataFrame
    
    Returns:
        DataFrame con las clasificaciones ABC para cada criterio y la combinación final
    """
    import itertools
    
    # Calcular clasificación ABC para cada criterio
    for criterio in criterios_seleccionados:
        col_name = criterios_map[criterio]
        A_cut = cortes_abc[criterio]['A']
        B_cut = cortes_abc[criterio]['B']
        by_item[f'ABC_{criterio}'] = abc_by_contribution(by_item[col_name], A_cut, B_cut)
    
    # Generar todas las combinaciones posibles
    abc_values = ['A', 'B', 'C']
    combinaciones = list(itertools.product(abc_values, repeat=len(criterios_seleccionados)))
    
    # Crear la clasificación combinada
    def create_combination_class(row):
        combination = ''.join([row[f'ABC_{criterio}'] for criterio in criterios_seleccionados])
        return combination
    
    by_item['Clase_SuperABC'] = by_item.apply(create_combination_class, axis=1)
    
    return by_item

# Map zone from combined class

def map_zone(clase: str) -> str:
    """
    Mapea una clase de Súper ABC a una zona de bodega.
    Para múltiples criterios, se basa en la prioridad de las letras A, B, C.
    """
    # Si tiene al menos una A, va a Oro
    if 'A' in clase:
        return 'Oro'
    # Si tiene al menos una B (y no A), va a Plata
    elif 'B' in clase:
        return 'Plata'
    # Si solo tiene C, va a Bronce
    else:
        return 'Bronce'

def policy_by_demand(cv: float, intermittency: float) -> str:
    if intermittency >= 0.5:
        return 'RTP-EOQ (items intermitentes)'
    if cv < 0.5:
        return 'ROP-OUL (alta estabilidad)'
    if cv < 1.0:
        return 'ROP-EOQ (variabilidad media)'
    return 'RTP-EOQ (alta variabilidad)'

# Fill rates nuevos (usar guion ASCII)

def target_fill_rate(zone: str) -> str:
    if zone == 'Oro':
        return '99-90%'
    if zone == 'Plata':
        return '89-80%'
    if zone == 'Bronce':
        return '70-80%'
    return '80-90%'

# Week floor
from datetime import datetime

def week_floor(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period('W-MON').dt.start_time

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title='Súper ABC & Perfiles', layout='wide')
st.title('📦 Súper ABC Interactivo & Perfiles de Órdenes')

st.markdown("""
Bienvenido a la aplicación **Súper ABC & Perfiles** 🚀  

Esta herramienta permite analizar los productos de tu portafolio mediante una clasificación **Súper ABC**, combinando dos criterios (ej. ventas y cubicaje).  
El flujo de uso es el siguiente:

1. **Carga de archivo**: Sube un archivo Excel/CSV con la información de tus productos (ventas, cubicaje, pedidos, etc.).  
2. **Definición de cortes**: Elige los porcentajes que delimitan las categorías A, B y C según tu criterio.  
3. **Clasificación Súper ABC**: Los productos se clasifican automáticamente en las categorías combinadas **AA..CC**.  
4. **Resumen por categoría**: Se muestra una tabla con:
   - Cantidad de ítems por clase  
   - Zona de bodega y política de inventario sugerida  
   - Fill Rate objetivo  
   - **IRA (Índice de Rotación Aceptable)** según la clase  
   - Ventas y porcentaje de participación  
5. **Perfiles adicionales**: Podrás ver indicadores sobre líneas por orden, cubicaje por orden, días de inventario y tablas cruzadas.  
6. **Exportación**: Toda la información puede descargarse en un PDF o CSV para reportes.  

ℹ️ Esta aplicación está pensada como apoyo para decisiones de **gestión de inventario y almacenamiento**, facilitando el análisis ABC tradicional y extendido.
""")

# -------------------------------
# Advertencia sobre formato del Excel
# -------------------------------
st.info("""
📂 **Configuración del archivo Excel requerida:**

El archivo debe contener **exactamente** las siguientes columnas (respetando los nombres, aunque la aplicación es tolerante a espacios y mayúsculas/minúsculas):

- `Artículo` → Identificador único del producto  
- `Unid. Vend` → Cantidad de unidades vendidas  
- `Monto venta` → Monto total de venta  
- `Volumen total (p3) o Volumen total (m3)` → Volumen total del producto. Puede estar en **pies³** o **metros³**. La unidad se selecciona en el panel lateral y se convertirá automáticamente para los cálculos internos.  
- `Num. Doc` → Número de documento / pedido  
- `Fecha Doc` → Fecha del documento/pedido en formato DD/MM/AAAA. 

⚠️ **Importante:** Si alguna columna no existe o tiene un nombre diferente, la aplicación no podrá procesar los datos correctamente.  
Asegúrate de seleccionar la unidad correcta en la barra lateral para que los cálculos de volumen sean consistentes.

📋 **Ejemplo de estructura del archivo Excel:**
```
| Artículo | Unid. Vend | Monto venta | Volumen total (p3) | Num. Doc | Fecha Doc |
|----------|------------|-------------|-------------------|----------|-----------|
| PROD001  | 100        | 1500.00     | 2.5               | DOC001   | 15/01/2024|
| PROD002  | 50         | 750.00      | 1.2               | DOC002   | 16/01/2024|
```
""")

with st.sidebar:
    st.header('1) Cargar datos')
    uploaded_file = st.file_uploader('Excel de ventas/ordenes', type=['xlsx','xls'])
    sheet_name = st.text_input('Hoja (opcional)', help='Si tu Excel tiene múltiples hojas, especifica cuál usar. Si no especificas, se usará la primera.')
    unit_vol = st.selectbox('Unidad de volumen', ['pies3 (p3)','metros3 (m3)'])
    vol_factor = 35.3147 if unit_vol == 'metros3 (m3)' else 1.0

    # Permitir al usuario definir el volumen de una tarima
    default_tarima = 42.38 if unit_vol == 'pies3 (p3)' else 1.2
    vol_tarima = st.number_input(
        'Volumen de una tarima completa',
        min_value=0.01,
        value=default_tarima,
        help='Define el volumen de una tarima en la unidad seleccionada'
    )
    # Guardar en session_state para usarlo en PDF y análisis
    st.session_state['vol_tarima'] = vol_tarima

    st.header('2) Criterios ABC (elige múltiples)')
    criterios = {
        'Popularidad': 'popularidad',
        'Rotacion': 'rotacion_sem',
        'Ventas': 'ventas',
        'Volumen': 'volumen'
    }
    
    # Permitir selección múltiple de criterios
    criterios_seleccionados = st.multiselect(
        'Selecciona los criterios a aplicar (mínimo 2):',
        list(criterios.keys()),
        default=['Popularidad', 'Ventas'],
        help='Puedes seleccionar 2 o más criterios. Se generarán todas las combinaciones posibles.'
    )
    
    # Validar que se seleccionen al menos 2 criterios
    if len(criterios_seleccionados) < 2:
        st.warning('⚠️ Debes seleccionar al menos 2 criterios para continuar.')
        st.stop()
    
    # Mostrar información sobre las combinaciones que se generarán
    num_combinaciones = 3 ** len(criterios_seleccionados)  # A, B, C para cada criterio
    st.info(f"📊 Se generarán {num_combinaciones} combinaciones posibles (A, B, C para cada criterio)")
    
    # Para compatibilidad con el código existente, mantener crit1 y crit2
    crit1 = criterios_seleccionados[0]
    crit2 = criterios_seleccionados[1] if len(criterios_seleccionados) > 1 else criterios_seleccionados[0]

    st.header('3) Cortes ABC por contribucion (A, B)')
    
    # Crear sliders dinámicos para cada criterio seleccionado
    cortes_abc = {}
    for i, criterio in enumerate(criterios_seleccionados):
        st.subheader(f'Criterio: {criterio}')
        A_cut = st.slider(f'A ({criterio})', 50, 95, 80, key=f'A_cut_{criterio}_{i}') / 100.0
        B_cut = st.slider(f'B ({criterio})', int(A_cut*100)+1, 99, 95, key=f'B_cut_{criterio}_{i}') / 100.0
        cortes_abc[criterio] = {'A': A_cut, 'B': B_cut}
    
    # Guardar en session_state usando claves únicas
    st.session_state['criterios_seleccionados'] = criterios_seleccionados
    st.session_state['cortes_abc'] = cortes_abc

    st.header('4) Exportar')
    want_csv = st.checkbox('Permitir descarga Excel', True)
    gen_pdf = st.checkbox('Generar informe PDF', False)

if uploaded_file is None:
    st.info('Sube un Excel para comenzar')
    st.stop()

# -------------------------------
# Leer datos
# -------------------------------
try:
    df = read_excel_bytes(uploaded_file.read(), sheet_name=sheet_name or None)
    
    # Verificar si df es un diccionario (múltiples hojas)
    if isinstance(df, dict):
        st.warning("⚠️ El archivo Excel contiene múltiples hojas.")
        st.write("**Hojas disponibles:**", list(df.keys()))
        
        if sheet_name and sheet_name in df:
            df = df[sheet_name]
            st.info(f"✅ Usando la hoja especificada: '{sheet_name}'")
        else:
            # Si no se especificó hoja, usar la primera
            primera_hoja = list(df.keys())[0]
            df = df[primera_hoja]
            st.info(f"✅ Usando la primera hoja: '{primera_hoja}'")
            st.write("💡 **Tip:** Puedes especificar una hoja específica en el campo 'Hoja (opcional)' en la barra lateral")
    
    # Verificar que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        st.error("❌ Error: No se pudo cargar el archivo como DataFrame")
        st.stop()
    
    # Mostrar información del archivo cargado
    st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
    st.info(f"📊 Dimensiones del archivo: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Mostrar las columnas disponibles
    st.subheader("📋 Columnas disponibles en el archivo:")
    columnas_disponibles = list(df.columns)
    st.write(columnas_disponibles)
    
    # Verificar si existe la columna 'Artículo'
    if 'Artículo' not in df.columns:
        st.error("❌ No se encontró la columna 'Artículo' en el archivo.")
        st.write("**Columnas disponibles:**", columnas_disponibles)
        st.write("**Por favor verifica que tu archivo Excel contenga una columna llamada 'Artículo'**")
        st.stop()
    
    # Limpiar espacios y mayúsculas/minúsculas
    df['Artículo_LIMPIO'] = df['Artículo'].astype(str).str.strip().str.upper()
    
except Exception as e:
    st.error(f'Error leyendo Excel: {e}')
    st.write("**Posibles causas:**")
    st.write("- El archivo no es un Excel válido")
    st.write("- El archivo está corrupto")
    st.write("- No tienes permisos para leer el archivo")
    st.write("- El archivo está siendo usado por otra aplicación")
    st.write("- El archivo tiene múltiples hojas y no se especificó cuál usar")
    st.stop()

# map columns tolerant
try:
    st.subheader("🔍 Verificando columnas requeridas...")
    
    # Verificar cada columna requerida
    columnas_requeridas = {
        'Unid. Vend': ['Unid. Vend', 'Unidades Vendidas', 'Cantidad', 'Qty'],
        'Monto venta': ['Monto venta', 'Monto Venta', 'Valor', 'Total'],
        'Volumen total (p3)': ['Volumen total (p3)', 'Volumen total (m3)', 'Volumen total', 'Volumen'],
        'Num. Doc': ['Num. Doc', 'Num Doc', 'Documento', 'Pedido', 'Order'],
        'Fecha Doc': ['Fecha Doc', 'Fecha Doc', 'Fecha', 'Date']
    }
    
    columnas_faltantes = []
    for col_principal, alternativas in columnas_requeridas.items():
        encontrada = False
        for alt in alternativas:
            if alt in df.columns:
                encontrada = True
                break
        if not encontrada:
            columnas_faltantes.append(f"{col_principal} (alternativas: {', '.join(alternativas)})")
    
    if columnas_faltantes:
        st.error("❌ Faltan las siguientes columnas requeridas:")
        for col in columnas_faltantes:
            st.write(f"- {col}")
        st.write("**Por favor verifica que tu archivo Excel contenga todas las columnas requeridas.**")
        st.stop()
    
    # Mapear columnas
    art = df['Artículo_LIMPIO']
    unid = pd.to_numeric(safe_col(df, 'Unid. Vend'), errors='coerce').fillna(0)
    monto = pd.to_numeric(safe_col(df, 'Monto venta'), errors='coerce').fillna(0)
    vol = pd.to_numeric(safe_col(df, 'Volumen total (p3)', alt_names=['Volumen total (m3)', 'Volumen total']), errors='coerce').fillna(0) * vol_factor
    numdoc = safe_col(df, 'Num. Doc').astype(str)
    fecha = pd.to_datetime(safe_col(df, 'Fecha Doc'), errors='coerce')
    
    st.success("✅ Todas las columnas requeridas fueron encontradas y mapeadas correctamente")
    
except Exception as e:
    st.error(f'Error mapeando columnas: {e}')
    st.write("**Posibles causas:**")
    st.write("- Los nombres de las columnas no coinciden exactamente")
    st.write("- Hay caracteres especiales o espacios extra en los nombres")
    st.write("- El formato de los datos no es el esperado")
    st.stop()

base = pd.DataFrame({
    'Articulo': art,
    'Unidades': unid,
    'Monto': monto,
    'Volumen_p3': vol,
    'NumDoc': numdoc,
    'Fecha': fecha,
    'Cajas_vendidas': pd.to_numeric(safe_col(df, 'Cajas vend.'), errors='coerce').fillna(0)
}).dropna(subset=['Fecha'])

# Guardar base en session_state para usarlo en PDF y perfiles
st.session_state['base'] = base

if len(base) == 0:
    st.error('No hay registros con fecha valida')
    st.stop()

st.write("Primeras filas de base:")
st.dataframe(base.head())
st.write("Suma Unidades:", base['Unidades'].sum())
st.write("Suma Cajas_vendidas:", base['Cajas_vendidas'].sum())

# -------------------------------
# Calcular Super ABC
# -------------------------------
st.subheader('▶️ Control de secciones')

if st.button('1) Calcular Súper ABC'):
    by_item = base.groupby('Articulo').agg(
        popularidad=('NumDoc','nunique'),
        unidades=('Unidades','sum'),
        ventas=('Monto','sum'),
        volumen=('Volumen_p3','sum'),
        lineas=('NumDoc','count')
    )
    # rotacion semanal
    days_range = (base['Fecha'].max() - base['Fecha'].min()).days + 1
    weeks_range = max(1, days_range/7)
    by_item['rotacion_sem'] = by_item['unidades'] / weeks_range

    # Usar la nueva función para generar combinaciones múltiples
    by_item = generate_super_abc_combinations(
        by_item, 
        criterios_seleccionados, 
        cortes_abc, 
        criterios
    )

    # Mostrar artículos con problemas de clasificación
    # Verificar si hay valores NaN en las clasificaciones ABC
    abc_columns = [f'ABC_{criterio}' for criterio in criterios_seleccionados]
    problemas_mask = by_item[abc_columns].isna().any(axis=1) | by_item['Clase_SuperABC'].str.contains('nan')
    problemas = by_item[problemas_mask]
    
    if not problemas.empty:
        st.warning(f"Hay {len(problemas)} artículos sin clase válida. Mira la tabla abajo para revisar:")
        st.dataframe(problemas)
    else:
        st.info("Todos los artículos tienen clase válida.")

    # stats semanales
    base['WeekStart'] = week_floor(base['Fecha'])
    weekly = base.groupby(['Articulo','WeekStart']).agg(units=('Unidades','sum')).reset_index()
    stats = weekly.pivot_table(index='Articulo', values='units', aggfunc=[np.mean, np.std, lambda x: (x==0).mean()])
    stats.columns = ['mean_week','std_week','intermittency']
    by_item = by_item.join(stats, how='left')
    by_item['cv'] = by_item['std_week'] / by_item['mean_week'].replace(0, np.nan)
    by_item['cv'] = by_item['cv'].fillna(np.inf)
    by_item['intermittency'] = by_item['intermittency'].fillna(1.0)

    by_item['Zona_Bodega'] = by_item['Clase_SuperABC'].apply(map_zone)
    by_item['Política_Inv'] = [policy_by_demand(cv, ii) for cv, ii in zip(by_item['cv'], by_item['intermittency'])]
    by_item['FillRate_obj'] = by_item['Zona_Bodega'].apply(target_fill_rate)
    by_item['Frecuencia_Recuento'] = by_item['Zona_Bodega'].apply(cycle_count_freq)

    st.session_state['by_item'] = by_item
    st.session_state['criterios_seleccionados'] = criterios_seleccionados
    st.session_state['crit1_name'] = crit1
    st.session_state['crit2_name'] = crit2
    st.success(f'Súper ABC calculado correctamente con {len(criterios_seleccionados)} criterios 🎯')

    # -------------------------------
    # Guardar by_item limpio como perfil
    # -------------------------------
    export_df = by_item.reset_index().copy()
    export_df.columns = [unicodedata.normalize('NFKD', str(c)).encode('ascii','ignore').decode('ascii') for c in export_df.columns]

    if 'FillRate_obj' in export_df.columns:
        export_df['FillRate_obj'] = export_df['FillRate_obj'].astype(str).str.replace('–','-', regex=False).str.replace('—','-', regex=False)

    export_df = sanitize_colnames(export_df)
    st.session_state['perfil_by_item_sanitizado'] = export_df


    # --- Comparación de artículos únicos para detectar pérdidas ---
    articulos_excel = set(df['Artículo'].astype(str).unique())
    articulos_base = set(base['Articulo'].unique())
    articulos_by_item = set(by_item.index)

    faltan_en_base = articulos_excel - articulos_base
    faltan_en_by_item = articulos_base - articulos_by_item

    st.write(f"Total artículos en Excel: {len(articulos_excel)}")
    st.write(f"Total artículos en base (con fecha válida): {len(articulos_base)}")
    st.write(f"Total artículos en by_item (agrupados): {len(articulos_by_item)}")

    if faltan_en_base:
        st.warning(f"Artículos en Excel pero no en base (probablemente por fecha vacía o inválida): {faltan_en_base}")
    if faltan_en_by_item:
        st.warning(f"Artículos en base pero no en by_item (posible error de agrupación): {faltan_en_by_item}")
    if not faltan_en_base and not faltan_en_by_item:
        st.info("No se pierden artículos en ninguna etapa del procesamiento.")
# -------------------------------
# Mostrar resumen y perfiles
# -------------------------------
def ira_by_class(clase: str) -> str:
    """
    Determina el IRA (Inventory Record Accuracy) basado en la clase de Súper ABC.
    Para múltiples criterios, se basa en la prioridad de las letras A, B, C.
    """
    # Contar la cantidad de cada letra
    count_a = clase.count('A')
    count_b = clase.count('B')
    count_c = clase.count('C')
    
    # Determinar IRA basado en la prioridad
    if count_a >= 2:  # Múltiples A
        return '> 95%'
    elif count_a == 1 and count_b >= 1:  # Una A y al menos una B
        return '94% - 95%'
    elif count_a == 1:  # Solo una A
        return '92% - 94%'
    elif count_b >= 2:  # Múltiples B
        return '90% - 92%'
    elif count_b == 1:  # Solo una B
        return '88% - 90%'
    elif count_c >= 2:  # Múltiples C
        return '86% - 88%'
    elif count_c == 1:  # Solo una C
        return '84% - 86%'
    else:
        return '< 80%'

if 'by_item' in st.session_state:
    by_item = st.session_state['by_item']

    if st.button('2) Mostrar tabla resumen y perfiles'):
        # Mostrar información sobre los criterios utilizados
        criterios_usados = st.session_state.get('criterios_seleccionados', [crit1, crit2])
        st.subheader(f'📋 Resumen por categoría - Criterios: {", ".join(criterios_usados)}')
        
        # Mostrar estadísticas de las combinaciones generadas
        combinaciones_unicas = by_item['Clase_SuperABC'].nunique()
        st.info(f"Se generaron {combinaciones_unicas} combinaciones únicas de clasificación ABC")
        
        summary = by_item.groupby('Clase_SuperABC').agg(
            Cantidad=('Clase_SuperABC','count'),
            Zona_Bodega=('Zona_Bodega','first'),
            Politica=('Política_Inv','first'),
            FillRate=('FillRate_obj','first'),
            Ventas=('ventas','sum'),
            Frecuencia_Recuento=('Frecuencia_Recuento','first')
        ).reset_index()

        # Insertar columna IRA después de FillRate
        summary['IRA'] = summary['Clase_SuperABC'].apply(ira_by_class)

        summary['Porcentaje'] = (summary['Cantidad']/summary['Cantidad'].sum()*100).round(2)
        total_sales = summary['Ventas'].sum()
        summary['% Ventas'] = (100 * summary['Ventas'] / (total_sales if total_sales>0 else 1)).round(2)

        # Ordenar categorías - para múltiples criterios, ordenar alfabéticamente
        summary = summary.sort_values('Clase_SuperABC')

        # Reordenar columnas para que IRA quede después de FillRate
        cols = ['Clase_SuperABC','Cantidad','Zona_Bodega','Politica','FillRate','IRA',
                'Frecuencia_Recuento','Ventas','Porcentaje','% Ventas']
        summary = summary[cols]

        st.dataframe(summary)
        st.session_state['perfil_resumen'] = summary

        # Perfil: lineas por orden (distribucion %)
        st.subheader('% de órdenes por # líneas')
        lines_per_order = base.groupby('NumDoc').agg(lineas=('Articulo','nunique')).reset_index()
        dist_lines = lines_per_order.groupby('lineas').size().rename('conteo').reset_index()
        total_orders = dist_lines['conteo'].sum()
        dist_lines['%_ordenes'] = 100 * dist_lines['conteo']/ (total_orders if total_orders>0 else 1)
        st.dataframe(dist_lines.sort_values('lineas'))
        fig_lines = px.bar(dist_lines.sort_values('lineas'), x='lineas', y='%_ordenes', labels={'lineas':'Líneas por orden','%_ordenes':'% de órdenes'})
        st.plotly_chart(fig_lines, use_container_width=True)

        st.session_state['perfil_lineas'] = dist_lines

        # Perfil: cubicaje por orden
        st.subheader('% de órdenes por rango de volumen (pies³)')
        cubic_per_order = base.groupby('NumDoc').agg(volumen_total=('Volumen_p3','sum')).reset_index()
        vol_bins = [-1,1,2,5,10,20,50,1e9]
        vol_labels = ['≤1','1-2','2-5','5-10','10-20','20-50','>50']
        cubic_per_order['vol_bin'] = pd.cut(cubic_per_order['volumen_total'], bins=vol_bins, labels=vol_labels)
        dist_cubic = cubic_per_order.groupby('vol_bin').size().rename('conteo').reset_index()
        total_orders2 = dist_cubic['conteo'].sum()
        dist_cubic['%_ordenes'] = 100 * dist_cubic['conteo']/ (total_orders2 if total_orders2>0 else 1)
        st.dataframe(dist_cubic)
        fig_cubic = px.bar(dist_cubic, x='vol_bin', y='%_ordenes', labels={'vol_bin':'Rango volumen (pies³)','%_ordenes':'% de órdenes'})
        st.plotly_chart(fig_cubic, use_container_width=True)

        st.session_state['perfil_cubicaje'] = dist_cubic

        # Distribucion por dia de la semana
        st.subheader('Distribución de órdenes por día de la semana')
        orders_dates = base.groupby('NumDoc').agg(fecha=('Fecha','max')).reset_index()
        orders_dates['dia'] = orders_dates['fecha'].dt.day_name()
        mapping_days = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miércoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sábado','Sunday':'Domingo'}
        orders_dates['dia'] = orders_dates['dia'].replace(mapping_days)
        day_order = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
        dist_days = orders_dates.groupby('dia').size().reindex(day_order).fillna(0).astype(int).rename('conteo').reset_index()
        dist_days['%_ordenes'] = 100 * dist_days['conteo'] / (dist_days['conteo'].sum() if dist_days['conteo'].sum()>0 else 1)
        st.dataframe(dist_days)
        fig_days = px.bar(dist_days, x='dia', y='%_ordenes', labels={'dia':'Día','%_ordenes':'% de órdenes'})
        st.plotly_chart(fig_days, use_container_width=True)

        st.session_state['perfil_dias'] = dist_days

        # Preparar datos
        lv = base.groupby('NumDoc').agg(
            lineas=('Articulo','nunique'),
            volumen_total=('Volumen_p3','sum')
        ).reset_index()

        # Parámetro: volumen de una tarima completa (ajusta según tu operación)
        VOLUMEN_TARIMA = st.session_state.get('vol_tarima', 42.38)

        # % de carga unitaria respecto a una tarima
        lv['%_carga_unidad'] = 100 * lv['volumen_total'] / VOLUMEN_TARIMA
        lv['%_carga_unidad'] = lv['%_carga_unidad'].clip(upper=100)  # máximo 100%

        # Bins para % de carga unitaria
        carga_bins = list(range(0, 105, 5))
        carga_labels = [f'{i}-{i+5}%' for i in range(0, 100, 5)]
        lv['r_carga'] = pd.cut(lv['%_carga_unidad'], bins=carga_bins, labels=carga_labels, right=True, include_lowest=True)
        
        # Distribución cruzada: % líneas de pedido vs % carga unitaria
        dist_incremento = lv.groupby(['r_carga']).agg(
            pedidos=('NumDoc', 'count'),
            lineas_prom=('lineas', 'mean')
        ).reset_index()
        dist_incremento['%_lineas_pedido'] = 100 * dist_incremento['pedidos'] / dist_incremento['pedidos'].sum()

        st.subheader('Distribución por incremento de pedidos (% carga unitaria vs % de líneas de pedido)')
        st.dataframe(dist_incremento.rename(columns={'%_lineas_pedido': '% de líneas de pedido'}))
        fig_incremento = px.bar(
            dist_incremento,
            x='r_carga',
            y='%_lineas_pedido',
            labels={'r_carga': '% de carga unitaria (tarima)', '%_lineas_pedido': '% de líneas de pedido'},
            title='% de líneas de pedido por % de carga unitaria'
        )
        st.plotly_chart(fig_incremento, use_container_width=True)

        st.session_state['perfil_carga'] = dist_incremento

        # -------------------------------
        # Tabla cruzada líneas x volumen por pedido
        # -------------------------------
        st.subheader('Tabla cruzada: Líneas por pedido vs pies³ por pedido')

        # Categorías
        line_labels = ['1','2-5','6-9','10+']
        lv['r_lineas'] = pd.cut(lv['lineas'], bins=[0,1,6,10,1e9], labels=line_labels, right=True, include_lowest=True)

        vol_labels = ['0-1','1-2','2-5','5-10','10-20','20+']
        lv['r_vol'] = pd.cut(lv['volumen_total'], bins=[0,1,2,5,10,20,1e9], labels=vol_labels, right=True, include_lowest=True)

        # Desglose de pedidos por rango de líneas (incluyendo volumen)
        st.subheader('Desglose de pedidos por rango de líneas')
        for rango in line_labels:
            pedidos_rango = lv[lv['r_lineas'] == rango][['NumDoc', 'lineas', 'volumen_total', 'r_vol']]
            st.markdown(f"**Rango {rango}: {len(pedidos_rango)} pedidos**")
            st.dataframe(pedidos_rango.reset_index(drop=True))

        # Conteo de pedidos por línea y volumen
        ct_counts = pd.crosstab(lv['r_lineas'], lv['r_vol'], dropna=False)

        # Totales de línea y % pedidos por línea
        ct_counts['Totales'] = ct_counts.sum(axis=1)
        ct_counts['% pedidos'] = (ct_counts['Totales'] / ct_counts['Totales'].sum() * 100).round(2)

        # 🔹 Total de líneas (sumando líneas, no volumen)
        pivot_lines = pd.pivot_table(
            lv,
            index='r_lineas',
            values='lineas',
            aggfunc='sum',
            fill_value=0
        )
        total_lines_global = pivot_lines['lineas'].sum()
        pivot_lines['% linea'] = (pivot_lines['lineas'] / (total_lines_global if total_lines_global>0 else 1) * 100).round(2)

        # Crear tabla final con columnas en orden deseado
        table_final = pd.DataFrame(index=line_labels, columns=vol_labels + ['Totales','% pedidos','Total_Linea','% linea'])
        table_final[vol_labels] = ct_counts[vol_labels]
        table_final['Totales'] = ct_counts['Totales']
        table_final['% pedidos'] = ct_counts['% pedidos']
        table_final['Total_Linea'] = pivot_lines['lineas']
        table_final['% linea'] = pivot_lines['% linea']

        # Fila Totales
        totales_row = table_final[vol_labels].sum()
        totales_row['Totales'] = table_final['Totales'].sum()
        totales_row['% pedidos'] = 100
        totales_row['Total_Linea'] = table_final['Total_Linea'].sum()
        totales_row['% linea'] = 100
        table_final.loc['Totales'] = totales_row

        # Fila % pedidos (por columna)
        pct_pedidos_row = (table_final.loc[line_labels, vol_labels].sum() / table_final['Totales'].sum() * 100).round(2)
        pct_pedidos_row['Totales'] = 100
        pct_pedidos_row['% pedidos'] = np.nan
        pct_pedidos_row['Total_Linea'] = np.nan
        pct_pedidos_row['% linea'] = np.nan
        table_final.loc['% pedidos'] = pct_pedidos_row

        # Fila Espacio total (volumen)
        espacio_total_row = lv.groupby('r_vol')['volumen_total'].sum()
        espacio_total_row = espacio_total_row.reindex(vol_labels, fill_value=0)
        espacio_total_row['Totales'] = espacio_total_row.sum()
        espacio_total_row['% pedidos'] = np.nan
        espacio_total_row['Total_Linea'] = np.nan       # No calcular para espacio total
        espacio_total_row['% linea'] = np.nan           # No calcular para espacio total
        table_final.loc['Espacio total'] = espacio_total_row

        # Renombrar índice
        table_final.index.name = 'Líneas por orden / Volumen por orden'

        # Mostrar tabla
        st.dataframe(table_final.round(2))

        st.session_state['perfil_cruzado'] = table_final

        # Pareto popularidad
        st.subheader('Pareto de popularidad de ítems (picks acumulados)')
        pareto = by_item.sort_values('popularidad', ascending=False)[['popularidad']].copy()
        pareto['cum_picks'] = pareto['popularidad'].cumsum()
        total_picks = pareto['popularidad'].sum()
        pareto['cum_pct_picks'] = 100 * pareto['cum_picks'] / (total_picks if total_picks>0 else 1)
        pareto['sku_rank'] = np.arange(1, len(pareto)+1)
        pareto['pct_sku'] = 100 * pareto['sku_rank'] / len(pareto)
        st.dataframe(pareto.head(20))
        fig_pareto = px.line(pareto, x='pct_sku', y='cum_pct_picks', labels={'pct_sku':'% de SKU (acumulado)','cum_pct_picks':'% de picks (acumulado)'}, title='Curva de Pareto – Popularidad')
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.session_state['perfil_pareto'] = pareto


    # -------------------------------
    # Exportar CSV (sanitizado)
    # -------------------------------

    # -------------------------------
    # Datos generales
    # -------------------------------
    file_name = st.session_state.get('file_name', uploaded_file.name if uploaded_file else 'Archivo no registrado')
    sheet_used = st.session_state.get('sheet_name', sheet_name or 'Hoja no registrada')
    vol_units = st.session_state.get('vol_units', unit_vol)
    criterios_usados = st.session_state.get('criterios_seleccionados', [crit1, crit2])
    cortes_abc = st.session_state.get('cortes_abc', {})
    
    # Para compatibilidad con código existente
    crit1 = criterios_usados[0] if criterios_usados else 'Popularidad'
    crit2 = criterios_usados[1] if len(criterios_usados) > 1 else criterios_usados[0] if criterios_usados else 'Ventas'
    
    # Obtener cortes de la nueva estructura
    A_cut_1 = cortes_abc.get(crit1, {}).get('A', 0.8) if cortes_abc else 0.8
    B_cut_1 = cortes_abc.get(crit1, {}).get('B', 0.95) if cortes_abc else 0.95
    A_cut_2 = cortes_abc.get(crit2, {}).get('A', 0.8) if cortes_abc else 0.8
    B_cut_2 = cortes_abc.get(crit2, {}).get('B', 0.95) if cortes_abc else 0.95

    # -------------------------------
    # Crear hoja Portada
    # -------------------------------
    # Crear datos de portada dinámicamente
    portada_campos = ['Documento leído', 'Hoja utilizada', 'Unidades de volumen', 'Criterios utilizados']
    portada_valores = [file_name, sheet_used, vol_units, ', '.join(criterios_usados)]
    
    # Agregar cortes para cada criterio
    for criterio in criterios_usados:
        if criterio in cortes_abc:
            portada_campos.extend([f'Corte A ({criterio})', f'Corte B ({criterio})'])
            portada_valores.extend([cortes_abc[criterio]['A'], cortes_abc[criterio]['B']])
    
    portada_data = {
        'Campo': portada_campos,
        'Valor': portada_valores
    }

    df_portada = pd.DataFrame(portada_data)


    if want_csv:
        if st.button("📥 Exportar perfiles a Excel"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                # Hoja Portada primero
                df_portada.to_excel(writer, sheet_name='Portada', index=False)
                for key, df in st.session_state.items():
                    if key.startswith("perfil_") and isinstance(df, pd.DataFrame):
                        hoja = key.replace("perfil_", "")[:30]  # hoja ≤ 31 chars
                        df.to_excel(writer, sheet_name=hoja, index=False)

            st.download_button(
                "📊 Descargar Excel con perfiles",
                data=buffer.getvalue(),
                file_name="perfiles_distribuciones.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        


if 'by_item' in st.session_state:
    by_item = st.session_state['by_item']
    base = st.session_state['base']

    st.header('🔮 Forecasting de Demanda por Artículo')

    # Selección de artículo
    articulos = sorted(base['Articulo'].unique())
    articulo_sel = st.selectbox('Selecciona Artículo para pronóstico', articulos, key='forecast_articulo')

    # Período y cantidad de forecast
    periodo_forecast = st.selectbox('Periodo de forecast', ['Mensual', 'Semanal'], index=0)
    n_periods = st.number_input(f'Períodos a pronosticar ({periodo_forecast.lower()})', min_value=1, max_value=52, value=4, step=1)

    # Unidad a pronosticar
    unidad_forecast = st.selectbox('Unidad a pronosticar', ['Unidades vendidas', 'Cajas vendidas'], index=0)
    columna_forecast = 'Unidades' if unidad_forecast=='Unidades vendidas' else 'Cajas_vendidas'

    # Filtrar datos
    base_art = base[base['Articulo']==articulo_sel].copy()
    if base_art.empty:
        st.warning("No hay registros para ese artículo.")
        st.stop()
    for col in ['Unidades','Cajas_vendidas']:
        base_art[col] = pd.to_numeric(base_art.get(col,0), errors='coerce').fillna(0)

    # Serie histórica
    orders_df = base_art.groupby('NumDoc').agg(Fecha=('Fecha','max'),
                                               Unidades=('Unidades','sum'),
                                               Cajas_vendidas=('Cajas_vendidas','sum')).reset_index()
    resample_freq = 'MS' if periodo_forecast=='Mensual' else 'W-MON'
    date_offset = pd.DateOffset(months=1) if periodo_forecast=='Mensual' else pd.DateOffset(weeks=1)
    ts_art = orders_df.set_index('Fecha')[columna_forecast].resample(resample_freq).sum().fillna(0)
    st.subheader("Serie histórica")
    st.line_chart(ts_art)

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import plotly.graph_objects as go

    # Modelos
    modelos = ['Media móvil (4 periodos)','Holt-Winters','Prophet','Random Forest']
    forecasts_dict = {}
    resultados = []

    for modelo in modelos:
        try:
            last_index = ts_art.index[-1]
            future_index = pd.date_range(start=last_index + date_offset, periods=n_periods, freq=resample_freq)
            forecast_future = None
            forecast_hist = None

            # ---------------- Media Móvil ----------------
            if modelo=='Media móvil (4 periodos)':
                ma = ts_art.rolling(window=4, min_periods=1).mean().shift(1)
                ma = ma.fillna(ts_art)  # reemplazar NaN iniciales
                forecast_future = pd.Series([ma.iloc[-1]]*n_periods, index=future_index)
                forecast_hist = ma

            # ---------------- Holt-Winters ----------------
            elif modelo=='Holt-Winters':
                if len(ts_art) >= 2:
                    # Detectar estacionalidad automáticamente si hay suficientes ciclos
                    period = None
                    if periodo_forecast=='Mensual' and len(ts_art) >= 24:
                        period = 12
                    elif periodo_forecast=='Semanal' and len(ts_art) >= 104:
                        period = 52

                    hw = sm.tsa.ExponentialSmoothing(
                        ts_art,
                        trend='add',
                        seasonal='add' if period else None,
                        seasonal_periods=period,
                        initialization_method="estimated"
                    ).fit()
                    forecast_future = pd.Series(hw.forecast(n_periods).values, index=future_index)
                    forecast_hist = hw.fittedvalues
                else:
                    st.info("Holt-Winters omitido por pocos datos.")

            # ---------------- Prophet ----------------
            elif modelo=='Prophet':
                from prophet import Prophet
                df_prophet = ts_art.reset_index().rename(columns={'Fecha':'ds', columna_forecast:'y'})
                
                if len(df_prophet) >= 3:
                    # Decidir automáticamente la estacionalidad
                    yearly = False
                    weekly = False
                    daily = False  # normalmente no se usa para datos semanales/mensuales

                    if periodo_forecast=='Mensual' and len(ts_art) >= 24:
                        yearly = True
                    if periodo_forecast=='Semanal':
                        if len(ts_art) >= 104:
                            yearly = True
                        if len(ts_art) >= 8:
                            weekly = True

                    m = Prophet(yearly_seasonality=yearly,
                                weekly_seasonality=weekly,
                                daily_seasonality=daily)
                    m.fit(df_prophet)
                    future_all = m.make_future_dataframe(periods=n_periods, freq=resample_freq)
                    forecast = m.predict(future_all)
                    # Forzar valores positivos para forecast futuro
                    forecast_future = pd.Series(np.maximum(forecast['yhat'].tail(n_periods).values, 0),
                                                index=forecast['ds'].tail(n_periods))
                    forecast_hist = pd.Series(forecast['yhat'].iloc[:len(ts_art)].values, index=ts_art.index)


            # ---------------- Random Forest ----------------
            elif modelo=='Random Forest':
                df_ml = ts_art.copy().reset_index()
                df_ml.rename(columns={'Fecha':'Periodo', columna_forecast:'y'}, inplace=True)
                # Reindexar a frecuencia continua y rellenar vacíos
                df_ml = df_ml.set_index('Periodo').asfreq(resample_freq, fill_value=0).reset_index()
                max_lag = 4
                for lag in range(1, max_lag+1):
                    df_ml[f'lag_{lag}'] = df_ml['y'].shift(lag)
                    df_ml[f'lag_{lag}'].fillna(df_ml['y'].iloc[0], inplace=True)  # Rellenar NaN iniciales

                X = df_ml[[f'lag_{i}' for i in range(1, max_lag+1)]].to_numpy()
                y = df_ml['y'].to_numpy()
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                rf.fit(X, y)

                # Forecast futuro
                last_values = list(df_ml.iloc[-1][[f'lag_{i}' for i in range(1, max_lag+1)]])
                preds_future = []
                for _ in range(n_periods):
                    pred = rf.predict([last_values])[0]
                    pred = max(pred, 0)
                    preds_future.append(pred)
                    last_values = [pred] + last_values[:-1]
                forecast_future = pd.Series(preds_future, index=future_index)

                # Forecast histórico
                forecast_hist = pd.Series(rf.predict(X), index=df_ml['Periodo'])
                forecast_hist = forecast_hist.reindex(ts_art.index, method='ffill')

            # Guardar resultados y métricas
            if forecast_future is not None:
                forecasts_dict[modelo] = {'future':forecast_future, 'hist':forecast_hist}
                if forecast_hist is not None and len(forecast_hist)==len(ts_art):
                    y_true = ts_art.values
                    y_pred = forecast_hist.values
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = np.mean(np.abs((y_true-y_pred)/(y_true+1e-9)))*100
                    # Detectar posibles valores absurdos
                    mape_warning = mape > 1000  # umbral arbitrario para advertencia
                    if mape_warning:
                        st.warning(f"⚠️ El MAPE del modelo '{modelo}' es extremadamente alto ({mape:.2f}%). Esto puede ocurrir por valores cercanos a cero en la serie histórica y puede no reflejar un error realista. Use el MAPE simétrico (SMAPE) como referencia.")
                    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-9))
                    resultados.append({'Modelo':modelo,'MAE':mae,'RMSE':rmse,'MAPE (%)':mape,'SMAPE (%)':smape})

        except Exception as e:
            st.warning(f"{modelo} omitido: {e}")
            
    # ----------- Tabla de métricas -----------
    df_resultados = pd.DataFrame(resultados)
    if not df_resultados.empty:
        st.subheader("📊 Comparación de métricas")
        st.dataframe(df_resultados.round(2).sort_values('RMSE'))

    # ----------- Selección de modelos a mostrar -----------
    modelos_disp = st.multiselect("Selecciona modelos a mostrar en la gráfica", list(forecasts_dict.keys()), default=list(forecasts_dict.keys()))

    # ----------- Gráfico interactivo con Plotly -----------
    st.subheader("📈 Comparativa interactiva de forecasts")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_art.index, y=ts_art.values, mode='lines+markers', name='Observado', line=dict(color='black', width=3)))
    for modelo in modelos_disp:
        data = forecasts_dict[modelo]
        if data['hist'] is not None:
            fig.add_trace(go.Scatter(x=data['hist'].index, y=data['hist'].values, mode='lines', name=f"{modelo} (hist)", line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=data['future'].index, y=data['future'].values, mode='lines+markers', name=f"{modelo} (futuro)"))
    fig.update_layout(hovermode='x unified', xaxis_title='Fecha', yaxis_title=columna_forecast)
    st.plotly_chart(fig, use_container_width=True)

    # ----------- Descarga ZIP -----------
    import io, zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        if not df_resultados.empty:
            zf.writestr("comparacion_metricas.csv", df_resultados.round(2).to_csv(index=False))
        all_forecasts = pd.DataFrame({m:data['future'] for m,data in forecasts_dict.items()})
        all_forecasts.index.name='Periodo'
        all_forecasts.reset_index(inplace=True)
        zf.writestr("forecasts_modelos.csv", all_forecasts.round(2).to_csv(index=False))
    st.download_button("📥 Descargar resultados completos (ZIP)", data=buffer.getvalue(),
                       file_name=f"forecast_completo_{articulo_sel}.zip", mime="application/zip")

    # -------------------------------
    # Forecasting por Categorías Súper ABC
    # -------------------------------
    st.header('🎯 Forecasting de Demanda por Categorías Súper ABC')
    
    st.markdown("""
    Esta sección permite hacer pronósticos de demanda agregada por categorías del Súper ABC, 
    lo cual es útil para planificación estratégica y gestión de inventarios a nivel de categoría.
    
    **Modelos disponibles:** Media móvil, Holt-Winters, Prophet y Random Forest (mismo conjunto que para SKU individual).
    """)
    
    # Verificar que tenemos datos de Súper ABC
    if 'Clase_SuperABC' not in by_item.columns:
        st.warning("⚠️ Primero debes calcular el Súper ABC para usar esta funcionalidad.")
    else:
        # Obtener categorías disponibles
        categorias_disponibles = sorted(by_item['Clase_SuperABC'].unique())
        
        # Selección de categorías
        col1, col2 = st.columns(2)
        
        with col1:
            # Permitir selección múltiple de categorías
            categorias_seleccionadas = st.multiselect(
                'Selecciona las categorías ABC a pronosticar:',
                categorias_disponibles,
                default=categorias_disponibles[:3] if len(categorias_disponibles) >= 3 else categorias_disponibles,
                help='Puedes seleccionar una o más categorías para comparar sus pronósticos'
            )
        
        with col2:
            # Parámetros de forecast
            periodo_forecast_cat = st.selectbox('Periodo de forecast', ['Mensual', 'Semanal'], index=0, key='forecast_cat_periodo')
            n_periods_cat = st.number_input(f'Períodos a pronosticar ({periodo_forecast_cat.lower()})', 
                                          min_value=1, max_value=52, value=6, step=1, key='forecast_cat_periods')
            unidad_forecast_cat = st.selectbox('Unidad a pronosticar', ['Unidades vendidas', 'Cajas vendidas'], 
                                             index=0, key='forecast_cat_unidad')
        
        if categorias_seleccionadas:
            columna_forecast_cat = 'Unidades' if unidad_forecast_cat=='Unidades vendidas' else 'Cajas_vendidas'
            
            # Agregar datos por categoría
            base_con_categoria = base.merge(
                by_item[['Clase_SuperABC']].reset_index(), 
                left_on='Articulo', 
                right_on='Articulo', 
                how='left'
            )
            
            # Filtrar solo las categorías seleccionadas
            base_categorias = base_con_categoria[base_con_categoria['Clase_SuperABC'].isin(categorias_seleccionadas)]
            
            if base_categorias.empty:
                st.warning("No hay datos para las categorías seleccionadas.")
            else:
                # Agregar por categoría y fecha
                resample_freq_cat = 'MS' if periodo_forecast_cat=='Mensual' else 'W-MON'
                date_offset_cat = pd.DateOffset(months=1) if periodo_forecast_cat=='Mensual' else pd.DateOffset(weeks=1)
                
                # Agregar datos por categoría y período
                ts_categorias = base_categorias.groupby(['Clase_SuperABC', 'Fecha'])[columna_forecast_cat].sum().reset_index()
                ts_categorias = ts_categorias.set_index('Fecha').groupby('Clase_SuperABC')[columna_forecast_cat].resample(resample_freq_cat).sum().fillna(0)
                
                # Mostrar series históricas
                st.subheader("📊 Series históricas por categoría")
                ts_categorias_pivot = ts_categorias.unstack(level=0).fillna(0)
                st.line_chart(ts_categorias_pivot)
                
                # Estadísticas por categoría
                st.subheader("📈 Estadísticas por categoría")
                stats_categorias = ts_categorias.groupby('Clase_SuperABC').agg([
                    'count', 'mean', 'std', 'min', 'max', 'sum'
                ]).round(2)
                stats_categorias.columns = ['Períodos', 'Promedio', 'Desv. Est.', 'Mínimo', 'Máximo', 'Total']
                st.dataframe(stats_categorias)
                
                # Forecasting por categoría
                st.subheader("🔮 Pronósticos por categoría")
                
                # Modelos para categorías (incluir Random Forest como en SKU individual)
                modelos_cat = ['Media móvil (4 periodos)', 'Holt-Winters', 'Prophet', 'Random Forest']
                forecasts_cat_dict = {}
                resultados_cat = []
                
                for categoria in categorias_seleccionadas:
                    if categoria in ts_categorias.index.get_level_values(0):
                        ts_cat = ts_categorias.loc[categoria]
                        
                        if len(ts_cat) < 2:
                            st.warning(f"Categoría {categoria}: Insuficientes datos para pronóstico")
                            continue
                        
                        st.write(f"**Pronosticando categoría: {categoria}**")
                        
                        # Crear índice futuro
                        last_index_cat = ts_cat.index[-1]
                        future_index_cat = pd.date_range(start=last_index_cat + date_offset_cat, 
                                                       periods=n_periods_cat, freq=resample_freq_cat)
                        
                        categoria_forecasts = {}
                        
                        for modelo in modelos_cat:
                            try:
                                forecast_future_cat = None
                                forecast_hist_cat = None
                                
                                # Media Móvil
                                if modelo == 'Media móvil (4 periodos)':
                                    ma_cat = ts_cat.rolling(window=4, min_periods=1).mean().shift(1)
                                    ma_cat = ma_cat.fillna(ts_cat)
                                    forecast_future_cat = pd.Series([ma_cat.iloc[-1]]*n_periods_cat, index=future_index_cat)
                                    forecast_hist_cat = ma_cat
                                
                                # Holt-Winters
                                elif modelo == 'Holt-Winters':
                                    if len(ts_cat) >= 2:
                                        period_cat = None
                                        if periodo_forecast_cat=='Mensual' and len(ts_cat) >= 24:
                                            period_cat = 12
                                        elif periodo_forecast_cat=='Semanal' and len(ts_cat) >= 104:
                                            period_cat = 52
                                        
                                        hw_cat = sm.tsa.ExponentialSmoothing(
                                            ts_cat,
                                            trend='add',
                                            seasonal='add' if period_cat else None,
                                            seasonal_periods=period_cat,
                                            initialization_method="estimated"
                                        ).fit()
                                        forecast_future_cat = pd.Series(hw_cat.forecast(n_periods_cat).values, index=future_index_cat)
                                        forecast_hist_cat = hw_cat.fittedvalues
                                
                                # Prophet
                                elif modelo == 'Prophet':
                                    df_prophet_cat = ts_cat.reset_index().rename(columns={'Fecha':'ds', columna_forecast_cat:'y'})
                                    
                                    if len(df_prophet_cat) >= 3:
                                        yearly_cat = False
                                        weekly_cat = False
                                        
                                        if periodo_forecast_cat=='Mensual' and len(ts_cat) >= 24:
                                            yearly_cat = True
                                        if periodo_forecast_cat=='Semanal':
                                            if len(ts_cat) >= 104:
                                                yearly_cat = True
                                            if len(ts_cat) >= 8:
                                                weekly_cat = True
                                        
                                        m_cat = Prophet(yearly_seasonality=yearly_cat,
                                                      weekly_seasonality=weekly_cat,
                                                      daily_seasonality=False)
                                        m_cat.fit(df_prophet_cat)
                                        future_all_cat = m_cat.make_future_dataframe(periods=n_periods_cat, freq=resample_freq_cat)
                                        forecast_cat = m_cat.predict(future_all_cat)
                                        forecast_future_cat = pd.Series(np.maximum(forecast_cat['yhat'].tail(n_periods_cat).values, 0),
                                                                      index=forecast_cat['ds'].tail(n_periods_cat))
                                        forecast_hist_cat = pd.Series(forecast_cat['yhat'].iloc[:len(ts_cat)].values, index=ts_cat.index)
                                
                                # Random Forest
                                elif modelo == 'Random Forest':
                                    df_ml_cat = ts_cat.copy().reset_index()
                                    df_ml_cat.rename(columns={'Fecha':'Periodo', columna_forecast_cat:'y'}, inplace=True)
                                    # Reindexar a frecuencia continua y rellenar vacíos
                                    df_ml_cat = df_ml_cat.set_index('Periodo').asfreq(resample_freq_cat, fill_value=0).reset_index()
                                    
                                    max_lag_cat = 4
                                    for lag in range(1, max_lag_cat+1):
                                        df_ml_cat[f'lag_{lag}'] = df_ml_cat['y'].shift(lag)
                                        df_ml_cat[f'lag_{lag}'].fillna(df_ml_cat['y'].iloc[0], inplace=True)  # Rellenar NaN iniciales
                                    
                                    X_cat = df_ml_cat[[f'lag_{i}' for i in range(1, max_lag_cat+1)]].to_numpy()
                                    y_cat = df_ml_cat['y'].to_numpy()
                                    
                                    rf_cat = RandomForestRegressor(n_estimators=200, random_state=42)
                                    rf_cat.fit(X_cat, y_cat)
                                    
                                    # Forecast futuro
                                    last_values_cat = list(df_ml_cat.iloc[-1][[f'lag_{i}' for i in range(1, max_lag_cat+1)]])
                                    preds_future_cat = []
                                    for _ in range(n_periods_cat):
                                        pred_cat = rf_cat.predict([last_values_cat])[0]
                                        pred_cat = max(pred_cat, 0)
                                        preds_future_cat.append(pred_cat)
                                        last_values_cat = [pred_cat] + last_values_cat[:-1]
                                    
                                    forecast_future_cat = pd.Series(preds_future_cat, index=future_index_cat)
                                    
                                    # Forecast histórico
                                    forecast_hist_cat = pd.Series(rf_cat.predict(X_cat), index=df_ml_cat['Periodo'])
                                    forecast_hist_cat = forecast_hist_cat.reindex(ts_cat.index, method='ffill')
                                
                                if forecast_future_cat is not None:
                                    categoria_forecasts[modelo] = {'future': forecast_future_cat, 'hist': forecast_hist_cat}
                                    
                                    # Calcular métricas
                                    if forecast_hist_cat is not None and len(forecast_hist_cat) == len(ts_cat):
                                        y_true_cat = ts_cat.values
                                        y_pred_cat = forecast_hist_cat.values
                                        mae_cat = mean_absolute_error(y_true_cat, y_pred_cat)
                                        rmse_cat = np.sqrt(mean_squared_error(y_true_cat, y_pred_cat))
                                        mape_cat = np.mean(np.abs((y_true_cat-y_pred_cat)/(y_true_cat+1e-9)))*100
                                        smape_cat = 100 * np.mean(2 * np.abs(y_true_cat - y_pred_cat) / (np.abs(y_true_cat) + np.abs(y_pred_cat) + 1e-9))
                                        
                                        resultados_cat.append({
                                            'Categoría': categoria,
                                            'Modelo': modelo,
                                            'MAE': mae_cat,
                                            'RMSE': rmse_cat,
                                            'MAPE (%)': mape_cat,
                                            'SMAPE (%)': smape_cat
                                        })
                                
                            except Exception as e:
                                st.warning(f"Categoría {categoria}, {modelo} omitido: {e}")
                        
                        forecasts_cat_dict[categoria] = categoria_forecasts
                
                # Mostrar resultados
                if resultados_cat:
                    st.subheader("📊 Comparación de métricas por categoría")
                    df_resultados_cat = pd.DataFrame(resultados_cat)
                    st.dataframe(df_resultados_cat.round(2).sort_values(['Categoría', 'RMSE']))
                    
                    # Gráfico comparativo
                    st.subheader("📈 Comparativa de pronósticos por categoría")
                    
                    # Seleccionar modelo para comparar
                    modelos_disponibles_cat = list(set([r['Modelo'] for r in resultados_cat]))
                    modelo_comparar = st.selectbox("Selecciona modelo para comparar categorías:", 
                                                 modelos_disponibles_cat, key='modelo_comparar_cat')
                    
                    fig_cat = go.Figure()
                    
                    # Colores para categorías
                    colors = px.colors.qualitative.Set3
                    
                    for i, categoria in enumerate(categorias_seleccionadas):
                        if categoria in forecasts_cat_dict and modelo_comparar in forecasts_cat_dict[categoria]:
                            data_cat = forecasts_cat_dict[categoria][modelo_comparar]
                            color = colors[i % len(colors)]
                            
                            # Serie histórica
                            ts_cat_plot = ts_categorias.loc[categoria]
                            fig_cat.add_trace(go.Scatter(
                                x=ts_cat_plot.index, 
                                y=ts_cat_plot.values, 
                                mode='lines+markers', 
                                name=f'{categoria} (hist)',
                                line=dict(color=color, width=2)
                            ))
                            
                            # Pronóstico histórico
                            if data_cat['hist'] is not None:
                                fig_cat.add_trace(go.Scatter(
                                    x=data_cat['hist'].index, 
                                    y=data_cat['hist'].values, 
                                    mode='lines', 
                                    name=f'{categoria} ({modelo_comparar} hist)',
                                    line=dict(color=color, dash='dot')
                                ))
                            
                            # Pronóstico futuro
                            fig_cat.add_trace(go.Scatter(
                                x=data_cat['future'].index, 
                                y=data_cat['future'].values, 
                                mode='lines+markers', 
                                name=f'{categoria} ({modelo_comparar} futuro)',
                                line=dict(color=color, width=3)
                            ))
                    
                    fig_cat.update_layout(
                        hovermode='x unified', 
                        xaxis_title='Fecha', 
                        yaxis_title=columna_forecast_cat,
                        title=f'Pronósticos por categoría - {modelo_comparar}'
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
                    
                    # Descarga de resultados por categorías
                    st.subheader("📥 Descargar resultados por categorías")
                    buffer_cat = io.BytesIO()
                    with zipfile.ZipFile(buffer_cat, 'w') as zf:
                        # Métricas por categoría
                        zf.writestr("metricas_por_categoria.csv", df_resultados_cat.round(2).to_csv(index=False))
                        
                        # Pronósticos por categoría
                        for categoria in categorias_seleccionadas:
                            if categoria in forecasts_cat_dict:
                                for modelo in forecasts_cat_dict[categoria]:
                                    data_cat = forecasts_cat_dict[categoria][modelo]
                                    forecast_df = pd.DataFrame({
                                        'Periodo': data_cat['future'].index,
                                        'Pronostico': data_cat['future'].values
                                    })
                                    zf.writestr(f"forecast_{categoria}_{modelo}.csv", forecast_df.to_csv(index=False))
                    
                    st.download_button(
                        "📊 Descargar pronósticos por categorías (ZIP)", 
                        data=buffer_cat.getvalue(),
                        file_name="forecasts_por_categorias.zip", 
                        mime="application/zip"
                    )
        else:
            st.info("Selecciona al menos una categoría para hacer pronósticos.")
        
        # -------------------------------
        # Análisis de Contribución por Categorías
        # -------------------------------
        st.header('📊 Análisis de Contribución por Categorías ABC')
        
        st.markdown("""
        Esta sección proporciona un análisis detallado de la contribución de cada categoría ABC 
        al total de ventas, volumen y popularidad, útil para entender el impacto de cada categoría.
        """)
        
        # Análisis de contribución
        contribucion_categorias = by_item.groupby('Clase_SuperABC').agg({
            'ventas': 'sum',
            'volumen': 'sum', 
            'popularidad': 'sum',
            'unidades': 'sum'
        }).round(2)
        
        # Agregar conteo de artículos (usando el índice)
        contribucion_categorias['Cantidad_Articulos'] = by_item.groupby('Clase_SuperABC').size()
        
        # Calcular porcentajes
        total_ventas = contribucion_categorias['ventas'].sum()
        total_volumen = contribucion_categorias['volumen'].sum()
        total_popularidad = contribucion_categorias['popularidad'].sum()
        total_unidades = contribucion_categorias['unidades'].sum()
        total_articulos = contribucion_categorias['Cantidad_Articulos'].sum()
        
        contribucion_categorias['% Ventas'] = (contribucion_categorias['ventas'] / total_ventas * 100).round(2)
        contribucion_categorias['% Volumen'] = (contribucion_categorias['volumen'] / total_volumen * 100).round(2)
        contribucion_categorias['% Popularidad'] = (contribucion_categorias['popularidad'] / total_popularidad * 100).round(2)
        contribucion_categorias['% Unidades'] = (contribucion_categorias['unidades'] / total_unidades * 100).round(2)
        contribucion_categorias['% Artículos'] = (contribucion_categorias['Cantidad_Articulos'] / total_articulos * 100).round(2)
        
        # Renombrar columnas
        contribucion_categorias.columns = ['Ventas', 'Volumen', 'Popularidad', 'Unidades', 'Cantidad Artículos', 
                                         '% Ventas', '% Volumen', '% Popularidad', '% Unidades', '% Artículos']
        
        st.subheader("📈 Tabla de Contribución por Categorías")
        st.dataframe(contribucion_categorias)
        
        # Gráficos de contribución
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 Contribución de Ventas por Categoría")
            fig_ventas = px.pie(
                values=contribucion_categorias['% Ventas'], 
                names=contribucion_categorias.index,
                title="Distribución de Ventas por Categoría ABC"
            )
            st.plotly_chart(fig_ventas, use_container_width=True)
        
        with col2:
            st.subheader("🥧 Contribución de Volumen por Categoría")
            fig_volumen = px.pie(
                values=contribucion_categorias['% Volumen'], 
                names=contribucion_categorias.index,
                title="Distribución de Volumen por Categoría ABC"
            )
            st.plotly_chart(fig_volumen, use_container_width=True)
        
        # Gráfico de barras comparativo
        st.subheader("📊 Comparación de Contribuciones")
        
        # Preparar datos para gráfico de barras
        contrib_data = contribucion_categorias[['% Ventas', '% Volumen', '% Popularidad', '% Unidades']].reset_index()
        contrib_data_melted = contrib_data.melt(
            id_vars=['Clase_SuperABC'], 
            value_vars=['% Ventas', '% Volumen', '% Popularidad', '% Unidades'],
            var_name='Métrica', 
            value_name='Porcentaje'
        )
        
        fig_barras = px.bar(
            contrib_data_melted, 
            x='Clase_SuperABC', 
            y='Porcentaje', 
            color='Métrica',
            title="Comparación de Contribuciones por Categoría ABC",
            barmode='group'
        )
        fig_barras.update_layout(xaxis_title="Categoría ABC", yaxis_title="Porcentaje (%)")
        st.plotly_chart(fig_barras, use_container_width=True)
        
        # Análisis de concentración (Pareto por categorías)
        st.subheader("📈 Análisis de Concentración (Pareto)")
        
        # Ordenar por ventas
        pareto_categorias = contribucion_categorias.sort_values('% Ventas', ascending=False)
        pareto_categorias['Ventas_Acumuladas'] = pareto_categorias['% Ventas'].cumsum()
        pareto_categorias['Categorias_Acumuladas'] = range(1, len(pareto_categorias) + 1)
        pareto_categorias['% Categorias'] = (pareto_categorias['Categorias_Acumuladas'] / len(pareto_categorias) * 100).round(2)
        
        # Gráfico de Pareto
        fig_pareto_cat = go.Figure()
        
        # Barras de ventas
        fig_pareto_cat.add_trace(go.Bar(
            x=pareto_categorias.index,
            y=pareto_categorias['% Ventas'],
            name='% Ventas',
            marker_color='lightblue'
        ))
        
        # Línea de acumulado
        fig_pareto_cat.add_trace(go.Scatter(
            x=pareto_categorias.index,
            y=pareto_categorias['Ventas_Acumuladas'],
            mode='lines+markers',
            name='% Ventas Acumulado',
            yaxis='y2',
            line=dict(color='red', width=3)
        ))
        
        fig_pareto_cat.update_layout(
            title="Análisis de Pareto - Concentración de Ventas por Categorías ABC",
            xaxis_title="Categorías ABC (ordenadas por ventas)",
            yaxis=dict(title="% Ventas", side="left"),
            yaxis2=dict(title="% Ventas Acumulado", side="right", overlaying="y"),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_pareto_cat, use_container_width=True)
        
        # Mostrar tabla de Pareto
        st.subheader("📋 Tabla de Análisis de Pareto")
        pareto_display = pareto_categorias[['% Ventas', 'Ventas_Acumuladas', '% Categorias']].copy()
        pareto_display.columns = ['% Ventas', '% Ventas Acumulado', '% Categorías']
        st.dataframe(pareto_display)
        
        # Insights automáticos
        st.subheader("💡 Insights Automáticos")
        
        # Categoría con mayor contribución
        top_categoria = pareto_categorias.index[0]
        top_ventas = pareto_categorias.iloc[0]['% Ventas']
        
        # Categorías que representan el 80% de las ventas
        categorias_80 = pareto_categorias[pareto_categorias['Ventas_Acumuladas'] <= 80]
        num_categorias_80 = len(categorias_80)
        
        # Categorías con baja contribución
        categorias_bajas = pareto_categorias[pareto_categorias['% Ventas'] < 5]
        
        insights = []
        insights.append(f"🎯 **Categoría líder**: {top_categoria} representa el {top_ventas}% de las ventas")
        insights.append(f"📊 **Concentración**: {num_categorias_80} categorías representan el 80% de las ventas")
        insights.append(f"📉 **Categorías de baja contribución**: {len(categorias_bajas)} categorías contribuyen menos del 5% cada una")
        
        if len(categorias_bajas) > 0:
            insights.append(f"🔍 **Categorías a revisar**: {', '.join(categorias_bajas.index)}")
        
        for insight in insights:
            st.info(insight)


# -------------------------------
# Generar PDF completo robusto y profesional (mejorado)
# -------------------------------
if gen_pdf:
    if not PDF_LIBS_AVAILABLE:
        st.error('Para generar PDFs instala: pip install reportlab matplotlib')
    else:
        if st.button('4) Generar informe PDF'):
            from reportlab.lib import colors
            from reportlab.platypus import TableStyle, Image, PageBreak
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import cm, mm
            from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
            import io, matplotlib.pyplot as plt

            # --- Pie de página con numeración
            def add_page_number(canvas, doc):
                page_num = canvas.getPageNumber()
                text = f"Página {page_num}"
                canvas.setFont('Helvetica', 8)
                canvas.drawRightString(200*mm, 10*mm, text)

            buffer = io.BytesIO()
            doc = BaseDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=25, leftMargin=25,
                topMargin=25, bottomMargin=18
            )
            frame = Frame(doc.leftMargin, doc.bottomMargin,
                          doc.width, doc.height, id='normal')
            template = PageTemplate(id='with-number',
                                    frames=frame,
                                    onPage=add_page_number)
            doc.addPageTemplates([template])

            styles = getSampleStyleSheet()
            elems = []

            # -------------------------------
            # Encabezado
            # -------------------------------
            elems.append(Paragraph('📊 Informe de Análisis - Súper ABC & Perfiles', styles['Title']))
            elems.append(Spacer(1, 14))

            # -------------------------------
            # Texto explicativo inicial
            # -------------------------------
            intro_text = """
            <b>Clasificación de zonas de bodega:</b><br/>
            - <b>Zona Oro (Close to door, close to floor):</b> Área de mayor valor, ubicada estratégicamente cerca de las puertas de entrada y salida de la bodega. Se destina a los productos de <b>alta rotación</b>, minimizando tiempo de viaje y esfuerzo de los operarios.<br/>
            - <b>Zona Plata (Close to floor):</b> Ubicada a una distancia media de las puertas. Se utiliza para productos de <b>rotación media</b>. El tiempo de acceso es moderado.<br/>
            - <b>Zona Bronce (Far from door, far from floor):</b> Área más alejada de las puertas. Reservada para productos de <b>baja rotación</b>. Aunque implica mayor tiempo de acceso, la baja frecuencia de movimiento lo justifica.<br/><br/>

            <b>Políticas de inventario:</b><br/>
            - <b>ROP-OUL:</b> Reordenar al alcanzar el punto de pedido (ROP), con un límite superior (OUL) para evitar exceso de inventario.<br/>
            - <b>RTP-EOQ:</b> Política de revisión periódica (RTP), aplicando el tamaño de lote económico (EOQ) como cantidad óptima de pedido.<br/>
            - <b>ROP-EOQ:</b> Política de reorden continuo (ROP), usando el EOQ como lote de reposición.<br/><br/>

            <b>Fill rate:</b> Métrica de nivel de servicio que mide el porcentaje de demanda atendida en el primer intento con el inventario disponible. Un fill rate alto indica capacidad de satisfacer pedidos sin generar faltantes.<br/><br/>

            <b>IRA (Inventory Record Accuracy):</b> KPI que mide la exactitud del inventario, comparando los registros teóricos del sistema con la realidad física del stock disponible en un almacén. Un IRA alto indica que la información del sistema es confiable, lo que permite una gestión de inventarios más eficiente, reduciendo pérdidas, excedentes y retrasos en los pedidos.  <br/><br/>

            <b>Recuento cíclico:</b> Estrategia de control de inventarios que consiste en revisar y contar de forma periódica subgrupos de productos a lo largo del año. Se enfoca más en artículos críticos o de mayor rotación (categoría A o AA), garantizando precisión de inventario sin necesidad de inventarios generales completos.
            """

            elems.append(Paragraph(intro_text, styles['Normal']))
            elems.append(Spacer(1, 14))

            # -------------------------------
            # Datos generales
            # -------------------------------
            file_name = st.session_state.get('file_name', uploaded_file.name if uploaded_file else 'Archivo no registrado')
            sheet_used = st.session_state.get('sheet_name', sheet_name or 'Hoja no registrada')
            vol_units = st.session_state.get('vol_units', unit_vol)
            criterios_usados = st.session_state.get('criterios_seleccionados', [crit1, crit2])
            cortes_abc = st.session_state.get('cortes_abc', {})
            
            # Para compatibilidad con código existente
            crit1 = criterios_usados[0] if criterios_usados else 'Popularidad'
            crit2 = criterios_usados[1] if len(criterios_usados) > 1 else criterios_usados[0] if criterios_usados else 'Ventas'
            
            # Obtener cortes de la nueva estructura
            A_cut_1 = cortes_abc.get(crit1, {}).get('A', 0.8) if cortes_abc else 0.8
            B_cut_1 = cortes_abc.get(crit1, {}).get('B', 0.95) if cortes_abc else 0.95
            A_cut_2 = cortes_abc.get(crit2, {}).get('A', 0.8) if cortes_abc else 0.8
            B_cut_2 = cortes_abc.get(crit2, {}).get('B', 0.95) if cortes_abc else 0.95

            general_info = f"""
            <b>Documento leído:</b> {file_name}<br/>
            <b>Hoja utilizada:</b> {sheet_used}<br/>
            <b>Unidades de volumen:</b> {vol_units}<br/>
            <b>Criterios utilizados:</b> {', '.join(criterios_usados)}<br/>
            """
            
            # Agregar cortes para cada criterio
            for criterio in criterios_usados:
                if criterio in cortes_abc:
                    general_info += f"<b>Corte A ({criterio}):</b> {cortes_abc[criterio]['A']*100:.1f}%<br/>"
                    general_info += f"<b>Corte B ({criterio}):</b> {cortes_abc[criterio]['B']*100:.1f}%<br/>"
            elems.append(Paragraph(general_info, styles['Normal']))
            elems.append(Spacer(1, 12))

            by_item = st.session_state['by_item']
            base = st.session_state['base']

            # -------------------------------
            # Tabla resumen Super ABC (columnas compactas)
            # -------------------------------
            summary_table = by_item.groupby('Clase_SuperABC').agg(
                Cantidad=('Clase_SuperABC','count'),
                Zona_Bodega=('Zona_Bodega','first'),
                Politica=('Política_Inv','first'),
                FillRate=('FillRate_obj','first'),
                Frecuencia_Recuento=('Frecuencia_Recuento','first'),
                Ventas=('ventas','sum')
            ).reset_index()

            summary_table['Porcentaje'] = (summary_table['Cantidad']/summary_table['Cantidad'].sum()*100).round(2)
            total_sales = summary_table['Ventas'].sum()
            summary_table['% Ventas'] = (100 * summary_table['Ventas'] / (total_sales if total_sales>0 else 1)).round(2)
            summary_table['Ventas'] = summary_table['Ventas'].round(2)

            # 👉 Definir IRA según categoría usando la nueva función
            summary_table['IRA'] = summary_table['Clase_SuperABC'].apply(ira_by_class)

            # Reordenar columnas para poner IRA después de FillRate
            cols = list(summary_table.columns)
            insert_pos = cols.index('FillRate') + 1
            cols = cols[:insert_pos] + ['IRA'] + cols[insert_pos:-1]  # dejamos % Ventas al final
            summary_table = summary_table[cols]

            # preparar datos y anchos
            data = [list(summary_table.columns)] + summary_table.round(2).astype(str).values.tolist()
            col_widths = []
            for col in summary_table.columns:
                if col in ['Cantidad','Zona_Bodega','FillRate','IRA']:
                    col_widths.append(45)
                elif col in ['Ventas','Porcentaje','% Ventas']:
                    col_widths.append(50)
                elif col in ['Clase_SuperABC','Frecuencia_Recuento']:
                    col_widths.append(70)
                else:
                    col_widths.append(97)

            t = Table(data, colWidths=col_widths, hAlign='CENTER')
            t.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('ALIGN',(0,0),(-1,-1),'CENTER')
            ]))
            elems.append(Paragraph('📑 Resumen por categoría (AA..CC)', styles['Heading2']))
            elems.append(t)
            elems.append(PageBreak())

            # -------------------------------
            # Función auxiliar para añadir figuras
            # -------------------------------
            def add_fig(fig, title='', width=450, height=240):
                img_buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(img_buf, format='png', dpi=130)
                plt.close(fig)
                img_buf.seek(0)
                elems.append(Paragraph(title, styles['Heading3']))
                elems.append(Image(img_buf, width=width, height=height))
                elems.append(Spacer(1, 12))
            # -------------------------------
            # Gráfica Pareto
            # -------------------------------

            pareto = by_item.sort_values('popularidad', ascending=False).copy()
            pareto['cum_picks'] = pareto['popularidad'].cumsum()
            total_picks = pareto['popularidad'].sum()
            pareto['cum_pct_picks'] = 100*pareto['cum_picks']/(total_picks if total_picks>0 else 1)
            pareto['pct_sku'] = 100 * np.arange(1,len(pareto)+1)/len(pareto)
            fig1, ax1 = plt.subplots(figsize=(6,3))
            ax1.plot(pareto['pct_sku'], pareto['cum_pct_picks'], marker='o')
            ax1.set_xlabel('% SKU (acumulado)')
            ax1.set_ylabel('% picks (acumulado)')
            ax1.set_title('Distribución de popularidad')
            add_fig(fig1, 'Pareto de popularidad')
                        
            pareto_intro = """
            Este perfil muestra qué porcentaje acumulado de los movimientos de picking corresponde a qué porcentaje acumulado de SKUs según el principio de Pareto (muchos triviales, pocos vitales). 
            Permite identificar los productos que concentran la mayor parte de la actividad y que deben recibir prioridad en la bodega.
            """
            elems.append(Paragraph(pareto_intro, styles['Normal']))
            elems.append(Spacer(1, 6))

            elems.append(PageBreak())

            # -------------------------------
            # Líneas por orden
            # -------------------------------
            lines_per_order = base.groupby('NumDoc').agg(lineas=('Articulo','nunique')).reset_index()
            dist_lines = lines_per_order.groupby('lineas').size().rename('conteo').reset_index()
            total_orders = dist_lines['conteo'].sum()
            dist_lines['%_ordenes'] = 100*dist_lines['conteo']/(total_orders if total_orders>0 else 1)
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.bar(dist_lines['lineas'].astype(str), dist_lines['%_ordenes'])
            ax2.set_xlabel('Líneas por orden')
            ax2.set_ylabel('% de órdenes')
            ax2.set_title('Distribución de líneas por orden')
            add_fig(fig2, 'Líneas por orden')
            
            lines_intro = """
            Este perfil muestra cuántas líneas (SKUs distintos) tiene cada pedido y qué porcentaje de órdenes corresponde a cada cantidad de líneas. 
            Permite evaluar la complejidad de los pedidos y planificar recursos de picking y personal.
            """
            elems.append(Paragraph(lines_intro, styles['Normal']))
            elems.append(Spacer(1, 6))
            elems.append(PageBreak())

            # -------------------------------
            # Cubicaje por orden
            # -------------------------------

            cubic_per_order = base.groupby('NumDoc').agg(volumen_total=('Volumen_p3','sum')).reset_index()
            vol_bins = [-1,1,2,5,10,20,50,1e9]
            vol_labels = ['≤1','1-2','2-5','5-10','10-20','20-50','>50']
            cubic_per_order['vol_bin'] = pd.cut(cubic_per_order['volumen_total'], bins=vol_bins, labels=vol_labels)
            dist_cubic = cubic_per_order.groupby('vol_bin').size().rename('conteo').reset_index()
            total_orders2 = dist_cubic['conteo'].sum()
            dist_cubic['%_ordenes'] = 100*dist_cubic['conteo']/(total_orders2 if total_orders2>0 else 1)
            fig3, ax3 = plt.subplots(figsize=(6,3))
            ax3.bar(dist_cubic['vol_bin'].astype(str), dist_cubic['%_ordenes'])
            ax3.set_xlabel('Rango volumen (pies³)')
            ax3.set_ylabel('% de órdenes')
            ax3.set_title('Distribución de volumen por orden')
            add_fig(fig3, 'Volumen por orden')

            cubic_intro = """
            El presente perfil ilustra mediante una gráfica el rango de volumen total de los pedidos y su porcentaje sobre el total de órdenes. 
            Es útil para dimensionar espacio de almacenamiento, cajas, pallets y vehículos de transporte, según requerimientos de espacio y rotación.
            """
            elems.append(Paragraph(cubic_intro, styles['Normal']))
            elems.append(Spacer(1, 6))
            elems.append(PageBreak())

            # Recalcular lv y dist_incremento para el PDF
            lv = base.groupby('NumDoc').agg(
                lineas=('Articulo','nunique'),
                volumen_total=('Volumen_p3','sum')
            ).reset_index()

            VOLUMEN_TARIMA = st.session_state.get('vol_tarima', 42.38)
            lv['%_carga_unidad'] = 100 * lv['volumen_total'] / VOLUMEN_TARIMA
            lv['%_carga_unidad'] = lv['%_carga_unidad'].clip(upper=100)
            carga_bins = list(range(0, 105, 5))
            carga_labels = [f'{i}-{i+5}%' for i in range(0, 100, 5)]
            lv['r_carga'] = pd.cut(lv['%_carga_unidad'], bins=carga_bins, labels=carga_labels, right=True, include_lowest=True)
            dist_incremento = lv.groupby(['r_carga']).agg(
                pedidos=('NumDoc', 'count'),
                lineas_prom=('lineas', 'mean')
            ).reset_index()
            dist_incremento['%_lineas_pedido'] = 100 * dist_incremento['pedidos'] / dist_incremento['pedidos'].sum()

            # Gráfica de incremento de pedidos (carga unitaria vs % líneas de pedido)
            fig_inc, ax_inc = plt.subplots(figsize=(6,3))
            ax_inc.bar(dist_incremento['r_carga'].astype(str), dist_incremento['%_lineas_pedido'])
            ax_inc.set_xlabel('% de carga unitaria (tarima)')
            ax_inc.set_ylabel('% de líneas de pedido')
            ax_inc.set_title('Distribución por incremento de pedidos')
            plt.setp(ax_inc.get_xticklabels(), rotation=60, ha='right', fontsize=7)  # Rota y reduce fuente
            add_fig(fig_inc, 'Distribución por incremento de pedidos')

            inc_intro = """
            Esta gráfica muestra la proporción de líneas de pedido según el porcentaje de carga unitaria (por ejemplo, respecto a una tarima completa).
            Permite visualizar cuántos pedidos representan cargas parciales o completas, facilitando la planificación logística y el uso eficiente de espacio.
            """
            elems.append(Paragraph(inc_intro, styles['Normal']))
            elems.append(Spacer(1, 6))
            elems.append(PageBreak())

            # -------------------------------
            # Distribución por día de la semana
            # -------------------------------

            orders_dates = base.groupby('NumDoc').agg(fecha=('Fecha','max')).reset_index()
            orders_dates['dia'] = orders_dates['fecha'].dt.day_name()
            mapping_days = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miércoles','Thursday':'Jueves',
                            'Friday':'Viernes','Saturday':'Sábado','Sunday':'Domingo'}
            orders_dates['dia'] = orders_dates['dia'].replace(mapping_days)
            day_order = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
            dist_days = orders_dates.groupby('dia').size().reindex(day_order).fillna(0).astype(int).rename('conteo').reset_index()
            dist_days['%_ordenes'] = 100*dist_days['conteo']/dist_days['conteo'].sum()
            fig4, ax4 = plt.subplots(figsize=(6,3))
            ax4.bar(dist_days['dia'], dist_days['%_ordenes'])
            ax4.set_xlabel('Día')
            ax4.set_ylabel('% de órdenes')
            ax4.set_title('Distribución de órdenes por día de la semana')
            add_fig(fig4, 'Órdenes por día de la semana')

            days_intro = """
            Este muestra cómo se distribuyen los pedidos a lo largo de la semana y su porcentaje sobre el total. 
            Permite planificar personal, turnos y recursos logísticos en función de los picos y valles de demanda, identificando qué días presentan mayor ingreso de órdenes.
            """
            elems.append(Paragraph(days_intro, styles['Normal']))
            elems.append(PageBreak())

            # -------------------------------
            # Tabla cruzada líneas x volumen con % pedidos, Totales y Total Línea
            # -------------------------------

            lv = base.groupby('NumDoc').agg(
                lineas=('Articulo','nunique'),
                volumen_total=('Volumen_p3','sum')
            ).reset_index()

            # Definir rangos (misma lógica que en Streamlit)
            line_labels = ['1','2-5','6-9','10+']
            vol_labels2 = ['0-1','1-2','2-5','5-10','10-20','20+']

            # Categorizar (igual que en la app)
            lv['r_lineas'] = pd.cut(lv['lineas'], bins=[0,1,5,9,1e9], labels=line_labels, right=True, include_lowest=True)
            lv['r_vol'] = pd.cut(lv['volumen_total'], bins=[0,1,2,5,10,20,1e9], labels=vol_labels2, right=True, include_lowest=True)

            # Conteos y totales
            ct_counts = pd.crosstab(lv['r_lineas'], lv['r_vol'], dropna=False)
            ct_counts = ct_counts.reindex(index=line_labels, columns=vol_labels2, fill_value=0)
            ct_counts['Totales'] = ct_counts.sum(axis=1)

            # 🔹 Total de líneas (sumando líneas, no volumen)
            pivot_lines = pd.pivot_table(
                lv, index='r_lineas',
                values='lineas', aggfunc='sum', fill_value=0
            ).reindex(index=line_labels, fill_value=0)
            pivot_lines['% linea'] = (pivot_lines['lineas'] / pivot_lines['lineas'].sum() * 100).round(2)

            # Volumen total (solo para fila de "Espacio total")
            pivot_vol = pd.pivot_table(
                lv, index='r_lineas', columns='r_vol',
                values='volumen_total', aggfunc='sum', fill_value=0
            ).reindex(index=line_labels, columns=vol_labels2, fill_value=0).round(2)

            # Construir tabla combinada
            data_cross = []

            # Encabezado combinado
            data_cross.append(
                ['Líneas por orden'] 
                + ['Volumen por pedido (pies³)']*len(vol_labels2) 
                + ['Totales','% pedidos','Total Línea','% línea']
            )
            data_cross.append(
                [''] + vol_labels2 + ['Totales','% pedidos','Total Línea','% línea']
            )

            # Filas por r_lineas
            for idx in line_labels:
                row_counts = ct_counts.loc[idx, vol_labels2].tolist()
                row_total = ct_counts.loc[idx, 'Totales']
                row_pct_pedidos = (row_total / ct_counts['Totales'].sum() * 100).round(2)
                row_total_linea = int(pivot_lines.loc[idx, 'lineas'])  # 🔹 ahora es la suma de líneas
                row_pct_linea = float(pivot_lines.loc[idx, '% linea'])
                data_cross.append([idx] + row_counts + [row_total, row_pct_pedidos, row_total_linea, row_pct_linea])

            # 👉 Fila de Totales
            tot_row_counts = ct_counts[vol_labels2].sum().tolist()
            tot_total = ct_counts['Totales'].sum()
            tot_pct_pedidos = 100.0
            tot_total_linea = int(pivot_lines['lineas'].sum())  # 🔹 total líneas global
            tot_pct_linea = 100.0
            data_cross.append(['Totales'] + tot_row_counts + [tot_total, tot_pct_pedidos, tot_total_linea, tot_pct_linea])

            # Fila de % pedidos (por columna de volumen + total)
            pct_pedidos_cols = (ct_counts[vol_labels2].sum() / ct_counts['Totales'].sum() * 100).round(2).tolist()
            pct_pedidos_total = round(sum(pct_pedidos_cols), 2)
            row_pct_pedidos = ['% pedidos'] + pct_pedidos_cols + [pct_pedidos_total, '', '', '']
            data_cross.append(row_pct_pedidos)

            # Fila de volumen total por columna
            vol_values = pivot_vol[vol_labels2].sum().round(2).tolist()
            row_vol_total = ['Espacio total'] + vol_values + [pivot_vol.values.sum().round(2), '', '', '']
            data_cross.append(row_vol_total)

            # Configurar tabla PDF
            col_widths_cross = [50] + [50]*len(vol_labels2) + [50,50,50,50]
            t_cross = Table(data_cross, colWidths=col_widths_cross, hAlign='CENTER')
            t_cross.setStyle(TableStyle([
                ('SPAN',(1,0),(len(vol_labels2),0)),  # unir fila 0 columnas de volumen
                ('SPAN',(len(vol_labels2)+1,0),(len(vol_labels2)+1,1)),  # Totales
                ('SPAN',(len(vol_labels2)+2,0),(len(vol_labels2)+2,1)),  # % pedidos
                ('SPAN',(len(vol_labels2)+3,0),(len(vol_labels2)+3,1)),  # Total Línea
                ('SPAN',(len(vol_labels2)+4,0),(len(vol_labels2)+4,1)),  # % línea
                ('GRID',(0,0),(-1,-1),0.5,colors.black),
                ('BACKGROUND',(0,0),(-1,1),colors.lightgrey),
                ('BACKGROUND',(0,-3),(-1,-3),colors.lightgrey),  # Totales fila
                ('BACKGROUND',(0,-2),(-1,-2),colors.whitesmoke),  # % pedidos
                ('BACKGROUND',(0,-1),(-1,-1),colors.whitesmoke),  # espacio total
                ('FONTSIZE',(0,0),(-1,-1),6),
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE')
            ]))
            elems.append(Paragraph('Tabla cruzada: líneas por orden vs volumen', styles['Heading2']))
            cross_intro = """
            Permite ver cuántos pedidos combinan cierta cantidad de líneas con un rango de volumen determinado, 
            junto con totales, porcentaje de pedidos y porcentaje de líneas. 
            Esto ayuda a identificar combinaciones de pedidos frecuentes o críticas y optimizar la disposición de la bodega y flujos de picking.
            """
            elems.append(Paragraph(cross_intro, styles['Normal']))
            elems.append(Spacer(1, 6))
            elems.append(t_cross)
            elems.append(Spacer(1, 10))


            # -------------------------------
            # Construir PDF
            # -------------------------------
            doc.build(elems)
            buffer.seek(0)
            st.download_button(
                '📄 Descargar Informe PDF',
                data=buffer.getvalue(),
                file_name='informe_super_abc_completo.pdf',
                mime='application/pdf'
            )

st.success('Listo. Ajusta cortes y vuelve a calcular según necesites.')
