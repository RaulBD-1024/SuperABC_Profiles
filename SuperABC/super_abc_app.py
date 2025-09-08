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
def read_excel_bytes(file_bytes: bytes, sheet_name=None) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine='openpyxl')

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

# Map zone from combined class

def map_zone(clase: str) -> str:
    if clase in {'AA','AB','AC'}:
        return 'Oro'
    if clase in {'BA','BB','BC'}:
        return 'Plata'
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
   - **IRA (Inventory Record Accuracy)** según la clase  
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
""")

with st.sidebar:
    st.header('1) Cargar datos')
    uploaded_file = st.file_uploader('Excel de ventas/ordenes', type=['xlsx','xls'])
    sheet_name = st.text_input('Hoja (opcional)')
    unit_vol = st.selectbox('Unidad de volumen', ['pies3 (p3)','metros3 (m3)'])
    vol_factor = 35.3147 if unit_vol == 'metros3 (m3)' else 1.0

    st.header('2) Criterios ABC (elige dos)')
    criterios = {
        'Popularidad': 'popularidad',
        'Rotacion': 'rotacion_sem',
        'Ventas': 'ventas',
        'Volumen': 'volumen'
    }
    crit1 = st.selectbox('Criterio 1', list(criterios.keys()), index=0)
    crit2 = st.selectbox('Criterio 2', [c for c in criterios.keys() if c != crit1], index=0)

    st.header('3) Cortes ABC por contribucion (A, B)')
    A_cut_1 = st.slider(f'A (criterio {crit1})', 50, 95, 80) / 100.0
    B_cut_1 = st.slider(f'B (criterio {crit1})', int(A_cut_1*100)+1, 99, 95) / 100.0
    A_cut_2 = st.slider(f'A (criterio {crit2})', 50, 95, 80) / 100.0
    B_cut_2 = st.slider(f'B (criterio {crit2})', int(A_cut_2*100)+1, 99, 95) / 100.0

    st.session_state['A_cut_1'] = A_cut_1
    st.session_state['B_cut_1'] = B_cut_1
    st.session_state['A_cut_2'] = A_cut_2
    st.session_state['B_cut_2'] = B_cut_2

    st.header('4) Exportar')
    want_csv = st.checkbox('Permitir descarga CSV', True)
    gen_pdf = st.checkbox('Generar informe PDF (requiere reportlab & matplotlib)', False)

if uploaded_file is None:
    st.info('Sube un Excel para comenzar')
    st.stop()

# -------------------------------
# Leer datos
# -------------------------------
try:
    df = read_excel_bytes(uploaded_file.read(), sheet_name=sheet_name or None)
except Exception as e:
    st.error(f'Error leyendo Excel: {e}')
    st.stop()

# map columns tolerant
try:
    art = safe_col(df, 'Artículo').astype(str)
    unid = pd.to_numeric(safe_col(df, 'Unid. Vend'), errors='coerce').fillna(0)
    monto = pd.to_numeric(safe_col(df, 'Monto venta'), errors='coerce').fillna(0)
    vol = pd.to_numeric(safe_col(df, 'Volumen total (p3)', alt_names=['Volumen total (m3)', 'Volumen total']), errors='coerce').fillna(0) * vol_factor
    numdoc = safe_col(df, 'Num. Doc').astype(str)
    fecha = pd.to_datetime(safe_col(df, 'Fecha Doc'), errors='coerce')
except Exception as e:
    st.error(f'Error mapeando columnas: {e}')
    st.stop()

base = pd.DataFrame({
    'Articulo': art,
    'Unidades': unid,
    'Monto': monto,
    'Volumen_p3': vol,
    'NumDoc': numdoc,
    'Fecha': fecha
}).dropna(subset=['Fecha'])

# Guardar base en session_state para usarlo en PDF y perfiles
st.session_state['base'] = base

if len(base) == 0:
    st.error('No hay registros con fecha valida')
    st.stop()

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

    key1 = criterios[crit1]
    key2 = criterios[crit2]

    by_item['ABC_1'] = abc_by_contribution(by_item[key1], A_cut_1, B_cut_1)
    by_item['ABC_2'] = abc_by_contribution(by_item[key2], A_cut_2, B_cut_2)
    by_item['Clase_SuperABC'] = by_item['ABC_1'].astype(str) + by_item['ABC_2'].astype(str)

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
    st.session_state['crit1_name'] = crit1
    st.session_state['crit2_name'] = crit2
    st.success('Súper ABC calculado correctamente 🎯')

# -------------------------------
# Mostrar resumen y perfiles
# -------------------------------
def ira_by_class(clase: str) -> str:
    mapping = {
        'AA': '> 95%',
        'AB': '94% - 95%',
        'AC': '92% - 94%',
        'BA': '90% - 92%',
        'BB': '88% - 90%',
        'BC': '86% - 88%',
        'CA': '84% - 86%',
        'CB': '82% - 84%',
        'CC': '< 80%'
    }
    return mapping.get(clase, 'N/A')

if 'by_item' in st.session_state:
    by_item = st.session_state['by_item']

    if st.button('2) Mostrar tabla resumen y perfiles'):
        st.subheader('📋 Resumen por categoría (AA..CC)')
        summary = by_item.groupby('Clase_SuperABC').agg(
            Cantidad=('ABC_1','count'),
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

        # Ordenar categorías
        order = [a+b for a in 'ABC' for b in 'ABC']
        summary['_ord'] = summary['Clase_SuperABC'].apply(lambda x: order.index(x) if x in order else 999)
        summary = summary.sort_values('_ord').drop(columns=['_ord'])

        # Reordenar columnas para que IRA quede después de FillRate
        cols = ['Clase_SuperABC','Cantidad','Zona_Bodega','Politica','FillRate','IRA',
                'Frecuencia_Recuento','Ventas','Porcentaje','% Ventas']
        summary = summary[cols]

        st.dataframe(summary)

        # Perfil: lineas por orden (distribucion %)
        st.subheader('% de órdenes por # líneas')
        lines_per_order = base.groupby('NumDoc').agg(lineas=('Articulo','nunique')).reset_index()
        dist_lines = lines_per_order.groupby('lineas').size().rename('conteo').reset_index()
        total_orders = dist_lines['conteo'].sum()
        dist_lines['%_ordenes'] = 100 * dist_lines['conteo']/ (total_orders if total_orders>0 else 1)
        st.dataframe(dist_lines.sort_values('lineas'))
        fig_lines = px.bar(dist_lines.sort_values('lineas'), x='lineas', y='%_ordenes', labels={'lineas':'Líneas por orden','%_ordenes':'% de órdenes'})
        st.plotly_chart(fig_lines, use_container_width=True)

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

        # Distribucion por dia de la semana
        st.subheader('Distribución de órdenes por día de la semana')
        orders_dates = base.groupby('NumDoc').agg(fecha=('Fecha','max')).reset_index()
        # day_name may produce English names depending on locale; map to Spanish
        orders_dates['dia'] = orders_dates['fecha'].dt.day_name()
        mapping_days = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miércoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sábado','Sunday':'Domingo'}
        orders_dates['dia'] = orders_dates['dia'].replace(mapping_days)
        day_order = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
        dist_days = orders_dates.groupby('dia').size().reindex(day_order).fillna(0).astype(int).rename('conteo').reset_index()
        dist_days['%_ordenes'] = 100 * dist_days['conteo'] / (dist_days['conteo'].sum() if dist_days['conteo'].sum()>0 else 1)
        st.dataframe(dist_days)
        fig_days = px.bar(dist_days, x='dia', y='%_ordenes', labels={'dia':'Día','%_ordenes':'% de órdenes'})
        st.plotly_chart(fig_days, use_container_width=True)

        # -------------------------------
        # Tabla cruzada líneas x volumen con columnas reorganizadas y correcciones de Totales
        # -------------------------------
        st.subheader('Tabla cruzada: Líneas por pedido vs pies³ por pedido')

        # Preparar datos
        lv = base.groupby('NumDoc').agg(
            lineas=('Articulo','nunique'),
            volumen_total=('Volumen_p3','sum')
        ).reset_index()

        # Categorías
        line_labels = ['1','2-5','6-9','10+']
        lv['r_lineas'] = pd.cut(lv['lineas'], bins=[0,1,6,10,1e9], labels=line_labels, right=False)

        vol_labels = ['0-1','1-2','2-5','5-10','10-20','20+']
        lv['r_vol'] = pd.cut(lv['volumen_total'], bins=[0,1,2,5,10,20,1e9], labels=vol_labels, right=False)

        # Conteo de pedidos por línea y volumen
        ct_counts = pd.crosstab(lv['r_lineas'], lv['r_vol'], dropna=False)

        # Totales de línea y % pedidos por línea
        ct_counts['Totales'] = ct_counts.sum(axis=1)
        ct_counts['% pedidos'] = (ct_counts['Totales'] / ct_counts['Totales'].sum() * 100).round(2)

        # Volumen total por línea y % línea
        pivot_vol = pd.pivot_table(
            lv,
            index='r_lineas',
            columns='r_vol',
            values='volumen_total',
            aggfunc='sum',
            fill_value=0
        )
        pivot_vol['Total_Linea'] = pivot_vol.sum(axis=1)
        total_vol_global = pivot_vol['Total_Linea'].sum()
        pivot_vol['% linea'] = (pivot_vol['Total_Linea'] / (total_vol_global if total_vol_global>0 else 1) * 100).round(2)

        # Crear tabla final con columnas en orden deseado
        table_final = pd.DataFrame(index=line_labels, columns=vol_labels + ['Totales','% pedidos','Total_Linea','% linea'])
        table_final[vol_labels] = ct_counts[vol_labels]
        table_final['Totales'] = ct_counts['Totales']
        table_final['% pedidos'] = ct_counts['% pedidos']
        table_final['Total_Linea'] = pivot_vol['Total_Linea']
        table_final['% linea'] = pivot_vol['% linea']

        # Fila Totales
        totales_row = table_final[vol_labels].sum()
        totales_row['Totales'] = table_final['Totales'].sum()
        totales_row['% pedidos'] = 100
        totales_row['Total_Linea'] = table_final['Total_Linea'].sum()
        totales_row['% linea'] = 100
        table_final.loc['Totales'] = totales_row

        # Fila % pedidos (por columna)
        pct_pedidos_row = (table_final[vol_labels].sum() / table_final['Totales'].sum() * 100).round(2)
        pct_pedidos_row['Totales'] = 100
        pct_pedidos_row['% pedidos'] = np.nan
        pct_pedidos_row['Total_Linea'] = np.nan
        pct_pedidos_row['% linea'] = np.nan
        table_final.loc['% pedidos'] = pct_pedidos_row

        # Fila Espacio total (volumen)
        espacio_total_row = pivot_vol[vol_labels].sum()
        espacio_total_row['Totales'] = espacio_total_row.sum()  # ahora sí suma propia de la fila
        espacio_total_row['% pedidos'] = np.nan
        espacio_total_row['Total_Linea'] = np.nan       # No calcular para espacio total
        espacio_total_row['% linea'] = np.nan           # No calcular para espacio total
        table_final.loc['Espacio total'] = espacio_total_row

        # Renombrar índice
        table_final.index.name = 'Líneas por orden / Volumen por orden'

        # Mostrar tabla
        st.dataframe(table_final.round(2))

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

    # -------------------------------
    # Exportar CSV (sanitizado)
    # -------------------------------
    if want_csv:
        if st.button('3) Exportar CSV'):
            export_df = by_item.reset_index().copy()
            # Normalizar nombres y datos
            export_df.columns = [unicodedata.normalize('NFKD', str(c)).encode('ascii','ignore').decode('ascii') for c in export_df.columns]
            # Asegurar FillRate con guion ASCII
            if 'FillRate_obj' in export_df.columns:
                export_df['FillRate_obj'] = export_df['FillRate_obj'].astype(str).str.replace('–','-', regex=False).str.replace('—','-', regex=False)
            export_df = sanitize_colnames(export_df)

            # Generar nombre de archivo y mantener extensión .csv
            raw_name = f"super_abc_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            fname = sanitize_filename(raw_name) + ".csv"

            st.download_button(
                '📥 Descargar CSV (sanitizado)',
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=fname,
                mime='text/csv'
            )
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
            crit1 = st.session_state.get('crit1_name', crit1)
            crit2 = st.session_state.get('crit2_name', crit2)
            A_cut_1 = st.session_state['A_cut_1']
            B_cut_1 = st.session_state['B_cut_1']
            A_cut_2 = st.session_state['A_cut_2']
            B_cut_2 = st.session_state['B_cut_2']

            general_info = f"""
            <b>Documento leído:</b> {file_name}<br/>
            <b>Hoja utilizada:</b> {sheet_used}<br/>
            <b>Unidades de volumen:</b> {vol_units}<br/>
            <b>Criterio principal:</b> {crit1}<br/>
            <b>Criterio secundario:</b> {crit2}<br/>
            <b>Corte A ({st.session_state['crit1_name']}):</b> {A_cut_1*100:.1f}%<br/>
            <b>Corte B ({st.session_state['crit1_name']}):</b> {B_cut_1*100:.1f}%<br/>
            <b>Corte A ({st.session_state['crit2_name']}):</b> {A_cut_2*100:.1f}%<br/>
            <b>Corte B ({st.session_state['crit2_name']}):</b> {B_cut_2*100:.1f}%<br/>
            """
            elems.append(Paragraph(general_info, styles['Normal']))
            elems.append(Spacer(1, 12))

            by_item = st.session_state['by_item']
            base = st.session_state['base']

            # -------------------------------
            # Tabla resumen Super ABC (columnas compactas)
            # -------------------------------
            summary_table = by_item.groupby('Clase_SuperABC').agg(
                Cantidad=('ABC_1','count'),
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

            # 👉 Definir IRA según categoría
            ira_map = {
                'AA': '> 95%',
                'AB': '94% - 95%',
                'AC': '92% - 94%',
                'BA': '90% - 92%',
                'BB': '88% - 90%',
                'BC': '86% - 88%',
                'CA': '84% - 86%',
                'CB': '82% - 84%',
                'CC': '< 80%'
            }
            summary_table['IRA'] = summary_table['Clase_SuperABC'].map(ira_map)

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

            # Tabla cruzada líneas x volumen con % pedidos, Totales y Total Línea
            # -------------------------------

            lv = base.groupby('NumDoc').agg(
                lineas=('Articulo','nunique'),
                volumen_total=('Volumen_p3','sum')
            ).reset_index()

            # Definir rangos
            line_labels = ['1','2-5','6-9','10+']
            vol_labels2 = ['0-1','1-2','2-5','5-10','10-20','20+']

            # Categorizar
            lv['r_lineas'] = pd.cut(lv['lineas'], bins=[0,1,6,10,1e9], labels=line_labels, right=False)
            lv['r_vol'] = pd.cut(lv['volumen_total'], bins=[0,1,2,5,10,20,1e9], labels=vol_labels2, right=False)

            # Conteos y totales
            ct_counts = pd.crosstab(lv['r_lineas'], lv['r_vol'], dropna=False)
            ct_counts = ct_counts.reindex(index=line_labels, columns=vol_labels2, fill_value=0)
            ct_counts['Totales'] = ct_counts.sum(axis=1)

            # Volumen total por r_lineas x r_vol
            pivot_vol = pd.pivot_table(
                lv, index='r_lineas', columns='r_vol',
                values='volumen_total', aggfunc='sum', fill_value=0
            ).reindex(index=line_labels, columns=vol_labels2, fill_value=0).round(2)
            pivot_vol['Total linea'] = pivot_vol.sum(axis=1)
            pivot_vol['% linea'] = (pivot_vol['Total linea'] / pivot_vol['Total linea'].sum() * 100).round(2)

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
                row_total_linea = pivot_vol.loc[idx, 'Total linea']
                row_pct_linea = pivot_vol.loc[idx, '% linea']
                data_cross.append([idx] + row_counts + [row_total, row_pct_pedidos, row_total_linea, row_pct_linea])

            # 👉 Fila de Totales (justo debajo de 10+)
            tot_row_counts = ct_counts[vol_labels2].sum().tolist()
            tot_total = ct_counts['Totales'].sum()
            tot_pct_pedidos = 100.0  # siempre 100%
            tot_total_linea = pivot_vol['Total linea'].sum().round(2)
            tot_pct_linea = 100.0  # siempre 100%
            data_cross.append(['Totales'] + tot_row_counts + [tot_total, tot_pct_pedidos, tot_total_linea, tot_pct_linea])

            # Fila de % pedidos (por columna de volumen + total)
            pct_pedidos_cols = (ct_counts[vol_labels2].sum() / ct_counts['Totales'].sum() * 100).round(2).tolist()
            pct_pedidos_total = round(sum(pct_pedidos_cols), 2)  # debe cerrar en 100
            row_pct_pedidos = ['% pedidos'] + pct_pedidos_cols + [pct_pedidos_total, '', '', '']
            data_cross.append(row_pct_pedidos)

            # Fila de volumen total por columna
            vol_values = pivot_vol[vol_labels2].sum().round(2).tolist()
            row_vol_total = ['Espacio total'] + vol_values + [pivot_vol['Total linea'].sum().round(2), '', '', '']
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
            junto con totales, porcentaje de pedidos y porcentaje de volumen por línea. 
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

