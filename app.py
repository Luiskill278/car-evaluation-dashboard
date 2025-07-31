import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# =====================
# Dashboard de Evaluación de Autos - UCI Dataset
# =====================


# Cargar el CSV limpio
df = pd.read_csv("car_data_limpio.csv")


# Configuración de la página
st.set_page_config(layout="wide")
st.title("🚗 Dashboard de Evaluación de Autos")


# Limpieza e interpolación (si hay valores faltantes)
if df.isnull().values.any():
    st.warning("⚠️ Datos faltantes detectados. Aplicando interpolación...")
    df.interpolate(method='linear', inplace=True)


# =====================
# Barra lateral con filtros
# =====================
st.sidebar.header("🎛️ Filtros")

precio_compra = st.sidebar.multiselect("Precio de Compra", sorted(df["Precio_Compra"].unique()), default=df["Precio_Compra"].unique())
costo_mant = st.sidebar.multiselect("Costo de Mantenimiento", sorted(df["Costo_Mantenimiento"].unique()), default=df["Costo_Mantenimiento"].unique())
capacidad = st.sidebar.multiselect("Capacidad de Personas", sorted(df["Capacidad_Personas"].unique()), default=df["Capacidad_Personas"].unique())
seguridad = st.sidebar.multiselect("Seguridad", sorted(df["Seguridad"].unique()), default=df["Seguridad"].unique())
num_puertas = st.sidebar.multiselect("Número de Puertas", sorted(df["Num_Puertas"].unique()), default=df["Num_Puertas"].unique())
tamano_maletero = st.sidebar.multiselect("Tamaño del Maletero", sorted(df["Tamano_Maletero"].unique()), default=df["Tamano_Maletero"].unique())

# Filtro adicional por tipo de aceptación
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtrar por tipo de aceptación**")
aceptaciones = sorted(df["Aceptacion"].unique())
aceptacion_sel = st.sidebar.multiselect("Aceptación", aceptaciones, default=aceptaciones)


# =====================
# Aplicar filtros
# =====================
df_filtrado = df[
    (df["Precio_Compra"].isin(precio_compra)) &
    (df["Costo_Mantenimiento"].isin(costo_mant)) &
    (df["Capacidad_Personas"].isin(capacidad)) &
    (df["Seguridad"].isin(seguridad)) &
    (df["Num_Puertas"].isin(num_puertas)) &
    (df["Tamano_Maletero"].isin(tamano_maletero)) &
    (df["Aceptacion"].isin(aceptacion_sel))
]


# =====================
# Indicadores clave (KPIs)
# =====================
st.markdown("## 📊 Indicadores Clave (KPIs)")
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
with col_kpi1:
    st.metric("% Autos Aceptados", f"{(df_filtrado['Aceptacion'].isin(['Aceptable','Buena','Muy buena']).mean()*100):.1f}%")
    st.caption("Porcentaje de autos con aceptación positiva")
with col_kpi2:
    st.metric("Precio Promedio", f"{df_filtrado['Precio_Compra'].mode()[0] if not df_filtrado.empty else '-'}")
    st.caption("Categoría de precio más frecuente")
with col_kpi3:
    st.metric("Nivel de Seguridad + común", f"{df_filtrado['Seguridad'].mode()[0] if not df_filtrado.empty else '-'}")
    st.caption("Nivel de seguridad predominante")

# =====================
# Mostrar tabla con los datos filtrados
# =====================
st.subheader("📋 Datos Filtrados")
st.dataframe(df_filtrado)


# =====================
# Visualizaciones principales
# =====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Aceptación según Precio de Compra")
    st.caption("Histograma que muestra la cantidad de autos por tipo de aceptación y categoría de precio.")
    grafico1 = px.histogram(df_filtrado, x="Aceptacion", color="Precio_Compra", barmode="group")
    st.plotly_chart(grafico1, use_container_width=True)

with col2:
    st.subheader("🥧 Distribución General de Aceptación")
    st.caption("Gráfico de pastel que representa la proporción de cada tipo de aceptación.")
    grafico2 = px.pie(df_filtrado, names="Aceptacion")
    st.plotly_chart(grafico2, use_container_width=True)

# =====================
# Gráfico de barras apiladas: Distribución de aceptación por nivel de seguridad
# =====================
st.subheader("🟦 Distribución de Aceptación por Nivel de Seguridad")
st.caption("Gráfico de barras apiladas que muestra cómo se distribuye la aceptación según el nivel de seguridad.")
grafico_stacked = px.histogram(df_filtrado, x="Seguridad", color="Aceptacion", barmode="stack")
st.plotly_chart(grafico_stacked, use_container_width=True)


# =====================
# Comparaciones adicionales
# =====================
st.subheader("📈 Comparaciones adicionales")
col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🔐 Aceptación por Nivel de Seguridad (agrupado)")
    st.caption("Histograma agrupado por nivel de seguridad y aceptación.")
    grafico3 = px.histogram(df_filtrado, x="Seguridad", color="Aceptacion", barmode="group")
    st.plotly_chart(grafico3, use_container_width=True)

with col4:
    st.markdown("### 👥 Capacidad vs Aceptación")
    st.caption("Histograma agrupado por capacidad de personas y aceptación.")
    grafico4 = px.histogram(df_filtrado, x="Capacidad_Personas", color="Aceptacion", barmode="group")
    st.plotly_chart(grafico4, use_container_width=True)


# =====================
# Dispersión: Precio vs Mantenimiento vs Seguridad
# =====================
st.subheader("🔍 Relación entre Precio, Seguridad y Aceptación")
st.caption("Gráfico de dispersión que relaciona precio de compra, costo de mantenimiento, seguridad y aceptación.")
grafico5 = px.scatter(df_filtrado, x="Precio_Compra", y="Costo_Mantenimiento", color="Aceptacion", symbol="Seguridad")
st.plotly_chart(grafico5, use_container_width=True)


# =====================
# Heatmap de correlaciones (si existen variables codificadas numéricamente)
# =====================
if df_filtrado.select_dtypes(include=["int64", "float64"]).shape[1] > 1:
    st.subheader("🔥 Mapa de Calor de Correlaciones")
    st.caption("Mapa de calor de correlaciones entre variables numéricas.")
    fig, ax = plt.subplots()
    sns.heatmap(df_filtrado.select_dtypes(include=["int64", "float64"]).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# =====================
# Footer
# =====================
st.markdown("---")
st.markdown("📌 **Proyecto desarrollado como parte de un sistema de evaluación de autos - UCI Dataset**")

# =====================
# Explicación de variables
# =====================
# Sección informativa sobre las variables del dataset
with st.expander("ℹ️ Explicación de variables", expanded=False):
    st.markdown("""
    - **Precio_Compra**: Categoría del precio de compra del auto (Muy alto, Alto, Medio, Bajo).
    - **Costo_Mantenimiento**: Categoría del costo de mantenimiento (Muy alto, Alto, Medio, Bajo).
    - **Num_Puertas**: Número de puertas del auto.
    - **Capacidad_Personas**: Número de personas que puede transportar.
    - **Tamano_Maletero**: Tamaño del maletero (Pequeño, Mediano, Grande).
    - **Seguridad**: Nivel de seguridad del auto (Baja, Media, Alta).
    - **Aceptacion**: Evaluación final del auto (No aceptable, Aceptable, Buena, Muy buena).
    """)
    st.caption("Esta sección describe cada variable y su influencia en la aceptación del auto.")

# =====================
# Descarga de datos filtrados
# =====================
# Permite al usuario descargar el dataframe filtrado como CSV
csv = df_filtrado.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Descargar datos filtrados en CSV",
    data=csv,
    file_name='autos_filtrados.csv',
    mime='text/csv',
    help="Descarga los datos actualmente filtrados en formato CSV."
)

# =====================
# Gráfico de radar: Comparación de un auto vs promedio
# =====================
# Permite seleccionar un auto y comparar sus características numéricas normalizadas contra el promedio del dataset
import numpy as np
import plotly.graph_objects as go
st.markdown("## 🕸️ Comparativa de Auto Seleccionado vs Promedio")
st.caption("Selecciona un auto para comparar visualmente sus características frente al promedio del dataset.")
if not df_filtrado.empty:
    idx_auto = st.selectbox("Selecciona el índice del auto a comparar:", df_filtrado.index, format_func=lambda x: f"Auto #{x}")
    auto_sel = df_filtrado.loc[idx_auto]
    radar_vars = ["Precio_Compra", "Costo_Mantenimiento", "Num_Puertas", "Capacidad_Personas", "Tamano_Maletero", "Seguridad"]
    radar_map = {
        "Precio_Compra": {"Muy alto":4, "Alto":3, "Medio":2, "Bajo":1},
        "Costo_Mantenimiento": {"Muy alto":4, "Alto":3, "Medio":2, "Bajo":1},
        "Tamano_Maletero": {"Grande":3, "Mediano":2, "Pequeño":1},
        "Seguridad": {"Alta":3, "Media":2, "Baja":1}
    }
    auto_vals = []
    prom_vals = []
    for var in radar_vars:
        if var in radar_map:
            auto_vals.append(radar_map[var].get(auto_sel[var], 0))
            prom_vals.append(df_filtrado[var].map(radar_map[var]).mean())
        else:
            auto_vals.append(pd.to_numeric(auto_sel[var], errors='coerce'))
            prom_vals.append(pd.to_numeric(df_filtrado[var], errors='coerce').mean())
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=auto_vals, theta=radar_vars, fill='toself', name='Auto Seleccionado'))
    fig_radar.add_trace(go.Scatterpolar(r=prom_vals, theta=radar_vars, fill='toself', name='Promedio'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("No hay autos filtrados para comparar.")

# =====================
# Tabla de autos destacados
# =====================
# Muestra los autos con mejor aceptación, menor precio, mayor seguridad, etc.
st.markdown("## ⭐ Autos Destacados")
st.caption("Tabla con autos destacados según aceptación, precio y seguridad.")
if not df_filtrado.empty:
    destacados = pd.DataFrame()
    mejores = df_filtrado[df_filtrado['Aceptacion'] == 'Muy buena']
    if not mejores.empty:
        destacados = pd.concat([destacados, mejores.head(3)])
    baratos = df_filtrado[df_filtrado['Precio_Compra'] == 'Bajo']
    if not baratos.empty:
        destacados = pd.concat([destacados, baratos.head(3)])
    seguros = df_filtrado[df_filtrado['Seguridad'] == 'Alta']
    if not seguros.empty:
        destacados = pd.concat([destacados, seguros.head(3)])
    destacados = destacados.drop_duplicates()
    st.dataframe(destacados)
else:
    st.info("No hay autos destacados para mostrar.")

# =====================
# Análisis de correlación interactivo
# =====================
# Permite al usuario elegir dos variables y ver su relación
st.markdown("## 🔗 Análisis de Correlación Interactivo")
st.caption("Selecciona dos variables para analizar su relación mediante un gráfico de dispersión.")
cols_corr = [c for c in df_filtrado.columns if c not in ["Aceptacion"]]
if len(cols_corr) >= 2:
    var_x = st.selectbox("Variable X", cols_corr, key="corr_x")
    var_y = st.selectbox("Variable Y", cols_corr, key="corr_y")
    if var_x and var_y:
        fig_corr = px.scatter(df_filtrado, x=var_x, y=var_y, color="Aceptacion")
        st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("No hay suficientes variables para análisis de correlación.")

# =====================
# Predicción de aceptación (modelo simple)
# =====================
# Entrena un modelo simple y permite predecir aceptación según selección de características
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
st.markdown("## 🤖 Predicción de Aceptación")
st.caption("Selecciona características y obtén una predicción de aceptación usando un modelo de machine learning básico.")
if not df.empty:
    df_model = df.copy()
    label_cols = ["Precio_Compra", "Costo_Mantenimiento", "Num_Puertas", "Capacidad_Personas", "Tamano_Maletero", "Seguridad", "Aceptacion"]
    encoders = {col: LabelEncoder().fit(df_model[col]) for col in label_cols}
    for col in label_cols:
        df_model[col] = encoders[col].transform(df_model[col])
    X = df_model.drop("Aceptacion", axis=1)
    y = df_model["Aceptacion"]
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    st.markdown("### Selecciona características para predecir aceptación:")
    user_input = {}
    for col in label_cols[:-1]:
        user_input[col] = st.selectbox(f"{col}", df[col].unique(), key=f"pred_{col}")
    if st.button("Predecir aceptación"):
        input_df = pd.DataFrame([user_input])
        for col in label_cols[:-1]:
            input_df[col] = encoders[col].transform(input_df[col])
        pred = model.predict(input_df)[0]
        pred_label = encoders["Aceptacion"].inverse_transform([pred])[0]
        st.success(f"Predicción de aceptación: **{pred_label}**")
else:
    st.info("No hay datos suficientes para entrenar el modelo de predicción.")
