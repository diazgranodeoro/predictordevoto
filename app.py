"""
🗳️ Predictor de Voto Electoral
Aplicación Streamlit para predecir intención de voto mediante Machine Learning
Basado en datos del CIS (Centro de Investigaciones Sociológicas)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Voto Electoral",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Forzar tema oscuro
st.markdown("""
    <style>
        /* Forzar modo oscuro */
        :root {
            color-scheme: dark;
        }
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Forzar texto blanco en todos los elementos */
        p, label, h1, h2, h3, li {
            color: #FAFAFA !important;
        }
        /* FORZAR texto NEGRO en selectbox y sus opciones */
        .stSelectbox [data-baseweb="select"] div,
        .stSelectbox [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] input,
        [data-baseweb="popover"] div,
        [data-baseweb="popover"] span,
        [data-baseweb="popover"] li,
        [role="option"],
        [role="option"] div,
        [role="option"] span {
            color: #000000 !important;
            background-color: #FFFFFF !important;
        }
        /* Dropdown de selectbox */
        [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("🗳️ Predictor de Voto Electoral")
st.markdown("### Predice la intención de voto basada en variables sociodemográficas")
st.markdown("---")

# Colores de los partidos políticos
# Colores oficiales de los 4 partidos principales
COLORES_PARTIDOS = {
    'PP': '#1e4a90',      # Azul PP
    'PSOE': '#FF0000',    # Rojo PSOE
    'Sumar': '#E61455',   # Magenta Sumar
    'VOX': '#73B446'      # Verde VOX
}

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    """Carga el modelo de predicción entrenado"""
    try:
        modelo = load_model('models/modelo_prediccion_voto')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.info("💡 Asegúrate de que el modelo esté guardado en 'models/modelo_prediccion_voto.pkl'")
        return None

# Función auxiliar para crear dataframe de predicción
def crear_dataframe_prediccion(grupo_edad, sexo, ccaa, tamuni, escideol, estudios, sitlab, participacion):
    """Convierte los parámetros de entrada en un DataFrame para el modelo"""
    return pd.DataFrame({
        'ESCIDEOL': [escideol],
        # GRUPO_EDAD
        'GRUPO_EDAD_30-39': [1 if grupo_edad == '30-39' else 0],
        'GRUPO_EDAD_40-49': [1 if grupo_edad == '40-49' else 0],
        'GRUPO_EDAD_50-59': [1 if grupo_edad == '50-59' else 0],
        'GRUPO_EDAD_60-69': [1 if grupo_edad == '60-69' else 0],
        'GRUPO_EDAD_70+': [1 if grupo_edad == '70+' else 0],
        # CCAA
        'CCAA_Aragón': [1 if ccaa == 'Aragón' else 0],
        'CCAA_Asturias': [1 if ccaa == 'Asturias' else 0],
        'CCAA_Balears': [1 if ccaa == 'Balears' else 0],
        'CCAA_Canarias': [1 if ccaa == 'Canarias' else 0],
        'CCAA_Cantabria': [1 if ccaa == 'Cantabria' else 0],
        'CCAA_Castilla y León': [1 if ccaa == 'Castilla y León' else 0],
        'CCAA_Castilla-La Mancha': [1 if ccaa == 'Castilla-La Mancha' else 0],
        'CCAA_Catalunya': [1 if ccaa == 'Catalunya' else 0],
        'CCAA_Ceuta y Melilla': [1 if ccaa == 'Ceuta y Melilla' else 0],
        'CCAA_Comunitat Valenciana': [1 if ccaa == 'Comunitat Valenciana' else 0],
        'CCAA_Euskadi': [1 if ccaa == 'Euskadi' else 0],
        'CCAA_Extremadura': [1 if ccaa == 'Extremadura' else 0],
        'CCAA_Galicia': [1 if ccaa == 'Galicia' else 0],
        'CCAA_La Rioja': [1 if ccaa == 'La Rioja' else 0],
        'CCAA_Madrid': [1 if ccaa == 'Madrid' else 0],
        'CCAA_Murcia': [1 if ccaa == 'Murcia' else 0],
        'CCAA_Navarra': [1 if ccaa == 'Navarra' else 0],
        # SEXO
        'SEXO_Mujer': [1 if sexo == 'Mujer' else 0],
        # TAMUNI
        'TAMUNI_0-10000': [1 if tamuni == '0-10.000' else 0],
        'TAMUNI_10.001-100.000': [1 if tamuni == '10.001-100.000' else 0],
        'TAMUNI_>100.000': [1 if tamuni == '>100.000' else 0],
        # PARTICIPACION
        'PARTICIPACIONG_Sí': [1 if participacion == 'Sí' else 0],
        # ESTUDIOS
        'ESTUDIOS_Formación Profesional': [1 if estudios == 'Formación Profesional' else 0],
        'ESTUDIOS_Secundaria': [1 if estudios == 'Secundaria' else 0],
        'ESTUDIOS_Sin estudios o primaria': [1 if estudios == 'Sin estudios o primaria' else 0],
        'ESTUDIOS_Superiores': [1 if estudios == 'Superiores' else 0],
        # SITLAB
        'SITLAB_En paro': [1 if sitlab == 'En paro' else 0],
        'SITLAB_Otra situación': [1 if sitlab == 'Otra situación' else 0],
        'SITLAB_Pensionista': [1 if sitlab == 'Pensionista' else 0],
        'SITLAB_Trabaja': [1 if sitlab == 'Trabaja' else 0]
    })

modelo = cargar_modelo()

if modelo is not None:
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs(["🔮 Predicción Individual", "📊 Análisis de Probabilidades", "⚖️ Comparar Perfiles"])
    
    # ============================================================================
    # PESTAÑA 1: PREDICCIÓN INDIVIDUAL
    # ============================================================================
    with tab1:
        # Crear dos columnas para el layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Datos Personales")
            
            # Grupo de edad
            grupo_edad = st.selectbox(
                "Grupo de edad",
                options=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'],
                index=1,  # Default: 30-39
                key="grupo_edad_tab1"
            )
            
            # Sexo
            sexo = st.radio("Sexo", options=["Hombre", "Mujer"], key="sexo_tab1")
            
            # CCAA
            ccaa_options = [
                'Andalucía', 'Aragón', 'Asturias', 'Balears', 'Canarias', 'Cantabria',
                'Castilla-La Mancha', 'Castilla y León', 'Catalunya', 'Comunitat Valenciana',
                'Extremadura', 'Galicia', 'Madrid', 'Murcia', 'Navarra', 'Euskadi',
                'La Rioja', 'Ceuta y Melilla'
            ]
            ccaa = st.selectbox("Comunidad Autónoma", options=ccaa_options, key="ccaa_tab1")
            
            # Tamaño del municipio
            tamuni = st.selectbox(
                "Tamaño del municipio",
                options=['0-10.000', '10.001-100.000', '>100.000'],
                key="tamuni_tab1"
            )
        
        with col2:
            st.subheader("🎓 Datos Socioeconómicos")
            
            # Escala ideológica
            escideol = st.slider(
                "Escala ideológica (1=Izquierda, 10=Derecha)",
                min_value=1, max_value=10, value=5, step=1,
                help="Posicionamiento político en el eje izquierda-derecha",
                key="escideol_tab1"
            )
            
            # Estudios
            estudios = st.selectbox(
                "Nivel de estudios",
                options=['Sin estudios o primaria', 'Secundaria', 'Formación Profesional', 'Superiores'],
                key="estudios_tab1"
            )
            
            # Situación laboral
            sitlab = st.selectbox(
                "Situación laboral",
                options=['Trabaja', 'En paro', 'Pensionista', 'Otra situación'],
                key="sitlab_tab1"
            )
            
            # Participación
            participacion = st.radio(
                "¿Participó en las últimas elecciones?",
                options=["Sí", "No"],
                key="participacion_tab1"
            )
        
        st.markdown("---")
        
        # Botón de predicción
        if st.button("🔮 Predecir Voto", type="primary", use_container_width=True):
            # Crear el dataframe con los valores ingresados
            nuevo_dato = crear_dataframe_prediccion(grupo_edad, sexo, ccaa, tamuni, escideol, estudios, sitlab, participacion)
            
            # Hacer la predicción
            with st.spinner('Realizando predicción...'):
                try:
                    prediccion = predict_model(modelo, data=nuevo_dato)
                    
                    # Mostrar resultado
                    st.success("✅ Predicción completada")
                    
                    # Crear tres columnas para el resultado
                    res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
                    
                    with res_col2:
                        st.markdown("### 🎯 Resultado de la Predicción")
                        
                        voto_predicho = prediccion['prediction_label'].values[0]
                        probabilidad = prediccion['prediction_score'].values[0]
                        
                    # Obtener el color del partido
                    color_partido = COLORES_PARTIDOS.get(voto_predicho, '#1f77b4')
                    
                    # Mostrar el partido predicho con estilo y color corporativo
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background-color: #1E1E1E; border-radius: 10px;'>
                        <h2 style='color: {color_partido}; margin: 0;'>{voto_predicho}</h2>
                        <p style='font-size: 18px; color: #FFFFFF; margin-top: 10px;'>
                            Probabilidad: <strong>{probabilidad:.1%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar barra de probabilidad con color del partido
                    st.markdown(f"""
                    <style>
                        .stProgress > div > div > div > div {{
                            background-color: {color_partido};
                        }}
                    </style>
                    """, unsafe_allow_html=True)
                    st.progress(probabilidad)
                    
                    # Información adicional
                    st.markdown("---")
                    st.info("""
                    **💡 Interpretación:**
                    - La predicción se basa en el modelo entrenado con datos del CIS 2025
                    - La probabilidad indica el nivel de confianza del modelo
                    - Probabilidad > 50%: Predicción con confianza moderada
                    - Probabilidad > 70%: Predicción con alta confianza
                    """)
                    
                    # Mostrar todas las probabilidades (si están disponibles)
                    if len([col for col in prediccion.columns if col.startswith('prediction_score')]) > 1:
                        st.markdown("### 📊 Probabilidades por partido")
                        
                        # Extraer todas las columnas de score
                        score_cols = [col for col in prediccion.columns if col.startswith('prediction_score')]
                        if score_cols:
                            scores_df = prediccion[score_cols].T
                            scores_df.columns = ['Probabilidad']
                            scores_df.index = [col.replace('prediction_score_', '') for col in score_cols]
                            scores_df = scores_df.sort_values('Probabilidad', ascending=False)
                            
                            st.dataframe(
                                scores_df.style.format({'Probabilidad': '{:.1%}'}),
                                use_container_width=True
                            )
                    
                except Exception as e:
                    st.error(f"❌ Error al realizar la predicción: {e}")
                    st.info("💡 Verifica que el modelo esté correctamente entrenado y guardado")
    
    # ============================================================================
    # PESTAÑA 2: ANÁLISIS DE PROBABILIDADES
    # ============================================================================
    with tab2:
        st.subheader("📊 Análisis de Probabilidades por Variable")
        st.markdown("Analiza cómo varía la predicción al cambiar una variable específica")
        
        # Configurar perfil base
        st.markdown("### ⚙️ Configurar Perfil Base")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            base_grupo_edad = st.selectbox("Grupo de edad base", 
                ['18-29', '30-39', '40-49', '50-59', '60-69', '70+'],
                index=1, key="base_grupo_edad")
            base_sexo = st.selectbox("Sexo base", ["Hombre", "Mujer"], key="base_sexo")
            base_ccaa = st.selectbox("CCAA base", ccaa_options, index=12, key="base_ccaa")  # Madrid por defecto
        
        with col2:
            base_estudios = st.selectbox("Estudios base", 
                ['Sin estudios o primaria', 'Secundaria', 'Formación Profesional', 'Superiores'],
                index=3, key="base_estudios")
            base_sitlab = st.selectbox("Situación laboral base",
                ['Trabaja', 'En paro', 'Pensionista', 'Otra situación'],
                key="base_sitlab")
        
        with col3:
            base_tamuni = st.selectbox("Tamaño municipio base",
                ['0-10.000', '10.001-100.000', '>100.000'],
                index=2, key="base_tamuni")
            base_participacion = st.selectbox("Participación base", ["Sí", "No"], key="base_participacion")
        
        # Seleccionar variable a analizar
        st.markdown("### 🔍 Variable a Analizar")
        variable_analizar = st.selectbox(
            "Selecciona la variable para ver cómo afecta la predicción:",
            ["Escala ideológica", "Grupo de edad", "Nivel de estudios", "Situación laboral", "CCAA"]
        )
        
        if st.button("📈 Generar Análisis", type="primary"):
            with st.spinner("Generando análisis..."):
                try:
                    resultados = []
                    
                    if variable_analizar == "Escala ideológica":
                        for ideol in range(1, 11):
                            df_temp = crear_dataframe_prediccion(
                                base_grupo_edad, base_sexo, base_ccaa, base_tamuni, 
                                ideol, base_estudios, base_sitlab, base_participacion
                            )
                            pred = predict_model(modelo, data=df_temp)
                            resultados.append({
                                'Variable': ideol,
                                'Partido': pred['prediction_label'].values[0],
                                'Probabilidad': pred['prediction_score'].values[0]
                            })
                    
                    elif variable_analizar == "Grupo de edad":
                        grupos_edad = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
                        for grupo_temp in grupos_edad:
                            df_temp = crear_dataframe_prediccion(
                                grupo_temp, base_sexo, base_ccaa, base_tamuni, 
                                5, base_estudios, base_sitlab, base_participacion
                            )
                            pred = predict_model(modelo, data=df_temp)
                            resultados.append({
                                'Variable': grupo_temp,
                                'Partido': pred['prediction_label'].values[0],
                                'Probabilidad': pred['prediction_score'].values[0]
                            })
                    
                    elif variable_analizar == "Nivel de estudios":
                        opciones = ['Sin estudios o primaria', 'Secundaria', 'Formación Profesional', 'Superiores']
                        for est in opciones:
                            df_temp = crear_dataframe_prediccion(
                                base_grupo_edad, base_sexo, base_ccaa, base_tamuni, 
                                5, est, base_sitlab, base_participacion
                            )
                            pred = predict_model(modelo, data=df_temp)
                            resultados.append({
                                'Variable': est,
                                'Partido': pred['prediction_label'].values[0],
                                'Probabilidad': pred['prediction_score'].values[0]
                            })
                    
                    elif variable_analizar == "Situación laboral":
                        opciones = ['Trabaja', 'En paro', 'Pensionista', 'Otra situación']
                        for sit in opciones:
                            df_temp = crear_dataframe_prediccion(
                                base_grupo_edad, base_sexo, base_ccaa, base_tamuni, 
                                5, base_estudios, sit, base_participacion
                            )
                            pred = predict_model(modelo, data=df_temp)
                            resultados.append({
                                'Variable': sit,
                                'Partido': pred['prediction_label'].values[0],
                                'Probabilidad': pred['prediction_score'].values[0]
                            })
                    
                    elif variable_analizar == "CCAA":
                        for ccaa_temp in ccaa_options[:10]:  # Primeras 10 para no saturar
                            df_temp = crear_dataframe_prediccion(
                                base_grupo_edad, base_sexo, ccaa_temp, base_tamuni, 
                                5, base_estudios, base_sitlab, base_participacion
                            )
                            pred = predict_model(modelo, data=df_temp)
                            resultados.append({
                                'Variable': ccaa_temp,
                                'Partido': pred['prediction_label'].values[0],
                                'Probabilidad': pred['prediction_score'].values[0]
                            })
                    
                    # Crear DataFrame de resultados
                    df_resultados = pd.DataFrame(resultados)
                    
                    # Gráfico con colores de partidos
                    fig = px.bar(df_resultados, x='Variable', y='Probabilidad', color='Partido',
                                title=f'Predicción según {variable_analizar}',
                                labels={'Variable': variable_analizar, 'Probabilidad': 'Probabilidad (%)'},
                                text='Probabilidad',
                                color_discrete_map=COLORES_PARTIDOS)
                    
                    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                    fig.update_layout(height=500)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de resultados
                    st.markdown("### 📋 Resultados Detallados")
                    st.dataframe(
                        df_resultados.style.format({'Probabilidad': '{:.1%}'}),
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error al generar análisis: {e}")
    
    # ============================================================================
    # PESTAÑA 3: COMPARAR PERFILES
    # ============================================================================
    with tab3:
        st.subheader("⚖️ Comparar Múltiples Perfiles")
        st.markdown("Compara las predicciones para diferentes perfiles de votantes")
        
        num_perfiles = st.slider("¿Cuántos perfiles quieres comparar?", min_value=2, max_value=5, value=2)
        
        perfiles = []
        cols = st.columns(num_perfiles)
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"### Perfil {i+1}")
                perfil = {
                    'nombre': st.text_input(f"Nombre", value=f"Perfil {i+1}", key=f"nombre_{i}"),
                    'grupo_edad': st.selectbox("Grupo de edad", 
                        ['18-29', '30-39', '40-49', '50-59', '60-69', '70+'],
                        index=1, key=f"grupo_edad_comp_{i}"),
                    'sexo': st.radio("Sexo", ["Hombre", "Mujer"], key=f"sexo_comp_{i}"),
                    'ccaa': st.selectbox("CCAA", ccaa_options, key=f"ccaa_comp_{i}"),
                    'escideol': st.slider("Ideología", 1, 10, 5, key=f"ideol_comp_{i}"),
                    'estudios': st.selectbox("Estudios", 
                        ['Sin estudios o primaria', 'Secundaria', 'Formación Profesional', 'Superiores'],
                        key=f"est_comp_{i}"),
                    'sitlab': st.selectbox("Situación", 
                        ['Trabaja', 'En paro', 'Pensionista', 'Otra situación'],
                        key=f"sit_comp_{i}"),
                    'tamuni': st.selectbox("Municipio",
                        ['0-10.000', '10.001-100.000', '>100.000'],
                        key=f"tam_comp_{i}"),
                    'participacion': st.radio("Participó", ["Sí", "No"], key=f"part_comp_{i}")
                }
                perfiles.append(perfil)
        
        st.markdown("---")
        
        if st.button("⚖️ Comparar Perfiles", type="primary", use_container_width=True):
            with st.spinner("Comparando perfiles..."):
                try:
                    comparacion = []
                    
                    for perfil in perfiles:
                        df_temp = crear_dataframe_prediccion(
                            perfil['grupo_edad'], perfil['sexo'], perfil['ccaa'], perfil['tamuni'],
                            perfil['escideol'], perfil['estudios'], perfil['sitlab'], perfil['participacion']
                        )
                        pred = predict_model(modelo, data=df_temp)
                        
                        comparacion.append({
                            'Perfil': perfil['nombre'],
                            'Voto Predicho': pred['prediction_label'].values[0],
                            'Probabilidad': pred['prediction_score'].values[0],
                            'Grupo de Edad': perfil['grupo_edad'],
                            'Ideología': perfil['escideol'],
                            'CCAA': perfil['ccaa']
                        })
                    
                    df_comp = pd.DataFrame(comparacion)
                    
                    # Gráfico de barras con colores de partidos
                    fig = px.bar(df_comp, x='Perfil', y='Probabilidad', color='Voto Predicho',
                                title='Comparación de Predicciones',
                                text='Probabilidad',
                                color_discrete_map=COLORES_PARTIDOS)
                    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                    fig.update_layout(height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla comparativa
                    st.markdown("### 📊 Tabla Comparativa")
                    st.dataframe(
                        df_comp.style.format({'Probabilidad': '{:.1%}'}),
                        use_container_width=True
                    )
                    
                    # Resumen
                    st.markdown("### 📝 Resumen")
                    partidos_unicos = df_comp['Voto Predicho'].unique()
                    st.write(f"**Partidos predichos:** {', '.join(partidos_unicos)}")
                    
                    if len(partidos_unicos) == 1:
                        st.success(f"✅ Todos los perfiles predicen voto a: **{partidos_unicos[0]}**")
                    else:
                        st.info("🔄 Los perfiles muestran predicciones diferentes")
                    
                except Exception as e:
                    st.error(f"❌ Error al comparar perfiles: {e}")

    # Sección de información
    st.markdown("---")
    with st.expander("ℹ️ Información sobre el modelo"):
        st.markdown("""
        ### Sobre este predictor
        
        Este modelo de predicción de voto electoral fue entrenado con datos de los **Barómetros del CIS 2025**.
        
        **Variables utilizadas:**
        - 📍 Comunidad Autónoma
        - 👤 Edad y Sexo
        - 🎓 Nivel de estudios
        - 💼 Situación laboral
        - 🏙️ Tamaño del municipio
        - 📊 Escala ideológica (1-10)
        - 🗳️ Participación electoral previa
        
        **Datos de entrenamiento:**
        - ~44.000 encuestas del CIS (11 meses de 2025)
        - Partidos predichos: PSOE, PP, VOX, Sumar, y otros
        
        **Limitaciones:**
        - El modelo refleja patrones históricos, no necesariamente comportamientos futuros
        - Puede tener sesgos inherentes a las encuestas del CIS
        - La precisión varía según el partido político
        """)

else:
    st.error("❌ No se pudo cargar el modelo. Verifica que el archivo 'models/modelo_prediccion_voto.pkl' exista.")
    st.info("""
    **Para usar esta aplicación:**
    1. Ejecuta el notebook 'limpiacsvs.ipynb' completamente
    2. Asegúrate de que el modelo se haya guardado en 'models/modelo_prediccion_voto.pkl'
    3. Recarga esta página
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #FFFFFF; font-size: 12px;'>
    Desarrollado por Rubén Díaz usando Streamlit y PyCaret | Datos: CIS 2025
</div>
""", unsafe_allow_html=True)
