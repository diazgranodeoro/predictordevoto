# 🗳️ Predictor de Voto Electoral
**Machine Learning aplicado a datos del CIS**

Proyecto completo de predicción de voto en elecciones españolas basado en datos del CIS (Centro de Investigaciones Sociológicas) usando PyCaret y Streamlit.

## 📋 Descripción

Este proyecto utiliza técnicas de Machine Learning para predecir la intención de voto de ciudadanos españoles basándose en variables sociodemográficas. Incluye:
- 📊 **Análisis completo** de ~44.000 encuestas del CIS (11 barómetros de 2025)
- 🤖 **Modelo de clasificación** multiclase con PyCaret
- 🌐 **Aplicación web interactiva** con Streamlit

## 🚀 Instalación

### ⚠️ Requisitos Importantes
- **Python 3.9, 3.10 o 3.11** (PyCaret NO funciona con Python 3.12+)
- Jupyter Notebook o VS Code con extensión de Python

### Instalación Rápida
```bash
# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows (en Linux/Mac: source venv/bin/activate)

# Instalar dependencias
pip install -r requirements.txt
```

📖 **Guía detallada:** Ver [INSTALACION.md](INSTALACION.md)

## 📊 Dataset

**Fuente:** Barómetros CIS 2025 (enero - diciembre, excepto agosto)  
**Total:** ~44.000 entrevistas (~4.000 por mes × 11 meses)  
**Formato original:** SPSS (.sav)

### Variables utilizadas (8 predictoras + 1 objetivo):
- 🗺️ **CCAA** - Comunidad Autónoma (18 categorías)
- 👤 **SEXO** - Hombre/Mujer
- 📅 **GRUPO_EDAD** - Grupos de edad (18-29, 30-39, 40-49, 50-59, 60-69, 70+)
- 📊 **ESCIDEOL** - Escala ideológica 1-10 (izquierda-derecha)
- 🎓 **ESTUDIOS** - Nivel educativo (4 categorías)
- 💼 **SITLAB** - Situación laboral (4 categorías)
- 🏙️ **TAMUNI** - Tamaño del municipio (3 categorías)
- 🗳️ **PARTICIPACIONG** - Participación electoral previa
- 🎯 **VOTOSIMG** - Voto en elecciones 2023 (variable objetivo)

**Partidos predichos (4 principales):** PP, PSOE, VOX, Sumar

### ¿Por qué solo 4 partidos?
El modelo se ha optimizado para los 4 partidos con mayor representación en los datos:
- ✅ **Suficientes casos** para entrenamiento robusto (~10.000 casos por partido)
- ✅ **Mejor accuracy:** 66.63% vs 25% del azar (4 clases)
- ✅ **Predicciones más confiables** al eliminar ruido de partidos minoritarios

## 📁 Estructura del proyecto

```
predictordevoto/
├── predictorvoto.ipynb          # Notebook principal con análisis completo
├── app.py                    # Aplicación web Streamlit
├── run_app.bat              # Script para ejecutar la app con Python 3.11
├── data/
│   ├── enero.sav - diciembre.sav  # Datos CIS 2025 (formato SPSS)
│   └── datos_limpios.csv    # Dataset procesado y listo para ML
├── models/
│   └── modelo_prediccion_voto.pkl # Modelo entrenado (Gradient Boosting)
├── requirements.txt          # Dependencias del proyecto
├── README.md                # Este archivo
├── INSTALACION.md           # Guía detallada de instalación
└── PRESENTACION.md          # Presentación del proyecto
```

## � Uso

### 1. Aplicación Web Interactiva (Recomendado)

**Ejecutar la app Streamlit:**

```bash
# Opción 1: Usar el script batch (ejecuta con Python 3.11)
.\run_app.bat

# Opción 2: Comando directo
py -3.11 -m streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

**Características de la app:**
- 🔮 **Predicción Individual:** Ingresa datos sociodemográficos y obtén predicción inmediata con probabilidad
- 📊 **Análisis de Probabilidades:** Visualiza cómo varía la predicción según edad, ideología, estudios, región, etc.
- ⚖️ **Comparar Perfiles:** Compara predicciones de 2-5 perfiles diferentes simultáneamente
- 🎨 **Colores corporativos:** PP (azul), PSOE (rojo), VOX (verde), Sumar (magenta)
- 🎯 **Solo 4 partidos:** PP, PSOE, VOX y Sumar (los más representados en los datos)

### 2. Análisis y Entrenamiento (Notebook Jupyter)

**Ejecutar el análisis completo en `predictorvoto.ipynb`:**

1. **Carga y preparación** - Importar 11 barómetros CIS 2025
2. **Limpieza y recodificación** - Transformar variables categóricas
3. **Preparación ML** - One-Hot Encoding y filtrado
4. **Análisis de correlaciones** - Verificar multicolinealidad
5. **Modelado con PyCaret** - Comparación automática de algoritmos
6. **Evaluación** - Matriz de confusión, ROC-AUC, Feature Importance
7. **Predicciones** - Ejemplos de uso del modelo

### 3. Usar el Modelo Programáticamente

```python
from pycaret.classification import load_model, predict_model
import pandas as pd

# Cargar modelo entrenado
modelo = load_model('models/modelo_prediccion_voto')

# Preparar datos (formato one-hot encoding)
nuevos_datos = pd.DataFrame({
    'ESCIDEOL': [5],
    'GRUPO_EDAD_30-39': [1],  # Grupo de edad 30-39
    'SEXO_Mujer': [1],
    'CCAA_Madrid': [1],
    # ... resto de variables (ver notebook para formato completo)
})

# Predecir
resultado = predict_model(modelo, data=nuevos_datos)
print(f"Partido predicho: {resultado['prediction_label'].values[0]}")
print(f"Probabilidad: {resultado['prediction_score'].values[0]:.2%}")
```

## 📈 Resultados del Modelo

### 🎯 Métricas de rendimiento:
- **Accuracy:** 66.63% (vs 25% del azar con 4 clases) → **+41.63 puntos sobre el azar**
- **Algoritmo:** Gradient Boosting Classifier (con tuning optimizado)
- **Validación:** Cross-validation 10-fold para garantizar robustez
- **Dataset de test:** 10% de los datos (~4.400 casos) para evaluación final

### 📊 Análisis detallado:

El notebook `predictorvoto.ipynb` genera:
- **Matriz de confusión**: Muestra que el modelo distingue bien izquierda (PSOE, Sumar) vs derecha (PP, VOX)
- **Feature Importance**: La escala ideológica (ESCIDEOL) es la variable más predictiva
- **Curvas AUC**: Rendimiento del clasificador por cada partido
- **Reporte de clasificación**: Métricas detalladas (precision, recall, F1-score) por partido

### 🔍 Patrones encontrados:

✅ **Lo que funciona:**
- Separación clara entre bloques ideológicos (izquierda vs derecha)
- La auto-ubicación ideológica es el mejor predictor del voto
- Variables sociodemográficas aportan información complementaria útil
- Accuracy de 66.63% es casi el triple que el azar (25%)

⚠️ **Confusiones comunes:**
- PP ↔ VOX: Ambos en la derecha, perfiles similares
- PSOE ↔ Sumar: Ambos en la izquierda, votantes con características parecidas

## ⚠️ Limitaciones y Consideraciones

1. **Sesgo del CIS**: Los datos del CIS pueden tener sesgos de representatividad
2. **Solo 4 partidos**: Partidos minoritarios (ERC, Junts, PNV, Bildu, etc.) no están incluidos
3. **Momento político**: Los datos son de 2025, el contexto político cambia constantemente
4. **Variables limitadas**: Solo se usan variables sociodemográficas básicas (no se incluyen actitudes políticas específicas)
5. **Accuracy limitado**: 54% es bueno para ciencias sociales, pero significa ~46% de error en predicciones individuales

## 💡 Mejoras futuras

- [ ] Incluir variables de actitudes políticas (confianza en instituciones, valoración de líderes)
- [ ] Agregar datos temporales para captar tendencias mensuales
- [ ] Implementar modelos ensemble más sofisticados (stacking, blending)
- [ ] Probar redes neuronales para captar interacciones complejas
- [ ] Análisis de subgrupos (jóvenes urbanos, pensionistas rurales, etc.)
- [ ] Desplegar la app en la nube (Streamlit Cloud, Azure, AWS)

## 📚 Referencias

- [Centro de Investigaciones Sociológicas (CIS)](http://www.cis.es/)
- [PyCaret Documentation](https://pycaret.org/)

## 👤 Autor

Rubén Díaz Grano de Oro

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo.
