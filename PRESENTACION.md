# 🗳️ Predictor de Voto Electoral
## Machine Learning aplicado a datos del CIS

**Autor:** Rubén Díaz Grano de Oro  
**Fecha:** Enero 2026

---

## 📊 RESUMEN EJECUTIVO

### Objetivo
Desarrollar un modelo de Machine Learning capaz de **predecir la intención de voto** en elecciones españolas basándose en variables sociodemográficas.

### Fuente de Datos
- **Dataset:** 11 Barómetros CIS 2025 (enero-diciembre, excepto agosto)
- **Tamaño:** ~44.000 entrevistas (~4.000/mes × 11 meses)
- **Variables:** 8 predictoras + 1 objetivo (VOTOSIMG)

### Metodología
- **Framework:** PyCaret 3.3.2 (wrapper de Scikit-learn)
- **Técnica:** Clasificación Multiclase (4 partidos principales)
- **Validación:** 90% Train / 10% Test + Cross-validation 10-fold
- **Modelo Final:** Gradient Boosting Classifier (con tuning optimizado)
- **Accuracy:** 66.63% (vs 25% del azar con 4 clases) → **+41.63 pp sobre el azar**

---

## 🎯 VARIABLES DEL MODELO

### Variables Predictoras (Features)
1. 🗺️ **CCAA** - Comunidad Autónoma (18 categorías)
2. 👥 **SEXO** - Hombre/Mujer
3. 📅 **GRUPO_EDAD** - Grupos de edad: 18-29, 30-39, 40-49, 50-59, 60-69, 70+
4. 💭 **ESCIDEOL** - Escala ideológica 1-10 (izquierda-derecha)
5. 🏙️ **TAMUNI** - Tamaño del municipio (3 categorías: 0-10.000, 10.001-100.000, >100.000)
6. ☑️ **PARTICIPACIONG** - Participación electoral previa (Sí/No)
7. 🎓 **ESTUDIOS** - Nivel educativo (4 categorías)
8. 💼 **SITLAB** - Situación laboral (4 categorías: Trabaja, En paro, Pensionista, Otra situación)

### Variable Objetivo (Target)
🎯 **VOTOSIMG** - Voto en Elecciones Generales 2023
- **4 partidos principales:** PP, PSOE, VOX, Sumar
- **Criterio de selección:** Solo partidos con más de 5.000 casos para garantizar entrenamiento robusto
- **Excluidos:** Votos en blanco, nulos, abstenciones, partidos minoritarios (ERC, Junts, PNV, Bildu, CC)

---

## 🔬 PIPELINE DE PROCESAMIENTO

```
┌─────────────────────────────────────────────────────────┐
│ 1. CARGA DE DATOS                                       │
│    ▸ 11 Barómetros CIS 2025 (formato SPSS .sav)        │
│    ▸ ~44.000 registros totales                         │
│    ▸ Pandas DataFrame consolidado                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. LIMPIEZA Y RECODIFICACIÓN                           │
│    ▸ Conversión de códigos numéricos → etiquetas       │
│    ▸ Creación de grupos de edad (7 categorías)         │
│    ▸ Simplificación de categorías                      │
│    ▸ Gestión de valores nulos (98, 99, NA)            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. PREPARACIÓN PARA ML                                  │
│    ▸ Filtrado: Solo 4 partidos principales             │
│    ▸ Eliminación de valores nulos                      │
│    ▸ One-Hot Encoding (variables categóricas)          │
│    ▸ ~34 features finales tras encoding                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. ANÁLISIS EXPLORATORIO                               │
│    ▸ Heatmap de correlaciones                          │
│    ▸ Verificación de multicolinealidad                 │
│    ▸ Distribución de variables                         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 5. MODELADO CON PYCARET                                │
│    ▸ Comparación automática de 15+ algoritmos          │
│    ▸ Selección: Gradient Boosting Classifier           │
│    ▸ Tuning con Random Search (50 iteraciones)         │
│    ▸ División 90/10 (train/test) + CV 10-fold         │
│    ▸ Resultado final: 66.63% accuracy                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 6. EVALUACIÓN Y MÉTRICAS                               │
│    ▸ Matriz de confusión                               │
│    ▸ Curvas ROC-AUC                                    │
│    ▸ Feature importance                                │
│    ▸ Classification report                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 7. DEPLOYMENT                                           │
│    ▸ Exportación del modelo (.pkl)                     │
│    ▸ Aplicación web Streamlit (app.py)                │
│    ▸ Interfaz con 3 pestañas interactivas             │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 RESULTADOS DEL MODELO

### 🎯 Métricas de Rendimiento
- ✅ **Dataset:** ~44.000 registros → filtrado a ~40.000 con 4 partidos
- ✅ **Split:** 90% entrenamiento (~36.000) / 10% test (~4.000)
- ✅ **Modelo:** Gradient Boosting Classifier (optimizado con tuning)
- ✅ **Features:** 34 variables (después de one-hot encoding)
- ✅ **Accuracy final:** **66.63%** (vs 25% del azar) → **Mejora de +41.63 puntos**
- ✅ **Validación:** Cross-validation 10-fold para evitar overfitting

### 🔍 Análisis de Resultados

#### ✅ Lo que funciona bien:
1. **Separación ideológica clara:**
   - El modelo distingue correctamente entre izquierda (PSOE, Sumar) y derecha (PP, VOX)
   - La escala ideológica (ESCIDEOL) es la variable más predictiva

2. **Predicciones realistas:**
   - Accuracy de 66.63% es casi el triple que el azar (25%)
   - El modelo captura patrones sociodemográficos relevantes
   - Buena generalización (evita overfitting)

3. **Robustez:**
   - ~40.000 casos de entrenamiento garantizan estabilidad
   - Validación cruzada confirma que el modelo no sobreajusta

#### ⚠️ Confusiones comunes:
- **PP ↔ VOX:** Ambos partidos de derecha, perfiles electorales similares
- **PSOE ↔ Sumar:** Ambos en la izquierda, votantes con características parecidas

### 📊 Partidos con Mejor Predicción
1. **PSOE** - Mayor representación en muestra (~11.000 casos)
2. **PP** - Segunda fuerza política (~10.000 casos)
3. **VOX** - Tercera fuerza (~9.500 casos)
4. **Sumar** - Izquierda alternativa (~9.500 casos)

### 💡 Interpretación de los Resultados

**Para análisis político:**
- Las variables sociodemográficas son indicadores útiles pero no definitivos del voto
- La auto-ubicación ideológica sigue siendo el mejor predictor
- El voto es un fenómeno complejo que no se reduce solo a características personales

**Para ciencia de datos:**
- 54% de accuracy es un resultado sólido en ciencias sociales (comportamiento humano es difícil de predecir)
- El modelo generaliza bien (no overfitting) gracias a no aplicar tuning agresivo
- Las 34 features capturan la información relevante sin sobrecomplicar el modelo

### 🚨 Limitaciones Identificadas
- ⚠️ **Solo 4 partidos:** Partidos minoritarios (PNV, Bildu, ERC, Junts, CC) no están incluidos por falta de casos suficientes
- ⚠️ **Sesgo del CIS:** Posible sesgo de representatividad en las encuestas telefónicas
- ⚠️ **Contexto temporal:** Datos de 2025, reflejan un momento político específico que puede cambiar
- ⚠️ **Variables limitadas:** Solo sociodemográficas básicas, sin variables de actitudes políticas específicas
- ⚠️ **Accuracy limitado:** 54% significa ~46% de error, apropiado para ciencias sociales pero no predicción perfecta

---

## 🚀 APLICACIÓN WEB STREAMLIT

### 🌐 Características Principales
La aplicación web (`app.py`) ofrece tres modos de uso interactivos:

#### Pestaña 1: 🔮 Predicción Individual
- **Interfaz intuitiva** con selectores para todas las variables
- **Grupos de edad:** 18-29, 30-39, 40-49, 50-59, 60-69, 70+
- **Predicción instantánea** con probabilidad asociada
- **Solo 4 partidos:** PP, PSOE, VOX, Sumar
- **Colores corporativos** de cada partido político
- **Barra de confianza** visual (alta, media, baja)

#### Pestaña 2: 📊 Análisis de Probabilidades
- **Perfil base configurable**
- **Análisis por variable:** Ideología, Grupo de edad, Estudios, Situación laboral, CCAA
- **Gráficos interactivos** con Plotly
- **Tabla de resultados detallados**

#### Pestaña 3: ⚖️ Comparar Perfiles
- **Comparación de 2-5 perfiles** simultáneamente
- **Visualización lado a lado**
- **Tabla comparativa** con probabilidades
Opción 1: Aplicación Web (Recomendado)
```bash
# Ejecutar con Python 3.11
.\run_app.bat

# O manualmente
py -3.11 -m streamlit run app.py
```

### Opción 2: Jupyter Notebook
Ejecutar `predictorvoto.ipynb` paso a paso para:
- Ver el análisis exploratorio completo
- Entrenar el modelo desde cero
- Evaluar métricas detalladas
- Exportar modelo actualizado

### Opción 3: Uso Programático

```python
from pycaret.classification import load_model, predict_model
import pandas as pd

# 1. Cargar el modelo entrenado
modelo = load_model('models/modelo_prediccion_voto')

# 2. Preparar datos del nuevo votante (formato one-hot encoding)
nuevo_votante = pd.DataFrame({
    'ESCIDEOL': [5],               # Ideología centro
    'GRUPO_EDAD_30-39': [1],      # Grupo de edad 30-39
    'CCAA_Madrid': [1],           # Vive en Madrid
    'SEXO_Mujer': [1],            # Mujer
    'ESTUDIOS_Superiores': [1],   # Estudios universitarios
    'SITLAB_Trabaja': [1],        # Empleada
    'TAMUNI_>100.000': [1],       # Municipio grande
    'PARTICIPACIONG_Sí': [1],     # Participó en elecciones previas
    # Resto de columnas en 0 (ver notebook para lista completa)n lugar de 70/30
4. ✅ **Aplicación web:** Streamlit con 3 pestañas interactivas
5. ✅ **Visualizaciones:** Colores corporativos de partidos
6. ✅ **Compatibilidad Python:** Script para ejecutar con Python 3.11

### 🔄 Mejoras Futuras Propuestas
1. 📊 **Variables adicionales:**
   - Actitudes políticas (confianza en instituciones, valoración de líderes)
   - Variables económicas (percepción de situación económica)
   - Datos temporales para captar tendencias mensuales

2. 🧠 **Modelos más avanzados:**
   - Ensemble methods personalizados (stacking, blending)
   - Redes neuronales para captar interacciones complejas

3. 🎯 **Análisis de subgrupos:**
   - Perfiles específicos (jóvenes urbanos, pensionistas rurales, etc.)
   - Análisis por comunidades autónomas

4. ☁️ **Deployment en la nube:**
   - Azure App Service o AWS Elastic Beanstalk
   - API REST con FastAPI para integraciones

---

## 💻 USO DEL MODELO

### Código de Ejemplo

```python
from pycaret.classification import load_model, predict_model
import pandas as pd

# 1. Cargar el modelo entrenado
modelo = load_model('models/modelo_prediccion_voto')

# 2. Preparar datos del nuevo votante
nuevo_votante = pd.DataFrame({
    'EDAD': [35],
    'PAISNAC': [1],  # Nacido en España
    'CCAA_Madrid': [1],  # Vive en Madrid
    'SEXO_Mujer': [1],  # Mujer
    'ESTUDIOS_Superiores': [1],  # Estudios universitarios
    'SITLAB_Trabaja': [1],  # Empleada
    # .predictorvoto.ipynb        # Notebook principal (análisis completo)
├── 🌐 app.py                     # Aplicación web Streamlit
├── ⚙️ run_app.bat                # Script para ejecutar con Python 3.11
├── 📄 README.md                  # Documentación técnica
├── 📋 PRESENTACION.md            # Este documento
├── 📋 INSTALACION.md             # Guía de instalación
├── 📦 requirements.txt           # Dependencias Python
├── ⚙️ config.py                  # Configuración centralizada (opcional)
│
├── data/
│   ├── enero.sav                 # Barómetro CIS enero 2025
│   ├── febrero.sav ... diciembre.sav  # 11 barómetros CIS
│   └── datos_limpios.csv         # Dataset procesado
│
└── models/
    └── modelo_prediccion_voto.pkl # Modelo Gradient Boosting

## 📚 ESTRUCTURA DEL PROYECTO

```
predictordevoto/
│
├── 📓 predictorvoto.ipynb        # Notebook principal (análisis completo con comentarios)
├── 🌐 app.py                     # Aplicación web Streamlit
├── ⚙️ run_app.bat                # Script para ejecutar con Python 3.11
├── 📄 README.md                  # Documentación técnica
├── 📋 PRESENTACION.md            # Este documento (presentación ejecutiva)
├── 📋 INSTALACION.md             # Guía de instalación paso a paso
├── 📦 requirements.txt           # Dependencias Python (PyCaret, Streamlit, etc.)
├── ⚙️ config.py                  # Configuración centralizada (opcional)
│
├── data/
│   ├── enero.sav                 # Barómetro CIS enero 2025
│   ├── febrero.sav ... diciembre.sav  # 11 barómetros CIS (SPSS)
│   └── datos_limpios.csv         # Dataset procesado (~40.000 filas, 34 columnas)
│
└── models/
    └── modelo_prediccion_voto.pkl # Gradient Boosting Classifier (66.63% accuracy)
```

---

## 🎓 LECCIONES APRENDIDAS

### Ventajas de PyCaret
✅ **Automatización completa:** Setup, comparación, tuning, evaluación en pocas líneas  
✅ **Comparación rápida:** 15+ algoritmos evaluados automáticamente  
✅ **Optimización integrada:** Tuning de hiperparámetros con Random Search  
✅ **Visualizaciones profesionales:** Confusion matrix, AUC curves, feature importance  
✅ **Facilidad de deployment:** Exporta modelos completos listos para producción
### Aprendizajes Técnicos
💡 **Grupos de edad:** 6 categorías mejoran interpretabilidad vs edad continua  
💡 **90/10 split:** Suficiente con ~44K registros (vs 70/30 tradicional)  
💡 **One-hot encoding:** Genera ~34 features manejables sin sobrecomplicar  
💡 **Streamlit:** Permite deployment rápido y profesional con interfaz intuitiva  
💡 **Modelo base vs tuning:** A veces menos es más - el modelo sin tuning generalizó mejor  
💡 **Validación cruzada:** CV 10-fold esencial para detectar overfitting

### Limitaciones del Dataset
⚠️ **Sesgo inherente:** Las encuestas del CIS pueden tener sesgos de representatividad  
⚠️ **Momento político:** Datos de 2025, el contexto político cambia constantemente  
⚠️ **Variables limitadas:** Solo sociodemográficas básicas, sin actitudes políticas  
⚠️ **Solo 4 partidos:** Partidos minoritarios excluidos por falta de casos suficientes  
⚠️ **Accuracy limitado:** 54% es bueno para ciencias sociales, pero no es predicción perfecta

### Consideraciones Éticas
- 🔒 **Privacidad:** Los datos del CIS son anónimos y de uso público
- ⚖️ **Sesgo:** El modelo puede perpetuar sesgos existentes en los datos
- 🎯 **Uso responsable:** No debe usarse para manipulación electoral o discriminación
- 📊 **Transparencia:** Limitaciones del modelo deben comunicarse claramente a usuarios finales

---

## 📞 CONTACTO Y RECURSOS

**Autor:** Rubén Díaz Grano de Oro  
**Proyecto:** Predictor de Voto Electoral con Machine Learning  
**Fecha:** Enero 2026  
**Accuracy del modelo:** 66.63% (Gradient Boosting Classifier)  
**Partidos predichos:** PP, PSOE, VOX, Sumar

### 📂 Archivos principales:
- 📓 **Análisis completo:** [predictorvoto.ipynb](predictorvoto.ipynb) (notebook con comentarios exhaustivos)
- 🌐 **Aplicación web:** `.\run_app.bat` o `py -3.11 -m streamlit run app.py`
- 📖 **Documentación:** [README.md](README.md) y [INSTALACION.md](INSTALACION.md)
- 💾 **Modelo entrenado:** `models/modelo_prediccion_voto.pkl`

---

## 📄 LICENCIA

Este proyecto es de **código abierto** y está disponible para uso **educativo y académico**.

---

**¿Preguntas o sugerencias?**

Consulta el notebook completo con todos los comentarios explicativos: [predictorvoto.ipynb](predictorvoto.ipynb)
