# 🚀 Guía de Instalación y Configuración

## ⚠️ Requisito Importante: Versión de Python

**PyCaret requiere Python 3.9, 3.10 o 3.11**

❌ **NO funciona con Python 3.12+**  
✅ **Recomendado: Python 3.11**

---

## 🔧 Instalación Paso a Paso

### Opción 1: Entorno Virtual con `venv` (Recomendado)

```bash
# 1. Verificar versión de Python
python --version
# Debe mostrar: Python 3.9.x, 3.10.x, o 3.11.x

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 4. Actualizar pip
python -m pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt

# 6. Verificar instalación de PyCaret
python -c "import pycaret; print(f'PyCaret {pycaret.__version__} instalado correctamente')"
```

### Opción 2: Anaconda/Miniconda

```bash
# 1. Crear entorno con Python 3.11
conda create -n predictor python=3.11

# 2. Activar el entorno
conda activate predictor

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 🌐 Ejecutar la Aplicación Streamlit

**⚠️ Importante:** La aplicación requiere Python 3.11 (no funciona con 3.12)

### Opción 1: Usar el script batch (Windows)
```bash
.\run_app.bat
```

### Opción 2: Comando manual
```bash
# Si tienes múltiples versiones de Python
py -3.11 -m streamlit run app.py

# Si Python 3.11 es tu versión por defecto
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### 🎯 Características de la App:
- **Pestaña 1:** Predicción individual con interfaz intuitiva
- **Pestaña 2:** Análisis de cómo varían las probabilidades según variables
- **Pestaña 3:** Comparación de múltiples perfiles

---

## 📓 Configuración de Jupyter Notebook

### Opción A: VS Code

```bash
# 1. Instalar extensiones necesarias en VS Code:
#    - Python
#    - Jupyter

# 2. Abrir el notebook
code predictorvoto.ipynb

# 3. Seleccionar el kernel del entorno virtual
#    (En VS Code: Click en "Select Kernel" → Python Environments → venv)
```

### Opción B: Jupyter Lab

```bash
# 1. Instalar Jupyter Lab en el entorno virtual
pip install jupyterlab

# 2. Registrar el kernel
python -m ipykernel install --user --name=predictor --display-name "Python (Predictor Voto)"

# 3. Iniciar Jupyter Lab
jupyter lab

# 4. Abrir predictorvoto.ipynb y seleccionar el kernel "Python (Predictor Voto)"
```

---

## ✅ Verificación de la Instalación

Ejecuta este script para verificar que todo está instalado correctamente:

```python
import sys
print(f"Python version: {sys.version}")

# Verificar librerías principales
try:
    import pycaret
    print(f"✅ PyCaret {pycaret.__version__}")
except ImportError:
    print("❌ PyCaret no instalado")

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__}")
except ImportError:
    print("❌ Pandas no instalado")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError:
    print("❌ NumPy no instalado")

try:
    import seaborn as sns
    print(f"✅ Seaborn {sns.__version__}")
except ImportError:
    print("❌ Seaborn no instalado")

try:
    import pyreadstat
    print(f"✅ Pyreadstat instalado")
except ImportError:
    print("❌ Pyreadstat no instalado")

print("\n🎉 ¡Todo listo para comenzar!")
```

---

## 🐛 Solución de Problemas Comunes

### Error: "PyCaret only supports python 3.9, 3.10, 3.11"

**Problema:** Estás usando Python 3.12 o superior.

**Solución:**
1. Instala Python 3.11 desde [python.org](https://www.python.org/downloads/)
2. Crea un nuevo entorno virtual con Python 3.11
3. Reinstala las dependencias

### Error: "No module named 'seaborn'"

**Solución:**
```bash
pip install seaborn
```

### Error al cargar datos: "FileNotFoundError: xxx.sav"

**Problema:** No se encuentra el archivo de datos.

**Solución:**
1. Verifica que los archivos `data/enero.sav`, `data/febrero.sav`, etc. existen
2. El proyecto necesita 11 archivos .sav (todos los meses de 2025 excepto agosto)
3. Asegúrate de estar en el directorio correcto del proyecto

### Error de memoria al entrenar modelos

**Solución:**
```python
# Reducir el número de modelos a comparar
best_models = compare_models(n_select=3, sort='Accuracy')

# Reducir iteraciones en tuning
tuned_model = tune_model(best_model, n_iter=20, optimize='Accuracy')
```

---

## 📚 Recursos Adicionales

- [Documentación oficial de PyCaret](https://pycaret.org/)
- [Tutorial de clasificación de PyCaret](https://pycaret.gitbook.io/docs/get-started/tutorials)
- [Datos del CIS](http://www.cis.es/)

---

## 🆘 Ayuda

Si encuentras problemas, revisa:
1. ✅ Versión de Python correcta (3.9-3.11)
2. ✅ Entorno virtual activado
3. ✅ Dependencias instaladas (`pip list`)
4. ✅ Kernel correcto seleccionado en Jupyter

---

**¡Disfruta explorando el predictor de voto!** 🗳️
