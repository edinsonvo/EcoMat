# 📊 Taller 2 — Modelo de Salarios de Mincer (Streamlit)

**Curso:** Economía Matemática Aplicada
**Autores:** Edinson Valencia y Santiago Lopez

Aplicación interactiva en **Streamlit** que estima e interpreta la ecuación de salarios de
Mincer usando microdatos de la Gran Encuesta Integrada de Hogares (GEIH), o datos sintéticos
calibrados con parámetros típicos del mercado laboral colombiano cuando no se cuenta con los
archivos del DANE.

## 🚀 Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá en el navegador (por defecto `http://localhost:8501`).

## 🗂️ Estructura del proyecto

```
mincer_streamlit/
├── app.py            # Aplicación Streamlit (todo el proyecto en un solo archivo)
├── requirements.txt  # Dependencias
└── README.md
```

## 🧭 Funcionalidades

- **Fuente de datos configurable** (barra lateral):
  - *Datos sintéticos calibrados*: genera un dataset con estructura GEIH, ajustando en vivo
    los parámetros β₁ (educación), β₂/β₃ (experiencia), brecha de género e informalidad, y el
    tamaño de muestra.
  - *Datos reales GEIH*: permite cargar los CSV oficiales del DANE de **Ocupados** y
    **Características generales, seguridad social en salud y educación**, hace el `merge` por
    las llaves `DIRECTORIO`, `SECUENCIA_P`, `HOGAR`, `ORDEN`, y deja mapear las columnas
    (ingreso, horas, educación, edad, sexo) desde selectores.
- **📊 Análisis Exploratorio (EDA):** distribución del salario y su logaritmo, salario por
  nivel educativo, curva empírica experiencia-salario por sexo, brecha de género y estadísticas
  descriptivas.
- **📐 Modelo de Mincer:** estimación OLS con errores robustos (HC3) vía `statsmodels`,
  coeficientes con IC 95%, tabla interpretable y controles opcionales (género, informalidad).
- **🎯 Pico salarial:** cálculo de X* = −β₂/(2β₃), retorno marginal de la experiencia y curva
  de Mincer estimada.
- **👥 Análisis por grupos:** re-estimación del modelo por sexo (y formalidad si está
  disponible), comparando retornos a la educación y picos salariales.
- **🔬 Diagnósticos MCO:** residuos vs ajustados, Q-Q plot, reales vs predichos, tests de
  Breusch-Pagan y Shapiro-Wilk.
- **📚 Teoría:** marco conceptual de la ecuación de Mincer, capital humano, no linealidad de la
  experiencia y su conexión con el desarrollo económico colombiano.

## 📥 Datos reales GEIH

1. Descarga desde [microdatos.dane.gov.co](https://microdatos.dane.gov.co) el mes de interés.
2. Necesitas el CSV de **Ocupados** y el de **Características generales, seguridad social en
   salud y educación**.
3. Cárgalos en la barra lateral; la app hace el merge y te deja seleccionar las columnas
   correspondientes a ingreso laboral, horas trabajadas, años de educación, edad y sexo.

## ☁️ Despliegue

Este proyecto puede desplegarse gratuitamente en [Streamlit Community Cloud](https://streamlit.io/cloud)
subiendo el repositorio (con `app.py` y `requirements.txt`) y apuntando la app a `app.py`.
