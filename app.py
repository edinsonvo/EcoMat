"""
Taller 2 — Economía Matemática Aplicada
Modelo de Salarios de Mincer con microdatos GEIH
Aplicación interactiva en Streamlit

Autores: Edinson Valencia y Santiago Lopez
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Taller 2 — Modelo de Mincer GEIH",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETA = {
    "azul": "#003366",
    "rojo": "#CC0000",
    "verde": "#2ecc71",
    "naranja": "#e67e22",
    "gris": "#7f8c8d",
}

CUSTOM_CSS = f"""
<style>
    .main-header {{
        background: linear-gradient(90deg, {PALETA['azul']} 0%, #005599 100%);
        padding: 1.6rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
    }}
    .main-header .header-texto h1 {{
        margin: 0;
        font-size: 1.7rem;
    }}
    .main-header .header-texto p {{
        margin: 0.2rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }}
    .main-header .header-logo img {{
        max-height: 72px;
        max-width: 180px;
        object-fit: contain;
        background: white;
        border-radius: 8px;
        padding: 6px 10px;
        flex-shrink: 0;
    }}
    .interp-box, .interp-box * {{
        background: #e8f4fd;
        color: #0b2545 !important;
    }}
    .interp-box {{
        padding: 14px 16px;
        border-radius: 8px;
        border-left: 4px solid #003366;
        margin: 8px 0;
    }}
    .formula-box, .formula-box * {{
        background: #fff3cd;
        color: #4a3b00 !important;
    }}
    .formula-box {{
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #ffc107;
    }}
    .conclusion-box, .conclusion-box * {{
        background: #d4edda;
        color: #14432a !important;
    }}
    .conclusion-box {{
        padding: 14px 16px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 8px 0;
    }}
    div[data-testid="stMetric"] {{
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 10px 14px;
    }}
    div[data-testid="stMetric"] * {{
        color: #1a1a1a !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #4a4a4a !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: #003366 !important;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
 
def _cargar_logo_base64():
    import base64
    from pathlib import Path
 
    directorio = Path(__file__).parent
    candidatos = [
        "logo_unal.png", "logo_unal.jpg", "logo_unal.svg",
        "logo_universidad.png", "logo.png", "logo.jpg", "logo.svg",
    ]
    for nombre in candidatos:
        ruta = directorio / nombre
        if ruta.exists():
            datos = base64.b64encode(ruta.read_bytes()).decode()
            ext = ruta.suffix.lstrip(".").lower()
            mime = "svg+xml" if ext == "svg" else ext
            return f"data:image/{mime};base64,{datos}"
    return None
 
 
LOGO_URI = _cargar_logo_base64()
LOGO_HTML = f'<div class="header-logo"><img src="{LOGO_URI}" alt="Logo institucional"></div>' if LOGO_URI else ""

st.markdown(
    f"""
    <div class="main-header">
        <h1>📊 Modelo de Salarios de Mincer — Microdatos GEIH</h1>
        <p>Taller 2 · Economía Matemática Aplicada</p>
        <p>👥 Autores: <b>Edinson Valencia</b> y <b>Santiago Lopez</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════
# MÓDULO 1 — GENERACIÓN DE DATOS SINTÉTICOS CALIBRADOS (GEIH-Colombia)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generar_datos_sinteticos(n=8000, semilla=2024, beta_educ=0.095, beta_exp=0.042,
                              beta_exp2=-0.0006, delta_mujer=-0.18, delta_informal=-0.22,
                              sigma=0.50):
    """
    Genera un dataset sintético que replica la estructura de la GEIH,
    calibrado con parámetros típicos del mercado laboral colombiano
    (fuente de calibración: DANE-GEIH, BID — Patrinos et al., 2021).

    ln(w_i) = β0 + β1*E_i + β2*X_i + β3*X_i² + δ_mujer*D_mujer + δ_inf*D_inf + ε_i
    """
    rng = np.random.default_rng(semilla)
    N = n

    educ_niveles = list(range(0, 19))
    educ_probs_raw = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.10, 0.10, 0.09,
                       0.08, 0.12, 0.04, 0.05, 0.06, 0.07, 0.07, 0.02, 0.02]
    educ_probs = np.array(educ_probs_raw) / sum(educ_probs_raw)
    educacion = rng.choice(educ_niveles, size=N, p=educ_probs)

    edad_min = educacion + 16
    edad = edad_min + rng.integers(0, 40, size=N)
    edad = np.clip(edad, 15, 70)
    experiencia = np.maximum(0, edad - educacion - 6)

    sexo = rng.choice([1, 2], size=N, p=[0.57, 0.43])          # 1=Hombre, 2=Mujer
    zona = rng.choice([1, 0], size=N, p=[0.80, 0.20])          # 1=Cabecera, 0=Rural

    prob_informal = 0.60 - 0.03 * educacion + 0.01 * (sexo == 2)
    informal = (rng.uniform(0, 1, N) < np.clip(prob_informal, 0.05, 0.95)).astype(int)

    beta_0 = 7.80
    log_salario = (beta_0 + beta_educ * educacion + beta_exp * experiencia
                   + beta_exp2 * experiencia ** 2
                   + delta_mujer * (sexo == 2).astype(int)
                   + delta_informal * informal
                   + rng.normal(0, sigma, N))

    salario_hora = np.exp(log_salario)

    df = pd.DataFrame({
        "DIRECTORIO": np.arange(1, N + 1),
        "P6040": edad,
        "P3271": sexo,
        "P3042": educacion,
        "AREA": zona,
        "P6500": salario_hora,
        "P6800": rng.choice([4, 8, 12, 16, 20, 24, 40, 48], size=N,
                             p=[0.05, 0.08, 0.07, 0.10, 0.10, 0.10, 0.35, 0.15]),
        "INFORMAL": informal,
        "_EDUCACION": educacion,
        "_EXPERIENCIA": experiencia,
    })
    return df


def _leer_csv_geih(bytes_archivo):
    """Lee un CSV de la GEIH de forma tolerante a separador y codificación,
    y normaliza los nombres de columnas (sin espacios ni BOM)."""
    import io

    tiene_bom = bytes_archivo[:3] == b"\xef\xbb\xbf"
    intentos = [
        dict(encoding="utf-8-sig", sep=None, engine="python"),
        dict(encoding="latin1", sep=None, engine="python"),
        dict(encoding="latin1", sep=";"),
        dict(encoding="latin1", sep=","),
    ]
    if not tiene_bom:
        # Si no hay BOM, priorizamos latin1 (codificación típica de los CSV del DANE)
        intentos[0], intentos[1] = intentos[1], intentos[0]

    ultimo_error = None
    for kwargs in intentos:
        try:
            df = pd.read_csv(io.BytesIO(bytes_archivo), low_memory=False, **kwargs)
            if df.shape[1] > 1:
                df.columns = [
                    str(c).strip().upper().replace("\ufeff", "").lstrip("Ï»¿")
                    for c in df.columns
                ]
                return df
        except Exception as e:
            ultimo_error = e
            continue
    raise ValueError(f"No fue posible leer el archivo CSV. Detalle: {ultimo_error}")


@st.cache_data(show_spinner=False)
def leer_archivos_geih(bytes_ocupados, bytes_caracteristicas):
    """Lee ambos módulos GEIH y devuelve los DataFrames crudos (sin combinar)."""
    df_ocup = _leer_csv_geih(bytes_ocupados)
    df_carac = _leer_csv_geih(bytes_caracteristicas)
    return df_ocup, df_carac


def combinar_geih(df_ocup, df_carac, llaves):
    """Combina (left-join) los módulos Ocupados + Características por las llaves indicadas."""
    llaves_validas = [k for k in llaves if k in df_ocup.columns and k in df_carac.columns]
    if not llaves_validas:
        raise ValueError(
            "Ninguna de las llaves seleccionadas está presente en ambos archivos. "
            "Revisa los nombres de columnas disponibles más abajo y elige llaves comunes."
        )
    df_merged = pd.merge(df_ocup, df_carac, on=llaves_validas, how="left",
                          suffixes=("_ocup", "_carac"))
    return df_merged, llaves_validas


def preparar_datos_mincer(df_raw, usar_reales, col_salario=None, col_horas=None,
                           col_educ=None, col_edad=None, col_sexo=None):
    """
    Construye las variables del modelo de Mincer:
    1. Salario/hora   2. ln(salario)   3. Experiencia potencial = max(0, edad-educ-6)
    4. Dummies de subgrupos   5. Filtros de calidad y outliers (percentiles 1-99)
    """
    df = df_raw.copy()

    if usar_reales:
        df["horas_mes"] = pd.to_numeric(df[col_horas], errors="coerce") * 4.33
        df["salario_hora"] = pd.to_numeric(df[col_salario], errors="coerce") / df["horas_mes"]
        df["educacion"] = pd.to_numeric(df[col_educ], errors="coerce")
        df["edad"] = pd.to_numeric(df[col_edad], errors="coerce")
        df["sexo"] = pd.to_numeric(df[col_sexo], errors="coerce")
        df["INFORMAL"] = df["INFORMAL"] if "INFORMAL" in df.columns else np.nan
    else:
        df["salario_hora"] = df["P6500"]
        df["educacion"] = df["_EDUCACION"]
        df["edad"] = df["P6040"]
        df["sexo"] = df["P3271"]

    df["ln_salario"] = np.log(df["salario_hora"])
    df["experiencia"] = np.maximum(0, df["edad"] - df["educacion"] - 6)
    df["experiencia2"] = df["experiencia"] ** 2
    df["mujer"] = (df["sexo"] == 2).astype(int)

    mask = (
        (df["salario_hora"] > 0)
        & (df["educacion"].between(0, 25))
        & (df["edad"].between(15, 65))
        & (df["experiencia"] >= 0)
        & (np.isfinite(df["ln_salario"]))
    )
    df_clean = df[mask].copy()

    p1, p99 = df_clean["salario_hora"].quantile([0.01, 0.99])
    df_clean = df_clean[(df_clean["salario_hora"] >= p1) & (df_clean["salario_hora"] <= p99)]

    df_clean["nivel_educ"] = pd.cut(
        df_clean["educacion"],
        bins=[-1, 5, 10, 11, 15, 16, 25],
        labels=["Primaria", "Sec. baja", "Bachillerato", "Técnico", "Universitario", "Posgrado"],
    )
    return df_clean, df_raw.shape[0] - df_clean.shape[0]


def estimar_mincer(df, controles=()):
    """Estima el modelo de Mincer vía OLS con errores robustos HC3."""
    vars_base = ["educacion", "experiencia", "experiencia2"]
    formula = "ln_salario ~ " + " + ".join(list(vars_base) + list(controles))
    modelo = smf.ols(formula, data=df).fit(cov_type="HC3")
    return modelo


def pico_salarial(modelo):
    b = modelo.params
    b2, b3 = b.get("experiencia", np.nan), b.get("experiencia2", np.nan)
    if pd.isna(b3) or b3 >= 0:
        return np.nan
    return -b2 / (2 * b3)


# ══════════════════════════════════════════════════════════════════════════
# BARRA LATERAL — FUENTE DE DATOS Y PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configuración de datos")
    fuente = st.radio(
        "Fuente de datos",
        ["Datos sintéticos calibrados (GEIH-Colombia)", "Cargar datos reales GEIH (DANE)"],
        index=0,
    )
    usar_reales = fuente.startswith("Cargar")

    st.markdown("---")

    if not usar_reales:
        st.subheader("🧪 Parámetros del generador")
        st.caption("Ecuación generadora: ln(w) = β₀ + β₁E + β₂X + β₃X² + δ_mujer·D + δ_inf·D + ε")
        n_obs = st.slider("Tamaño de muestra (N)", 1000, 15000, 8000, step=500)
        beta_educ = st.slider("Retorno educación (β₁)", 0.02, 0.20, 0.095, 0.005)
        beta_exp = st.slider("Coef. experiencia (β₂)", 0.01, 0.08, 0.042, 0.002)
        beta_exp2 = st.slider("Coef. experiencia² (β₃ × 1000)", -1.5, -0.1, -0.6, 0.05) / 1000
        delta_mujer = st.slider("Brecha de género (δ_mujer)", -0.40, 0.0, -0.18, 0.01)
        delta_inf = st.slider("Penalización informalidad (δ_inf)", -0.40, 0.0, -0.22, 0.01)
        semilla = st.number_input("Semilla aleatoria", value=2024, step=1)

        df_raw = generar_datos_sinteticos(
            n=n_obs, semilla=semilla, beta_educ=beta_educ, beta_exp=beta_exp,
            beta_exp2=beta_exp2, delta_mujer=delta_mujer, delta_informal=delta_inf,
        )
        df, n_eliminados = preparar_datos_mincer(df_raw, usar_reales=False)
    else:
        st.subheader("📥 Cargar archivos GEIH (DANE)")
        st.caption("Se necesitan los CSV de **Ocupados** y de **Características generales, "
                   "seguridad social en salud y educación** del mismo mes de la GEIH.")
        f_ocup = st.file_uploader("CSV — Ocupados", type=["csv"])
        f_carac = st.file_uploader("CSV — Características generales", type=["csv"])

        if f_ocup is not None and f_carac is not None:
            try:
                df_ocup_raw, df_carac_raw = leer_archivos_geih(f_ocup.getvalue(), f_carac.getvalue())
            except Exception as e:
                st.error(f"Error al leer los archivos: {e}")
                st.stop()

            cols_ocup = set(df_ocup_raw.columns)
            cols_carac = set(df_carac_raw.columns)
            comunes = sorted(cols_ocup & cols_carac)
            llaves_default = [k for k in ["DIRECTORIO", "SECUENCIA_P", "SECUENCIA_ENCUESTA",
                                           "HOGAR", "ORDEN"] if k in comunes]

            with st.expander(f"🔎 Columnas detectadas — Ocupados ({len(cols_ocup)}) / "
                              f"Características ({len(cols_carac)})"):
                cc1, cc2 = st.columns(2)
                cc1.caption("Ocupados")
                cc1.code("\n".join(sorted(cols_ocup)), language=None)
                cc2.caption("Características")
                cc2.code("\n".join(sorted(cols_carac)), language=None)

            if not comunes:
                st.error(
                    "Los dos archivos no comparten ninguna columna con el mismo nombre, por lo "
                    "que no se puede hacer el merge automático. Revisa en el panel de arriba los "
                    "nombres exactos de columnas de cada archivo (por ejemplo, en algunos meses de "
                    "la GEIH la llave de persona se llama `SECUENCIA_P` y en otros `SECUENCIA_ENCUESTA`), "
                    "y verifica que ambos CSV correspondan al mismo mes/año de la encuesta."
                )
                st.stop()

            llaves = st.multiselect(
                "Llaves de merge (columnas presentes en ambos archivos)",
                options=comunes,
                default=llaves_default if llaves_default else comunes[:1],
                help="Identifican de forma única cada hogar/persona para combinar los dos módulos.",
            )
            if not llaves:
                st.warning("Selecciona al menos una llave de merge para continuar.")
                st.stop()

            try:
                df_merged, llaves_usadas = combinar_geih(df_ocup_raw, df_carac_raw, llaves)
                st.success(f"Merge exitoso ({', '.join(llaves_usadas)}): "
                           f"{df_merged.shape[0]:,} filas × {df_merged.shape[1]} columnas")

                cols = list(df_merged.columns)

                def _idx(nombre_pref, opciones):
                    return opciones.index(nombre_pref) if nombre_pref in opciones else 0

                st.markdown("**Mapeo de variables** (según diccionario GEIH):")
                col_salario = st.selectbox("Ingreso laboral (ej. INGLABO / P6500)", cols,
                                            index=_idx("P6500", cols))
                col_horas = st.selectbox("Horas trabajadas (ej. P6800)", cols,
                                          index=_idx("P6800", cols))
                col_educ = st.selectbox("Años de educación (ej. P3042 / P6210)", cols,
                                         index=_idx("P3042", cols))
                col_edad = st.selectbox("Edad (ej. P6040)", cols,
                                         index=_idx("P6040", cols))
                col_sexo = st.selectbox("Sexo (ej. P3271: 1=Hombre, 2=Mujer)", cols,
                                         index=_idx("P3271", cols))

                df, n_eliminados = preparar_datos_mincer(
                    df_merged, usar_reales=True, col_salario=col_salario, col_horas=col_horas,
                    col_educ=col_educ, col_edad=col_edad, col_sexo=col_sexo,
                )
                if len(df) == 0:
                    st.error("Tras la limpieza no quedaron observaciones válidas. Revisa el "
                              "mapeo de columnas (ingreso, horas, educación, edad, sexo).")
                    st.stop()
            except Exception as e:
                st.error(f"Error al procesar los datos: {e}")
                st.stop()
        else:
            st.info("⬆️ Carga ambos archivos CSV para continuar. Mientras tanto, se muestra "
                    "una vista con datos sintéticos de demostración.")
            df_raw = generar_datos_sinteticos()
            df, n_eliminados = preparar_datos_mincer(df_raw, usar_reales=False)

    st.markdown("---")
    st.subheader("🧮 Controles del modelo")
    controles_sel = st.multiselect(
        "Variables de control adicionales",
        options=["mujer", "INFORMAL"],
        default=[],
        help="Se añaden a la especificación clásica de Mincer (educación, experiencia, experiencia²).",
    )
    controles_validos = [c for c in controles_sel if c in df.columns and df[c].notna().any()]

    st.markdown("---")
    st.caption(f"✅ Observaciones limpias: **{len(df):,}**  ·  Eliminadas en limpieza: {n_eliminados:,}")

# Modelo principal (reutilizado en varias pestañas)
modelo = estimar_mincer(df, controles=controles_validos)
X_star = pico_salarial(modelo)

# ══════════════════════════════════════════════════════════════════════════
# PESTAÑAS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════

tab_eda, tab_modelo, tab_pico, tab_grupos, tab_diag, tab_teoria = st.tabs(
    ["📊 Análisis Exploratorio", "📐 Modelo de Mincer", "🎯 Pico Salarial",
     "👥 Análisis por Grupos", "🔬 Diagnósticos MCO", "📚 Teoría"]
)

# ────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown(
        """
        <div class="interp-box">
        <b>📐 Justificación matemática:</b> el modelo de Mincer usa ln(w) como variable dependiente
        porque los salarios tienen distribución log-normal, no normal. La especificación
        semi-logarítmica permite interpretar β₁ directamente como la tasa de retorno porcentual
        de un año adicional de educación.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Salario/hora promedio", f"${df['salario_hora'].mean():,.0f} COP")
    c2.metric("🎓 Escolaridad promedio", f"{df['educacion'].mean():.1f} años")
    c3.metric("👩 % Mujeres en la muestra", f"{(df['mujer'].mean()*100):.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="salario_hora", nbins=60, color_discrete_sequence=[PALETA["rojo"]])
        fig.update_layout(title="Distribución del Salario/hora (sesgada a la derecha)",
                           xaxis_title="Pesos COP / hora", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="ln_salario", nbins=60, histnorm="probability density",
                            color_discrete_sequence=[PALETA["azul"]])
        mu, sd = df["ln_salario"].mean(), df["ln_salario"].std()
        xs = np.linspace(df["ln_salario"].min(), df["ln_salario"].max(), 200)
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mu, sd), mode="lines",
                                  name="Normal teórica", line=dict(color=PALETA["naranja"], width=2.5)))
        fig.update_layout(title="Distribución de ln(Salario/hora) — aprox. normal",
                           xaxis_title="ln(Pesos COP / hora)", yaxis_title="Densidad")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        sal_educ = df.groupby("educacion")["ln_salario"].mean().reset_index()
        fig = px.line(sal_educ, x="educacion", y="ln_salario", markers=True,
                       color_discrete_sequence=[PALETA["verde"]])
        fig.add_vline(x=11, line_dash="dash", line_color=PALETA["rojo"], annotation_text="Bachillerato")
        fig.add_vline(x=16, line_dash="dash", line_color=PALETA["naranja"], annotation_text="Universidad")
        fig.update_layout(title="ln(Salario) promedio por años de educación",
                           xaxis_title="Años de educación", yaxis_title="ln(Salario/hora) promedio")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(df, x="nivel_educ", y="ln_salario", color="nivel_educ",
                     category_orders={"nivel_educ": ["Primaria", "Sec. baja", "Bachillerato",
                                                      "Técnico", "Universitario", "Posgrado"]})
        fig.update_layout(title="Distribución salarial por nivel educativo", showlegend=False,
                           xaxis_title="Nivel educativo", yaxis_title="ln(Salario/hora)")
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        fig = go.Figure()
        for genero, color, etiq in [(0, PALETA["azul"], "Hombres"), (1, PALETA["rojo"], "Mujeres")]:
            perfil = df[df["mujer"] == genero].groupby("experiencia")["ln_salario"].mean().reset_index()
            perfil = perfil[perfil["experiencia"] <= 40]
            fig.add_trace(go.Scatter(x=perfil["experiencia"], y=perfil["ln_salario"],
                                      mode="lines", name=etiq, line=dict(color=color, width=2.5)))
        fig.update_layout(title="Perfil Experiencia-Salario por Sexo (curva de Mincer empírica)",
                           xaxis_title="Experiencia potencial (años)", yaxis_title="ln(Salario/hora)")
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        if df["mujer"].nunique() > 1:
            brecha = df.groupby(["nivel_educ", "mujer"], observed=True)["ln_salario"].mean().unstack()
            if 0 in brecha.columns and 1 in brecha.columns:
                brecha["brecha"] = brecha[0] - brecha[1]
                fig = px.bar(brecha.reset_index(), x="brecha", y="nivel_educ", orientation="h",
                             color=brecha["brecha"] > 0,
                             color_discrete_map={True: PALETA["azul"], False: PALETA["verde"]})
                fig.update_layout(title="Brecha de género salarial (ln-salario H − M)",
                                   xaxis_title="Diferencia en ln(salario)", yaxis_title="",
                                   showlegend=False)
                fig.add_vline(x=0, line_color="black")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay variación de género suficiente en la muestra para calcular la brecha.")

    st.subheader("📋 Estadísticas descriptivas")
    vars_desc = ["ln_salario", "salario_hora", "educacion", "experiencia", "edad", "mujer"]
    st.dataframe(df[vars_desc].describe().T.round(3), use_container_width=True)

# ────────────────────────────────────────────────────────────────────────
# TAB 2 — MODELO DE MINCER
# ────────────────────────────────────────────────────────────────────────
with tab_modelo:
    st.markdown(
        """
        <div class="formula-box">
        Especificación semi-logarítmica de Mincer, estimada por <b>Mínimos Cuadrados
        Ordinarios (MCO)</b> con errores estándar robustos a heterocedasticidad (HC3).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"\ln(w_i) = \beta_0 + \beta_1 E_i + \beta_2 X_i + \beta_3 X_i^2 + \varepsilon_i")
    st.caption("w = salario/hora · E = años de educación · X = experiencia potencial")

    col_izq, col_der = st.columns([1, 1.4])
    with col_izq:
        st.subheader("📊 Coeficientes con IC 95%")
        ci = modelo.conf_int()
        coef_plot = pd.DataFrame({
            "coef": modelo.params[1:], "lower": ci.iloc[1:, 0], "upper": ci.iloc[1:, 1]
        })
        nombres_bonitos = {"educacion": "Educación (β₁)", "experiencia": "Experiencia (β₂)",
                            "experiencia2": "Experiencia² (β₃)", "mujer": "Mujer (δ)",
                            "INFORMAL": "Informal (δ)"}
        coef_plot.index = [nombres_bonitos.get(i, i) for i in coef_plot.index]
        fig = go.Figure()
        for i, row in coef_plot.iterrows():
            color = PALETA["verde"] if row["coef"] > 0 else PALETA["rojo"]
            fig.add_trace(go.Scatter(x=[row["lower"], row["upper"]], y=[i, i], mode="lines",
                                      line=dict(color="black", width=2), showlegend=False))
            fig.add_trace(go.Scatter(x=[row["coef"]], y=[i], mode="markers",
                                      marker=dict(color=color, size=12), showlegend=False))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Coeficientes estimados", xaxis_title="Valor del coeficiente")
        st.plotly_chart(fig, use_container_width=True)

        st.metric("R² ajustado", f"{modelo.rsquared_adj*100:.1f}%",
                   help=f"N = {int(modelo.nobs):,} observaciones")

    with col_der:
        st.subheader("📄 Resultados MCO — Tabla interpretable")
        rc = pd.DataFrame({
            "Coeficiente": modelo.params, "Error Std (HC3)": modelo.bse,
            "t-valor": modelo.tvalues, "p-valor": modelo.pvalues,
        })
        rc["Significancia"] = np.where(rc["p-valor"] < 0.001, "***",
                                np.where(rc["p-valor"] < 0.01, "**",
                                np.where(rc["p-valor"] < 0.05, "*", "")))
        nombres_interp = {"Intercept": "β₀ — Intercepto", "educacion": "β₁ — Años educación",
                           "experiencia": "β₂ — Experiencia (lineal)",
                           "experiencia2": "β₃ — Experiencia² (cuadrático)",
                           "mujer": "δ_mujer — Dummy mujer", "INFORMAL": "δ_inf — Dummy informal"}
        rc.index = [nombres_interp.get(i, i) for i in rc.index]
        st.dataframe(rc.round(5), use_container_width=True)
        st.caption("*** p<0.001 · ** p<0.01 · * p<0.05 — Errores estándar robustos HC3")

        b = modelo.params
        b1 = b.get("educacion", np.nan)
        st.markdown(
            f"""
            <div class="interp-box">
            <b>β₁ (Educación) = {b1:.4f}</b><br>
            → Cada año adicional de educación incrementa el salario en aproximadamente
            <b>{b1*100:.2f}%</b> (interpretación exacta: {(np.exp(b1)-1)*100:.2f}%).
            </div>
            <div class="conclusion-box">
            <b>R² ajustado = {modelo.rsquared_adj:.3f}</b><br>
            El modelo explica el {modelo.rsquared_adj*100:.1f}% de la varianza salarial observada.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ────────────────────────────────────────────────────────────────────────
# TAB 3 — PICO SALARIAL
# ────────────────────────────────────────────────────────────────────────
with tab_pico:
    st.markdown(
        """
        <div class="formula-box">
        La condición de primer orden identifica el punto donde el crecimiento salarial
        respecto a la experiencia se detiene (máximo de la curva de Mincer).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"\frac{\partial \ln(w)}{\partial X} = \beta_2 + 2\beta_3 X = 0 \;\;\Rightarrow\;\; X^{*} = -\frac{\beta_2}{2\beta_3}")
    st.caption("La segunda derivada 2β₃ < 0 confirma que se trata de un máximo "
               "(función cóncava en la experiencia).")

    b = modelo.params
    b2, b3 = b.get("experiencia", np.nan), b.get("experiencia2", np.nan)
    rm5 = (b2 + 2 * b3 * 5) * 100 if pd.notna(b3) else np.nan
    rm20 = (b2 + 2 * b3 * 20) * 100 if pd.notna(b3) else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Pico salarial (X*)", f"{X_star:.1f} años" if pd.notna(X_star) else "N/A")
    c2.metric("📈 Retorno marginal a 5 años exp.", f"{rm5:.2f}%" if pd.notna(rm5) else "N/A")
    c3.metric("📉 Retorno marginal a 20 años exp.", f"{rm20:.2f}%" if pd.notna(rm20) else "N/A")

    col1, col2 = st.columns([1.4, 1])
    with col1:
        edu_med = df["educacion"].median()
        exp_seq = np.linspace(0, 50, 200)
        b0 = b.get("Intercept", 0)
        b1 = b.get("educacion", 0)
        pred_ln = b0 + b1 * edu_med + b2 * exp_seq + b3 * exp_seq ** 2
        pred_sal = np.exp(pred_ln)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=exp_seq, y=pred_sal, mode="lines",
                                  line=dict(color=PALETA["azul"], width=3), name="Predicción"))
        if pd.notna(X_star):
            sal_pico = np.exp(b0 + b1 * edu_med + b2 * X_star + b3 * X_star ** 2)
            fig.add_vline(x=X_star, line_dash="dash", line_color=PALETA["rojo"])
            fig.add_trace(go.Scatter(x=[X_star], y=[sal_pico], mode="markers+text",
                                      marker=dict(color=PALETA["rojo"], size=12),
                                      text=[f"X*={X_star:.1f} años"], textposition="top right",
                                      name="Pico salarial"))
        fig.update_layout(title=f"Curva de Mincer estimada (educación fija = {edu_med:.0f} años)",
                           xaxis_title="Años de experiencia", yaxis_title="Salario/hora predicho (COP)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ret_mg = (b2 + 2 * b3 * exp_seq) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=exp_seq, y=ret_mg, mode="lines",
                                  line=dict(color=PALETA["verde"], width=2.5), name="Retorno marginal"))
        fig.add_hline(y=0, line_color="black")
        if pd.notna(X_star):
            fig.add_vline(x=X_star, line_dash="dash", line_color=PALETA["rojo"])
        fig.update_layout(title="∂ln(w)/∂X = β₂ + 2β₃X",
                           xaxis_title="Experiencia (años)", yaxis_title="Retorno marginal (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
            <div class="interp-box">
            <b>Fórmula:</b> ∂ln(w)/∂X = β₂ + 2β₃X<br>
            Cuando X > X*, el retorno marginal de la experiencia se vuelve negativo.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ────────────────────────────────────────────────────────────────────────
# TAB 4 — ANÁLISIS POR GRUPOS
# ────────────────────────────────────────────────────────────────────────
with tab_grupos:
    st.markdown(
        """
        <div class="interp-box">
        La ecuación de Mincer puede estimarse por subgrupos para identificar heterogeneidad
        en los retornos al capital humano. En Colombia, los retornos a la educación difieren
        significativamente por género y tipo de contrato (formal/informal).
        </div>
        """,
        unsafe_allow_html=True,
    )

    grupos = {"Hombres": df[df["mujer"] == 0], "Mujeres": df[df["mujer"] == 1]}
    if "INFORMAL" in df.columns and df["INFORMAL"].notna().any():
        grupos["Formales"] = df[df["INFORMAL"] == 0]
        grupos["Informales"] = df[df["INFORMAL"] == 1]

    modelos_grupos = {nm: estimar_mincer(g) for nm, g in grupos.items() if len(g) > 100}

    color_map = {"Hombres": PALETA["azul"], "Mujeres": PALETA["rojo"],
                 "Formales": PALETA["verde"], "Informales": PALETA["naranja"]}

    col1, col2 = st.columns(2)
    with col1:
        edu_med = df["educacion"].median()
        exp_seq = np.linspace(0, 50, 200)
        fig = go.Figure()
        for nm in ["Hombres", "Mujeres"]:
            if nm not in modelos_grupos:
                continue
            bm = modelos_grupos[nm].params
            pred = (bm.get("Intercept", 0) + bm.get("educacion", 0) * edu_med
                    + bm.get("experiencia", 0) * exp_seq + bm.get("experiencia2", 0) * exp_seq ** 2)
            fig.add_trace(go.Scatter(x=exp_seq, y=pred, mode="lines", name=nm,
                                      line=dict(color=color_map[nm], width=2.5)))
        fig.update_layout(title="Curvas de Mincer por sexo",
                           xaxis_title="Experiencia (años)", yaxis_title="ln(Salario) predicho")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        retornos = {nm: m.params.get("educacion", np.nan) * 100 for nm, m in modelos_grupos.items()}
        df_ret = pd.DataFrame({"Grupo": list(retornos.keys()), "Retorno": list(retornos.values())})
        fig = px.bar(df_ret, x="Grupo", y="Retorno", color="Grupo",
                     color_discrete_map=color_map, text_auto=".2f")
        fig.update_layout(title="Retorno a la educación por grupo (%)", showlegend=False,
                           yaxis_title="% aumento salarial por año adicional")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Tabla comparativa de modelos por grupo")
    filas = []
    for nm, m in modelos_grupos.items():
        bm = m.params
        xst = pico_salarial(m)
        filas.append({
            "Grupo": nm, "N": int(m.nobs),
            "β₁ Educación (%)": round(bm.get("educacion", np.nan) * 100, 2),
            "β₂ Experiencia": round(bm.get("experiencia", np.nan), 4),
            "β₃ Exp²": round(bm.get("experiencia2", np.nan), 6),
            "X* Pico (años)": round(xst, 1) if pd.notna(xst) else np.nan,
            "R² ajustado": round(m.rsquared_adj, 4),
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────────
# TAB 5 — DIAGNÓSTICOS MCO
# ────────────────────────────────────────────────────────────────────────
with tab_diag:
    st.markdown(
        """
        <div class="interp-box">
        Los supuestos clásicos de MCO (Gauss-Markov) son fundamentales para la validez de los
        estimadores. Evaluamos: normalidad de residuos, homocedasticidad y bondad de ajuste.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fitted = modelo.fittedvalues
    resid = modelo.resid
    muestra_idx = np.random.default_rng(0).choice(len(fitted), size=min(2000, len(fitted)), replace=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(x=fitted.iloc[muestra_idx], y=resid.iloc[muestra_idx], opacity=0.35,
                          color_discrete_sequence=[PALETA["azul"]])
        fig.add_hline(y=0, line_color=PALETA["rojo"], line_width=2)
        fig.update_layout(title="Residuos vs Valores Ajustados (homocedasticidad)",
                           xaxis_title="Valores ajustados", yaxis_title="Residuos")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        qq = stats.probplot(resid, dist="norm")
        teor, real = qq[0][0], qq[0][1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=teor, y=real, mode="markers",
                                  marker=dict(color=PALETA["azul"], opacity=0.4, size=5)))
        fig.add_trace(go.Scatter(x=teor, y=teor * qq[1][0] + qq[1][1], mode="lines",
                                  line=dict(color=PALETA["rojo"], width=2.5)))
        fig.update_layout(title="Q-Q Plot de residuos (normalidad)", showlegend=False,
                           xaxis_title="Cuantiles teóricos normales", yaxis_title="Cuantiles de los residuos")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        real_y = modelo.model.endog
        fig = px.scatter(x=fitted.iloc[muestra_idx], y=real_y[muestra_idx], opacity=0.35,
                          color_discrete_sequence=[PALETA["azul"]])
        lims = [min(fitted.min(), real_y.min()), max(fitted.max(), real_y.max())]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                  line=dict(color=PALETA["rojo"], width=2)))
        fig.update_layout(title=f"Reales vs Predichos (R²={modelo.rsquared:.3f})", showlegend=False,
                           xaxis_title="ln(Salario) predicho", yaxis_title="ln(Salario) real")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.subheader("🧪 Tests estadísticos")
        try:
            bp_stat, bp_pval, _, _ = het_breuschpagan(resid, modelo.model.exog)
        except Exception:
            bp_stat, bp_pval = np.nan, np.nan
        res_sample = np.random.default_rng(0).choice(resid, size=min(5000, len(resid)), replace=False)
        sw_stat, sw_pval = stats.shapiro(res_sample)

        tabla_tests = pd.DataFrame({
            "Test": ["Breusch-Pagan (homocedasticidad)", "Shapiro-Wilk (normalidad residuos)"],
            "Estadístico": [round(bp_stat, 3), round(sw_stat, 4)],
            "p-valor": [round(bp_pval, 4), round(sw_pval, 4)],
            "Conclusión": [
                "✅ Homocedasticidad" if bp_pval > 0.05 else "⚠️ Heterocedasticidad detectada",
                "✅ Normalidad" if sw_pval > 0.05 else "⚠️ No normalidad (esperable con N grande)",
            ],
        })
        st.dataframe(tabla_tests, use_container_width=True, hide_index=True)
        st.caption("H₀ Breusch-Pagan: varianza constante de los errores. "
                   "H₀ Shapiro-Wilk: los residuos siguen una distribución normal.")

# ────────────────────────────────────────────────────────────────────────
# TAB 6 — TEORÍA
# ────────────────────────────────────────────────────────────────────────
with tab_teoria:
    st.subheader("📚 Marco teórico — Economía Matemática")

    st.markdown("#### 1. La ecuación de Mincer y el capital humano")
    st.markdown(
        """
        **Jacob Mincer (1974)** propuso que el logaritmo del salario puede modelarse como una
        función lineal de la educación y cuadrática de la experiencia. Esta especificación se
        deriva de la **teoría del capital humano** (Becker, Mincer): las personas invierten en
        educación y experiencia esperando un retorno futuro, análogo al retorno de un activo
        financiero.
        """
    )
    st.latex(r"\ln(w_i) = \beta_0 + \beta_1 E_i + \beta_2 X_i + \beta_3 X_i^2 + \varepsilon_i")

    st.markdown("#### 2. Crecimiento exponencial del salario")
    st.markdown(
        """
        Al despejar w se obtiene una función exponencial, lo que implica que la educación y la
        experiencia generan incrementos **porcentuales**, no absolutos, del salario:
        """
    )
    st.latex(r"w = e^{\beta_0 + \beta_1 S + \beta_2 X + \beta_3 X^2}")
    st.markdown(
        "Por ejemplo, si β₁ = 0.08, un año adicional de educación se asocia con un aumento "
        "salarial cercano al 8%, consistente con la teoría del capital humano de Gary Becker "
        "y Jacob Mincer."
    )

    st.markdown("#### 3. Comportamiento no lineal de la experiencia")
    st.markdown(
        """
        El término cuadrático X² permite modelar **retornos decrecientes**: al inicio de la vida
        laboral los salarios crecen rápido, luego más lento y finalmente se estabilizan,
        generando una trayectoria salarial cóncava, ampliamente documentada en mercados
        laborales reales.
        """
    )
    st.latex(r"\frac{\partial \ln(w)}{\partial X} = \beta_2 + 2\beta_3 X")
    st.markdown(
        "Si β₃ < 0, el retorno marginal disminuye con el tiempo, reflejando agotamiento del "
        "aprendizaje, estabilización profesional o límites de productividad marginal."
    )

    st.markdown("#### 4. Punto máximo salarial")
    st.latex(r"X^{*} = -\frac{\beta_2}{2\beta_3}")
    st.markdown(
        "Representa la experiencia aproximada en la que el crecimiento salarial deja de "
        "aumentar. La condición de segundo orden (2β₃ < 0) confirma que se trata de un máximo."
    )

    st.markdown("#### 5. Interpretación de los coeficientes")
    st.markdown(
        """
        Debido al uso del logaritmo, los coeficientes se interpretan como **cambios
        porcentuales aproximados**, no como cambios absolutos. Esto hace del modelo una
        herramienta especialmente útil para analizar desigualdad salarial, retornos educativos,
        informalidad, productividad laboral y movilidad social — temas centrales en el análisis
        del desarrollo económico colombiano.
        """
    )

    st.markdown("#### 6. Fuente de datos: la GEIH")
    st.markdown(
        """
        La **Gran Encuesta Integrada de Hogares (GEIH)**, ejecutada por el DANE, es la principal
        fuente de microdatos laborales de Colombia. Este taller combina los módulos de
        **Ocupados** (ingresos y variables laborales) y de **Características generales,
        seguridad social en salud y educación** (edad, sexo, nivel educativo), mediante un
        merge por las llaves `DIRECTORIO`, `SECUENCIA_P`, `HOGAR` y `ORDEN`.
        """
    )

    st.markdown("#### 7. Aplicación al desarrollo económico colombiano")
    st.markdown(
        """
        El modelo de Mincer, más allá de estimar retornos individuales, permite conectar la
        matemática económica con debates de **desarrollo y competitividad regional**: los
        retornos heterogéneos por género, formalidad y zona urbano/rural evidencian brechas
        estructurales del mercado laboral que son centrales en la discusión sobre productividad
        y capital humano en Colombia.
        """
    )

st.markdown("---")
st.caption(
    "Taller 2 — Economía Matemática Aplicada · Modelo de Salarios de Mincer con microdatos GEIH · "
    "Autores: Edinson Valencia y Santiago Lopez"
)
