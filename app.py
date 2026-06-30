import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# 1. Configuración de la página (DEBE SER LA PRIMERA LÍNEA DE STREAMLIT)
st.set_page_config(
    page_title="IntegralStudio | Área entre Curvas",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inyección de CSS Limpio y Seguro
st.markdown("""
    <style>
        .stApp { background-color: #0b0f19 !important; }
        section[data-testid="stSidebar"] {
            background-color: #0d1326 !important;
            border-right: 1px solid #1e293b;
        }
        .crypto-box {
            background: linear-gradient(135deg, #0e1726 0%, #090d16 100%);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #10b981;
            margin-bottom: 20px;
        }
        .math-box {
            background: linear-gradient(135deg, #0e1726 0%, #090d16 100%);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #1e293b;
            margin-bottom: 20px;
        }
        .pro-badge {
            color: #10b981; font-weight: bold; background: rgba(16, 185, 129, 0.1);
            padding: 2px 8px; border-radius: 6px;
        }
        .metric-title { color: #94a3b8; font-size: 0.95rem; font-weight: 600; }
        .metric-value { color: #10b981; font-size: 2.5rem; font-weight: 800; }
        .metric-status { color: #059669; font-size: 0.85rem; }
        .team-box {
            margin-top: 50px; padding: 12px; border-radius: 8px;
            background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .team-title { color: #64748b; font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; }
        .team-member { color: #94a3b8; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

# 3. Menú de Configuración Lateral (Sidebar)
with st.sidebar:
    st.markdown("## ⚡ **IntegralStudio**")
    st.markdown("### **Configuración del Modelo**")
    st.markdown("---")
    
    eje_sel = st.radio("Variable de Integración:", ("Respecto a x", "Respecto a y"), horizontal=True)
    var_str = 'x' if eje_sel == "Respecto a x" else 'y'
    
    f_input = st.text_input(f"Función Superior f({var_str}):", value=f"{var_str} + 2")
    g_input = st.text_input(f"Función Inferior g({var_str}):", value=f"{var_str}**2")
    
    st.markdown("#### Intervalos de Evaluación [a, b]")
    st.caption("Déjalos vacíos para calcular la intersección automáticamente.")
    lim_a = st.number_input("Límite inferior (a):", value=None, placeholder="Intersección auto", step=0.5)
    lim_b = st.number_input("Límite superior (b):", value=None, placeholder="Intersección auto", step=0.5)
    
    st.markdown("---")
    
    st.markdown("""
        <div class='team-box'>
            <div class='team-title'>DESARROLLADO POR:</div>
            <div class='team-member'>• Mina Montaño Elsa Domenika</div>
            <div class='team-member'>• Amat Briones Maykell Gonzalo</div>
            <div class='team-member'>• Bralle Torres Stefano Samuel</div>
            <div class='team-member'>• Medrano Aguirre Steven David</div>
        </div>
    """, unsafe_allow_html=True)

# 4. Panel Principal (Dashboard)
st.markdown("# ⚡ **IntegralStudio** <span class='pro-badge'>&lt;Pro&gt;</span>", unsafe_allow_html=True)
st.markdown("---")

try:
    var_sym = sp.Symbol(var_str)
    f_expr = sp.sympify(f_input)
    g_expr = sp.sympify(g_input)
    
    funcion_resta = f_expr - g_expr
    
    if lim_a is None or lim_b is None:
        puntos_reales = []
        try:
            # 1. Intentar resolver algebraicamente (método exacto)
            puntos_interseccion = sp.solve(funcion_resta, var_sym)
            puntos_reales = [float(p.evalf()) for p in puntos_interseccion if p.is_real]
        except Exception:
            pass # Si falla, pasamos al método numérico
        
        # 2. Si no encontró suficientes puntos, intentar numéricamente
        if len(puntos_reales) < 2:
            f_num_solve = sp.lambdify(var_sym, funcion_resta, 'numpy')
            for i in np.linspace(-10, 10, 20):
                try:
                    raiz = brentq(f_num_solve, i, i + 1)
                    puntos_reales.append(raiz)
                except ValueError:
                    continue
        
        # 3. Limpiar y ordenar los puntos encontrados
        if len(puntos_reales) >= 2:
            puntos_reales = sorted(list(set(np.round(puntos_reales, 3))))
        
        # 4. Filtro de seguridad (Si definitivamente no se cruzan)
        if len(puntos_reales) < 2:
            st.warning("⚠️ **Nota:** No se detectaron suficientes puntos de intersección para cerrar el área.")
            st.info("Por favor, ingresa los límites **a** y **b** manualmente en la barra lateral.")
            st.stop()
            
        lim_a_calc = float(puntos_reales[0])
        lim_b_calc = float(puntos_reales[-1])
        st.info(f"💡 Se calcularon los límites automáticamente: **a = {lim_a_calc:.3f}**, **b = {lim_b_calc:.3f}**")
        
    else:
        lim_a_calc = float(lim_a)
        lim_b_calc = float(lim_b)

    # Cálculo del área exacta
    area_exacta = sp.integrate(funcion_resta, (var_sym, lim_a_calc, lim_b_calc))
    area_num = float(area_exacta.evalf())
    
    col_izq, col_der = st.columns([4, 5])
    
    with col_izq:
        st.markdown(f"""
            <div class='crypto-box'>
                <div class='metric-title'>📐 Métrica de Resultado</div>
                <div class='metric-value'>{area_num:.4f} u²</div>
                <div class='metric-status'>✓ Computado con Éxito</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='math-box'>", unsafe_allow_html=True)
        st.markdown("#### 📜 Expresión Formal (LaTeX)")
        st.markdown("**Planteamiento Integral:**")
        st.latex(f"A = \\int_{{{lim_a_calc:.2f}}}^{{{lim_b_calc:.2f}}} \\left[ ({sp.latex(f_expr)}) - ({sp.latex(g_expr)}) \\right] d{var_str}")
        st.markdown("**Solución Analítica:**")
        st.latex(f"A = {sp.latex(area_exacta)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_der:
        f_num = sp.lambdify(var_sym, f_expr, 'numpy')
        g_num = sp.lambdify(var_sym, g_expr, 'numpy')
        
        f_eval = lambda t: np.ones_like(t) * f_num(t) if np.isscalar(f_num(t)) else f_num(t)
        g_eval = lambda t: np.ones_like(t) * g_num(t) if np.isscalar(g_num(t)) else g_num(t)
        
        ancho = lim_b_calc - lim_a_calc
        pad = ancho * 0.4 if ancho != 0 else 1.0
        t_vals = np.linspace(lim_a_calc - pad, lim_b_calc + pad, 500)
        t_fill = np.linspace(lim_a_calc, lim_b_calc, 300)
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#0e1726')
        ax.set_facecolor('#0e1726')
        
        if var_str == 'x':
            ax.plot(t_vals, f_eval(t_vals), label=f'f(x) = {f_input}', color='#0ea5e9', linewidth=2.5, zorder=4)
            ax.plot(t_vals, g_eval(t_vals), label=f'g(x) = {g_input}', color='#ef4444', linewidth=2.5, zorder=4)
            ax.plot(t_vals, f_eval(t_vals), color='#0ea5e9', linewidth=7, alpha=0.2, zorder=3)
            ax.plot(t_vals, g_eval(t_vals), color='#ef4444', linewidth=7, alpha=0.2, zorder=3)
            ax.fill_between(t_fill, f_eval(t_fill), g_eval(t_fill), color='#10b981', alpha=0.25, label='Área Integrada', zorder=1)
            ax.axvline(lim_a_calc, color='#ffffff', linestyle='--', alpha=0.4)
            ax.axvline(lim_b_calc, color='#ffffff', linestyle='--', alpha=0.4)
        else:
            ax.plot(f_eval(t_vals), t_vals, label=f'f(y) = {f_input}', color='#0ea5e9', linewidth=2.5, zorder=4)
            ax.plot(g_eval(t_vals), t_vals, label=f'g(y) = {g_input}', color='#ef4444', linewidth=2.5, zorder=4)
            ax.plot(f_eval(t_vals), t_vals, color='#0ea5e9', linewidth=7, alpha=0.2, zorder=3)
            ax.plot(g_eval(t_vals), t_vals, color='#ef4444', linewidth=7, alpha=0.2, zorder=3)
            ax.fill_betweenx(t_fill, f_eval(t_fill), g_eval(t_fill), color='#10b981', alpha=0.25, label='Área Integrada', zorder=1)
            ax.axhline(lim_a_calc, color='#ffffff', linestyle='--', alpha=0.4)
            ax.axhline(lim_b_calc, color='#ffffff', linestyle='--', alpha=0.4)
        
        ax.grid(True, color='#1e293b', linestyle=':', alpha=0.6)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
            
        ax.legend(frameon=True, facecolor='#090d16', edgecolor='#1e293b')
        ax.set_title(f"Representación Vectorial del Área (Respecto a {var_str})", fontsize=11, color='#ffffff', weight='bold', pad=10)
        st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Error matemático: Por favor revisa la sintaxis de las funciones. Detalles: {e}")