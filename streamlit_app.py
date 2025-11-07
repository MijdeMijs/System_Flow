import streamlit as st
import numpy as np
import graphviz as gr
import matplotlib.pyplot as plt

st.title("Deterministic System Flow")

# ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₀ 

# --- Data ---
# Define Gaussian PDF
def normal_dist(x, mean, sd):
    prob_density = (1 / (np.sqrt(2 * np.pi) * sd)) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density

# Set the x points for calculation
x_values = np.linspace(-10, 10, 300) 

# --- Model parameters ---
st.header("Model parameters", divider="rainbow")

# --- Sliders first row (C, T, M1) ---
row0_col0, row0_col1, row0_col2 = st.columns(3)

with row0_col0:
    st.subheader("C node")
    C_mean = st.slider("C (μ)", -3.0, 3.0, 0.0, 0.1)
    C_sd = st.slider("C (σ)", 0.1, 2.0, 1.0, 0.1)

with row0_col1:
    st.subheader("T node")
    beta_5 = st.slider("β₅ (C → T coefficient)", -3.0, 3.0, 0.0, 0.1)
    eT_mean = st.slider("ϵT (μ)", -3.0, 3.0, 0.0, 0.1)
    eT_sd = st.slider("ϵT (σ)", 0.1, 2.0, 1.0, 0.1)

with row0_col2:
    st.subheader("M₁ node")
    beta_2 = st.slider("β₂ (T → M₁ coefficient)", -3.0, 3.0, 0.0, 0.1)
    beta_4 = st.slider("β₄ (Z₂ → M₁ coefficient)", -3.0, 3.0, 0.0, 0.1)
    eM1_mean = st.slider("ϵM₁ (μ)", -3.0, 3.0, 0.0, 0.1)
    eM1_sd = st.slider("ϵM₁ (σ)", 0.1, 2.0, 1.0, 0.1)

# --- Sliders second row (Z1, Z2, M2) ---
row1_col0, row1_col1, row1_col2 = st.columns(3)

with row1_col0:
    st.subheader("Z₁ node")
    Z1_mean = st.slider("Z₁ (μ)", -3.0, 3.0, 0.0, 0.1)
    Z1_sd = st.slider("Z₁ (σ)", 0.1, 2.0, 1.0, 0.1)

with row1_col1:
    st.subheader("Z₂ node")
    Z2_mean = st.slider("Z₂ (μ)", -3.0, 3.0, 0.0, 0.1)
    Z2_sd = st.slider("Z₂ (σ)", 0.1, 2.0, 1.0, 0.1)

with row1_col2:
    st.subheader("M₂ node")
    beta_3 = st.slider("β₃ (T → M₂ coefficient)", -3.0, 3.0, 0.0, 0.1)
    eM2_mean = st.slider("ϵM₂ (μ)", -3.0, 3.0, 0.0, 0.1)
    eM2_sd = st.slider("ϵM₂ (σ)", 0.1, 2.0, 1.0, 0.1)

# --- Sliders third row (Y) ---
row2_col0, row2_col1, row2_col2 = st.columns(3)

with row2_col0:
    st.subheader("Y node")
    beta_0 = st.slider("β₀ (T → Y coefficient)", -3.0, 3.0, 0.0, 0.1)
    beta_1 = st.slider("β₁ (Z₁ → Y coefficient)", -3.0, 3.0, 0.0, 0.1)

with row2_col1:
    st.subheader("")
    beta_6 = st.slider("β₆ (C → Y coefficient)", -3.0, 3.0, 0.0, 0.1)
    beta_7 = st.slider("β₇ (M₁ → Y coefficient)", -3.0, 3.0, 0.0, 0.1)

with row2_col2:
    st.subheader("")
    beta_8 = st.slider("β₈ (M₂ → Y coefficient)", -3.0, 3.0, 0.0, 0.1)
    eY_mean = st.slider("ϵY (μ)", -3.0, 3.0, 0.0, 0.1)
    eY_sd = st.slider("ϵY (σ)", 0.1, 2.0, 1.0, 0.1)    

st.divider()   

# --- Calculations ---

# C node
C = normal_dist(x=x_values, mean=C_mean, sd=C_sd)

# T node
T_mean = beta_5 * C_mean + eT_mean
T_sd = np.sqrt(beta_5**2 * C_sd**2 + eT_sd**2)
T = normal_dist(x=x_values, mean=T_mean, sd=T_sd)

# Z1 node
Z1 = normal_dist(x=x_values, mean=Z1_mean, sd=Z1_sd)

# Z2 node
Z2 = normal_dist(x=x_values, mean=Z2_mean, sd=Z2_sd)

# M1 node
M1_mean = beta_2 * T_mean + beta_4 * Z2_mean + eM1_mean 
M1_sd = np.sqrt(beta_2**2 * T_sd**2 + beta_4 * Z2_sd + eM1_sd**2)
M1 = normal_dist(x=x_values, mean=M1_mean, sd=M1_sd)

# M2 node
M2_mean = beta_3 * T_mean + eM2_mean
M2_sd = np.sqrt(beta_3**2 * T_sd**2 + eM2_sd**2)
M2 = normal_dist(x=x_values, mean=M2_mean, sd=M2_sd)

# Y node
Y_mean = (beta_0 * T_mean 
          + beta_1 * Z1_mean 
          + beta_6 * C_mean 
          + beta_7 * M1_mean 
          + beta_8 * M2_mean 
          + eY_mean)
Y_sd = np.sqrt(beta_0 * T_sd 
               + beta_1 * Z1_sd 
               + beta_6 * C_sd 
               + beta_7 * M1_sd 
               + beta_8 * M2_sd 
               + eY_sd)
Y = normal_dist(x=x_values, mean=Y_mean, sd=Y_sd)

# --- Directed Acyclic Graph ---
st.header("Directed Acyclic Graph (DAG)", divider="rainbow")

# --- Helper function ---
def edge_color(beta, default_color="black"):
    """Return white if beta == 0, else default color."""
    return "grey92" if beta == 0 else default_color

g = gr.Digraph()

g.attr(rankdir='LR')

# ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₀

# Solid edges
g.edge("C", "T", label=f"β₅ = {beta_5}", color=edge_color(beta_5))
g.edge("Z₂", "M₁", label=f"β₄ = {beta_4}", color=edge_color(beta_4))
g.edge("T", "M₁", label=f"β₂ = {beta_2}", color=edge_color(beta_2))
g.edge("T", "M₂", label=f"β₃ = {beta_3}", color=edge_color(beta_3))

g.edge("Z₁", "Y", label=f"β₁ = {beta_1}", color=edge_color(beta_1))
g.edge("M₁", "Y", label=f"β₇ = {beta_7}", color=edge_color(beta_7))
g.edge("M₂", "Y", label=f"β₈ = {beta_8}", color=edge_color(beta_8))
g.edge("C", "Y", label=f"β₆ = {beta_6}", color=edge_color(beta_6))
g.edge("T", "Y", label=f"β₀ = {beta_0}", color=edge_color(beta_0))

st.graphviz_chart(g)

# --- Plotting ---
st.header("Probability Densities", divider="rainbow")

# --- Create two stacked subplots ---
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, figsize=(6, 6), sharex=True)

# --- First plot: C ---
ax1.plot(x_values, C, color='blue', label='C')
ax1.set_xlim(-10, 10)
ax1.set_xticks(np.arange(-10, 11, 2))
ax1.set_ylim(0, 0.5)
ax1.set_yticks(np.arange(0, 0.6, 0.1))
ax1.set_ylabel('P(C)')
ax1.legend()
ax1.grid(True)

# --- Second plot: T ---
ax2.plot(x_values, T, color='red', label='T')
ax2.set_xlim(-10, 10)
ax2.set_xticks(np.arange(-10, 11, 2))
ax2.set_ylim(0, 0.5)
ax2.set_yticks(np.arange(0, 0.6, 0.1))
# ax2.set_xlabel('X')
ax2.set_ylabel('P(T)')
ax2.legend()
ax2.grid(True)

# --- Third plot: M1 ---
ax3.plot(x_values, M1, color='lime', label='M₁')
ax3.set_xlim(-10, 10)
ax3.set_xticks(np.arange(-10, 11, 2))
ax3.set_ylim(0, 0.5)
ax3.set_yticks(np.arange(0, 0.6, 0.1))
# ax3.set_xlabel('X')
ax3.set_ylabel('P(M₁)')
ax3.legend()
ax3.grid(True)

# --- Fourth plot: M2 ---
ax4.plot(x_values, M2, color='lime', label='M₂')
ax4.set_xlim(-10, 10)
ax4.set_xticks(np.arange(-10, 11, 2))
ax4.set_ylim(0, 0.5)
ax4.set_yticks(np.arange(0, 0.6, 0.1))
# ax4.set_xlabel('X')
ax4.set_ylabel('P(M₂)')
ax4.legend()
ax4.grid(True)

# --- Fifth plot: Y ---
ax5.plot(x_values, Y, color='magenta', label='Y')
ax5.set_xlim(-10, 10)
ax5.set_xticks(np.arange(-10, 11, 2))
ax5.set_ylim(0, 0.5)
ax5.set_yticks(np.arange(0, 0.6, 0.1))
ax5.set_xlabel('X')
ax5.set_ylabel('P(Y)')
ax5.legend()
ax5.grid(True)

# --- Layout fix for Streamlit display ---
plt.tight_layout()

# ✅ Display stacked plots in Streamlit
st.pyplot(fig)
