import streamlit as st
import numpy as np
import graphviz as gr
import matplotlib.pyplot as plt

st.title("Interactive Deterministic Function Visualization")

# ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₀ 

# --- Data ---
# Define Gaussian PDF
def normal_dist(x, mean, sd):
    prob_density = (1 / (np.sqrt(2 * np.pi) * sd)) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density

# Set the x points for calculation
x_values = np.linspace(-5, 5, 250) 

# --- Model parameters ---
st.header("Model parameters", divider="rainbow")

# --- First row (C, T, M1) ---
row0_col0, row0_col1, row0_col2 = st.columns(3)

with row0_col0:
    st.subheader("C node")
    C_mean = st.slider("C (μ)", -3.0, 3.0, 0.0, 0.1)
    C_sd = st.slider("C (σ)", 0.1, 2.0, 1.0, 0.1)
    C = normal_dist(x=x_values, mean=C_mean, sd=C_sd)

with row0_col1:
    st.subheader("T node")
    beta_5 = st.slider("β₅ (C → T coefficient)", -3.0, 3.0, 0.0, 0.1)
    eT_sd = st.slider("T (σ)", 0.1, 2.0, 1.0, 0.1)
    T_mean = beta_5 * C_mean #+ eT_mean
    T_sd = np.sqrt(beta_5**2 * C_sd**2 + eT_sd**2)
    T = normal_dist(x=x_values, mean=T_mean, sd=T_sd)

with row0_col2:
    st.subheader("M₁ node")
    beta_0 = st.slider("β₂ (T → M₁ coefficient)", -3.0, 3.0, 0.0, 0.1)
    beta_2 = st.slider("β₄ (Z₂ → M₁ coefficient)", -3.0, 3.0, 0.0, 0.1)
    eM1_sd = st.slider("M₁ (σ)", 0.1, 2.0, 1.0, 0.1)
    # M2_mean = f(T(C))
    # M2_sd = 
    # M2 = normal_dist(x=x_values, mean=..., sd=...)

# --- Second row (Z1, Z2, M2) ---
row1_col0, row1_col1, row1_col2 = st.columns(3)

with row1_col0:
    st.subheader("Z₁ node")
    # beta_2 = st.slider("β₂ (coefficient)", -3.0, 3.0, 0.0, 0.1)
    eZ1_mean = st.slider("Z₁ (μ)", -3.0, 3.0, 0.0, 0.1)
    eZ1_sd = st.slider("Z₁ (σ)", 0.1, 2.0, 1.0, 0.1)
    # M2_mean = f(T(C))
    # M2_sd = 
    # M2 = normal_dist(x=x_values, mean=..., sd=...)

with row1_col1:
    st.subheader("Z₂ node")
    # beta_2 = st.slider(" (coefficient)", -3.0, 3.0, 0.0, 0.1)
    eZ1_mean = st.slider("Z₂ (μ)", -3.0, 3.0, 0.0, 0.1)
    eZ1_sd = st.slider("Z₂ (σ)", 0.1, 2.0, 1.0, 0.1)
    # M2_mean = f(T(C))
    # M2_sd = 
    # M2 = normal_dist(x=x_values, mean=..., sd=...)

with row1_col2:
    st.subheader("M₂ node")
    beta_1 = st.slider("β₃ (T → M₂ coefficient)", -3.0, 3.0, 0.0, 0.1)
    eM2_sd = st.slider("M₂ (σ)", 0.1, 2.0, 1.0, 0.1)
    # M2_mean = f(T(C))
    # M2_sd = 
    # M2 = normal_dist(x=x_values, mean=..., sd=...)

# --- Third row (Y) ---
row2_col0, row2_col1, row2_col2 = st.columns(3)

with row2_col0:
    st.subheader("Y node")
    beta_2 = st.slider("β₀ (T → Y coefficient)", -3.0, 3.0, 0.0, 0.1)
    beta_3 = st.slider("β₁ (Z₁ → Y coefficient)", -3.0, 3.0, 0.0, 0.1)

with row2_col1:
    st.subheader("")
    beta_4 = st.slider("β₆ (C → Y coefficient)", -3.0, 3.0, 0.0, 0.1)
    beta_6 = st.slider("β₇ (M₁ → Y coefficient)", -3.0, 3.0, 0.0, 0.1)

with row2_col2:
    st.subheader("")
    beta_7 = st.slider("β₈ (M₂ → Y coefficient)", -3.0, 3.0, 0.0, 0.1)
    Y_sd = st.slider("Y (σ)", 0.1, 2.0, 1.0, 0.1)    

st.divider()   

# --- Directed Acyclic Graph ---
st.header("Directed Acyclic Graph (DAG)", divider="rainbow")

# --- Helper function ---
def edge_color(beta, default_color="black"):
    """Return white if beta == 0, else default color."""
    return "grey92" if beta == 0 else default_color

g = gr.Digraph()

g.attr(rankdir='LR')

beta_off = 1

# ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₀

# Solid edges
g.edge("C", "T", label=f"β₅ = {beta_5}", color=edge_color(beta_5))
g.edge("Z₂", "M₁", label=f"β₄ = {beta_off}", color=edge_color(beta_off))
g.edge("T", "M₁", label=f"β₂ = {beta_off}", color=edge_color(beta_off))
g.edge("T", "M₂", label=f"β₃ = {beta_off}", color=edge_color(beta_off))

g.edge("Z₁", "Y", label=f"β₁ = {beta_off}", color=edge_color(beta_off))
g.edge("M₁", "Y", label=f"β₇ = {beta_off}", color=edge_color(beta_off))
g.edge("M₂", "Y", label=f"β₈ = {beta_off}", color=edge_color(beta_off))
g.edge("C", "Y", label=f"β₆ = {beta_off}", color=edge_color(beta_off))
g.edge("T", "Y", label=f"β₀ = {beta_off}", color=edge_color(beta_off))

st.graphviz_chart(g)

# --- Plotting ---
st.header("Probability Densities", divider="rainbow")

# --- Create two stacked subplots ---
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 3), sharex=True)

# --- First plot: C ---
ax1.plot(x_values, C, color='blue', label='C')
ax1.set_xlim(-5, 5)
ax1.set_xticks(np.arange(-5, 6, 1))
ax1.set_ylim(0, 0.5)
ax1.set_yticks(np.arange(0, 0.6, 0.1))
ax1.set_ylabel('P(C)')
ax1.legend()
ax1.grid(True)

# --- Second plot: T ---
ax2.plot(x_values, T, color='red', label='T')
ax2.set_xlim(-5, 5)
ax2.set_xticks(np.arange(-5, 6, 1))
ax2.set_ylim(0, 0.5)
ax2.set_yticks(np.arange(0, 0.6, 0.1))
ax2.set_xlabel('X')
ax2.set_ylabel('P(T)')
ax2.legend()
ax2.grid(True)

# --- Layout fix for Streamlit display ---
plt.tight_layout()

# ✅ Display stacked plots in Streamlit
st.pyplot(fig)
