import streamlit as st
import numpy as np
import graphviz as gr
import matplotlib.pyplot as plt

st.title("Interactive Deterministic Function Visualization")

g = gr.Digraph()

g.attr(rankdir='LR')

# Solid edges
g.edge("T", "Y")
g.edge("C", "T")
g.edge("C", "Y")

# Nodes
g.node("Y", label='Y')
g.node("C", label='C')
g.node("T", label='T')

st.graphviz_chart(g)

st.header("Parameters")

# --- Data domain ---
def normal_dist(x, mean, sd):
    prob_density = (1 / (np.sqrt(2 * np.pi) * sd)) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density

x_values = np.linspace(-5, 5, 250) 

# --- Two columns for parameters ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("C Parameters")
    C_mean = st.slider("C (μ)", -5.0, 5.0, 0.0, 0.1)
    C_sd = st.slider("C (σ)", 0.1, 2.0, 1.0, 0.1)
    C = normal_dist(x=x_values, mean=C_mean, sd=C_sd)

with col2:
    st.subheader("T Parameters")
    beta_5 = st.slider("β₅ (coefficient)", -5.0, 5.0, 0.0, 0.1)
    eT_mean = st.slider("T (μ)", -5.0, 5.0, 0.0, 0.1)
    eT_sd = st.slider("T (σ)", 0.1, 2.0, 1.0, 0.1)
    T_mean = beta_5 * C_mean + eT_mean
    T_sd = np.sqrt(beta_5**2 * C_sd**2 + eT_sd**2)
    T = normal_dist(x=x_values, mean=T_mean, sd=T_sd)

# --- Plotting ---
fig, ax = plt.subplots()

ax.plot(x_values, C, label='C')
ax.plot(x_values, T, label='T')

ax.set_xlim(-5, 5)
ax.set_xticks(np.arange(-5, 6, 1))
ax.set_xlabel('X')
ax.set_ylim(0, 0.5)
ax.set_ylabel('Probability Density')
ax.legend()
ax.grid(True)

# ✅ Display in Streamlit
st.pyplot(fig)
