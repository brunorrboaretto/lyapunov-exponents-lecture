
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interpretação do espectro de Lyapunov

(a) Foco estável: oscilador amortecido
(b) Ciclo limite: modelo de FitzHugh-Nagumo
(c) Atrator caótico: circuito de Chua

Identidade visual:
  navy   #0B192B
  cyan   #49C6E5
  orange #FFB454
  white  #F2F5F7
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# Identidade visual
# ============================================================

BG = "#0B192B"
BG_DARK = "#081426"
CYAN = "#49C6E5"
ORANGE = "#FFB454"
WHITE = "#F2F5F7"
MUTED = "#9FB3C8"
GRID = "#31404D"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": True,
    "font.size": 11,
    "axes.titlecolor": WHITE,
    "text.color": WHITE,
})


# ============================================================
# (a) Oscilador amortecido: foco estável
#
# dx/dt = y
# dy/dt = -x - gamma*y
# ============================================================

gamma = 0.8

def damped_oscillator(t, X):
    x, y = X
    return [y, -x - gamma*y]


t_eval = np.linspace(0, 35, 5000)

sol_focus = solve_ivp(
    damped_oscillator,
    (t_eval[0], t_eval[-1]),
    [2.0, 0.0],
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
)

x_focus, y_focus = sol_focus.y


# ============================================================
# (b) FitzHugh-Nagumo: ciclo limite estável
#
# dv/dt = v - v^3/3 - w + I
# dw/dt = eps*(v + a - b*w)
# ============================================================

a = 0.7
b = 0.8
eps = 0.08
I = 0.5

def fitzhugh_nagumo(t, X):
    v, w = X

    dv = v - v**3/3 - w + I
    dw = eps*(v + a - b*w)

    return [dv, dw]


sol_fhn = solve_ivp(
    fitzhugh_nagumo,
    (0, 500),
    [-1.0, -0.5],
    dense_output=True,
    rtol=1e-9,
    atol=1e-11,
    max_step=0.1,
)

t_fhn = np.linspace(300, 500, 10000)
v_fhn, w_fhn = sol_fhn.sol(t_fhn)


# ============================================================
# (c) Circuito de Chua: atrator caótico
# ============================================================

alpha = 15.6
beta = 28.0
m0 = -1.143
m1 = -0.714

def chua(t, X):
    x, y, z = X

    h = m1*x + 0.5*(m0-m1)*(
        abs(x+1) - abs(x-1)
    )

    dx = alpha*(y-x-h)
    dy = x-y+z
    dz = -beta*y

    return [dx, dy, dz]


sol_chua = solve_ivp(
    chua,
    (0, 500),
    [0.1, 0.0, 0.0],
    rtol=1e-8,
    atol=1e-10,
    max_step=0.05,
    dense_output=True,
)

t_chua = np.linspace(100, 500, 60000)
x_chua, y_chua, z_chua = sol_chua.sol(t_chua)


# ============================================================
# Figura
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 5.2),
    facecolor=BG,
)

#fig.text(
#    0.035, 0.955,
#    "Interpretação do espectro de Lyapunov",
#    fontsize=24,
#    color=WHITE,
#    fontweight="bold",
#    ha="left",
#)


# ============================================================
# Configuração comum dos painéis
# ============================================================

titles = [
    "Foco estável",
    "Ciclo limite estável",
    "Atrator caótico",
]

exponents = [
    r"$\lambda_1,\ldots,\lambda_n<0$",

    r"$\lambda_1=0,\quad"
    r"\lambda_2,\ldots,\lambda_n<0$",

    r"$\lambda_1>0,\quad\lambda_2=0,\quad"
    r"\lambda_3,\ldots,\lambda_n<0$",
]
for ax, title, exponent in zip(axes, titles, exponents):

    ax.set_facecolor(BG)

    ax.set_title(
        title + "\n" + exponent,
        fontsize=22,
        color=WHITE,
        fontweight="bold",
        pad=13,
        linespacing=1.6,
    )

    ax.tick_params(
        colors=MUTED,
        labelsize=12,
        length=3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_color(MUTED)
    ax.spines["left"].set_color(MUTED)

    ax.grid(
        True,
        color=GRID,
        alpha=0.5,
        linewidth=0.5,
    )

    ax.set_axisbelow(True)


# ============================================================
# Painel (a): foco estável
# ============================================================

ax = axes[0]

ax.plot(
    x_focus,
    y_focus,
    color=CYAN,
    lw=1.8,
    alpha=0.95,
)

# Condição inicial
ax.scatter(
    [x_focus[0]],
    [y_focus[0]],
    color=ORANGE,
    s=38,
    edgecolors=WHITE,
    linewidths=0.5,
    zorder=5,
)

# Ponto fixo
ax.scatter(
    [0],
    [0],
    color=WHITE,
    s=35,
    edgecolors=ORANGE,
    linewidths=1.0,
    zorder=6,
)

ax.set_xlabel(r"$x$", fontsize=17, labelpad=5,color=WHITE)
ax.set_ylabel(r"$y$", fontsize=17, labelpad=5,color=WHITE)

ax.set_xlim(-2.3, 2.3)
ax.set_ylim(-1.8, 1.8)

ax.set_aspect("equal", adjustable="box")


# ============================================================
# Painel (b): ciclo limite de FitzHugh-Nagumo
# ============================================================

ax = axes[1]

ax.plot(
    v_fhn,
    w_fhn,
    color=CYAN,
    lw=1.5,
    alpha=0.95,
)

# Ponto inicial da janela representada
ax.scatter(
    [v_fhn[0]],
    [w_fhn[0]],
    color=ORANGE,
    s=30,
    edgecolors=WHITE,
    linewidths=0.4,
    zorder=5,
)

ax.set_xlabel(r"$v$", fontsize=17, labelpad=5,color=WHITE)
ax.set_ylabel(r"$w$", fontsize=17, labelpad=5,color=WHITE)


# ============================================================
# Painel (c): atrator de Chua
# ============================================================

ax = axes[2]

ax.plot(
    x_chua,
    y_chua,
    color=CYAN,
    lw=0.45,
    alpha=0.85,
)

ax.set_xlabel(r"$x$", fontsize=17, labelpad=5,color=WHITE)
ax.set_ylabel(r"$y$", fontsize=17, labelpad=5,color=WHITE)


# ============================================================
# Acabamento
# ============================================================

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.78,
    bottom=0.19,
    wspace=0.20,
)

fig.savefig(
    "espectro_lyapunov_interpretacao.png",
    dpi=300,
    facecolor=BG,
    bbox_inches="tight",
)

#plt.show()
