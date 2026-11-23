#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Lorenz: sensibilidade às condições iniciais
Dois painéis 3D:
  esquerda: regime não caótico (rho = 10), convergência para ponto fixo;
  direita: regime caótico clássico (rho = 28), atrator de Lorenz.

Identidade visual alinhada aos vídeos dos pêndulos:
  navy   #0B192B
  cyan   #49C6E5
  orange #FFB454
  white  #F2F5F7

Dependências:
    numpy, scipy, matplotlib
Para MP4:
    ffmpeg instalado no sistema
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import to_rgba


# ============================================================
# Parâmetros editáveis
# ============================================================
SIGMA = 10.0
BETA = 8.0 / 3.0

RHO_STABLE = 10.0
RHO_CHAOTIC = 28.0

N_CONDITIONS = 10
DELTA_X0 = 0.01        # amplitude total das perturbações em x
T_VIDEO = 60.0         # duração física simulada
FPS = 60

TRAIL_SECONDS = 4.0   # duração da cauda visível, em unidades de tempo
TRAIL_MIN_ALPHA = 0.015  # opacidade na ponta antiga (0 a 1)
TRAIL_MAX_ALPHA = 0.95   # opacidade perto do ponto atual (0 a 1)
TRAIL_FADE_POWER = 1.5   # maior: a cauda antiga desaparece mais rápido
DPI = 120

OUT_MP4 = "sensibilidade_condicoes_iniciais_lorenz.mp4"
OUT_GIF = "sensibilidade_condicoes_iniciais_lorenz.gif"

# Aparência
BG = "#0B192B"
BG_DARK = "#081426"
CYAN = "#49C6E5"
ORANGE = "#FFB454"
WHITE = "#F2F5F7"
MUTED = "#9FB3C8"
GRID = "#31404D"

# Cores ordenadas para distinguir as dez condições iniciais
COLORS = plt.cm.cool(np.linspace(0.05, 0.95, N_CONDITIONS))


def lorenz(t, state, rho):
    """Equações de Lorenz."""
    x, y, z = state
    return (
        SIGMA * (y - x),
        x * (rho - z) - y,
        x * y - BETA * z,
    )


def integrate_trajectory(initial_condition, rho, t_eval):
    """Integra uma trajetória com tolerâncias adequadas à visualização."""
    sol = solve_ivp(
        lorenz,
        (float(t_eval[0]), float(t_eval[-1])),
        initial_condition,
        args=(rho,),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        max_step=0.02,
    )
    if not sol.success:
        raise RuntimeError(f"Falha na integração: {sol.message}")
    return sol.y.T


# ============================================================
# Integração das trajetórias
# ============================================================
nframes = int(T_VIDEO * FPS)
t_eval = np.linspace(0.0, T_VIDEO, nframes)

# As dez condições iniciais diferem apenas ligeiramente em x.
# Todas partem da mesma região e, no painel esquerdo, convergem
# para o mesmo ponto fixo estável.
x_offsets = np.linspace(-DELTA_X0 / 2, DELTA_X0 / 2, N_CONDITIONS)
base_initial = np.array([1.0, 1.0, 1.0])

initial_conditions = [
    base_initial + np.array([dx, 0.0, 0.0])
    for dx in x_offsets
]

trajectories_stable = [
    integrate_trajectory(ic, RHO_STABLE, t_eval)
    for ic in initial_conditions
]
trajectories_chaotic = [
    integrate_trajectory(ic, RHO_CHAOTIC, t_eval)
    for ic in initial_conditions
]


# ============================================================
# Figura e identidade visual
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": True,
    "font.size": 11,
    "axes.titlecolor": WHITE,
    "text.color": WHITE,
})

fig = plt.figure(figsize=(14, 7.5), dpi=DPI, facecolor=BG)
ax_left = fig.add_subplot(1, 2, 1, projection="3d")
ax_right = fig.add_subplot(1, 2, 2, projection="3d")
axs = [ax_left, ax_right]

fig.text(
    0.035, 0.945, "Sensibilidade às condições iniciais — sistema de Lorenz",
    fontsize=27, color=WHITE, fontweight="bold", ha="left"
)
fig.text(
    0.037, 0.902,
    rf"{N_CONDITIONS} trajetórias; perturbação inicial máxima "
    rf"$\Delta x_0={DELTA_X0:.3f}$",
    fontsize=20, color=MUTED, ha="left"
)
fig.text(
    0.965, 0.105, "Prof. Bruno Boaretto",
    fontsize=16, color=MUTED, ha="right"
)

titles = [
    rf"Regime não caótico  ($\rho={RHO_STABLE:g}$)",
    rf"Atrator caótico  ($\rho={RHO_CHAOTIC:g}$)",
]
subtitles = [
    "Convergência para um ponto fixo estável",
    "Afastamento das trajetórias inicialmente próximas",
]

for ax, title, subtitle in zip(axs, titles, subtitles):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=20, pad=18, color=WHITE, fontweight="bold")
    ax.text2D(
        0.5, 0.96, subtitle,
        transform=ax.transAxes, ha="center", va="top",
        color=MUTED, fontsize=14
    )
    ax.set_xlabel(r"$x$", labelpad=3, color=WHITE,fontsize=14)
    ax.set_ylabel(r"$y$", labelpad=3, color=WHITE,fontsize=14)
    ax.set_zlabel(r"$z$", labelpad=3, color=WHITE,fontsize=14)
    ax.tick_params(colors=MUTED, labelsize=8, pad=0)
    ax.xaxis.pane.set_facecolor(BG_DARK)
    ax.yaxis.pane.set_facecolor(BG_DARK)
    ax.zaxis.pane.set_facecolor(BG_DARK)
    ax.xaxis.pane.set_edgecolor(GRID)
    ax.yaxis.pane.set_edgecolor(GRID)
    ax.zaxis.pane.set_edgecolor(GRID)
    ax.xaxis._axinfo["grid"]["color"] = (0.19, 0.25, 0.31, 0.45)
    ax.yaxis._axinfo["grid"]["color"] = (0.19, 0.25, 0.31, 0.45)
    ax.zaxis._axinfo["grid"]["color"] = (0.19, 0.25, 0.31, 0.45)
    ax.view_init(elev=24, azim=-58)

# Limites escolhidos para comparar a convergência e o atrator.
ax_left.set_xlim(-2, 9)
ax_left.set_ylim(-2, 9)
ax_left.set_zlim(0, 14)

ax_right.set_xlim(-22, 22)
ax_right.set_ylim(-28, 28)
ax_right.set_zlim(0, 55)

# Traços com desvanecimento: cada trajetória mantém apenas uma janela temporal.
lines_left, points_left = [], []
lines_right, points_right = [], []

for ax, trajectories, lines, points in [
    (ax_left, trajectories_stable, lines_left, points_left),
    (ax_right, trajectories_chaotic, lines_right, points_right),
]:
    for color, traj in zip(COLORS, trajectories):
        line = Line3DCollection(
            [], linewidths=1.7, colors=[to_rgba(color)], zorder=3
        )
        ax.add_collection3d(line, autolim=False)

        point, = ax.plot(
            [], [], [], marker="o", ms=3.8,
            color=color, mec=WHITE, mew=0.25,
            linestyle="None", zorder=4
        )
        lines.append(line)
        points.append(point)

# Pequenos indicadores de início das trajetórias
for ax in axs:
    ax.scatter(
        [base_initial[0]], [base_initial[1]], [base_initial[2]],
        s=18, color=ORANGE, edgecolors=WHITE, linewidths=0.4, depthshade=False
    )

# Um contador discreto de tempo
time_label = fig.text(
    0.50, 0.075, r"$t=0.00$",
    fontsize=18, color=CYAN, ha="center"
)

# Converte a duração física da cauda em número de amostras.
trail_frames = max(2, int(TRAIL_SECONDS / T_VIDEO * nframes))


def update(frame):
    start = max(0, frame - trail_frames)

    for trajectories, lines, points in [
        (trajectories_stable, lines_left, points_left),
        (trajectories_chaotic, lines_right, points_right),
    ]:
        for traj, line, point in zip(trajectories, lines, points):
            recent = traj[start:frame + 1]

            if len(recent) >= 2:
                # Segmentos 3D consecutivos, em vez da trajetória completa.
                segments = np.stack((recent[:-1], recent[1:]), axis=1)
                line.set_segments(segments)

                # Transparência gradual: antiga = apagada; atual = visível.
                fade = np.linspace(0.0, 1.0, len(segments))
                alpha = TRAIL_MIN_ALPHA + (
                    TRAIL_MAX_ALPHA - TRAIL_MIN_ALPHA
                ) * fade**TRAIL_FADE_POWER

                rgb = np.asarray(to_rgba(line.get_colors()[0])[:3])
                rgba = np.column_stack([
                    np.tile(rgb, (len(segments), 1)),
                    alpha
                ])
                line.set_color(rgba)
            else:
                line.set_segments([])

            p = traj[frame]
            point.set_data([p[0]], [p[1]])
            point.set_3d_properties([p[2]])

    time_label.set_text(rf"$t={t_eval[frame]:.2f}$")
    return (
        lines_left + points_left
        + lines_right + points_right
        + [time_label]
    )


# Margens controladas manualmente para manter o enquadramento fixo no vídeo.
fig.subplots_adjust(
    left=0.015, right=0.985, top=0.82, bottom=0.105, wspace=0.03
)

ani = FuncAnimation(
    fig, update, frames=nframes, interval=1000 / FPS, blit=False
)

if shutil.which("ffmpeg"):
    print(f"Salvando {OUT_MP4} ...")
    ani.save(
        OUT_MP4,
        writer=FFMpegWriter(
            fps=FPS,
            bitrate=6000,
            extra_args=["-pix_fmt", "yuv420p"],
        ),
        dpi=DPI,
    )
    output = OUT_MP4
else:
    print("ffmpeg não encontrado; salvando GIF (mais lento e maior).")
    ani.save(OUT_GIF, writer=PillowWriter(fps=20), dpi=90)
    output = OUT_GIF

print(f"Arquivo salvo: {output}")
plt.close(fig)
