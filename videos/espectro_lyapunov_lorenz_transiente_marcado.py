#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Espectro de Lyapunov do sistema de Lorenz — animação.

Esquerda: x(t), y(t), z(t), em ciano.
Direita: estimativas acumuladas de lambda_1, lambda_2 e lambda_3.

Método: integração do sistema de Lorenz e das equações variacionais,
com ortogonalização QR periódica (implementação do procedimento
de Gram-Schmidt para o cálculo do espectro).

Dependências: numpy, scipy, matplotlib e ffmpeg (para MP4).
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.integrate import solve_ivp


# ========================= PARÂMETROS EDITÁVEIS =========================

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
IC = np.array([1.0, 1.0, 1.0])

T_MAX = 100.0
T_TRANSIENT = 20.0       # transiente descartado para estimar os expoentes
QR_INTERVAL = 0.10       # intervalo entre ortogonalizações
SAMPLES_PER_QR = 5       # resolução da trajetória entre ortogonalizações

FPS = 30
VIDEO_SECONDS = 20
DPI = 140
BITRATE = 6000

OUT_MP4 = "espectro_lyapunov_lorenz.mp4"
OUT_GIF = "espectro_lyapunov_lorenz.gif"

# Identidade visual
BG = "#0B192B"
BG_DARK = "#081426"
CYAN = "#49C6E5"
MAGENTA = "#E038A3"      # expoente positivo
NEUTRAL = "#B8C0CC"      # expoente aproximadamente nulo
GOLD = "#B58A45"         # expoente negativo
WHITE = "#F2F5F7"
MUTED = "#9FB3C8"
GRID = "#31404D"


# ========================= LORENZ E JACOBIANA =========================

def lorenz_rhs(x):
    xx, yy, zz = x
    return np.array([
        SIGMA * (yy - xx),
        xx * (RHO - zz) - yy,
        xx * yy - BETA * zz
    ])


def lorenz_jacobian(x):
    xx, yy, zz = x
    return np.array([
        [-SIGMA, SIGMA, 0.0],
        [RHO - zz, -1.0, -xx],
        [yy, xx, -BETA]
    ])


def augmented_rhs(t, state):
    """Integra x e os nove elementos da matriz tangente Q."""
    x = state[:3]
    Q = state[3:].reshape(3, 3)
    dx = lorenz_rhs(x)
    dQ = lorenz_jacobian(x) @ Q
    return np.concatenate([dx, dQ.ravel()])


# ========================= INTEGRAÇÃO + QR =========================

def simulate_spectrum():
    n_intervals = int(round(T_MAX / QR_INTERVAL))
    if not np.isclose(n_intervals * QR_INTERVAL, T_MAX):
        raise ValueError("T_MAX deve ser múltiplo de QR_INTERVAL.")
    if not np.isclose(T_TRANSIENT / QR_INTERVAL,
                      round(T_TRANSIENT / QR_INTERVAL)):
        raise ValueError("T_TRANSIENT deve ser múltiplo de QR_INTERVAL.")

    n_transient = int(round(T_TRANSIENT / QR_INTERVAL))
    if n_transient >= n_intervals:
        raise ValueError("T_TRANSIENT deve ser menor que T_MAX.")

    x = IC.astype(float).copy()
    Q = np.eye(3)

    trajectory_t = [0.0]
    trajectory_x = [x.copy()]
    # Eixo temporal completo para os expoentes. Durante o transiente,
    # os valores permanecem como NaN (não são exibidos no gráfico).
    spectrum_t = [0.0]
    spectrum_values = [np.full(3, np.nan)]
    accumulated_logs = np.zeros(3)
    elapsed_observation = 0.0

    print("Integrando Lorenz e equações variacionais...")
    for k in range(n_intervals):
        t0 = k * QR_INTERVAL
        tf = (k + 1) * QR_INTERVAL
        state0 = np.concatenate([x, Q.ravel()])

        # Pontos internos para desenhar a trajetória suavemente.
        t_eval = np.linspace(t0, tf, SAMPLES_PER_QR + 1)[1:]
        sol = solve_ivp(
            augmented_rhs, (t0, tf), state0,
            t_eval=t_eval, method="DOP853",
            rtol=1e-9, atol=1e-11,
            max_step=QR_INTERVAL / 4.0
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        for j in range(sol.t.size):
            trajectory_t.append(float(sol.t[j]))
            trajectory_x.append(sol.y[:3, j].copy())

        x = sol.y[:3, -1].copy()
        Q_evolved = sol.y[3:, -1].reshape(3, 3)

        # Q_evolved = Q_new R. Ajustamos os sinais para diag(R) > 0.
        Q_new, R = np.linalg.qr(Q_evolved)
        signs = np.sign(np.diag(R))
        signs[signs == 0.0] = 1.0
        Q_new = Q_new * signs[np.newaxis, :]
        R = signs[:, np.newaxis] * R

        stretch = np.maximum(np.abs(np.diag(R)), np.finfo(float).tiny)

        # Durante o transiente continuamos ortogonalizando,
        # mas ainda não acumulamos os expoentes.
        if k >= n_transient:
            accumulated_logs += np.log(stretch)
            elapsed_observation += QR_INTERVAL
            spectrum_t.append(tf)
            spectrum_values.append(accumulated_logs / elapsed_observation)
        else:
            spectrum_t.append(tf)
            spectrum_values.append(np.full(3, np.nan))

        # Reinicia os vetores tangentes na base ortonormal Q_new.
        Q = Q_new

        if (k + 1) % max(1, n_intervals // 10) == 0:
            print(f"  Integração: {100*(k+1)/n_intervals:5.1f}%")

    return (
        np.asarray(trajectory_t),
        np.asarray(trajectory_x),
        np.asarray(spectrum_t),
        np.asarray(spectrum_values)
    )


# ========================= ANIMAÇÃO =========================

def main():
    t, traj, t_spec, lambdas = simulate_spectrum()

    print("\nEstimativas finais:")
    for i, value in enumerate(lambdas[-1], start=1):
        print(f"  lambda_{i} = {value:.6f}")

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "text.usetex": True,
        "font.size": 11,
        "text.color": WHITE,
        "axes.labelcolor": WHITE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })

    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.55, 1.0],
        left=0.075, right=0.965, top=0.83, bottom=0.17,
        wspace=0.12, hspace=0.30
    )

    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 0], sharex=ax_x)
    ax_z = fig.add_subplot(gs[2, 0], sharex=ax_x)
    ax_l1 = fig.add_subplot(gs[0, 1])
    ax_l2 = fig.add_subplot(gs[1, 1], sharex=ax_l1)
    ax_l3 = fig.add_subplot(gs[2, 1], sharex=ax_l1)

    # Título
    fig.text(0.075, 0.945, "Espectro de expoentes de Lyapunov",
             fontsize=25, color=WHITE, fontweight="bold", ha="left")
    fig.text(
        0.077, 0.895,
        rf"Sistema de Lorenz: $\sigma={SIGMA:g}$, "
        rf"$\rho={RHO:g}$, $\beta={BETA:.3g}$",
        fontsize=16, color=MUTED, ha="left"
    )
    fig.text(0.965, 0.075, "Prof. Bruno Boaretto",
             fontsize=15, color=MUTED, ha="right")

    # Três variáveis da trajetória de referência.
    left_axes = [ax_x, ax_y, ax_z]
    variables = [(0, r"$x(t)$"), (1, r"$y(t)$"), (2, r"$z(t)$")]
    trajectory_lines = []
    trajectory_cursors = []

    for ax, (idx, ylabel) in zip(left_axes, variables):
        ax.set_facecolor(BG_DARK)
        ax.set_ylabel(ylabel, fontsize=16, labelpad=9)
        ax.grid(True, color=GRID, alpha=0.55, linewidth=0.65)
        ax.tick_params(labelsize=13, length=3)
        lo, hi = traj[:, idx].min(), traj[:, idx].max()
        margin = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlim(0.0, T_MAX)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        line, = ax.plot([], [], color=CYAN, lw=1.6)
        cursor = ax.axvline(0.0, color=WHITE, lw=0.8, alpha=0.5)
        ax.axvline(T_TRANSIENT, color="#D7DEE8", lw=1.0,
                   ls="--", alpha=0.75, zorder=2)
        trajectory_lines.append(line)
        trajectory_cursors.append(cursor)

    ax_z.set_xlabel(r"Tempo $t$", fontsize=13)
    ax_x.tick_params(labelbottom=False)
    ax_y.tick_params(labelbottom=False)

    # Três estimativas acumuladas do espectro.
    lyap_axes = [ax_l1, ax_l2, ax_l3]
    lyap_colors = [MAGENTA, NEUTRAL, GOLD]
    lyap_labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    y_limits = [(-0.2, 2.8), (-0.8, 0.8), (-20.0, -5.0)]

    exponent_lines = []
    exponent_points = []
    exponent_cursors = []

    for ax, color, label, ylim in zip(
        lyap_axes, lyap_colors, lyap_labels, y_limits
    ):
        ax.set_facecolor(BG_DARK)
        ax.set_ylabel(label, fontsize=17, color=color, labelpad=8)
        ax.set_xlim(0.0, T_MAX)
        ax.set_ylim(*ylim)
        ax.grid(True, color=GRID, alpha=0.55, linewidth=0.65)
        ax.tick_params(labelsize=13, length=3)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.0, color=MUTED, lw=0.8, ls=":", alpha=0.7)
        ax.axvline(T_TRANSIENT, color="#D7DEE8", lw=1.0,
                   ls="--", alpha=0.75, zorder=2)

        line, = ax.plot([], [], color=color, lw=2.0)
        point, = ax.plot([], [], marker="o", ms=4.5, color=color,
                         mec=WHITE, mew=0.35, linestyle="None")
        cursor = ax.axvline(0.0, color=WHITE, lw=0.8, alpha=0.45)
        exponent_lines.append(line)
        exponent_points.append(point)
        exponent_cursors.append(cursor)

    ax_l3.set_xlabel(r"Tempo $t$", fontsize=13)
    ax_l1.tick_params(labelbottom=False)
    ax_l2.tick_params(labelbottom=False)

    fig.text(0.35, 0.845, "Trajetória de referência",
             color=CYAN, fontsize=17, ha="center")
    fig.text(0.82, 0.85, "Convergência dos expoentes",
             color=WHITE, fontsize=17, ha="center")

    time_label = fig.text(0.52, 0.115, r"$t=0.00$",
                          fontsize=15, color=CYAN, ha="center")

    nframes = int(VIDEO_SECONDS * FPS)
    frame_indices = np.linspace(0, len(t) - 1, nframes).astype(int)

    def update(frame):
        k = frame_indices[frame]
        current_time = t[k]

        for j, (idx, _) in enumerate(variables):
            trajectory_lines[j].set_data(t[:k + 1], traj[:k + 1, idx])
            trajectory_cursors[j].set_xdata([current_time, current_time])

        has_estimate = False
        if len(t_spec) > 0:
            spec_k = np.searchsorted(t_spec, current_time, side="right") - 1
            spec_k = int(np.clip(spec_k, 0, len(t_spec) - 1))
            has_estimate = current_time >= T_TRANSIENT

        if has_estimate:
            for j in range(3):
                exponent_lines[j].set_data(
                    t_spec[:spec_k + 1], lambdas[:spec_k + 1, j]
                )
                exponent_points[j].set_data(
                    [t_spec[spec_k]], [lambdas[spec_k, j]]
                )
                exponent_cursors[j].set_xdata(
                    [current_time, current_time]
                )
        else:
            for j in range(3):
                exponent_lines[j].set_data([], [])
                exponent_points[j].set_data([], [])
                exponent_cursors[j].set_xdata(
                    [current_time, current_time]
                )

        time_label.set_text(rf"$t={current_time:.2f}$")
        return (
            trajectory_lines + trajectory_cursors
            + exponent_lines + exponent_points + exponent_cursors
            + [time_label]
        )

    ani = FuncAnimation(
        fig, update, frames=nframes,
        interval=1000 / FPS, blit=False
    )

    if shutil.which("ffmpeg"):
        print(f"\nSalvando vídeo: {OUT_MP4}")
        writer = FFMpegWriter(
            fps=FPS, bitrate=BITRATE,
            extra_args=["-pix_fmt", "yuv420p"]
        )
        ani.save(OUT_MP4, writer=writer, dpi=DPI)
        output = OUT_MP4
    else:
        print("\nffmpeg não encontrado. Salvando GIF.")
        ani.save(OUT_GIF, writer=PillowWriter(fps=FPS), dpi=90)
        output = OUT_GIF

    print(f"Arquivo salvo: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
