from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 2.2,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)

COLOR_SPLIT = "#A23B32"
COLOR_DILATION = "#0B5CAD"
COLOR_NEUTRAL = "#555555"

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"


def spectral_ratio(matrix: np.ndarray) -> float:
    numerator = np.linalg.norm(matrix, 2) ** 2
    denominator = np.linalg.norm(
        matrix @ matrix.conj().T + matrix.conj().T @ matrix, 2
    )
    return float(numerator / denominator)


def output_directories() -> tuple[Path, ...]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return (FIGURE_DIR,)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for folder in output_directories():
        fig.savefig(folder / f"{stem}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")


def pauli_family(gamma: float) -> np.ndarray:
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    return z + 1j * gamma * x


def jordan_family(gamma: float) -> np.ndarray:
    return np.array([[1, gamma], [0, 1]], dtype=complex)


def neutral_family() -> np.ndarray:
    return np.array([[0, 2], [0, 0]], dtype=complex)


def random_qsp_like_matrices(num_samples: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrices = rng.normal(size=(num_samples, 2, 2)) + 1j * rng.normal(
        size=(num_samples, 2, 2)
    )
    return np.array([spectral_ratio(matrix) for matrix in matrices])


def plot_variance_ratio_landscape() -> None:
    gamma = np.linspace(0.0, 10.0, 500)
    pauli_ratios = np.array([spectral_ratio(pauli_family(value)) for value in gamma])
    jordan_ratios = np.array([spectral_ratio(jordan_family(value)) for value in gamma])
    random_ratios = random_qsp_like_matrices(5000)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"wspace": 0.24})

    axes[0].plot(
        gamma,
        pauli_ratios,
        color=COLOR_DILATION,
    )
    axes[0].plot(
        gamma,
        jordan_ratios,
        color=COLOR_SPLIT,
        linestyle="--",
    )
    axes[0].axhline(0.5, color=COLOR_NEUTRAL, linestyle=":")
    axes[0].axhline(1.0, color=COLOR_NEUTRAL, linestyle="-.")
    axes[0].set_xlabel(r"Family parameter $\gamma$")
    axes[0].set_ylabel(r"Ratio $R=\|M\|_\infty^2/\|MM^\dagger + M^\dagger M\|_\infty$")
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0.48, 1.02)
    axes[0].set_title("Representative operator families")

    pauli_anchor_x = 3.5
    pauli_anchor_y = float(np.interp(pauli_anchor_x, gamma, pauli_ratios))
    axes[0].annotate(
        "Pauli-type family\n" + r"$M_{\rm P}(\gamma)=Z+i\gamma X$",
        xy=(pauli_anchor_x, pauli_anchor_y),
        xytext=(5.6, 0.93),
        color=COLOR_DILATION,
        fontsize=9.4,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.22),
        arrowprops=dict(arrowstyle="-", color=COLOR_DILATION, linewidth=1.2),
    )

    jordan_anchor_x = 7.8
    jordan_anchor_y = float(np.interp(jordan_anchor_x, gamma, jordan_ratios))
    axes[0].annotate(
        "Jordan-type family\n" + r"$M_{\rm J}(\gamma)=I+\gamma|0\rangle\langle 1|$",
        xy=(jordan_anchor_x, jordan_anchor_y),
        xytext=(5.9, 0.565),
        color=COLOR_SPLIT,
        fontsize=9.4,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.22),
        arrowprops=dict(arrowstyle="-", color=COLOR_SPLIT, linewidth=1.2),
    )
    axes[0].text(
        9.7,
        1.0,
        r"Upper bound $1$",
        fontsize=9.2,
        color=COLOR_NEUTRAL,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.15),
    )
    axes[0].text(
        9.7,
        0.5,
        r"Lower bound $1/2$",
        fontsize=9.2,
        color=COLOR_NEUTRAL,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.15),
    )

    axes[1].hist(
        random_ratios,
        bins=40,
        color=COLOR_DILATION,
        alpha=0.75,
        density=True,
        edgecolor="white",
    )
    axes[1].axvline(0.5, color=COLOR_NEUTRAL, linestyle=":")
    axes[1].axvline(1.0, color=COLOR_NEUTRAL, linestyle="-.")
    axes[1].axvline(
        float(np.mean(random_ratios)),
        color=COLOR_SPLIT,
        linestyle="--",
        label=rf"Sample mean = {np.mean(random_ratios):.3f}",
    )
    axes[1].set_xlabel(r"Ratio $R$ for random complex $2\times 2$ matrices")
    axes[1].set_ylabel("Probability density")
    axes[1].set_xlim(0.48, 1.02)
    axes[1].set_title("Random survey over QSP-sized operators")
    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(0.03, 0.995),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="none",
    )

    fig.suptitle(
        "Dilation-to-splitting variance ratio stays in [1/2, 1] but is strongly problem-dependent",
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    save_figure(fig, "Figure_1_Variance_Bound")
    plt.close(fig)


def qsp_jordan_signal(gamma: float) -> float:
    return 0.5 + 0.25 * gamma


def effective_cost_ratio(
    matrix: np.ndarray,
    signal_magnitude: float,
    cnot_error: float,
    prep_cost: float = 20.0,
    dilation_cnots: int = 3,
    spam_error: float = 0.002,
    target_rmse: float = 0.15,
) -> float:
    split_factor = np.linalg.norm(
        matrix @ matrix.conj().T + matrix.conj().T @ matrix, 2
    )
    dilation_factor = np.linalg.norm(matrix, 2) ** 2

    eta_split = 1.0 - 2.0 * spam_error
    eta_dilation = (1.0 - 2.0 * spam_error) ** 2 * (1.0 - cnot_error) ** dilation_cnots

    split_budget = target_rmse**2 - ((1.0 - eta_split) * abs(signal_magnitude)) ** 2
    dilation_budget = target_rmse**2 - (
        (1.0 - eta_dilation) * abs(signal_magnitude)
    ) ** 2

    if split_budget <= 0 or dilation_budget <= 0:
        return np.nan

    split_shots = split_factor / split_budget
    dilation_shots = dilation_factor / dilation_budget

    split_cost = prep_cost * split_shots
    dilation_cost = (prep_cost + dilation_cnots) * dilation_shots
    return float(dilation_cost / split_cost)


def plot_breakeven_analysis() -> None:
    gamma = np.linspace(0.0, 6.0, 300)
    cnot_errors = [0.0, 0.005, 0.01, 0.02]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    for p in cnot_errors:
        ratios = [
            effective_cost_ratio(jordan_family(g), qsp_jordan_signal(g), p)
            for g in gamma
        ]
        axes[0].plot(gamma, ratios, label=rf"$p_{{\rm CNOT}}={p:.3f}$")

    axes[0].axhline(1.0, color=COLOR_NEUTRAL, linestyle="--", label="Break-even")
    axes[0].set_xlabel(r"QSP coefficient parameter $\gamma$")
    axes[0].set_ylabel(r"Runtime ratio $C_{\rm dil}/C_{\rm split}$")
    axes[0].set_xlim(0, 6)
    axes[0].set_ylim(0.45, 1.55)
    axes[0].set_title("QSP Jordan family under noise")
    axes[0].legend(loc="upper left", frameon=False)

    cnot_axis = np.linspace(0.0, 0.03, 300)
    favorable = [
        effective_cost_ratio(jordan_family(1.0), qsp_jordan_signal(1.0), p)
        for p in cnot_axis
    ]
    neutral = [
        effective_cost_ratio(neutral_family(), 0.5, p)
        for p in cnot_axis
    ]

    axes[1].plot(
        cnot_axis,
        favorable,
        color=COLOR_DILATION,
        label=r"Advantageous QSP family ($\gamma=1$)",
    )
    axes[1].plot(
        cnot_axis,
        neutral,
        color=COLOR_SPLIT,
        linestyle="--",
        label=r"Neutral rank-one family",
    )
    axes[1].axhline(1.0, color=COLOR_NEUTRAL, linestyle="--", label="Break-even")
    axes[1].set_xlabel(r"CNOT error probability $p_{\rm CNOT}$")
    axes[1].set_ylabel(r"Runtime ratio $C_{\rm dil}/C_{\rm split}$")
    axes[1].set_xlim(0.0, 0.03)
    axes[1].set_ylim(0.45, 1.55)
    axes[1].set_title("Preparation-dominated cost model")
    axes[1].legend(loc="upper left", frameon=False)

    fig.suptitle(
        "Noise-aware break-even analysis for QSP-oriented unitary dilation",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "Figure_3_Breakeven")
    plt.close(fig)


if __name__ == "__main__":
    plot_variance_ratio_landscape()
    plot_breakeven_analysis()
