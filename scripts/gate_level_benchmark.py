from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorcircuit as tc
from matplotlib.colors import TwoSlopeNorm
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import schur, sqrtm
from tensorcircuit import channels

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 12.5,
        "axes.titlesize": 13.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
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

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

COLOR_SPLIT = "#A23B32"
COLOR_DILATION = "#0B5CAD"
COLOR_GENEXP = "#2A7F62"
COLOR_NEUTRAL = "#555555"
LIGHT_BLUE = "#EAF3FF"
LIGHT_RED = "#FDEEEE"
LIGHT_GREEN = "#EEF8F3"

jax.config.update("jax_enable_x64", True)
TC_BACKEND = tc.set_backend("jax")

ONE_QUBIT_DEPOLARIZING = 1.0e-4
TWO_QUBIT_DEPOLARIZING = 2.0e-3
ONE_TO_TWO_QUBIT_NOISE_RATIO = ONE_QUBIT_DEPOLARIZING / TWO_QUBIT_DEPOLARIZING
CNOT_BUDGETS = np.logspace(3.0, 5.0, 220)
SENSITIVITY_TWO_QUBIT_NOISES = np.array(
    [2.5e-4, 5.0e-4, 1.0e-3, 1.5e-3, 2.0e-3, 3.0e-3, 4.0e-3, 5.0e-3]
)
SENSITIVITY_WORKLOAD_LAYERS = (1, 2, 3, 4)
ENSEMBLE_SAMPLE_COUNT = 24
ENSEMBLE_TARGET_SPECTRAL_NORM = 2.0
ENSEMBLE_SEED = 17
SEARCH_PARAM_INDICES = (3, 8)
SEARCH_COARSE_GRID = np.linspace(-np.pi, np.pi, 5)
SEARCH_FINE_DELTAS = (np.pi / 4.0, np.pi / 8.0)
SEARCH_BUDGETS = np.logspace(4.3, 5.5, 7)
SEARCH_SCORE_PHASE = float(np.arctan(5.0))
SEARCH_TRIALS = 96
SEARCH_DENSE_GRID_SIZE = 33
SEARCH_SEED = 23
ZZ_OPERATOR = np.diag([1.0, -1.0, -1.0, 1.0]).astype(complex)
IDEAL_MATCH_TOLERANCE = 1.0e-6

# Fixed workload angles for a reproducible three-qubit benchmark circuit.
WORKLOAD_ANGLES = (
    0.31,
    -0.52,
    0.73,
    0.44,
    -0.28,
    0.61,
    0.35,
    -0.47,
    0.18,
    -0.59,
    0.84,
    0.27,
)
WORKLOAD_LAYER_PERTURBATION = np.array(
    [0.05, -0.04, 0.03, 0.02, -0.02, 0.04, 0.02, -0.03, 0.03, -0.04, 0.05, 0.02]
)


@dataclass
class GateCounts:
    qubits: int
    u_gates: float
    cx_gates: float
    depth: float


@dataclass
class MethodStatistics:
    family: str
    method: str
    workload_layers: int
    target_real: float
    target_imag: float
    ideal_mean_real: float
    ideal_mean_imag: float
    ideal_variance: float
    noisy_mean_real: float
    noisy_mean_imag: float
    noisy_variance: float
    noisy_bias_squared: float
    cx_per_shot: float
    u_per_shot: float
    depth_per_shot: float
    qubits: int


@dataclass
class CompiledBenchmark:
    family: str
    workload_layers: int
    matrix: np.ndarray
    full_operator: np.ndarray
    target: complex
    prep_split_instructions: list[tuple[str, list[int], list[float]]]
    prep_dilation_instructions: list[tuple[str, list[int], list[float]]]
    prep_genexp_instructions: list[tuple[str, list[int], list[float]]]
    split_hermitian_basis_instructions: list[tuple[str, list[int], list[float]]]
    split_antihermitian_basis_instructions: list[tuple[str, list[int], list[float]]]
    dilation_basis_instructions: list[tuple[str, list[int], list[float]]]
    genexp_hermitian_basis_instructions: list[tuple[str, list[int], list[float]]]
    genexp_antihermitian_basis_instructions: list[tuple[str, list[int], list[float]]]
    split_hermitian_eigs: np.ndarray
    split_antihermitian_eigs: np.ndarray
    dilation_eigs: np.ndarray
    genexp_hermitian_eigs: np.ndarray
    genexp_antihermitian_eigs: np.ndarray
    split_counts: GateCounts
    dilation_counts: GateCounts
    genexp_counts: GateCounts
    ideal_split_mean: complex
    ideal_split_variance: float
    ideal_dilation_mean: complex
    ideal_dilation_variance: float
    ideal_genexp_mean: complex
    ideal_genexp_variance: float


@dataclass
class SensitivityPoint:
    family: str
    workload_layers: int
    one_qubit_noise: float
    two_qubit_noise: float
    split_noisy_factor: float
    dilation_noisy_factor: float
    genexp_noisy_factor: float
    dil_vs_split_noisy_ratio: float
    dil_vs_genexp_noisy_ratio: float
    split_cx_per_shot: float
    dilation_cx_per_shot: float
    genexp_cx_per_shot: float


@dataclass
class EnsemblePoint:
    sample_id: int
    workload_layers: int
    one_qubit_noise: float
    two_qubit_noise: float
    operator_ratio: float
    dil_vs_split_ideal_ratio: float
    dil_vs_split_noisy_ratio: float
    dil_vs_genexp_ideal_ratio: float
    dil_vs_genexp_noisy_ratio: float
    genexp_vs_split_ideal_ratio: float
    genexp_vs_split_noisy_ratio: float
    split_ideal_factor: float
    dilation_ideal_factor: float
    genexp_ideal_factor: float
    split_noisy_factor: float
    dilation_noisy_factor: float
    genexp_noisy_factor: float
    split_cx_per_shot: float
    dilation_cx_per_shot: float
    genexp_cx_per_shot: float


@dataclass
class SearchBudgetPoint:
    method: str
    total_cnot_budget: float
    shots_per_evaluation: int
    mean_reported_best_score: float
    mean_selected_true_score: float
    mean_regret: float
    task_rmse: float
    schedule_cx_cost: float


def output_directories() -> tuple[Path, ...]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return (FIGURE_DIR,)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for folder in output_directories():
        fig.savefig(folder / f"{stem}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_csv(rows: list[object], filename: str) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def jordan_family(gamma: float) -> np.ndarray:
    return np.array([[1.0, gamma], [0.0, 1.0]], dtype=complex)


def neutral_family() -> np.ndarray:
    return np.array([[0.0, 2.0], [0.0, 0.0]], dtype=complex)


def random_normalized_ensemble(
    num_samples: int = ENSEMBLE_SAMPLE_COUNT,
    seed: int = ENSEMBLE_SEED,
    target_spectral_norm: float = ENSEMBLE_TARGET_SPECTRAL_NORM,
) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    matrices: list[tuple[str, np.ndarray]] = []
    for sample_id in range(num_samples):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        matrix *= target_spectral_norm / np.linalg.norm(matrix, 2)
        matrices.append((f"random_norm2_{sample_id:02d}", matrix))
    return matrices


def qsp_full_operator(matrix: np.ndarray) -> np.ndarray:
    return np.kron(ZZ_OPERATOR, matrix)


def dilation_operator(matrix: np.ndarray) -> np.ndarray:
    spectral_norm = np.linalg.norm(matrix, 2)
    normalized = matrix / spectral_norm
    top_right = sqrtm(np.eye(2) - normalized @ normalized.conj().T)
    bottom_left = sqrtm(np.eye(2) - normalized.conj().T @ normalized)
    return spectral_norm * np.block(
        [
            [normalized, top_right],
            [bottom_left, -normalized.conj().T],
        ]
    )


def transpile_instruction_list(circuit: QuantumCircuit) -> tuple[list[tuple[str, list[int], list[float]]], GateCounts]:
    transpiled = transpile(circuit, basis_gates=["u", "cx"], optimization_level=3)
    instructions: list[tuple[str, list[int], list[float]]] = []
    for item in transpiled.data:
        instructions.append(
            (
                item.operation.name,
                [qubit._index for qubit in item.qubits],
                [float(parameter) for parameter in item.operation.params],
            )
        )

    ops = transpiled.count_ops()
    return instructions, GateCounts(
        qubits=transpiled.num_qubits,
        u_gates=float(ops.get("u", 0)),
        cx_gates=float(ops.get("cx", 0)),
        depth=float(transpiled.depth()),
    )


def workload_angles_for_layer(
    layer_index: int,
    override_angles: np.ndarray | None = None,
) -> np.ndarray:
    base = np.asarray(override_angles if override_angles is not None else WORKLOAD_ANGLES, dtype=float)
    return base + layer_index * WORKLOAD_LAYER_PERTURBATION


def append_workload_layer(
    circuit: QuantumCircuit,
    qubits: tuple[int, int, int],
    angles: np.ndarray,
) -> None:
    ancilla, system_0, system_1 = qubits
    a = angles

    circuit.ry(a[0], system_0)
    circuit.rx(a[1], system_1)
    circuit.cx(ancilla, system_0)
    circuit.cx(system_0, system_1)

    circuit.rz(a[2], ancilla)
    circuit.ry(a[3], system_0)
    circuit.rx(a[4], system_1)
    circuit.cx(system_1, ancilla)
    circuit.cx(ancilla, system_0)

    circuit.rz(a[5], system_0)
    circuit.ry(a[6], system_1)
    circuit.cx(system_0, system_1)
    circuit.cx(system_1, ancilla)

    circuit.rx(a[7], ancilla)
    circuit.ry(a[8], system_0)
    circuit.rz(a[9], system_1)
    circuit.cx(ancilla, system_0)
    circuit.cx(system_0, system_1)

    circuit.ry(a[10], ancilla)
    circuit.rx(a[11], system_1)


def build_workload_circuit(
    total_qubits: int,
    qubits: tuple[int, int, int] = (0, 1, 2),
    workload_layers: int = 1,
    override_angles: np.ndarray | None = None,
) -> QuantumCircuit:
    circuit = QuantumCircuit(total_qubits)
    circuit.h(qubits[0])
    for layer_index in range(workload_layers):
        append_workload_layer(
            circuit,
            qubits,
            workload_angles_for_layer(layer_index, override_angles=override_angles),
        )
    return circuit


def build_genexp_prep_circuit(
    workload_layers: int = 1,
    override_angles: np.ndarray | None = None,
) -> QuantumCircuit:
    circuit = QuantumCircuit(7)
    circuit.compose(
        build_workload_circuit(
            7,
            qubits=(0, 1, 2),
            workload_layers=workload_layers,
            override_angles=override_angles,
        ),
        inplace=True,
    )
    circuit.compose(
        build_workload_circuit(
            7,
            qubits=(3, 4, 5),
            workload_layers=workload_layers,
            override_angles=override_angles,
        ),
        inplace=True,
    )
    circuit.h(6)
    circuit.cswap(6, 0, 3)
    circuit.cswap(6, 1, 4)
    circuit.cswap(6, 2, 5)
    return circuit


def build_basis_circuit(
    total_qubits: int,
    unitary: np.ndarray,
    target_qubits: list[int],
    x_basis_qubits: list[int] | None = None,
) -> tuple[list[tuple[str, list[int], list[float]]], GateCounts]:
    circuit = QuantumCircuit(total_qubits)
    circuit.unitary(Operator(unitary), target_qubits)
    for qubit in x_basis_qubits or []:
        circuit.h(qubit)
    return transpile_instruction_list(circuit)


def apply_instruction_list(
    circuit: tc.AbstractCircuit,
    instructions: list[tuple[str, list[int], list[float]]],
    noisy: bool,
    one_qubit_noise: float,
    two_qubit_noise: float,
) -> None:
    for gate_name, qubits, parameters in instructions:
        if gate_name == "u":
            circuit.u(qubits[0], theta=parameters[0], phi=parameters[1], lbd=parameters[2])
            if noisy and one_qubit_noise > 0.0:
                circuit.apply_general_kraus(
                    channels.generaldepolarizingchannel(one_qubit_noise, 1),
                    qubits[0],
                )
        elif gate_name == "cx":
            circuit.cx(qubits[0], qubits[1])
            if noisy and two_qubit_noise > 0.0:
                circuit.apply_general_kraus(
                    channels.generaldepolarizingchannel(two_qubit_noise, 2),
                    qubits[0],
                    qubits[1],
                )
        else:
            raise ValueError(f"Unsupported transpiled gate: {gate_name}")


def simulate_probabilities(
    num_qubits: int,
    instructions: list[tuple[str, list[int], list[float]]],
    noisy: bool,
    one_qubit_noise: float,
    two_qubit_noise: float,
) -> np.ndarray:
    circuit = tc.DMCircuit(num_qubits) if noisy else tc.Circuit(num_qubits)
    apply_instruction_list(
        circuit,
        instructions,
        noisy=noisy,
        one_qubit_noise=one_qubit_noise,
        two_qubit_noise=two_qubit_noise,
    )
    if noisy:
        probabilities = np.asarray(circuit.probability())
    else:
        state = np.asarray(circuit.state()).reshape(-1)
        probabilities = np.abs(state) ** 2
    return probabilities / np.sum(probabilities)


def expectation_from_statevector(
    matrix: np.ndarray,
    workload_layers: int = 1,
    override_angles: np.ndarray | None = None,
) -> complex:
    state = Statevector.from_instruction(
        build_workload_circuit(
            3,
            qubits=(0, 1, 2),
            workload_layers=workload_layers,
            override_angles=override_angles,
        )
    ).data
    operator = qsp_full_operator(matrix)
    return complex(np.vdot(state, operator @ state))


def combine_counts(first: GateCounts, second: GateCounts) -> GateCounts:
    return GateCounts(
        qubits=max(first.qubits, second.qubits),
        u_gates=first.u_gates + second.u_gates,
        cx_gates=first.cx_gates + second.cx_gates,
        depth=first.depth + second.depth,
    )


def average_counts(weighted_counts: list[tuple[float, GateCounts]]) -> GateCounts:
    return GateCounts(
        qubits=max(counts.qubits for _, counts in weighted_counts),
        u_gates=float(sum(weight * counts.u_gates for weight, counts in weighted_counts)),
        cx_gates=float(sum(weight * counts.cx_gates for weight, counts in weighted_counts)),
        depth=float(sum(weight * counts.depth for weight, counts in weighted_counts)),
    )


def distribution_moments(values: np.ndarray, probabilities: np.ndarray) -> tuple[complex, float]:
    mean = np.sum(probabilities * values)
    second_moment = np.sum(probabilities * np.abs(values) ** 2)
    return complex(mean), float(np.real(second_moment - abs(mean) ** 2))


def projected_distribution_stats(
    values: np.ndarray,
    probabilities: np.ndarray,
    phase: float,
) -> tuple[float, float]:
    scores = np.real(np.exp(-1.0j * phase) * values)
    mean = float(np.sum(probabilities * scores))
    second = float(np.sum(probabilities * scores**2))
    return mean, float(second - mean**2)


def zz_from_bits(first_system_bit: int, second_system_bit: int) -> int:
    eigen_first = 1 if first_system_bit == 0 else -1
    eigen_second = 1 if second_system_bit == 0 else -1
    return eigen_first * eigen_second


def split_outcome_distribution(
    prep_instructions: list[tuple[str, list[int], list[float]]],
    hermitian_basis: list[tuple[str, list[int], list[float]]],
    antihermitian_basis: list[tuple[str, list[int], list[float]]],
    hermitian_eigs: np.ndarray,
    antihermitian_eigs: np.ndarray,
    noisy: bool,
    one_qubit_noise: float,
    two_qubit_noise: float,
) -> tuple[complex, float]:
    probabilities_h = simulate_probabilities(
        3,
        prep_instructions + hermitian_basis,
        noisy=noisy,
        one_qubit_noise=one_qubit_noise,
        two_qubit_noise=two_qubit_noise,
    )
    probabilities_a = simulate_probabilities(
        3,
        prep_instructions + antihermitian_basis,
        noisy=noisy,
        one_qubit_noise=one_qubit_noise,
        two_qubit_noise=two_qubit_noise,
    )

    values: list[complex] = []
    probabilities: list[float] = []
    for index, probability in enumerate(probabilities_h):
        ancilla = (index >> 2) & 1
        system_0 = (index >> 1) & 1
        system_1 = index & 1
        values.append(2.0 * zz_from_bits(system_0, system_1) * hermitian_eigs[ancilla])
        probabilities.append(0.5 * float(probability))

    for index, probability in enumerate(probabilities_a):
        ancilla = (index >> 2) & 1
        system_0 = (index >> 1) & 1
        system_1 = index & 1
        values.append(2.0j * zz_from_bits(system_0, system_1) * antihermitian_eigs[ancilla])
        probabilities.append(0.5 * float(probability))
    return np.asarray(values, dtype=complex), np.asarray(probabilities, dtype=float)


def dilation_outcome_distribution(
    prep_instructions: list[tuple[str, list[int], list[float]]],
    basis_instructions: list[tuple[str, list[int], list[float]]],
    dilation_eigs: np.ndarray,
    noisy: bool,
    one_qubit_noise: float,
    two_qubit_noise: float,
) -> tuple[complex, float]:
    probabilities = simulate_probabilities(
        4,
        prep_instructions + basis_instructions,
        noisy=noisy,
        one_qubit_noise=one_qubit_noise,
        two_qubit_noise=two_qubit_noise,
    )
    values: list[complex] = []
    weights: list[float] = []
    for index, probability in enumerate(probabilities):
        ancilla = (index >> 3) & 1
        system_0 = (index >> 2) & 1
        system_1 = (index >> 1) & 1
        extra_ancilla = index & 1
        pair_index = ancilla + 2 * extra_ancilla
        values.append(zz_from_bits(system_0, system_1) * dilation_eigs[pair_index])
        weights.append(float(probability))
    return np.asarray(values, dtype=complex), np.asarray(weights, dtype=float)


def genexp_outcome_distribution(
    prep_instructions: list[tuple[str, list[int], list[float]]],
    hermitian_basis: list[tuple[str, list[int], list[float]]],
    antihermitian_basis: list[tuple[str, list[int], list[float]]],
    hermitian_eigs: np.ndarray,
    antihermitian_eigs: np.ndarray,
    noisy: bool,
    one_qubit_noise: float,
    two_qubit_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities_h = simulate_probabilities(
        7,
        prep_instructions + hermitian_basis,
        noisy=noisy,
        one_qubit_noise=one_qubit_noise,
        two_qubit_noise=two_qubit_noise,
    )
    probabilities_a = simulate_probabilities(
        7,
        prep_instructions + antihermitian_basis,
        noisy=noisy,
        one_qubit_noise=one_qubit_noise,
        two_qubit_noise=two_qubit_noise,
    )

    values: list[complex] = []
    probabilities: list[float] = []
    for index, probability in enumerate(probabilities_h):
        q0 = (index >> 6) & 1
        q1 = (index >> 5) & 1
        q2 = (index >> 4) & 1
        left_register = q2 * 4 + q1 * 2 + q0
        ancilla_x = index & 1
        x_sign = 1.0 if ancilla_x == 0 else -1.0
        values.append(2.0 * x_sign * hermitian_eigs[left_register])
        probabilities.append(0.5 * float(probability))

    for index, probability in enumerate(probabilities_a):
        q0 = (index >> 6) & 1
        q1 = (index >> 5) & 1
        q2 = (index >> 4) & 1
        left_register = q2 * 4 + q1 * 2 + q0
        ancilla_x = index & 1
        x_sign = 1.0 if ancilla_x == 0 else -1.0
        values.append(2.0j * x_sign * antihermitian_eigs[left_register])
        probabilities.append(0.5 * float(probability))
    return np.asarray(values, dtype=complex), np.asarray(probabilities, dtype=float)


def operator_variance_ratio(matrix: np.ndarray) -> float:
    split_factor = np.linalg.norm(matrix @ matrix.conj().T + matrix.conj().T @ matrix, 2)
    dilation_factor = np.linalg.norm(matrix, 2) ** 2
    return float(dilation_factor / split_factor)


def compile_family_benchmark(
    family: str,
    matrix: np.ndarray,
    workload_layers: int = 1,
    override_angles: np.ndarray | None = None,
) -> CompiledBenchmark:
    full_operator = qsp_full_operator(matrix)
    direct_target = expectation_from_statevector(
        matrix,
        workload_layers=workload_layers,
        override_angles=override_angles,
    )

    hermitian = 0.5 * (matrix + matrix.conj().T)
    antihermitian = (matrix - matrix.conj().T) / (2.0j)
    hermitian_full = 0.5 * (full_operator + full_operator.conj().T)
    antihermitian_full = (full_operator - full_operator.conj().T) / (2.0j)
    split_hermitian_eigs, split_hermitian_vecs = np.linalg.eigh(hermitian)
    split_antihermitian_eigs, split_antihermitian_vecs = np.linalg.eigh(antihermitian)
    genexp_hermitian_eigs, genexp_hermitian_vecs = np.linalg.eigh(hermitian_full)
    genexp_antihermitian_eigs, genexp_antihermitian_vecs = np.linalg.eigh(antihermitian_full)

    dilation = dilation_operator(matrix)
    schur_form, schur_vectors = schur(dilation, output="complex")
    dilation_eigs = np.diag(schur_form)

    prep_split_instructions, prep_split_counts = transpile_instruction_list(
        build_workload_circuit(
            3,
            qubits=(0, 1, 2),
            workload_layers=workload_layers,
            override_angles=override_angles,
        )
    )
    prep_dilation_instructions, prep_dilation_counts = transpile_instruction_list(
        build_workload_circuit(
            4,
            qubits=(0, 1, 2),
            workload_layers=workload_layers,
            override_angles=override_angles,
        )
    )
    prep_genexp_instructions, prep_genexp_counts = transpile_instruction_list(
        build_genexp_prep_circuit(
            workload_layers=workload_layers,
            override_angles=override_angles,
        )
    )
    split_hermitian_basis_instructions, split_hermitian_basis_counts = build_basis_circuit(
        total_qubits=3,
        unitary=split_hermitian_vecs.conj().T,
        target_qubits=[0],
    )
    split_antihermitian_basis_instructions, split_antihermitian_basis_counts = build_basis_circuit(
        total_qubits=3,
        unitary=split_antihermitian_vecs.conj().T,
        target_qubits=[0],
    )
    dilation_basis_instructions, dilation_basis_counts = build_basis_circuit(
        total_qubits=4,
        unitary=schur_vectors.conj().T,
        target_qubits=[0, 3],
    )
    genexp_hermitian_basis_instructions, genexp_hermitian_basis_counts = build_basis_circuit(
        total_qubits=7,
        unitary=genexp_hermitian_vecs.conj().T,
        target_qubits=[0, 1, 2],
        x_basis_qubits=[6],
    )
    genexp_antihermitian_basis_instructions, genexp_antihermitian_basis_counts = build_basis_circuit(
        total_qubits=7,
        unitary=genexp_antihermitian_vecs.conj().T,
        target_qubits=[0, 1, 2],
        x_basis_qubits=[6],
    )

    split_counts = combine_counts(
        prep_split_counts,
        average_counts(
            [
                (0.5, split_hermitian_basis_counts),
                (0.5, split_antihermitian_basis_counts),
            ]
        ),
    )
    dilation_counts = combine_counts(prep_dilation_counts, dilation_basis_counts)
    genexp_counts = combine_counts(
        prep_genexp_counts,
        average_counts(
            [
                (0.5, genexp_hermitian_basis_counts),
                (0.5, genexp_antihermitian_basis_counts),
            ]
        ),
    )

    ideal_split_values, ideal_split_probabilities = split_outcome_distribution(
        prep_split_instructions,
        split_hermitian_basis_instructions,
        split_antihermitian_basis_instructions,
        split_hermitian_eigs,
        split_antihermitian_eigs,
        noisy=False,
        one_qubit_noise=0.0,
        two_qubit_noise=0.0,
    )
    ideal_dilation_values, ideal_dilation_probabilities = dilation_outcome_distribution(
        prep_dilation_instructions,
        dilation_basis_instructions,
        dilation_eigs,
        noisy=False,
        one_qubit_noise=0.0,
        two_qubit_noise=0.0,
    )
    ideal_genexp_values, ideal_genexp_probabilities = genexp_outcome_distribution(
        prep_genexp_instructions,
        genexp_hermitian_basis_instructions,
        genexp_antihermitian_basis_instructions,
        genexp_hermitian_eigs,
        genexp_antihermitian_eigs,
        noisy=False,
        one_qubit_noise=0.0,
        two_qubit_noise=0.0,
    )

    ideal_split_mean, ideal_split_variance = distribution_moments(
        ideal_split_values,
        ideal_split_probabilities,
    )
    ideal_dilation_mean, ideal_dilation_variance = distribution_moments(
        ideal_dilation_values,
        ideal_dilation_probabilities,
    )
    ideal_genexp_mean, ideal_genexp_variance = distribution_moments(
        ideal_genexp_values,
        ideal_genexp_probabilities,
    )

    for method_name, mean in (
        ("split", ideal_split_mean),
        ("dilation", ideal_dilation_mean),
        ("genexp", ideal_genexp_mean),
    ):
        if abs(mean - ideal_split_mean) > IDEAL_MATCH_TOLERANCE:
            raise ValueError(
                f"Ideal {method_name} realization failed for {family}: "
                f"|{mean} - {ideal_split_mean}| = {abs(mean - ideal_split_mean)}"
            )
    target = ideal_split_mean

    return CompiledBenchmark(
        family=family,
        workload_layers=workload_layers,
        matrix=matrix,
        full_operator=full_operator,
        target=target,
        prep_split_instructions=prep_split_instructions,
        prep_dilation_instructions=prep_dilation_instructions,
        prep_genexp_instructions=prep_genexp_instructions,
        split_hermitian_basis_instructions=split_hermitian_basis_instructions,
        split_antihermitian_basis_instructions=split_antihermitian_basis_instructions,
        dilation_basis_instructions=dilation_basis_instructions,
        genexp_hermitian_basis_instructions=genexp_hermitian_basis_instructions,
        genexp_antihermitian_basis_instructions=genexp_antihermitian_basis_instructions,
        split_hermitian_eigs=split_hermitian_eigs,
        split_antihermitian_eigs=split_antihermitian_eigs,
        dilation_eigs=dilation_eigs,
        genexp_hermitian_eigs=genexp_hermitian_eigs,
        genexp_antihermitian_eigs=genexp_antihermitian_eigs,
        split_counts=split_counts,
        dilation_counts=dilation_counts,
        genexp_counts=genexp_counts,
        ideal_split_mean=ideal_split_mean,
        ideal_split_variance=float(ideal_split_variance),
        ideal_dilation_mean=ideal_dilation_mean,
        ideal_dilation_variance=float(ideal_dilation_variance),
        ideal_genexp_mean=ideal_genexp_mean,
        ideal_genexp_variance=float(ideal_genexp_variance),
    )


def evaluate_compiled_benchmark(
    benchmark: CompiledBenchmark,
    one_qubit_noise: float = ONE_QUBIT_DEPOLARIZING,
    two_qubit_noise: float = TWO_QUBIT_DEPOLARIZING,
) -> list[MethodStatistics]:
    noisy_split_values, noisy_split_probabilities = split_outcome_distribution(
        benchmark.prep_split_instructions,
        benchmark.split_hermitian_basis_instructions,
        benchmark.split_antihermitian_basis_instructions,
        benchmark.split_hermitian_eigs,
        benchmark.split_antihermitian_eigs,
        noisy=True,
        one_qubit_noise=one_qubit_noise,
        two_qubit_noise=two_qubit_noise,
    )
    noisy_dilation_values, noisy_dilation_probabilities = dilation_outcome_distribution(
        benchmark.prep_dilation_instructions,
        benchmark.dilation_basis_instructions,
        benchmark.dilation_eigs,
        noisy=True,
        one_qubit_noise=one_qubit_noise,
        two_qubit_noise=two_qubit_noise,
    )
    noisy_genexp_values, noisy_genexp_probabilities = genexp_outcome_distribution(
        benchmark.prep_genexp_instructions,
        benchmark.genexp_hermitian_basis_instructions,
        benchmark.genexp_antihermitian_basis_instructions,
        benchmark.genexp_hermitian_eigs,
        benchmark.genexp_antihermitian_eigs,
        noisy=True,
        one_qubit_noise=one_qubit_noise,
        two_qubit_noise=two_qubit_noise,
    )

    noisy_split_mean, noisy_split_variance = distribution_moments(
        noisy_split_values,
        noisy_split_probabilities,
    )
    noisy_dilation_mean, noisy_dilation_variance = distribution_moments(
        noisy_dilation_values,
        noisy_dilation_probabilities,
    )
    noisy_genexp_mean, noisy_genexp_variance = distribution_moments(
        noisy_genexp_values,
        noisy_genexp_probabilities,
    )

    target = benchmark.target
    return [
        MethodStatistics(
            family=benchmark.family,
            method="split",
            workload_layers=benchmark.workload_layers,
            target_real=float(np.real(target)),
            target_imag=float(np.imag(target)),
            ideal_mean_real=float(np.real(benchmark.ideal_split_mean)),
            ideal_mean_imag=float(np.imag(benchmark.ideal_split_mean)),
            ideal_variance=float(benchmark.ideal_split_variance),
            noisy_mean_real=float(np.real(noisy_split_mean)),
            noisy_mean_imag=float(np.imag(noisy_split_mean)),
            noisy_variance=float(noisy_split_variance),
            noisy_bias_squared=float(abs(noisy_split_mean - target) ** 2),
            cx_per_shot=benchmark.split_counts.cx_gates,
            u_per_shot=benchmark.split_counts.u_gates,
            depth_per_shot=benchmark.split_counts.depth,
            qubits=benchmark.split_counts.qubits,
        ),
        MethodStatistics(
            family=benchmark.family,
            method="dilation",
            workload_layers=benchmark.workload_layers,
            target_real=float(np.real(target)),
            target_imag=float(np.imag(target)),
            ideal_mean_real=float(np.real(benchmark.ideal_dilation_mean)),
            ideal_mean_imag=float(np.imag(benchmark.ideal_dilation_mean)),
            ideal_variance=float(benchmark.ideal_dilation_variance),
            noisy_mean_real=float(np.real(noisy_dilation_mean)),
            noisy_mean_imag=float(np.imag(noisy_dilation_mean)),
            noisy_variance=float(noisy_dilation_variance),
            noisy_bias_squared=float(abs(noisy_dilation_mean - target) ** 2),
            cx_per_shot=benchmark.dilation_counts.cx_gates,
            u_per_shot=benchmark.dilation_counts.u_gates,
            depth_per_shot=benchmark.dilation_counts.depth,
            qubits=benchmark.dilation_counts.qubits,
        ),
        MethodStatistics(
            family=benchmark.family,
            method="genexp",
            workload_layers=benchmark.workload_layers,
            target_real=float(np.real(target)),
            target_imag=float(np.imag(target)),
            ideal_mean_real=float(np.real(benchmark.ideal_genexp_mean)),
            ideal_mean_imag=float(np.imag(benchmark.ideal_genexp_mean)),
            ideal_variance=float(benchmark.ideal_genexp_variance),
            noisy_mean_real=float(np.real(noisy_genexp_mean)),
            noisy_mean_imag=float(np.imag(noisy_genexp_mean)),
            noisy_variance=float(noisy_genexp_variance),
            noisy_bias_squared=float(abs(noisy_genexp_mean - target) ** 2),
            cx_per_shot=benchmark.genexp_counts.cx_gates,
            u_per_shot=benchmark.genexp_counts.u_gates,
            depth_per_shot=benchmark.genexp_counts.depth,
            qubits=benchmark.genexp_counts.qubits,
        ),
    ]


def rmse_curve(mse_factor: float, cx_per_shot: float, budgets: np.ndarray) -> np.ndarray:
    shots = budgets / cx_per_shot
    return np.sqrt(mse_factor / shots)


def budget_normalized_mse_factor(row: MethodStatistics) -> float:
    return row.cx_per_shot * (row.noisy_variance + row.noisy_bias_squared)


def budget_normalized_ideal_factor(row: MethodStatistics) -> float:
    return row.cx_per_shot * row.ideal_variance


def plot_gate_level_figure(rows: list[MethodStatistics]) -> None:
    grouped: dict[tuple[str, str], MethodStatistics] = {
        (row.family, row.method): row for row in rows
    }
    methods = [
        ("split", "Randomized split", LIGHT_RED, COLOR_SPLIT),
        ("dilation", "Coherent dilation", LIGHT_BLUE, COLOR_DILATION),
        ("genexp", "Genexp baseline", LIGHT_GREEN, COLOR_GENEXP),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.3, 5.0),
        gridspec_kw={"width_ratios": [1.12, 1.22, 1.22]},
    )

    categories = ["Qubits", "U gates", "CX gates", "Depth"]
    positions = np.arange(len(categories))
    width = 0.24
    for offset, (method, label, face, edge) in enumerate(methods):
        row = grouped[("jordan_gamma_1", method)]
        values = [row.qubits, row.u_per_shot, row.cx_per_shot, row.depth_per_shot]
        axes[0].bar(
            positions + (offset - 1) * width,
            values,
            width=width,
            color=face,
            edgecolor=edge,
            label=label,
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(categories, rotation=18)
    axes[0].set_title("Per-shot compiled resources")
    axes[0].legend(frameon=False, loc="upper left")

    def add_rmse_panel(ax: plt.Axes, family_key: str, title: str) -> None:
        for method, label, _, edge in methods:
            row = grouped[(family_key, method)]
            ideal_curve = rmse_curve(row.ideal_variance, row.cx_per_shot, CNOT_BUDGETS)
            noisy_curve = rmse_curve(
                row.noisy_variance + row.noisy_bias_squared,
                row.cx_per_shot,
                CNOT_BUDGETS,
            )
            ax.plot(CNOT_BUDGETS, noisy_curve, color=edge, label=f"{label}, noisy")
            ax.plot(CNOT_BUDGETS, ideal_curve, color=edge, linestyle="--", label=f"{label}, ideal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Total CNOT budget $B_{\rm CX}$")
        ax.set_title(title)

    add_rmse_panel(axes[1], "jordan_gamma_1", "Beneficial family: Jordan $\\gamma=1$")
    axes[1].set_ylabel(r"End-to-end RMSE")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8.0)

    add_rmse_panel(axes[2], "neutral_rank_one", "Neutral family: rank-one control")

    fig.suptitle(
        "Compiled three-way benchmark on the Holmes-QSP measurement primitive",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "Figure_4_GateLevel_EndToEnd")


def plot_sensitivity_figure(points: list[SensitivityPoint]) -> None:
    grouped = {
        (point.family, point.workload_layers, point.two_qubit_noise): point
        for point in points
    }
    families = [
        ("jordan_gamma_1", "Jordan $\\gamma=1$"),
        ("neutral_rank_one", "Rank-one control"),
    ]

    all_ratios = np.array(
        [
            ratio
            for point in points
            for ratio in (point.dil_vs_split_noisy_ratio, point.dil_vs_genexp_noisy_ratio)
        ],
        dtype=float,
    )
    ratio_norm = TwoSlopeNorm(
        vmin=float(np.min(all_ratios)),
        vcenter=1.0,
        vmax=float(np.max(all_ratios)),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.8), sharex=True, sharey=True)
    for row_index, (family, family_title) in enumerate(families):
        dil_vs_split = np.array(
            [
                [
                    grouped[(family, layers, noise)].dil_vs_split_noisy_ratio
                    for noise in SENSITIVITY_TWO_QUBIT_NOISES
                ]
                for layers in SENSITIVITY_WORKLOAD_LAYERS
            ]
        )
        dil_vs_genexp = np.array(
            [
                [
                    grouped[(family, layers, noise)].dil_vs_genexp_noisy_ratio
                    for noise in SENSITIVITY_TWO_QUBIT_NOISES
                ]
                for layers in SENSITIVITY_WORKLOAD_LAYERS
            ]
        )
        matrices = [dil_vs_split, dil_vs_genexp]
        titles = [
            f"{family_title}: dilation / split",
            f"{family_title}: dilation / genexp",
        ]
        for col_index, (matrix, title) in enumerate(zip(matrices, titles)):
            ax = axes[row_index, col_index]
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                norm=ratio_norm,
            )
            ax.set_title(title)
            if row_index == 1:
                ax.set_xlabel(r"Two-qubit noise $p_{2\mathrm{q}}$")
            if col_index == 0:
                ax.set_ylabel("Shared workload layers")
            ax.set_xticks(np.arange(len(SENSITIVITY_TWO_QUBIT_NOISES)))
            ax.set_xticklabels(
                [f"{value:.1e}" for value in SENSITIVITY_TWO_QUBIT_NOISES],
                rotation=35,
                ha="right",
            )
            ax.set_yticks(np.arange(len(SENSITIVITY_WORKLOAD_LAYERS)))
            ax.set_yticklabels([str(value) for value in SENSITIVITY_WORKLOAD_LAYERS])
            for inner_row in range(len(SENSITIVITY_WORKLOAD_LAYERS)):
                for inner_col in range(len(SENSITIVITY_TWO_QUBIT_NOISES)):
                    value = matrix[inner_row, inner_col]
                    text_color = "white" if abs(value - 1.0) > 0.14 else "black"
                    ax.text(
                        inner_col,
                        inner_row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8.4,
                        color=text_color,
                    )

    colorbar_axis = fig.add_axes([0.915, 0.17, 0.022, 0.64])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Budget-normalized MSE ratio")
    fig.suptitle(
        "Sensitivity of coherent dilation against theorem and practical baselines",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.15, top=0.9, wspace=0.14, hspace=0.18)
    save_figure(fig, "Figure_5_GateLevel_Sensitivity")


def generate_gate_level_sensitivity() -> list[SensitivityPoint]:
    points: list[SensitivityPoint] = []
    families = [
        ("jordan_gamma_1", jordan_family(1.0)),
        ("neutral_rank_one", neutral_family()),
    ]

    compiled_benchmarks = {
        (family, layers): compile_family_benchmark(
            family,
            matrix,
            workload_layers=layers,
        )
        for family, matrix in families
        for layers in SENSITIVITY_WORKLOAD_LAYERS
    }

    for family, _ in families:
        for layers in SENSITIVITY_WORKLOAD_LAYERS:
            benchmark = compiled_benchmarks[(family, layers)]
            for two_qubit_noise in SENSITIVITY_TWO_QUBIT_NOISES:
                one_qubit_noise = ONE_TO_TWO_QUBIT_NOISE_RATIO * two_qubit_noise
                rows = evaluate_compiled_benchmark(
                    benchmark,
                    one_qubit_noise=one_qubit_noise,
                    two_qubit_noise=two_qubit_noise,
                )
                row_lookup = {row.method: row for row in rows}
                split_factor = budget_normalized_mse_factor(row_lookup["split"])
                dilation_factor = budget_normalized_mse_factor(row_lookup["dilation"])
                genexp_factor = budget_normalized_mse_factor(row_lookup["genexp"])
                points.append(
                    SensitivityPoint(
                        family=family,
                        workload_layers=layers,
                        one_qubit_noise=float(one_qubit_noise),
                        two_qubit_noise=float(two_qubit_noise),
                        split_noisy_factor=float(split_factor),
                        dilation_noisy_factor=float(dilation_factor),
                        genexp_noisy_factor=float(genexp_factor),
                        dil_vs_split_noisy_ratio=float(dilation_factor / split_factor),
                        dil_vs_genexp_noisy_ratio=float(dilation_factor / genexp_factor),
                        split_cx_per_shot=float(row_lookup["split"].cx_per_shot),
                        dilation_cx_per_shot=float(row_lookup["dilation"].cx_per_shot),
                        genexp_cx_per_shot=float(row_lookup["genexp"].cx_per_shot),
                    )
                )

    save_csv(points, "gate_level_sensitivity_summary.csv")
    plot_sensitivity_figure(points)
    return points


def plot_ensemble_phase_diagram(points: list[EnsemblePoint]) -> None:
    beneficial_split = np.zeros((len(SENSITIVITY_WORKLOAD_LAYERS), len(SENSITIVITY_TWO_QUBIT_NOISES)))
    beneficial_genexp = np.zeros_like(beneficial_split)
    median_split = np.zeros_like(beneficial_split)
    median_genexp = np.zeros_like(beneficial_split)

    for row_index, layers in enumerate(SENSITIVITY_WORKLOAD_LAYERS):
        for col_index, noise in enumerate(SENSITIVITY_TWO_QUBIT_NOISES):
            subset = [
                point
                for point in points
                if point.workload_layers == layers and point.two_qubit_noise == noise
            ]
            split_ratios = np.array([point.dil_vs_split_noisy_ratio for point in subset], dtype=float)
            genexp_ratios = np.array([point.dil_vs_genexp_noisy_ratio for point in subset], dtype=float)
            beneficial_split[row_index, col_index] = float(np.mean(split_ratios < 1.0))
            beneficial_genexp[row_index, col_index] = float(np.mean(genexp_ratios < 1.0))
            median_split[row_index, col_index] = float(np.median(split_ratios))
            median_genexp[row_index, col_index] = float(np.median(genexp_ratios))

    fig = plt.figure(figsize=(13.2, 7.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.055], wspace=0.34, hspace=0.22)
    axes = np.array(
        [
            [
                fig.add_subplot(grid[0, 0]),
                fig.add_subplot(grid[0, 1]),
            ],
            [
                fig.add_subplot(grid[1, 0]),
                fig.add_subplot(grid[1, 1]),
            ],
        ],
        dtype=object,
    )
    axes[0, 1].sharex(axes[0, 0])
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 0].sharex(axes[0, 0])
    axes[1, 0].sharey(axes[0, 0])
    axes[1, 1].sharex(axes[0, 0])
    axes[1, 1].sharey(axes[0, 0])
    cax_top = fig.add_subplot(grid[0, 2])
    cax_bottom = fig.add_subplot(grid[1, 2])
    beneficial_data = [beneficial_split, beneficial_genexp]
    median_data = [median_split, median_genexp]
    comparison_labels = ["dilation / split", "dilation / genexp"]
    beneficial_image = None
    median_image = None

    for col_index, label in enumerate(comparison_labels):
        beneficial_image = axes[0, col_index].imshow(
            beneficial_data[col_index],
            origin="lower",
            aspect="auto",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, col_index].set_title(f"Beneficial fraction: {label}")
        axes[0, col_index].set_ylabel("Shared workload layers")

        median_image = axes[1, col_index].imshow(
            median_data[col_index],
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(
                vmin=float(np.min(median_data)),
                vcenter=1.0,
                vmax=float(np.max(median_data)),
            ),
        )
        axes[1, col_index].set_title(f"Median noisy ratio: {label}")
        axes[1, col_index].set_xlabel(r"Two-qubit noise $p_{2\mathrm{q}}$")
        axes[1, col_index].set_ylabel("Shared workload layers")

        for ax in (axes[0, col_index], axes[1, col_index]):
            ax.set_xticks(np.arange(len(SENSITIVITY_TWO_QUBIT_NOISES)))
            ax.set_xticklabels(
                [f"{value:.1e}" for value in SENSITIVITY_TWO_QUBIT_NOISES],
                rotation=35,
                ha="right",
            )
            ax.set_yticks(np.arange(len(SENSITIVITY_WORKLOAD_LAYERS)))
            ax.set_yticklabels([str(value) for value in SENSITIVITY_WORKLOAD_LAYERS])

        for inner_row in range(len(SENSITIVITY_WORKLOAD_LAYERS)):
            for inner_col in range(len(SENSITIVITY_TWO_QUBIT_NOISES)):
                frac_value = beneficial_data[col_index][inner_row, inner_col]
                axes[0, col_index].text(
                    inner_col,
                    inner_row,
                    f"{frac_value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="white" if frac_value > 0.55 else "black",
                )
                median_value = median_data[col_index][inner_row, inner_col]
                axes[1, col_index].text(
                    inner_col,
                    inner_row,
                    f"{median_value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="white" if abs(median_value - 1.0) > 0.14 else "black",
                )

    colorbar_left = fig.colorbar(beneficial_image, cax=cax_top)
    colorbar_left.set_label("Fraction with ratio < 1")
    colorbar_right = fig.colorbar(median_image, cax=cax_bottom)
    colorbar_right.set_label("Median compiled noisy ratio")
    fig.suptitle(
        "Operator-ensemble deployment boundary for coherent dilation against both baselines",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.14, top=0.9)
    save_figure(fig, "Figure_6_Ensemble_PhaseDiagram")


def plot_predictor_figure(points: list[EnsemblePoint]) -> None:
    operator_ratios = np.array([point.operator_ratio for point in points], dtype=float)
    dil_vs_split = np.array([point.dil_vs_split_noisy_ratio for point in points], dtype=float)
    dil_vs_split_ideal = np.array([point.dil_vs_split_ideal_ratio for point in points], dtype=float)
    dil_vs_genexp = np.array([point.dil_vs_genexp_noisy_ratio for point in points], dtype=float)
    dil_vs_genexp_ideal = np.array([point.dil_vs_genexp_ideal_ratio for point in points], dtype=float)
    colors = np.array([point.workload_layers for point in points], dtype=float)

    corr_operator = float(np.corrcoef(operator_ratios, dil_vs_split)[0, 1])
    corr_split_ideal = float(np.corrcoef(dil_vs_split_ideal, dil_vs_split)[0, 1])
    corr_genexp_ideal = float(np.corrcoef(dil_vs_genexp_ideal, dil_vs_genexp)[0, 1])

    fig = plt.figure(figsize=(15.4, 4.8))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.06], wspace=0.33)
    axes = np.array(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]),
        ],
        dtype=object,
    )
    cax = fig.add_subplot(grid[0, 3])
    scatter_left = axes[0].scatter(
        operator_ratios,
        dil_vs_split,
        c=colors,
        cmap="viridis",
        alpha=0.72,
        s=24,
        edgecolors="none",
    )
    axes[0].axhline(1.0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.2)
    axes[0].set_xlabel(r"Operator-level ratio $R_{\rm split\rightarrow dil}$")
    axes[0].set_ylabel(r"Noisy ratio $G_{\rm noisy}^{\rm dil/split}$")
    axes[0].set_title(rf"Operator-only screen ($\rho={corr_operator:.2f}$)")

    axes[1].scatter(
        dil_vs_split_ideal,
        dil_vs_split,
        c=colors,
        cmap="viridis",
        alpha=0.72,
        s=24,
        edgecolors="none",
    )
    axes[1].axhline(1.0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.2)
    axes[1].axvline(1.0, color=COLOR_NEUTRAL, linestyle=":", linewidth=1.2)
    axes[1].set_xlabel(r"Ideal proxy $G_{\rm ideal}^{\rm dil/split}$")
    axes[1].set_ylabel(r"Noisy ratio $G_{\rm noisy}^{\rm dil/split}$")
    axes[1].set_title(rf"Split comparison ($\rho={corr_split_ideal:.2f}$)")

    axes[2].scatter(
        dil_vs_genexp_ideal,
        dil_vs_genexp,
        c=colors,
        cmap="viridis",
        alpha=0.72,
        s=24,
        edgecolors="none",
    )
    axes[2].axhline(1.0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.2)
    axes[2].axvline(1.0, color=COLOR_NEUTRAL, linestyle=":", linewidth=1.2)
    axes[2].set_xlabel(r"Ideal proxy $G_{\rm ideal}^{\rm dil/genexp}$")
    axes[2].set_ylabel(r"Noisy ratio $G_{\rm noisy}^{\rm dil/genexp}$")
    axes[2].set_title(rf"Genexp comparison ($\rho={corr_genexp_ideal:.2f}$)")

    fig.colorbar(scatter_left, cax=cax, label="Shared workload layers")
    fig.suptitle(
        "Compilation-aware ideal proxies predict deployment far better than the operator-only theorem screen",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.86)
    save_figure(fig, "Figure_7_Compiled_Predictors")


def generate_operator_ensemble_artifacts() -> list[EnsemblePoint]:
    points: list[EnsemblePoint] = []
    matrices = random_normalized_ensemble()
    compiled_benchmarks = {
        (family, layers): compile_family_benchmark(
            family,
            matrix,
            workload_layers=layers,
        )
        for family, matrix in matrices
        for layers in SENSITIVITY_WORKLOAD_LAYERS
    }

    matrix_lookup = {family: matrix for family, matrix in matrices}
    for family, _ in matrices:
        matrix = matrix_lookup[family]
        ratio = operator_variance_ratio(matrix)
        for layers in SENSITIVITY_WORKLOAD_LAYERS:
            benchmark = compiled_benchmarks[(family, layers)]
            baseline_rows = evaluate_compiled_benchmark(
                benchmark,
                one_qubit_noise=0.0,
                two_qubit_noise=0.0,
            )
            baseline_lookup = {row.method: row for row in baseline_rows}
            split_ideal_factor = budget_normalized_ideal_factor(baseline_lookup["split"])
            dilation_ideal_factor = budget_normalized_ideal_factor(baseline_lookup["dilation"])
            genexp_ideal_factor = budget_normalized_ideal_factor(baseline_lookup["genexp"])
            sample_id = int(family.rsplit("_", 1)[-1])

            for two_qubit_noise in SENSITIVITY_TWO_QUBIT_NOISES:
                one_qubit_noise = ONE_TO_TWO_QUBIT_NOISE_RATIO * two_qubit_noise
                rows = evaluate_compiled_benchmark(
                    benchmark,
                    one_qubit_noise=one_qubit_noise,
                    two_qubit_noise=two_qubit_noise,
                )
                row_lookup = {row.method: row for row in rows}
                split_noisy_factor = budget_normalized_mse_factor(row_lookup["split"])
                dilation_noisy_factor = budget_normalized_mse_factor(row_lookup["dilation"])
                genexp_noisy_factor = budget_normalized_mse_factor(row_lookup["genexp"])
                points.append(
                    EnsemblePoint(
                        sample_id=sample_id,
                        workload_layers=layers,
                        one_qubit_noise=float(one_qubit_noise),
                        two_qubit_noise=float(two_qubit_noise),
                        operator_ratio=float(ratio),
                        dil_vs_split_ideal_ratio=float(dilation_ideal_factor / split_ideal_factor),
                        dil_vs_split_noisy_ratio=float(dilation_noisy_factor / split_noisy_factor),
                        dil_vs_genexp_ideal_ratio=float(dilation_ideal_factor / genexp_ideal_factor),
                        dil_vs_genexp_noisy_ratio=float(dilation_noisy_factor / genexp_noisy_factor),
                        genexp_vs_split_ideal_ratio=float(genexp_ideal_factor / split_ideal_factor),
                        genexp_vs_split_noisy_ratio=float(genexp_noisy_factor / split_noisy_factor),
                        split_ideal_factor=float(split_ideal_factor),
                        dilation_ideal_factor=float(dilation_ideal_factor),
                        genexp_ideal_factor=float(genexp_ideal_factor),
                        split_noisy_factor=float(split_noisy_factor),
                        dilation_noisy_factor=float(dilation_noisy_factor),
                        genexp_noisy_factor=float(genexp_noisy_factor),
                        split_cx_per_shot=float(row_lookup["split"].cx_per_shot),
                        dilation_cx_per_shot=float(row_lookup["dilation"].cx_per_shot),
                        genexp_cx_per_shot=float(row_lookup["genexp"].cx_per_shot),
                    )
                )

    save_csv(points, "operator_ensemble_summary.csv")
    plot_ensemble_phase_diagram(points)
    plot_predictor_figure(points)
    return points


def search_angles(theta_1: float, theta_2: float) -> np.ndarray:
    angles = np.asarray(WORKLOAD_ANGLES, dtype=float).copy()
    angles[SEARCH_PARAM_INDICES[0]] = theta_1
    angles[SEARCH_PARAM_INDICES[1]] = theta_2
    return angles


def score_from_expectation(expectation: complex, phase: float = SEARCH_SCORE_PHASE) -> float:
    return float(np.real(np.exp(-1.0j * phase) * expectation))


def dense_search_optimum(
    matrix: np.ndarray,
    workload_layers: int = 1,
) -> tuple[float, tuple[float, float]]:
    grid = np.linspace(-np.pi, np.pi, SEARCH_DENSE_GRID_SIZE)
    best_score = -np.inf
    best_point = (0.0, 0.0)
    for theta_1 in grid:
        for theta_2 in grid:
            expectation = expectation_from_statevector(
                matrix,
                workload_layers=workload_layers,
                override_angles=search_angles(theta_1, theta_2),
            )
            score = score_from_expectation(expectation)
            if score > best_score:
                best_score = score
                best_point = (float(theta_1), float(theta_2))
    return float(best_score), best_point


def build_search_schedule(
    matrix: np.ndarray,
    workload_layers: int = 1,
) -> list[tuple[float, float]]:
    schedule: list[tuple[float, float]] = []
    stage_scores: dict[tuple[float, float], float] = {}

    def clipped(point: tuple[float, float]) -> tuple[float, float]:
        return (
            float(np.clip(point[0], -np.pi, np.pi)),
            float(np.clip(point[1], -np.pi, np.pi)),
        )

    def add_stage(points: list[tuple[float, float]]) -> tuple[float, float]:
        best_point = points[0]
        best_score = -np.inf
        for point in points:
            if point not in stage_scores:
                expectation = expectation_from_statevector(
                    matrix,
                    workload_layers=workload_layers,
                    override_angles=search_angles(*point),
                )
                stage_scores[point] = score_from_expectation(expectation)
                schedule.append(point)
            if stage_scores[point] > best_score:
                best_score = stage_scores[point]
                best_point = point
        return best_point

    coarse_points = [
        (float(theta_1), float(theta_2))
        for theta_1 in SEARCH_COARSE_GRID
        for theta_2 in SEARCH_COARSE_GRID
    ]
    center = add_stage(coarse_points)
    for delta in SEARCH_FINE_DELTAS:
        stage_points = [
            clipped((center[0] + offset_1, center[1] + offset_2))
            for offset_1 in (-delta, 0.0, delta)
            for offset_2 in (-delta, 0.0, delta)
        ]
        center = add_stage(stage_points)
    return schedule


def benchmark_score_distribution(
    benchmark: CompiledBenchmark,
    method: str,
    phase: float = SEARCH_SCORE_PHASE,
    one_qubit_noise: float = ONE_QUBIT_DEPOLARIZING,
    two_qubit_noise: float = TWO_QUBIT_DEPOLARIZING,
) -> tuple[float, float]:
    if method == "split":
        values, probabilities = split_outcome_distribution(
            benchmark.prep_split_instructions,
            benchmark.split_hermitian_basis_instructions,
            benchmark.split_antihermitian_basis_instructions,
            benchmark.split_hermitian_eigs,
            benchmark.split_antihermitian_eigs,
            noisy=True,
            one_qubit_noise=one_qubit_noise,
            two_qubit_noise=two_qubit_noise,
        )
    elif method == "dilation":
        values, probabilities = dilation_outcome_distribution(
            benchmark.prep_dilation_instructions,
            benchmark.dilation_basis_instructions,
            benchmark.dilation_eigs,
            noisy=True,
            one_qubit_noise=one_qubit_noise,
            two_qubit_noise=two_qubit_noise,
        )
    elif method == "genexp":
        values, probabilities = genexp_outcome_distribution(
            benchmark.prep_genexp_instructions,
            benchmark.genexp_hermitian_basis_instructions,
            benchmark.genexp_antihermitian_basis_instructions,
            benchmark.genexp_hermitian_eigs,
            benchmark.genexp_antihermitian_eigs,
            noisy=True,
            one_qubit_noise=one_qubit_noise,
            two_qubit_noise=two_qubit_noise,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return projected_distribution_stats(values, probabilities, phase=phase)


def route_schedule_cx_cost(
    benchmarks: dict[tuple[float, float], CompiledBenchmark],
    method: str,
    schedule: list[tuple[float, float]],
) -> float:
    return float(
        sum(
            {
                "split": benchmarks[point].split_counts.cx_gates,
                "dilation": benchmarks[point].dilation_counts.cx_gates,
                "genexp": benchmarks[point].genexp_counts.cx_gates,
            }[method]
            for point in schedule
        )
    )


def plot_budgeted_search_figure(rows: list[SearchBudgetPoint], optimum_score: float) -> None:
    grouped: dict[str, list[SearchBudgetPoint]] = {
        method: sorted(
            [row for row in rows if row.method == method],
            key=lambda row: row.total_cnot_budget,
        )
        for method in ("split", "dilation", "genexp")
    }
    method_styles = {
        "split": ("Randomized split", COLOR_SPLIT),
        "dilation": ("Coherent dilation", COLOR_DILATION),
        "genexp": ("Genexp baseline", COLOR_GENEXP),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), sharex=True)
    for method, method_rows in grouped.items():
        label, color = method_styles[method]
        budgets = np.array([row.total_cnot_budget for row in method_rows], dtype=float)
        reported = np.array([row.mean_reported_best_score for row in method_rows], dtype=float)
        selected = np.array([row.mean_selected_true_score for row in method_rows], dtype=float)
        regrets = np.array([row.mean_regret for row in method_rows], dtype=float)
        rmse = np.array([row.task_rmse for row in method_rows], dtype=float)

        axes[0].plot(budgets, reported, color=color, label=f"{label}, reported")
        axes[0].plot(budgets, selected, color=color, linestyle="--", label=f"{label}, true")
        axes[1].plot(budgets, regrets, color=color, label=label)
        axes[2].plot(budgets, rmse, color=color, label=label)

    axes[0].axhline(optimum_score, color=COLOR_NEUTRAL, linestyle=":", linewidth=1.4, label="Dense-grid optimum")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"Total CNOT budget $B_{\rm CX}$")
    axes[0].set_ylabel("Best-found score")
    axes[1].set_ylabel("True regret")
    axes[2].set_ylabel("Task RMSE")
    axes[0].set_title("Best-found objective")
    axes[1].set_title("Optimization regret")
    axes[2].set_title("End-to-end task error")
    axes[0].legend(frameon=False, fontsize=7.9, loc="lower right")
    axes[1].legend(frameon=False, fontsize=8.2, loc="upper right")
    axes[2].legend(frameon=False, fontsize=8.2, loc="upper right")
    fig.suptitle(
        "Budgeted parameter search on a two-parameter Holmes-QSP objective",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "Figure_8_Search_Workflow")


def generate_budgeted_search_artifacts() -> list[SearchBudgetPoint]:
    matrix = jordan_family(1.0)
    schedule = build_search_schedule(matrix)
    optimum_score, _ = dense_search_optimum(matrix)
    benchmarks = {
        point: compile_family_benchmark(
            family=f"search_theta_{point[0]:+.3f}_{point[1]:+.3f}",
            matrix=matrix,
            workload_layers=1,
            override_angles=search_angles(*point),
        )
        for point in schedule
    }
    true_scores = {
        point: score_from_expectation(benchmarks[point].target)
        for point in schedule
    }
    noisy_stats = {
        point: {
            method: benchmark_score_distribution(benchmarks[point], method)
            for method in ("split", "dilation", "genexp")
        }
        for point in schedule
    }
    schedule_costs = {
        method: route_schedule_cx_cost(benchmarks, method, schedule)
        for method in ("split", "dilation", "genexp")
    }

    rng = np.random.default_rng(SEARCH_SEED)
    rows: list[SearchBudgetPoint] = []
    for method in ("split", "dilation", "genexp"):
        for budget in SEARCH_BUDGETS:
            shots_per_eval = max(1, int(np.floor(budget / schedule_costs[method])))
            reported_best_scores: list[float] = []
            selected_true_scores: list[float] = []
            for _ in range(SEARCH_TRIALS):
                best_estimate = -np.inf
                best_true_score = -np.inf
                for point in schedule:
                    mean_score, variance_score = noisy_stats[point][method]
                    estimate = float(
                        rng.normal(
                            loc=mean_score,
                            scale=np.sqrt(max(variance_score, 1.0e-12) / shots_per_eval),
                        )
                    )
                    if estimate > best_estimate:
                        best_estimate = estimate
                        best_true_score = true_scores[point]
                reported_best_scores.append(best_estimate)
                selected_true_scores.append(best_true_score)

            reported_array = np.asarray(reported_best_scores, dtype=float)
            selected_array = np.asarray(selected_true_scores, dtype=float)
            rows.append(
                SearchBudgetPoint(
                    method=method,
                    total_cnot_budget=float(budget),
                    shots_per_evaluation=shots_per_eval,
                    mean_reported_best_score=float(np.mean(reported_array)),
                    mean_selected_true_score=float(np.mean(selected_array)),
                    mean_regret=float(np.mean(optimum_score - selected_array)),
                    task_rmse=float(np.sqrt(np.mean((selected_array - optimum_score) ** 2))),
                    schedule_cx_cost=float(schedule_costs[method]),
                )
            )

    save_csv(rows, "budgeted_search_summary.csv")
    plot_budgeted_search_figure(rows, optimum_score)
    return rows


def generate_gate_level_artifacts() -> list[MethodStatistics]:
    rows = []
    compiled_benchmarks = [
        compile_family_benchmark("jordan_gamma_1", jordan_family(1.0)),
        compile_family_benchmark("neutral_rank_one", neutral_family()),
    ]
    for benchmark in compiled_benchmarks:
        rows.extend(
            evaluate_compiled_benchmark(
                benchmark,
                one_qubit_noise=ONE_QUBIT_DEPOLARIZING,
                two_qubit_noise=TWO_QUBIT_DEPOLARIZING,
            )
        )
    save_csv(rows, "gate_level_resource_summary.csv")
    plot_gate_level_figure(rows)
    generate_gate_level_sensitivity()
    generate_operator_ensemble_artifacts()
    generate_budgeted_search_artifacts()
    return rows


if __name__ == "__main__":
    generate_gate_level_artifacts()
