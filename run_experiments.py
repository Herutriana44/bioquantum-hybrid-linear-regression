from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless (CI) safe

import matplotlib.pyplot as plt
import numpy as np
from brian2 import NeuronGroup, StateMonitor, defaultclock, ms, run, start_scope
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler as AerSampler
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error


RESULTS_DIR = Path("results")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_results_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _save_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + os.linesep, encoding="utf-8")


def _save_csv_xy(path: Path, x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> None:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape. got {x.shape} vs {y.shape}")
    header = f"{x_name},{y_name}"
    arr = np.column_stack([x, y])
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def _save_csv_matrix(path: Path, columns: dict[str, np.ndarray]) -> None:
    keys = list(columns.keys())
    cols = [np.asarray(columns[k]).reshape(-1) for k in keys]
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError("All columns must have same length")
    mat = np.column_stack(cols)
    np.savetxt(path, mat, delimiter=",", header=",".join(keys), comments="")


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def simple_neuron_operator(x: float) -> float:
    """
    Simple Neuron Operator for Brain Organoids Intelligence implements
    (copied from the notebook; run in Brian2 each call).
    """

    start_scope()  # Brian2: safe repeated simulations
    defaultclock.dt = 1 * ms
    eqs = """
    dv/dt = (-v + I) / (10*ms) : 1
    I : 1
    """
    g = NeuronGroup(1, eqs, method="euler")

    g.I = x
    g.v = 0.0

    mon = StateMonitor(g, "v", record=True)
    run(10 * ms)

    return float(mon.v[0][-1])


def float_to_qubit_prob(x: float, shots: int = 1000) -> float:
    """
    Encode a float value as a Y-rotation on a single qubit, simulate, and return
    the probability of measuring |1>.
    (copied from the notebook; Aer shot noise included)
    """

    sampler = AerSampler()
    qc = QuantumCircuit(1, 1)
    qc.ry(x * np.pi, 0)
    qc.measure(0, 0)
    job = sampler.run(qc, shots=shots)
    result = job.result()
    quasi_dist = result.quasi_dists[0]
    prob_1 = float(quasi_dist.get(1, 0.0))
    return prob_1


@dataclass(frozen=True)
class RunMeta:
    created_utc: str
    python_seed: int
    brian2_dt_ms: float
    qubit_shots: int


def experiment_synthetic(meta: RunMeta, out_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(meta.python_seed)

    # generate synthetic data (same form as notebook)
    x = np.linspace(0, 10, 100)
    y = 3 * x + 2 + rng.normal(0, 1, size=x.shape)

    _save_csv_xy(out_dir / "synthetic_data.csv", x, y, "x", "y")

    # circuit visualization (notebook's "qc.draw(output='mpl')" cell)
    qc = QuantumCircuit(1, 1)
    qc.ry(0.5 * np.pi, 0)
    qc.measure(0, 0)
    fig = qc.draw(output="mpl")
    if hasattr(fig, "savefig"):
        fig.savefig(out_dir / "quantum_circuit.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Original data plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Original Data")
    plt.grid(True, alpha=0.25)
    _save_fig(out_dir / "synthetic_original_data.png")

    # Model comparison experiment (notebook)
    n = len(x)

    # --- (1) Baseline: classical linear regression y ~ b0 + b1 * x
    x_classical = np.column_stack([np.ones(n), x])
    beta_classical, *_ = np.linalg.lstsq(x_classical, y, rcond=None)
    y_pred_classical = x_classical @ beta_classical

    # --- (2) Quantum feature: P(|1>) from the circuit (shot noise from Aer, same as reference)
    q_feat = np.array([float_to_qubit_prob(float(xv), shots=meta.qubit_shots) for xv in x])
    x_quantum = np.column_stack([np.ones(n), q_feat])
    beta_quantum, *_ = np.linalg.lstsq(x_quantum, y, rcond=None)
    y_pred_quantum = x_quantum @ beta_quantum

    # --- (3) Bio-inspired only
    bio_feat = np.array([simple_neuron_operator(float(xv)) for xv in x])
    x_bio = np.column_stack([np.ones(n), bio_feat])
    beta_bio, *_ = np.linalg.lstsq(x_bio, y, rcond=None)
    y_pred_bio = x_bio @ beta_bio

    # --- (4) Bio–quantum hybrid: y ~ b0 + b1 * q + b2 * bio (original formulation)
    x_hybrid = np.column_stack([np.ones(n), q_feat, bio_feat])
    beta_hybrid, *_ = np.linalg.lstsq(x_hybrid, y, rcond=None)
    y_pred_hybrid = x_hybrid @ beta_hybrid

    b_hybrid = float(beta_hybrid[0])
    wqc_fit = float(beta_hybrid[1])
    wboi_fit = float(beta_hybrid[2])

    mses = {
        "Classical (baseline)": float(mean_squared_error(y, y_pred_classical)),
        "Classical + quantum noise (q feature)": float(mean_squared_error(y, y_pred_quantum)),
        "Classical + bio-inspired only": float(mean_squared_error(y, y_pred_bio)),
        "Bio–quantum hybrid": float(mean_squared_error(y, y_pred_hybrid)),
    }

    _save_json(out_dir / "synthetic_mse.json", mses)
    _save_text(
        out_dir / "synthetic_mse.txt",
        "MSE by model:\n" + "\n".join([f"  {k}: {v:.6f}" for k, v in mses.items()]),
    )

    _save_csv_matrix(
        out_dir / "synthetic_features_and_predictions.csv",
        {
            "x": x,
            "y": y,
            "q_feat": q_feat,
            "bio_feat": bio_feat,
            "y_pred_classical": y_pred_classical,
            "y_pred_quantum": y_pred_quantum,
            "y_pred_bio": y_pred_bio,
            "y_pred_hybrid": y_pred_hybrid,
        },
    )

    # Smooth curves for plotting (sample x grid)
    xs = np.linspace(float(x.min()), float(x.max()), 120)
    q_line = np.array([float_to_qubit_prob(float(xv), shots=meta.qubit_shots) for xv in xs])
    bio_line = np.array([simple_neuron_operator(float(xv)) for xv in xs])

    y_line_cl = beta_classical[0] + beta_classical[1] * xs
    y_line_q = beta_quantum[0] + beta_quantum[1] * q_line
    y_line_b = beta_bio[0] + beta_bio[1] * bio_line
    y_line_h = beta_hybrid[0] + beta_hybrid[1] * q_line + beta_hybrid[2] * bio_line

    # Plot predictions curves (matches notebook)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x, y, alpha=0.45, label="Data", s=28, edgecolors="none")
    ax.plot(xs, y_line_cl, lw=2, label="(1) Classical (baseline)")
    ax.plot(xs, y_line_q, lw=2, label="(2) + quantum noise (q feature)")
    ax.plot(xs, y_line_b, lw=2, label="(3) + bio-inspired only")
    ax.plot(xs, y_line_h, lw=2.5, label="(4) Bio–quantum hybrid")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predictions: classical → quantum → bio → hybrid")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(out_dir / "synthetic_predictions_curves.png")

    # Bar chart of MSE (matches notebook)
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    names_short = ["(1) Classical", "(2) Quantum", "(3) Bio", "(4) Hybrid"]
    mse_vals = [mses["Classical (baseline)"], mses["Classical + quantum noise (q feature)"], mses["Classical + bio-inspired only"], mses["Bio–quantum hybrid"]]
    colors = ["#1abc9c", "#3498db", "#e67e22", "#9b59b6"]
    ax2.bar(range(4), mse_vals, color=colors)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(names_short, rotation=12, ha="right")
    ax2.set_ylabel("MSE")
    ax2.set_title("Mean squared error (lower is better)")
    m0 = max(mse_vals) if mse_vals else 1.0
    for i, v in enumerate(mse_vals):
        ax2.text(i, v + 0.02 * m0, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    _save_fig(out_dir / "synthetic_mse_bar.png")

    # Single-point prediction (notebook's hybrid_predict_example)
    x_test = 10.0
    boi_operation = simple_neuron_operator(x_test)
    quantum_operation = float_to_qubit_prob(x_test, shots=meta.qubit_shots)
    y_pred_hybrid_pt = b_hybrid + wqc_fit * quantum_operation + wboi_fit * boi_operation
    _save_text(
        out_dir / "synthetic_hybrid_single_point.txt",
        f"x_test = {x_test} -> y_pred (hybrid) = {y_pred_hybrid_pt}",
    )

    # Data vs hybrid predictions at each sample (notebook's hybrid_predict_curve)
    predicted_y_hybrid = b_hybrid + wqc_fit * q_feat + wboi_fit * bio_feat
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "o", label="Data", alpha=0.5)
    plt.plot(x, predicted_y_hybrid, "-", label="Hybrid (fitted)", lw=2)
    plt.title("Data vs bio–quantum hybrid prediction (least squares)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_fig(out_dir / "synthetic_data_vs_hybrid.png")

    return {
        "beta_classical": beta_classical.tolist(),
        "beta_quantum": beta_quantum.tolist(),
        "beta_bio": beta_bio.tolist(),
        "beta_hybrid": beta_hybrid.tolist(),
        "hybrid_coeffs": {"b": b_hybrid, "wqc": wqc_fit, "wboi": wboi_fit},
        "mse": mses,
    }


def experiment_iris(meta: RunMeta, out_dir: Path) -> dict[str, Any]:
    iris = load_iris()

    # x: sepal length (cm), y: petal length (cm) — both continuous
    x_raw = iris.data[:, 0].astype(float)
    y = iris.data[:, 2].astype(float)

    # Scale x to [0, 1] for quantum / neuron encoding (same as notebook)
    xr_min, xr_max = float(x_raw.min()), float(x_raw.max())
    x = (x_raw - xr_min) / (xr_max - xr_min + 1e-12)

    _save_csv_matrix(
        out_dir / "iris_data.csv",
        {"x_raw_sepal_length_cm": x_raw, "x_norm_0_1": x, "y_petal_length_cm": y, "species_class": iris.target},
    )

    n = len(x)

    # --- (1) Classical
    x_c = np.column_stack([np.ones(n), x])
    b_c, *_ = np.linalg.lstsq(x_c, y, rcond=None)
    y_p_c = x_c @ b_c

    # --- (2) Quantum noise (q feature)
    q = np.array([float_to_qubit_prob(float(xv), shots=meta.qubit_shots) for xv in x])
    x_q = np.column_stack([np.ones(n), q])
    b_q, *_ = np.linalg.lstsq(x_q, y, rcond=None)
    y_p_q = x_q @ b_q

    # --- (3) Bio-inspired
    bio = np.array([simple_neuron_operator(float(xv)) for xv in x])
    x_b = np.column_stack([np.ones(n), bio])
    b_b, *_ = np.linalg.lstsq(x_b, y, rcond=None)
    y_p_b = x_b @ b_b

    # --- (4) Hybrid
    x_h = np.column_stack([np.ones(n), q, bio])
    b_h, *_ = np.linalg.lstsq(x_h, y, rcond=None)
    y_p_h = x_h @ b_h

    mses = {
        "Classical (baseline)": float(mean_squared_error(y, y_p_c)),
        "Classical + quantum noise (q feature)": float(mean_squared_error(y, y_p_q)),
        "Classical + bio-inspired only": float(mean_squared_error(y, y_p_b)),
        "Bio–quantum hybrid": float(mean_squared_error(y, y_p_h)),
    }

    _save_json(out_dir / "iris_mse.json", mses)
    _save_text(
        out_dir / "iris_mse.txt",
        "Iris — MSE by model (target: petal length):\n"
        + "\n".join([f"  {k}: {v:.6f}" for k, v in mses.items()]),
    )

    xs = np.linspace(0.0, 1.0, 120)
    q_line = np.array([float_to_qubit_prob(float(xv), shots=meta.qubit_shots) for xv in xs])
    bio_line = np.array([simple_neuron_operator(float(xv)) for xv in xs])

    y_l_c = b_c[0] + b_c[1] * xs
    y_l_q = b_q[0] + b_q[1] * q_line
    y_l_b = b_b[0] + b_b[1] * bio_line
    y_l_h = b_h[0] + b_h[1] * q_line + b_h[2] * bio_line

    # Curves plot (matches notebook)
    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(
        x,
        y,
        alpha=0.5,
        label="Iris (sepal length → petal length)",
        s=35,
        edgecolors="none",
        c=iris.target,
        cmap="viridis",
    )
    ax.plot(xs, y_l_c, lw=2, label="(1) Classical")
    ax.plot(xs, y_l_q, lw=2, label="(2) + quantum noise")
    ax.plot(xs, y_l_b, lw=2, label="(3) + bio-inspired")
    ax.plot(xs, y_l_h, lw=2.5, label="(4) Bio–quantum hybrid")
    ax.set_xlabel("Sepal length (normalized to [0, 1])")
    ax.set_ylabel("Petal length (cm)")
    ax.set_title("Iris: comparison of four regression models")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Species class (0/1/2)")
    plt.tight_layout()
    _save_fig(out_dir / "iris_predictions_curves.png")

    # MSE bars (matches notebook)
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    mse_vals = [mses["Classical (baseline)"], mses["Classical + quantum noise (q feature)"], mses["Classical + bio-inspired only"], mses["Bio–quantum hybrid"]]
    colors = ["#1abc9c", "#3498db", "#e67e22", "#9b59b6"]
    ax2.bar(range(4), mse_vals, color=colors)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["(1) Classical", "(2) Quantum", "(3) Bio", "(4) Hybrid"], rotation=12, ha="right")
    ax2.set_ylabel("MSE")
    ax2.set_title("Iris — mean squared error (petal length)")
    m0 = max(mse_vals) if mse_vals else 1.0
    for i, v in enumerate(mse_vals):
        ax2.text(i, v + 0.02 * m0, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    _save_fig(out_dir / "iris_mse_bar.png")

    _save_csv_matrix(
        out_dir / "iris_features_and_predictions.csv",
        {
            "x_norm_0_1": x,
            "y_petal_length_cm": y,
            "q_feat": q,
            "bio_feat": bio,
            "y_pred_classical": y_p_c,
            "y_pred_quantum": y_p_q,
            "y_pred_bio": y_p_b,
            "y_pred_hybrid": y_p_h,
            "species_class": iris.target,
        },
    )

    return {
        "b_classical": b_c.tolist(),
        "b_quantum": b_q.tolist(),
        "b_bio": b_b.tolist(),
        "b_hybrid": b_h.tolist(),
        "mse": mses,
        "x_raw_min_max": [xr_min, xr_max],
    }


def main() -> None:
    meta = RunMeta(
        created_utc=_now_utc_iso(),
        python_seed=123,
        brian2_dt_ms=float((defaultclock.dt / ms) if hasattr(defaultclock, "dt") else 1.0),
        qubit_shots=1000,
    )

    _reset_results_dir(RESULTS_DIR)
    _save_json(RESULTS_DIR / "run_meta.json", asdict(meta))

    synthetic_dir = RESULTS_DIR / "synthetic"
    iris_dir = RESULTS_DIR / "iris"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    iris_dir.mkdir(parents=True, exist_ok=True)

    syn_summary = experiment_synthetic(meta, synthetic_dir)
    iris_summary = experiment_iris(meta, iris_dir)

    _save_json(
        RESULTS_DIR / "summary.json",
        {
            "meta": asdict(meta),
            "synthetic": syn_summary,
            "iris": iris_summary,
        },
    )

    print("Done. Results written to:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()

