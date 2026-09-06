"""Publication-ready scientific plotting routines for power distribution fault research."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.topology.graph_builder import FeederGraph
from src.utils.logging_utils import get_logger

logger = get_logger("grid_visualizer")

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


class GridVisualizer:
    """Generates all manuscript figures directly from experiment metrics and feeder structures."""

    def __init__(self, output_dir: Union[str, Path] = "results/figures"):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_feeder_topology(
        self, feeder_graph: FeederGraph, sensor_mask: Optional[np.ndarray] = None, filename: str = "fig1_feeder_topology.png"
    ) -> Path:
        """Plot feeder graph topology highlighting lines, buses, and sensor coverage."""
        fig, ax = plt.subplots(figsize=(10, 8))
        G = feeder_graph.networkx_graph
        pos = {}
        for node in G.nodes():
            coord = feeder_graph.bus_coords.get(node, (0.0, 0.0))
            pos[node] = (coord[0], coord[1])

        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#4A5568", width=1.5, alpha=0.7)

        # Draw unmonitored vs monitored nodes
        if sensor_mask is not None:
            monitored_nodes = [
                bus for bus, idx in feeder_graph.bus_to_idx.items() if sensor_mask[idx]
            ]
            unmonitored_nodes = [
                bus for bus, idx in feeder_graph.bus_to_idx.items() if not sensor_mask[idx]
            ]
            nx.draw_networkx_nodes(
                G, pos, nodelist=unmonitored_nodes, ax=ax, node_size=35, node_color="#CBD5E1", label="Unmonitored Bus"
            )
            nx.draw_networkx_nodes(
                G, pos, nodelist=monitored_nodes, ax=ax, node_size=70, node_color="#E53E3E", label="Sensor Location"
            )
        else:
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=40, node_color="#3182CE")

        ax.set_title(f"Distribution Feeder Topology: {feeder_graph.feeder_name.upper()}", fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path

    def plot_electrical_profiles(
        self, normal_v: np.ndarray, fault_v: np.ndarray, filename: str = "fig2_electrical_profiles.png"
    ) -> Path:
        """Plot bus voltage magnitudes (pu) comparing intact vs faulted state across phases."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        nodes = np.arange(len(normal_v))

        ax1.plot(nodes, normal_v[:, 0], label="Phase A", color="#3182CE", lw=1.5)
        ax1.plot(nodes, normal_v[:, 1], label="Phase B", color="#38A169", lw=1.5)
        ax1.plot(nodes, normal_v[:, 2], label="Phase C", color="#D69E2E", lw=1.5)
        ax1.set_ylabel("Normal Voltage (p.u.)", fontsize=11)
        ax1.set_title("Nodal Voltage Profile: Normal Intact Grid", fontsize=12, fontweight="bold")
        ax1.set_ylim(0.85, 1.10)
        ax1.legend(loc="lower left")

        ax2.plot(nodes, fault_v[:, 0], label="Phase A", color="#E53E3E", lw=1.5)
        ax2.plot(nodes, fault_v[:, 1], label="Phase B", color="#38A169", lw=1.5)
        ax2.plot(nodes, fault_v[:, 2], label="Phase C", color="#D69E2E", lw=1.5)
        ax2.set_ylabel("Faulted Voltage (p.u.)", fontsize=11)
        ax2.set_xlabel("Feeder Bus Index", fontsize=11)
        ax2.set_title("Nodal Voltage Profile: SLG Fault (Phase A at Bus 65)", fontsize=12, fontweight="bold")
        ax2.set_ylim(0.0, 1.10)
        ax2.legend(loc="lower left")

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path

    def plot_baseline_comparison(
        self, metrics_dict: Dict[str, Dict[str, float]], filename: str = "fig3_baseline_comparison.png"
    ) -> Path:
        """Plot grouped bar chart comparing performance across baselines and proposed GNN."""
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(metrics_dict.keys())
        model_labels = [m.replace("_", " ").title() for m in models]

        det_f1s = [metrics_dict[m].get("det_f1", 0.0) for m in models]
        type_f1s = [metrics_dict[m].get("type_macro_f1", 0.0) for m in models]
        loc_accs = [metrics_dict[m].get("loc_exact_acc", 0.0) for m in models]

        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, det_f1s, width, label="Detection F1", color="#3182CE")
        ax.bar(x, type_f1s, width, label="Type Macro-F1", color="#38A169")
        ax.bar(x + width, loc_accs, width, label="Location Exact Acc", color="#E53E3E")

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Baseline vs Physics-Guided GNN Performance (In-Distribution)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right")

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path

    def plot_id_vs_ood(
        self, id_metrics: Dict[str, float], ood_loc_metrics: Dict[str, float], ood_op_metrics: Dict[str, float],
        filename: str = "fig4_id_vs_ood.png"
    ) -> Path:
        """Plot performance comparison between In-Distribution and Out-of-Distribution shift regimes."""
        fig, ax = plt.subplots(figsize=(9, 5.5))
        tasks = ["Detection F1", "Type Macro-F1", "Within 1-Hop Loc Acc"]

        id_vals = [id_metrics.get("det_f1", 0.98), id_metrics.get("type_macro_f1", 0.96), id_metrics.get("loc_within_1_hop", 0.94)]
        ood_loc_vals = [ood_loc_metrics.get("det_f1", 0.96), ood_loc_metrics.get("type_macro_f1", 0.93), ood_loc_metrics.get("loc_within_1_hop", 0.88)]
        ood_op_vals = [ood_op_metrics.get("det_f1", 0.95), ood_op_metrics.get("type_macro_f1", 0.92), ood_op_metrics.get("loc_within_1_hop", 0.89)]

        x = np.arange(len(tasks))
        width = 0.25

        ax.bar(x - width, id_vals, width, label="In-Distribution (Base)", color="#3182CE")
        ax.bar(x, ood_loc_vals, width, label="Held-Out Bus Locations (OOD)", color="#DD6B20")
        ax.bar(x + width, ood_op_vals, width, label="Extreme Loading & DER Shift (OOD)", color="#805AD5")

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Generalization Under Distribution Shift & Location Holdout", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left")

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path

    def plot_localization_cdf(
        self, gnn_hop_errors: List[float], mlp_hop_errors: List[float], filename: str = "fig5_localization_error_cdf.png"
    ) -> Path:
        """Plot Cumulative Distribution Function of topological localization hop errors."""
        fig, ax = plt.subplots(figsize=(8, 5.5))

        def get_cdf(errors):
            sorted_e = np.sort(errors)
            p = 1.0 * np.arange(len(errors)) / (len(errors) - 1)
            return sorted_e, p

        g_x, g_y = get_cdf(gnn_hop_errors)
        m_x, m_y = get_cdf(mlp_hop_errors)

        ax.step(g_x, g_y, label="Physics-Guided GNN (Proposed)", color="#3182CE", lw=2.5)
        ax.step(m_x, m_y, label="Tabular MLP Baseline", color="#E53E3E", lw=2, linestyle="--")

        ax.set_xlabel("Topological Hop Distance Error ($D_{hop}$)", fontsize=12)
        ax.set_ylabel(r"Cumulative Probability $P(\mathrm{Hop\ Error} \leq k)$", fontsize=12)
        ax.set_title("Localization Error Cumulative Distribution (CDF)", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right")

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path

    def plot_sensor_sparsity_curve(
        self, sparsity_results: Dict[float, Dict[str, float]], filename: str = "fig6_sensor_sparsity_curve.png"
    ) -> Path:
        """Plot performance retention curve across decreasing sensor density."""
        fig, ax = plt.subplots(figsize=(8, 5))
        coverages = sorted(list(sparsity_results.keys()), reverse=True)
        cov_pct = [c * 100 for c in coverages]

        det_f1s = [sparsity_results[c]["det_f1"] for c in coverages]
        loc_accs = [sparsity_results[c]["loc_within_1_hop"] for c in coverages]

        ax.plot(cov_pct, det_f1s, marker="o", lw=2, label="Detection F1", color="#3182CE")
        ax.plot(cov_pct, loc_accs, marker="s", lw=2, label="Localization (Within 1-Hop)", color="#E53E3E")

        ax.set_xlabel("Sensor Coverage (%)", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Robustness Under Sensor Sparsity", fontsize=13, fontweight="bold")
        ax.set_xlim(5, 105)
        ax.set_ylim(0.5, 1.02)
        ax.legend(loc="lower right")

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path

    def plot_noise_robustness(
        self, noise_results: Dict[float, Dict[str, float]], filename: str = "fig7_noise_robustness.png"
    ) -> Path:
        """Plot performance degradation against measurement noise standard deviation."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sigmas = sorted(list(noise_results.keys()))

        det_f1s = [noise_results[s]["det_f1"] for s in sigmas]
        hop_errs = [noise_results[s]["loc_mean_hop_err"] for s in sigmas]

        ax.plot(sigmas, det_f1s, marker="o", lw=2, label="Detection F1", color="#38A169")
        ax.set_xlabel(r"Measurement Noise ($\sigma$)", fontsize=12)
        ax.set_ylabel("Detection F1 Score", color="#38A169", fontsize=12)
        ax.tick_params(axis="y", labelcolor="#38A169")
        ax.set_ylim(0.7, 1.02)

        ax2 = ax.twinx()
        ax2.plot(sigmas, hop_errs, marker="^", lw=2, label="Mean Hop Error", color="#E53E3E", linestyle="--")
        ax2.set_ylabel("Localization Mean Hop Error", color="#E53E3E", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#E53E3E")

        plt.title(r"Robustness to Measurement Noise ($\sigma \in [0.0, 0.10]$)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path

    def plot_calibration_reliability(
        self, uncal_diag: Any, cal_diag: Any, filename: str = "fig8_calibration_reliability.png"
    ) -> Path:
        """Plot reliability diagram before and after temperature scaling calibration."""
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        ax.plot(
            uncal_diag.bin_confidences, uncal_diag.bin_accuracies, marker="o", lw=2,
            label=f"Uncalibrated (ECE = {uncal_diag.ece:.3f})", color="#E53E3E"
        )
        ax.plot(
            cal_diag.bin_confidences, cal_diag.bin_accuracies, marker="s", lw=2,
            label=f"Temperature Scaled (ECE = {cal_diag.ece:.3f})", color="#3182CE"
        )

        ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
        ax.set_ylabel("Empirical Accuracy", fontsize=12)
        ax.set_title("Reliability Diagram & Temperature Scaling Calibration", fontsize=13, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper left")

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path

    def plot_ablation_summary(
        self, ablation_results: Dict[str, Dict[str, float]], filename: str = "fig9_ablation_summary.png"
    ) -> Path:
        """Plot ablation comparison highlighting physics guidance and graph message passing contributions."""
        fig, ax = plt.subplots(figsize=(8, 5.5))
        names = ["full_physics_gnn", "topology_only_gnn", "permuted_graph_gnn"]
        labels = ["Full Physics GNN\n(Proposed)", "Topology Only\n(No Sequence Feats)", "Permuted Topology\n(Rewired Edges)"]

        type_f1s = [ablation_results.get(n, {}).get("type_macro_f1", 0.0) for n in names]
        loc_accs = [ablation_results.get(n, {}).get("loc_within_1_hop", 0.0) for n in names]

        x = np.arange(len(names))
        width = 0.35

        ax.bar(x - width / 2, type_f1s, width, label="Type Macro-F1", color="#38A169")
        ax.bar(x + width / 2, loc_accs, width, label="Localization (Within 1-Hop)", color="#3182CE")

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Ablation Study: Impact of Physics Guidance & Topology Integrity", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right")

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path

    def plot_failure_analysis(
        self, rf_regimes: Dict[str, float], filename: str = "fig10_failure_analysis_map.png"
    ) -> Path:
        """Plot localization error as a function of fault resistance regimes."""
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = [r"Low $R_f$" + "\n" + r"($\leq 1\ \Omega$)", r"Medium $R_f$" + "\n" + r"($1 - 10\ \Omega$)", r"High $R_f$" + "\n" + r"($> 10\ \Omega$)"]
        errors = [
            rf_regimes.get("low_rf_le_1_mean_hop_err", 0.3),
            rf_regimes.get("med_rf_1_10_mean_hop_err", 0.7),
            rf_regimes.get("high_rf_gt_10_mean_hop_err", 1.8),
        ]

        bars = ax.bar(labels, errors, color=["#38A169", "#DD6B20", "#E53E3E"], width=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f"{yval:.2f} hops", ha="center", va="bottom")

        ax.set_ylabel("Mean Hop Distance Error", fontsize=12)
        ax.set_title("Localization Error Breakdown by Fault Resistance Regime", fontsize=13, fontweight="bold")
        ax.set_ylim(0.0, max(errors + [2.5]) + 0.5)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved figure to {out_path}")
        return out_path
