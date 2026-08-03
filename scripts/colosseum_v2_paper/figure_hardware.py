import pandas as pd
import numpy as np
import argparse
import matplotlib
import matplotlib.lines
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import isnan



""" Compares sim performance vs hardware performance

python scripts/colosseum_v2_paper/parse_logs.py --results-paths logs/act_clip/single_arm.csv --output-path logs/act_clip/single_arm

# Example usage:
python scripts/colosseum_v2_paper/figure_hardware.py \
    --sim-csv-filepath logs/act_clip/single_arm.formatted.csv \
    --out-dir logs/act_clip/
"""

# Real-world success rates (% = successes/20 * 100). "x" = not evaluated (key omitted).
# Size columns "small"/"big" are averaged into mo_size when both are present.
# Task name map: LiftCube→RaiseCube, LiftPeg→LiftPegUpright,
# LiftDish→PickDishFromRack, PickCan→PickSodaFromCabinet.
# "Language-none" maps to language_none.
HARDWARE_ROWS = [
    {
        "Task": "RaiseCube",
        "none": 8 / 20 * 100,
        "light_color": 7 / 20 * 100,
        "mo_size": (7 + 1) / 2 / 20 * 100,  # small=7/20, big=1/20
        "background_color": 7 / 20 * 100,
        "table_color": 0 / 20 * 100,
        "mo_color": 1 / 20 * 100,
        "distractor_object": 7 / 20 * 100,
        "language_none": 6 / 20 * 100,
    },
    {
        "Task": "RotateArrow",
        "none": 18 / 20 * 100,
        "light_color": 12 / 20 * 100,
        "mo_size": (17 + 18) / 2 / 20 * 100,  # small=17/20, big=18/20
        "background_color": 17 / 20 * 100,
        "table_color": 3 / 20 * 100,
        "mo_color": 13 / 20 * 100,
        "distractor_object": 7 / 20 * 100,
        "language_none": 17 / 20 * 100,
    },
    {
        "Task": "LiftPegUpright",
        "none": 7 / 20 * 100,
        "light_color": 4 / 20 * 100,
        "mo_size": (5 + 3) / 2 / 20 * 100,  # small=5/20, big=3/20
        "background_color": 5 / 20 * 100,
        "table_color": 1 / 20 * 100,
        "mo_color": 2 / 20 * 100,
        "distractor_object": 6 / 20 * 100,
        "language_none": 4 / 20 * 100,
    },
    {
        "Task": "PickDishFromRack",
        "none": 20 / 20 * 100,
        "light_color": 16 / 20 * 100,
        "background_color": 14 / 20 * 100,
        "table_color": 16 / 20 * 100,
        "mo_color": 15 / 20 * 100,
        "distractor_object": 19 / 20 * 100,
        "language_none": 19 / 20 * 100,
    },
    {
        "Task": "PickSodaFromCabinet",
        "none": 6 / 20 * 100,
        "light_color": 3 / 20 * 100,
        "background_color": 6 / 20 * 100,
        "table_color": 4 / 20 * 100,
        "mo_color": 1 / 20 * 100,
        "distractor_object": 7 / 20 * 100,
        "language_none": 8 / 20 * 100,
    },
]

# HARDWARE_ROWS = [
#     {
#         "Task": "RaiseCube",
#         "none": 55,
#         "mo_size": 42.5,
#         "light_color": 10,
#         "distractor_object": 35,
#         "background_color": 50,
#         "mo_color": 0,
#     },
#     {
#         "Task": "RotateArrow",
#         "none": 65,
#         "light_color": 25,
#         "mo_size": 57.5,
#         "distractor_object": 50,
#         "background_color": 50,
#         "mo_color": 50,
#     },
#     {
#         "Task": "LiftPegUpright",
#         "none": 60,
#         "light_color": 25,
#         "distractor_object": 50,
#         "background_color": 30,
#         "mo_size": 15,
#         "mo_color": 45,
#     },
#     # {
#     #     "Task": "OpenDrawer",
#     #     "none": 80,
#     #     "light_color": 5,
#     #     "distractor_object": 70,
#     #     "background_color": 65,
#     #     "mo_color": 25,
#     # },
# ]

PERTURBATION_SET_DISPLAY_NAMES = {
    "none".lower(): "None",
    "all".lower(): "All",
    "MO_color".lower(): "MO Color",
    "RO_color".lower(): "RO Color",
    "MO_texture".lower(): "MO Texture",
    "RO_texture".lower(): "RO Texture",
    "MO_size".lower(): "MO Size",
    "RO_size".lower(): "RO Size",
    "table_color".lower(): "Table Color",
    "light_color".lower(): "Light Color",
    "table_texture".lower(): "Table Texture",
    "distractor_object".lower(): "Distractor Object",
    "background_texture".lower(): "Background Texture",
    "background_color".lower(): "Background Color",
    "camera_pose".lower(): "Camera Pose",
    "MO_mass".lower(): "MO Mass",
    "language_none".lower(): "Language None",
    "language_paraphrase".lower(): "Language Paraphrase",
    "language_other_task".lower(): "Language Other Task",
    "language_random".lower(): "Language Random",
}


def calculate_spearman_correlation(x_vals, y_vals) -> float:
    if len(x_vals) < 2 or len(set(x_vals)) < 2 or len(set(y_vals)) < 2:
        return float("nan")
    rho = pd.Series(x_vals, dtype="float64").corr(pd.Series(y_vals, dtype="float64"), method="spearman")
    return float(rho) if not pd.isna(rho) else float("nan")


def calculate_pearson_correlation(x_vals, y_vals) -> float:
    if len(x_vals) < 2 or len(set(x_vals)) < 2 or len(set(y_vals)) < 2:
        return float("nan")
    r = pd.Series(x_vals, dtype="float64").corr(pd.Series(y_vals, dtype="float64"), method="pearson")
    return float(r) if not pd.isna(r) else float("nan")


def plot_by_task(sim_csv_filepath: str, out_dir: str):

    sim_df = pd.read_csv(sim_csv_filepath)
    if "Task" not in sim_df.columns:
        raise ValueError(f"Expected a 'Task' column in {sim_csv_filepath}, got columns: {list(sim_df.columns)}")

    # Index both tables by Task so we can use .at[] lookups safely.
    # If sim has duplicate tasks, average them.
    sim_df = sim_df.set_index("Task")
    hardware_df = pd.DataFrame(HARDWARE_ROWS).set_index("Task")

    task_names = (
        "RaiseCube",
        "RotateArrow",
        "LiftPegUpright",
        "PickDishFromRack",
        "PickSodaFromCabinet",
    )
    perturbation_names = (
        "none",
        "mo_size",
        "light_color",
        "distractor_object",
        "background_color",
        "table_color",
        "mo_color",
        "language_none",
    )
    print("Sim:")
    print(sim_df)
    print("\nHardware:")
    print(hardware_df)

    LABEL_FONTSIZE = 15
    LEGEND_FONTSIZE = 12
    TICK_FONTSIZE = 12

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))

    task_colors = {
        "RaiseCube": "#FF8C00",            # vivid orange
        "RotateArrow": "#0066FF",          # strong blue
        "LiftPegUpright": "#00A651",       # bright green-teal
        "PickDishFromRack": "#CC00CC",     # vivid magenta
        "PickSodaFromCabinet": "#A0522D",  # sienna
    }
    perturbation_markers = {
        "none": "o",                # circle
        "mo_size": "X",             # x (filled)
        "light_color": "s",         # square
        "distractor_object": "^",   # triangle_up
        "background_color": "P",    # plus (filled)
        "table_color": "v",         # triangle_down
        "mo_color": "D",            # diamond
        "language_none": "*",       # star
    }
    marker_display_names = {
        "o": "circle",
        "X": "x (filled)",
        "s": "square",
        "^": "triangle up",
        "P": "plus (filled)",
        "v": "triangle down",
        "D": "diamond",
        "*": "star",
    }
    print()
    print("Perturbation markers:")
    for name, marker in perturbation_markers.items():
        display = PERTURBATION_SET_DISPLAY_NAMES.get(name.lower(), name)
        print(f"  {display}: '{marker}' ({marker_display_names.get(marker, marker)})")
    all_x1s = []
    all_x2s = []
    deltas_by_perturbation = {var: [] for var in perturbation_names}
    deltas_by_task = {task: [] for task in task_names}

    print()
    print("-------------")
    all_abs_errors = []
    for task_name in task_names:

        task_x1s = []
        task_x2s = []
        task_pert_names = []
        none_scatter = None

        for perturbation_name in perturbation_names:

            hw_val = hardware_df.at[task_name, perturbation_name]
            sim_val = sim_df.at[task_name, perturbation_name]
            delta = hw_val - sim_val
            if isnan(hw_val) or isnan(sim_val):
                continue

            deltas_by_perturbation[perturbation_name].append(delta)
            deltas_by_task[task_name].append(delta)
            task_x1s.append(hw_val)
            task_x2s.append(sim_val)
            task_pert_names.append(perturbation_name)
            all_x1s.append(hw_val)
            all_x2s.append(sim_val)

            color = task_colors[task_name]
            if perturbation_name == "none":
                none_scatter = ax.scatter(
                    hw_val,
                    sim_val,
                    label=f"{task_name}",
                    color=color,
                    marker=perturbation_markers[perturbation_name],
                    s=125,
                )
            else:
                ax.scatter(
                    hw_val,
                    sim_val,
                    label=None,
                    color=color,
                    marker=perturbation_markers[perturbation_name],
                    s=125,
                )

        print()
        print(f"{task_name}: {task_x1s} {task_x2s}")
        assert none_scatter is not None
        best_fit_line = np.polyfit(task_x1s, task_x2s, 1)
        x1_range = np.arange(min(task_x1s), max(task_x1s))
        task_x2s_arr = np.array(task_x2s)
        y_pred = np.polyval(best_fit_line, task_x1s)
        R_squared = 1 - (np.sum((task_x2s_arr - y_pred) ** 2) / np.sum((task_x2s_arr - np.mean(task_x2s_arr)) ** 2))
        spearman_rho = calculate_spearman_correlation(task_x1s, task_x2s)
        pearson_r = calculate_pearson_correlation(task_x1s, task_x2s)
        print(f"  R-squared: {R_squared:.5f}")
        print(f"  Pearson r: {pearson_r:.5f}")
        print(f"  Spearman rho: {spearman_rho:.5f}")
        print("  Absolute error from trend line:")
        abs_errors = [
            (pert, hw_val, sim_val, pred, abs(sim_val - pred))
            for pert, hw_val, sim_val, pred in zip(task_pert_names, task_x1s, task_x2s, y_pred)
        ]
        abs_errors.sort(key=lambda x: x[4], reverse=True)
        for pert, hw_val, sim_val, pred, abs_err in abs_errors:
            print(f"\t{pert}\t|{sim_val:.1f} - {pred:.1f}|\t=\t{abs_err:.2f}")
            all_abs_errors.append((task_name, pert, hw_val, sim_val, pred, abs_err))
        ax.plot(x1_range, np.polyval(best_fit_line, x1_range), color=color, linestyle="--")
        none_scatter.set_label(f"{task_name} (R²={R_squared:.3f}, ρ={spearman_rho:.3f})")

    # Average trend across all (task x perturbation) points.
    all_x1s_arr = np.array(all_x1s, dtype=float)
    all_x2s_arr = np.array(all_x2s, dtype=float)
    avg_fit = np.polyfit(all_x1s_arr, all_x2s_arr, 1)
    avg_pred = np.polyval(avg_fit, all_x1s_arr)
    avg_r2 = 1 - (np.sum((all_x2s_arr - avg_pred) ** 2) / np.sum((all_x2s_arr - np.mean(all_x2s_arr)) ** 2))
    avg_pearson = calculate_pearson_correlation(all_x1s, all_x2s)
    avg_spearman = calculate_spearman_correlation(all_x1s, all_x2s)
    x1_range = np.arange(min(all_x1s), max(all_x1s))
    ax.plot(
        x1_range,
        np.polyval(avg_fit, x1_range),
        color="black",
        linestyle="-",
        linewidth=2.0,
        label=f"Average (R²={avg_r2:.3f}, ρ={avg_spearman:.3f})",
    )
    print()
    print(f"Average: n={len(all_x1s)}")
    print(f"  R-squared: {avg_r2:.5f}")
    print(f"  Pearson r: {avg_pearson:.5f}")
    print(f"  Spearman rho: {avg_spearman:.5f}")

    print()
    print("All (task x perturbation) absolute errors from trend lines:")
    all_abs_errors.sort(key=lambda x: x[5], reverse=True)
    for task_name, pert, hw_val, sim_val, pred, abs_err in all_abs_errors:
        print(f"{task_name}.{pert}\t{abs_err:.2f}")

    print()
    print("Deltas by perturbation:")
    for var in perturbation_names:
        print(f"    {var}: {deltas_by_perturbation[var]}")

    print()
    print("Deltas by task:")
    for task in task_names:
        print(f"    {task}: {deltas_by_task[task]}")


    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True,alpha=0.5)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_xlabel("Hardware Success Rate", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Sim Success Rate", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    out_path = Path(out_dir) / "hardware_vs_sim.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)
    

def plot_by_perturbation(sim_csv_filepath: str, out_dir: str):
    """This function lumps the success rates for each task by perturbation
    """

    sim_df = pd.read_csv(sim_csv_filepath)
    if "Task" not in sim_df.columns:
        raise ValueError(f"Expected a 'Task' column in {sim_csv_filepath}, got columns: {list(sim_df.columns)}")

    sim_df = sim_df.set_index("Task")
    hardware_df = pd.DataFrame(HARDWARE_ROWS).set_index("Task")

    task_names = (
        "RaiseCube",
        "RotateArrow",
        "LiftPegUpright",
        "PickDishFromRack",
        "PickSodaFromCabinet",
    )
    perturbation_names = (
        "none",
        "mo_size",
        "light_color",
        "distractor_object",
        "background_color",
        "table_color",
        "mo_color",
        "language_none",
    )

    LABEL_FONTSIZE = 15
    LEGEND_FONTSIZE = 12
    TICK_FONTSIZE = 12

    perturbation_colors = {
        "none":               "#333333",
        "mo_size":            "#E6194B",
        "light_color":        "#F58231",
        "distractor_object":  "#3CB44B",
        "background_color":   "#4363D8",
        "table_color":        "#42D4F4",
        "mo_color":           "#911EB4",
        "language_none":      "#F032E6",
    }
    task_markers = {
        "RaiseCube":          "o",
        "RotateArrow":        "X",
        "LiftPegUpright":     "s",
        "PickDishFromRack":   "^",
        "PickSodaFromCabinet":"D",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 103)

    all_x1s = []
    all_x2s = []

    for perturbation_name in perturbation_names:
        var_hw = []
        var_sim = []
        first_scatter = None

        display_name = PERTURBATION_SET_DISPLAY_NAMES.get(perturbation_name.lower(), perturbation_name)
        color = perturbation_colors[perturbation_name]

        for task_name in task_names:
            hw_val = hardware_df.at[task_name, perturbation_name]
            sim_val = sim_df.at[task_name, perturbation_name]

            if isnan(hw_val) or isnan(sim_val):
                continue

            var_hw.append(hw_val)
            var_sim.append(sim_val)
            all_x1s.append(hw_val)
            all_x2s.append(sim_val)

            sc = ax.scatter(
                hw_val,
                sim_val,
                label=display_name if first_scatter is None else None,
                color=color,
                marker=task_markers[task_name],
                s=125,
                zorder=3,
            )
            if first_scatter is None:
                first_scatter = sc

        if len(var_hw) >= 2:
            best_fit_line = np.polyfit(var_hw, var_sim, 1)
            x_range = np.linspace(min(var_hw), max(var_hw), 100)
            y_pred = np.polyval(best_fit_line, var_hw)
            var_sim_arr = np.array(var_sim)
            ss_res = np.sum((var_sim_arr - y_pred) ** 2)
            ss_tot = np.sum((var_sim_arr - np.mean(var_sim_arr)) ** 2)
            R_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            ax.plot(x_range, np.polyval(best_fit_line, x_range), color=color, linestyle="--")
            if first_scatter is not None:
                first_scatter.set_label(f"{display_name} (R²={R_squared:.2f})")

    all_x1s_arr = np.array(all_x1s)
    all_x2s_arr = np.array(all_x2s)
    best_fit_line = np.polyfit(all_x1s_arr, all_x2s_arr, 1)
    x1_range = np.arange(min(all_x1s), max(all_x1s))
    y_pred = np.polyval(best_fit_line, all_x1s_arr)
    R_squared = 1 - (np.sum((all_x2s_arr - y_pred) ** 2) / np.sum((all_x2s_arr - np.mean(all_x2s_arr)) ** 2))
    ax.plot(x1_range, np.polyval(best_fit_line, x1_range), color="grey", linestyle="--", label=f"All (R²={R_squared:.2f})")
    # ax.text(0.05, 0.95, f"R²={R_squared:.2f}", fontsize=LABEL_FONTSIZE, transform=ax.transAxes, verticalalignment="top", horizontalalignment="left")

    # Marker legend for tasks
    task_legend_handles = [
        matplotlib.lines.Line2D(
            [], [],
            marker=task_markers[t],
            color="gray",
            linestyle="None",
            markersize=9,
            label=t,
        )
        for t in task_names
    ]
    perturbation_legend = ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", title="Perturbation")
    ax.add_artist(perturbation_legend)
    ax.legend(handles=task_legend_handles, fontsize=LEGEND_FONTSIZE, loc="lower right", title="Task")

    ax.grid(True, alpha=0.5)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Hardware Success Rate", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Sim Success Rate", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    out_path = Path(out_dir) / "hardware_vs_sim_by_perturbation.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_percent_of_none(sim_csv_filepath: str, out_dir: str):
    """Bar chart of success rate as a percent of the 'none' baseline (Real vs Sim).

    One subplot per task, plus a final subplot averaged across tasks.
    """

    sim_df = pd.read_csv(sim_csv_filepath)
    if "Task" not in sim_df.columns:
        raise ValueError(f"Expected a 'Task' column in {sim_csv_filepath}, got columns: {list(sim_df.columns)}")

    sim_df = sim_df.set_index("Task")
    hardware_df = pd.DataFrame(HARDWARE_ROWS).set_index("Task")

    task_names = (
        "RaiseCube",
        "RotateArrow",
        "LiftPegUpright",
        "PickDishFromRack",
        "PickSodaFromCabinet",
    )
    perturbation_names = (
        "mo_size",
        "light_color",
        "distractor_object",
        "background_color",
        "table_color",
        "mo_color",
        "language_none",
    )
    pert_labels = [PERTURBATION_SET_DISPLAY_NAMES.get(p.lower(), p) for p in perturbation_names]

    LABEL_FONTSIZE = 13
    LEGEND_FONTSIZE = 10
    TICK_FONTSIZE = 9

    subplot_titles = list(task_names) + ["Average"]
    n_subplots = len(subplot_titles)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 3.2 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]

    width = 0.35
    x = np.arange(len(perturbation_names))
    y_max = 120.0

    # Collect per-task relative rates so we can also average them.
    hw_by_task: dict[str, list[float]] = {}
    sim_by_task: dict[str, list[float]] = {}

    # print()
    # print("Percent of none (per task):")
    for task_name in task_names:
        hw_none = hardware_df.at[task_name, "none"] if "none" in hardware_df.columns else float("nan")
        sim_none = sim_df.at[task_name, "none"] if "none" in sim_df.columns else float("nan")

        hw_vals = []
        sim_vals = []
        for perturbation_name in perturbation_names:
            hw_val = (
                hardware_df.at[task_name, perturbation_name]
                if perturbation_name in hardware_df.columns
                else float("nan")
            )
            sim_val = (
                sim_df.at[task_name, perturbation_name]
                if perturbation_name in sim_df.columns
                else float("nan")
            )

            hw_rel = (
                100.0 * hw_val / hw_none
                if not isnan(hw_val) and not isnan(hw_none) and hw_none > 0
                else float("nan")
            )
            sim_rel = (
                100.0 * sim_val / sim_none
                if not isnan(sim_val) and not isnan(sim_none) and sim_none > 0
                else float("nan")
            )
            hw_vals.append(hw_rel)
            sim_vals.append(sim_rel)

        hw_by_task[task_name] = hw_vals
        sim_by_task[task_name] = sim_vals

        # print(f"  {task_name}:")
        # for p, hw_rel, sim_rel in zip(perturbation_names, hw_vals, sim_vals):
        #     hw_s = f"{hw_rel:.1f}%" if not isnan(hw_rel) else "nan"
        #     sim_s = f"{sim_rel:.1f}%" if not isnan(sim_rel) else "nan"
        #     print(f"    {p}: real={hw_s}, sim={sim_s}")

    # Average across tasks (nanmean so missing task/pert pairs are skipped).
    hw_avg = [
        float(np.nanmean([hw_by_task[t][i] for t in task_names]))
        for i in range(len(perturbation_names))
    ]
    sim_avg = [
        float(np.nanmean([sim_by_task[t][i] for t in task_names]))
        for i in range(len(perturbation_names))
    ]
    # print("  Average:")
    # for p, hw_rel, sim_rel in zip(perturbation_names, hw_avg, sim_avg):
    #     hw_s = f"{hw_rel:.1f}%" if not isnan(hw_rel) else "nan"
    #     sim_s = f"{sim_rel:.1f}%" if not isnan(sim_rel) else "nan"
    #     print(f"    {p}: real={hw_s}, sim={sim_s}")

    series_by_title = {t: (hw_by_task[t], sim_by_task[t]) for t in task_names}
    series_by_title["Average"] = (hw_avg, sim_avg)

    for ax, title in zip(axes, subplot_titles):
        hw_vals, sim_vals = series_by_title[title]
        for v in hw_vals + sim_vals:
            if not isnan(v):
                y_max = max(y_max, v * 1.1)

        ax.bar(x - width / 2, hw_vals, width, label="Real", color="#4363D8")
        ax.bar(x + width / 2, sim_vals, width, label="Sim", color="#F58231")
        ax.axhline(100.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_ylabel("% of None SR", fontsize=LABEL_FONTSIZE)
        ax.set_title(title, fontsize=LABEL_FONTSIZE, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")

    for ax in axes:
        ax.set_ylim(0, y_max)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(pert_labels, rotation=30, ha="right")
    axes[-1].set_xlabel("Perturbation", fontsize=LABEL_FONTSIZE)
    plt.tight_layout()

    out_path = Path(out_dir) / "hardware_vs_sim_percent_of_none.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-csv-filepath", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="plots/colosseum_v2_hardware")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing plot images (prints summaries only).")
    args = parser.parse_args()
    plot_by_task(args.sim_csv_filepath, args.out_dir)
    plot_by_perturbation(args.sim_csv_filepath, args.out_dir)
    plot_percent_of_none(args.sim_csv_filepath, args.out_dir)
