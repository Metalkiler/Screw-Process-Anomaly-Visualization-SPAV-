import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Union
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize as Color_Normalize


def plot_reconstruction(actual_df: pd.DataFrame, reconstruction_df: pd.DataFrame,
                        score_df: pd.DataFrame, detected_anomalies: pd.Series = None,
                        process_idx: int = None, file_path: str = "profile_reconstruction.png",
                        x: str = "angle", y: str = "torque", x_reconstr: str = "reconstr_angle",
                        y_reconstr: str = "reconstr_torque", actual_line_color: str = "black",
                        reconstructed_line_color: str = "deepskyblue", anomaly_color: str = "orange",
                        figsize=(12, 5), dpi: int = 200, save: bool = True):
    if process_idx:
        n = actual_df.loc[process_idx].shape[0]
        anomaly_score = score_df.loc[process_idx, :].mean().mean()
        actual_df = actual_df.loc[process_idx]
        reconstruction_df = reconstruction_df.loc[process_idx]
    else:
        n = actual_df.shape[0]
        anomaly_score = score_df.mean().mean()
    anomaly_score = float(anomaly_score)

    plt.figure(figsize=figsize, dpi=dpi)
    # Plot original data (Actual)
    sns.lineplot(data=actual_df, x=x, y=y, color=actual_line_color, label='Actual')
    # Plot reconstructed data
    inv_tf_data = reconstruction_df[[x_reconstr, y_reconstr]].copy()
    sns.lineplot(x=inv_tf_data[x_reconstr], y=inv_tf_data[y_reconstr], color=reconstructed_line_color, linestyle="--",
                 label="Reconstructed")
    # Plot detected anomalies as points if present
    if detected_anomalies is not None:
        sns.scatterplot(data=detected_anomalies, x=x, y=y, color=anomaly_color, label="Anomaly")

    plt.xlabel("Angle", fontsize=12)
    plt.ylabel("Torque", fontsize=12)

    if process_idx:
        plt.title(
            f'Actual vs Reconstructed Values: Cycle {process_idx} - Anomaly Score = {anomaly_score:.4f}, $N_{{rows}}$={n}',
            fontsize=14)
    else:
        plt.title(f'Actual vs Reconstructed Values: Anomaly Score = {anomaly_score:.4f}, $N_{{rows}}$={n}', fontsize=14)
    plt.legend(fontsize=10)

    if save:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close('all')


def plot_global_error(df: pd.DataFrame, score_df: pd.DataFrame, score_column: str = "ae",
                      x: str = "angle", y: str = "torqueun", cycle_ids: list = None, normalize: bool = True,
                      cmap: str = "viridis", xlabel: str = "Angle", ylabel: str = "Torque",
                      error_name: str = "Absolute Error (Normalized)", line_width: float = 0.01,
                      figsize: tuple = (14, 8), dpi: int = 300, show_grid: bool = False, title_fontsize: int = 20,
                      label_fontsize: int = 18, ticks_fontsize: int = 14, label_padding: float = 12.0,
                      title_padding: float = 12.0, theme: str = "default", save: bool = True,
                      title: str = "Global Plot of Evaluated Processes Colored by Anomaly Score",
                      file_path: str = "global_error_plot.png"):
    with plt.style.context(theme):
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if not cycle_ids or len(cycle_ids) == 0:
            cycle_ids = df.index.get_level_values(0).unique()
        errors = score_df[score_column].to_numpy(dtype=float)

        if normalize:
            min_err, max_err = errors.min(), errors.max()
            denom = max_err - min_err if max_err > min_err else 1.0
            errors_norm = (errors - min_err) / denom
            norm = Color_Normalize(vmin=0, vmax=1)  # normalized scale
            errors_to_use = pd.DataFrame(errors_norm, index=df.index)
        else:
            norm = Color_Normalize(vmin=errors.min(), vmax=errors.max())
            errors_to_use = pd.DataFrame(errors, index=df.index)

        cmap_obj = plt.get_cmap(cmap)

        all_segments = []
        all_segment_errors = []

        for process in cycle_ids:
            process_data = df.loc[process].copy()
            x_values = process_data[x].copy().to_numpy()
            y_values = process_data[y].copy().to_numpy()

            # Create segments for this line (each segment is two points)
            points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # For each segment, assign the error score of the process
            # error is associated to each segment, and we choose to average
            # between the starting and end points.
            segment_errors = (errors_to_use.loc[process].values[:-1] + errors_to_use.loc[process][1:]) / 2

            all_segments.append(segments)
            all_segment_errors.append(segment_errors)

        # Concatenate all process segments and corresponding error arrays
        all_segments = np.vstack(all_segments)
        all_segment_errors = np.concatenate(all_segment_errors).flatten()

        # Create one big LineCollection for all lines
        lc = LineCollection(all_segments,
                            cmap=cmap_obj,
                            norm=norm,
                            linewidth=line_width)
        lc.set_array(all_segment_errors)
        ax.add_collection(lc)
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label(error_name, fontsize=label_fontsize, labelpad=label_padding)
        cbar.ax.tick_params(labelsize=ticks_fontsize)

        ax.grid(visible=show_grid)
        # Set axis limits (important, collections don't autoscale)
        ax.set_xlim(df[x].min(), df[x].max())
        ax.set_ylim(df[y].min(), df[y].max() * 1.05)  # add padding to height
        plt.xlabel(xlabel, fontsize=label_fontsize, labelpad=label_padding)
        plt.ylabel(ylabel, fontsize=label_fontsize, labelpad=label_padding)
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.title(title, fontsize=title_fontsize, pad=title_padding)
        plt.tight_layout()
        if save:
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()
            plt.close()


def plot_local_error(df: pd.DataFrame, score_df: pd.DataFrame, score_column: str = "ae",
                     x: str = "angle", y: str = "torque",
                     cmap: str = "inferno", normalize=True, xlabel: str = "Angle", ylabel: str = "Torque",
                     file_path: str = "local_error_plot.png", error_name: str = "Absolute Error (Normalized)",
                     title: str = "Local Plot of Evaluated Processes Colored by Anomaly Score", line_width: float = 2.,
                     figsize: tuple = (14, 6), dpi: int = 200, show_grid: bool = False, title_fontsize: int = 20,
                     label_fontsize: int = 18, ticks_fontsize: int = 14, label_padding: float = 12.0,
                     title_padding: float = 12.0, theme: str = "default", save: bool = True):
    with plt.style.context(theme):
        x_values = df[x]
        y_values = df[y]
        errors = score_df[score_column]

        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if normalize:
            errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors)).astype(float)
            norm = plt.Normalize(vmin=min(errors), vmax=max(errors))
        else:
            norm = None
        # Color segments by error (averaged between adjacent points)
        segment_errors = (errors.values[:-1] + errors.values[1:]) / 2

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=line_width,
        )
        lc.set_array(segment_errors)
        line = ax.add_collection(lc)

        cbar = plt.colorbar(line, ax=ax)
        cbar.set_label(error_name, size=label_fontsize, labelpad=label_padding)
        cbar.ax.tick_params(labelsize=ticks_fontsize)

        ax.grid(visible=show_grid)
        ax.set_xlim(x_values.min(), x_values.max())
        ax.set_ylim(y_values.min(), y_values.max() * 1.05)  # add padding to height
        ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=label_padding)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=label_padding)
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.title(title, fontsize=title_fontsize, pad=title_padding)
        plt.tight_layout()
        if save:
            plt.savefig(file_path)
        else:
            plt.show()
        plt.close()


def plot_detected_anomalies(df: Union[pd.DataFrame, pd.Series], detected_anomalies_df: Union[pd.DataFrame, pd.Series],
                            x: str = "angle", y: str = "torque", file_path: str = "detected_anomalies.png",
                            line_color: str = "black", anomaly_color: str = "orange",
                            title: str = "Anomalies Detected During Model Evaluation",
                            figsize: tuple = (14, 6), dpi: int = 300, show_grid: bool = False,
                            title_fontsize: int = 20, xlabel: str = "Angle", ylabel: str = "Torque",
                            label_fontsize: int = 18, ticks_fontsize: int = 14, label_padding: float = 12.0,
                            title_padding: float = 12.0, theme: str = "default", save: bool = True):
    with plt.style.context(theme):
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        sns.lineplot(data=df, x=x, y=y, color=line_color)
        sns.scatterplot(data=detected_anomalies_df, x=x, y=y, color=anomaly_color, label="Anomaly")

        ax.grid(visible=show_grid)
        plt.xlabel(xlabel, fontsize=label_fontsize, labelpad=label_padding)
        plt.ylabel(ylabel, fontsize=label_fontsize, labelpad=label_padding)
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.title(title, fontsize=title_fontsize, pad=title_padding)
        plt.tight_layout()
        if save:
            plt.savefig(file_path)
        else:
            plt.show()
        plt.close()


def plot_kde_density(feat_df: pd.DataFrame, detected_anomalies: pd.DataFrame,
                     x: str = "angle", y: str = "torque", cmap: str = "terrain",
                     bw_adjust: float = 0.3, thresh: float = 0.5,
                     levels: int = 100, alpha: float = 0.9, line_color: str = "black",
                     file_path: str = "kde_density_detected_anomalies.png",
                     title: str = "Density Plot of Anomalies Detected During Model Evaluation",
                     figsize: tuple = (14, 6), dpi: int = 300, show_grid: bool = False,
                     title_fontsize: int = 20, xlabel: str = "Angle", ylabel: str = "Torque",
                     label_fontsize: int = 18, ticks_fontsize: int = 14, label_padding: float = 12.0,
                     title_padding: float = 12.0, theme: str = "default", save: bool = True):
    with plt.style.context(theme):
        feat_df_anomalies = feat_df[[x, y]].copy()
        feat_df_anomalies = feat_df.loc[detected_anomalies.index]

        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        try:
            # KDEplot 2D with anomalous point density
            kde = sns.kdeplot(
                data=feat_df_anomalies,
                x=x,
                y=y,
                fill=True,
                cmap=cmap,
                bw_adjust=bw_adjust,  # Adjusts smoothness (smaller value = more detail)
                thresh=thresh,  # Hides regions with very lower density (under thresh)
                levels=levels,  # Adjust levels of detail (100 is very detailed)
                alpha=alpha
            )

            # seaborn’s kdeplot() returns the axes, not the contour set or scalar mappable needed for a colorbar
            # Usually, the first collection (collections) of this object (when fill=True) represents the filled area with the colormap.
            cbar = plt.colorbar(kde.collections[0], ax=ax)
            cbar.set_label("Anomaly Density", size=label_fontsize, labelpad=label_padding)
            # cbar.set_ticks([min_val, mid_val, max_val])
            cbar.ax.tick_params(labelsize=ticks_fontsize)
        except ValueError as e:
            print(e)
            print("Ensure that the detected_anomalies DataFrame has a sufficient amount of data points.")
            print(f"Detected Anomalies DataFrame:\n{detected_anomalies}")
            print("Returning plot without anomaly densities...")

        plt.plot(feat_df[x], feat_df[y], color=line_color, linewidth=1, label='True Process')

        # Layout
        ax.grid(visible=show_grid)
        plt.xlabel(xlabel, fontsize=label_fontsize, labelpad=label_padding)
        plt.ylabel(ylabel, fontsize=label_fontsize, labelpad=label_padding)
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.title(title, fontsize=title_fontsize, pad=title_padding)
        plt.tight_layout()
        if save:
            plt.savefig(file_path)
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    from auxiliary_functions import *
    import pyscrew
    import json
    import os

    print(os.getcwd())

    scenario = "s04"
    dir_results = f"results-ae-h2o-pyscrew-{scenario}"
    if os.path.exists("s04.csv"):
        screw_df = pd.read_csv(f"{scenario}.csv", index_col=[0, 1])
    else:
        data = pyscrew.get_data(scenario=scenario)
        df = pd.DataFrame(data)

        screw_df = df[["torque_values", "angle_values", "workpiece_result"]].copy()
        screw_df["workpiece_result"] = screw_df["workpiece_result"].map({"OK": 0, "NOK": 1})
        screw_df = screw_df.rename({"angle_values": "angle",
                                    "torque_values": "torque",
                                    "workpiece_result": "screwgof"}, axis=1)

        exploded_screw_df = screw_df.apply(pd.Series.explode).reset_index()
        exploded_screw_df = exploded_screw_df.rename({"index": "cycle_id"}, axis=1)

        screw_df = exploded_screw_df.set_index(
            ['cycle_id', exploded_screw_df.groupby('cycle_id').cumcount()])
        screw_df.to_csv(f"{scenario}.csv")

    print(screw_df)
    h2o.init()
    n_iter = 1
    # TODO Func load trained model / results?
    #model = h2o.load_model(os.path.join(dir_results, f"best-models-h2o/ae_h2o_{n_iter}"))
    model = tf.keras.models.load_model("modelBestAE_Products20250701-144726.keras")

    with open((os.path.join(dir_results, f"train-test-indices/train_test_indices_{n_iter}.json")), "r") as file:
        loaded_data = json.load(file)

    with open((os.path.join(dir_results, "ae_h2o_pipes.pkl")), "rb") as file:
        loaded_pipes = pickle.load(file)
    pipe = loaded_pipes[n_iter]
    columns = ["angle", "torqueun"]

    loaded_test_indices = loaded_data['test_indices']
    test_df = screw_df.loc[loaded_test_indices]
    # Need to apply processing pipe fit on train data to match what the model was trained on
    test_df_tf = pd.DataFrame(pipe.transform(test_df[columns]), columns=columns, index=test_df.index)
    y_true = screw_df.loc[loaded_test_indices]["screwgof"]
    anomaly_df_tf = get_anomaly_scores(model, test_df_tf, modelType="DFFN", yData = y_true)

    fpr, tpr, thresholds = roc_curve(anomaly_df_tf.Actual, anomaly_df_tf.ae, pos_label=1)
    scoreAUC = auc(fpr, tpr)

    # listAUC.append(scoreAUC)
    print("AUC", scoreAUC)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % auc(fpr, tpr),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig("/results/ROC.png")
    plt.close()
    threshold = find_nearest_threshold(anomaly_df_tf.Actual, anomaly_df_tf["ae"], threshold=5.5, force=True)

    plot_global_error(test_df, anomaly_df_tf, file_path="/results/global_error_plot.png", y = "torqueun", line_width=1)  # , cycle_ids=[105, 500])

    idx_anomaly = 815
    idx_good = 725


    test_df_local_true_anomaly = screw_df.loc[
        idx_anomaly].copy()  # unnormalized true local angle-torque values for plots
    y_true_local_anomaly = test_df_local_true_anomaly.pop("screwgof")
    #score_df_local_anomaly = get_score_df(model, test_df_tf, y_true, cycle_id=idx_anomaly, y = "torqueun")
    score_df_local_anomaly = get_anomaly_scores_single_ae(model, test_df_local_true_anomaly, yData=y_true_local_anomaly,cycleid=idx_anomaly)
    test_df_local_true_good = screw_df.loc[idx_good].copy()  # unnormalized true local angle-torque values for plots
    y_true_local_good = test_df_local_true_good.pop("screwgof")
    score_df_local_good = get_anomaly_scores_single_ae(model, test_df_local_true_good, yData=y_true_local_good,cycleid=idx_good)

    test_df_local_pred_anomaly = pd.concat(
        [test_df_local_true_anomaly.copy(), score_df_local_anomaly.copy()], axis=1)
    threshold_anomaly_idx_anomaly = classify_anomaly_by_threshold(dataset=score_df_local_anomaly,
                                                                  threshold_value=threshold)
    detected_anomalies_anomaly = test_df_local_pred_anomaly.iloc[threshold_anomaly_idx_anomaly].copy()

    test_df_local_pred_good = pd.concat(
        [test_df_local_true_good.copy(), score_df_local_good.copy()], axis=1)
    threshold_anomaly_idx_good = classify_anomaly_by_threshold(dataset=score_df_local_good,
                                                               threshold_value=threshold)
    detected_anomalies_good = test_df_local_pred_good.iloc[threshold_anomaly_idx_good].copy()

    plot_local_error(test_df_local_pred_anomaly, score_df_local_anomaly,
                     file_path="/results/local_error_plot_anomaly.png",
                     title=f"Local Plot of Evaluated Processes Colored by Anomaly Score (Process ID = {idx_anomaly})", y="torqueun")
    plot_local_error(test_df_local_pred_good, score_df_local_good, file_path="/results/local_error_plot_good.png",
                     title=f"Local Plot of Evaluated Processes Colored by Anomaly Score (Process ID = {idx_good})",y="torqueun")

    plot_detected_anomalies(test_df_local_pred_anomaly, detected_anomalies_anomaly,
                            title=f"Anomalies Detected During Evaluation (Threshold = {threshold:.4f})",
                            file_path="/results/detected_anomalies_anomaly.png",
                            title_fontsize=30, label_fontsize=24, ticks_fontsize=20,y="torqueun")
    plot_detected_anomalies(test_df_local_pred_good, detected_anomalies_good,
                            title=f"Anomalies Detected During Evaluation (Threshold = {threshold:.4f})",
                            file_path="/results/detected_anomalies_good.png",
                            title_fontsize=30, label_fontsize=24, ticks_fontsize=20,y="torqueun")

    plot_kde_density(test_df_local_pred_anomaly, detected_anomalies_anomaly,
                     title=f"Anomaly Density Plot (Threshold = {threshold:.4f})",
                     file_path="/results/kde_density_detected_anomalies_anomaly.png",
                     title_fontsize=30, label_fontsize=26, ticks_fontsize=22,y="torqueun")

    # plot_kde_density(test_df_local_pred_good, detected_anomalies_good,
    #                title=f"Anomaly Density Plot (Threshold = {threshold:.4f})",
    #               file_path="/results/kde_density_detected_anomalies_good.png",
    #              title_fontsize=30, label_fontsize=26, ticks_fontsize=22)

    test_df_local_reconstruction_anomaly = get_reconstruction_df_ae(model, test_df_local_true_anomaly)
    test_df_local_reconstruction_good = get_reconstruction_df_ae(model, test_df_local_true_good)
    #test_df_reconstruction = get_reconstruction_df_ae(model, test_df)

    plot_reconstruction(test_df_local_true_anomaly, test_df_local_reconstruction_anomaly, score_df_local_anomaly,
                        file_path="/results/profile_reconstruction_anomaly.png", y="torqueun", x_reconstr="angle", y_reconstr="torqueun")

    plot_reconstruction(test_df_local_true_good, test_df_local_reconstruction_good, score_df_local_good,
                        file_path="/results/profile_reconstruction_good.png", y="torqueun", x_reconstr="angle", y_reconstr="torqueun")

    h2o.cluster().shutdown()


