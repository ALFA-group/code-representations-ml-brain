import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(dataset):
    return pd.read_csv(f"../tables/raw/{dataset}.csv")


def update_names(data):
    data.loc[data.Feature == "MD+lang", "Feature"] = "MD+L"
    data.loc[data.Feature == "MD+vis", "Feature"] = "MD+V"
    data.loc[data.Feature == "lang+vis", "Feature"] = "L+V"
    data.loc[data.Feature == "MD", "Feature"] = "MD"
    data.loc[data.Feature == "lang", "Feature"] = "Language"
    data.loc[data.Feature == "vis", "Feature"] = "Visual"
    data.loc[data.Feature == "aud", "Feature"] = "Auditory"
    data.loc[data.Feature == "projection", "Feature"] = "Token Projection"
    data.loc[data.Feature == "bow", "Feature"] = "Bag Of Words"
    data.loc[data.Feature == "tfidf", "Feature"] = "TF-IDF"
    data.loc[data.Feature == "seq2seq", "Feature"] = "Seq2Seq"
    data.loc[data.Feature == "xlnet", "Feature"] = "XLNet"
    data.loc[data.Feature == "transformer", "Feature"] = "CodeTransformer"
    data.loc[data.Feature == "roberta", "Feature"] = "CodeBERTa"
    data.loc[data.Feature == "bert", "Feature"] = "CodeBERT"
    data.loc[data.Feature == "gpt2", "Feature"] = "CodeGPT"
    data.loc[data.Target == "code", "Target"] = "Code vs. Sentence"
    data.loc[data.Target == "lang", "Target"] = "Variable Language"
    data.loc[data.Target == "content", "Target"] = "Data Type"
    data.loc[data.Target == "structure", "Target"] = "Control Flow"
    data.loc[data.Target == "lines", "Target"] = "Dynamic Analysis"
    data.loc[data.Target == "bytes", "Target"] = "Bytecode Operations"
    data.loc[data.Target == "nodes", "Target"] = "Node Count"
    data.loc[data.Target == "tokens", "Target"] = "Static Analysis"
    data.loc[data.Target == "halstead", "Target"] = "Halstead Difficulty"
    data.loc[data.Target == "cyclomatic", "Target"] = "Cyclomatic Complexity"
    data.loc[data.Target == "projection", "Target"] = "Token Projection"
    data.loc[data.Target == "bow", "Target"] = "Bag Of Words"
    data.loc[data.Target == "tfidf", "Target"] = "TF-IDF"
    data.loc[data.Target == "seq2seq", "Target"] = "Seq2Seq"
    data.loc[data.Target == "xlnet", "Target"] = "XLNet"
    data.loc[data.Target == "transformer", "Target"] = "CodeTransformer"
    data.loc[data.Target == "roberta", "Target"] = "CodeBERTa"
    data.loc[data.Target == "bert", "Target"] = "CodeBERT"
    data.loc[data.Target == "gpt2", "Target"] = "CodeGPT"
    return data


def make_base_plot(data, dataset):
    bar_width = 0.16
    cidx = 0
    ax = plt.subplot(111)
    for i, rep in enumerate(data["Feature"].unique()):
        samples = data[data["Feature"] == rep]
        scores = samples["Score"].values
        error = samples["95CI"].values
        if not i:
            r = np.arange(len(scores)) - 0.5 * bar_width
        else:
            r = np.array([x + bar_width for x in r])
        if "ablation" in dataset:
            color = np.array(
                [1.0 - (cidx * 0.2), 0.05 + (cidx * 0.15), 0 + (0.2 * cidx)]
            )
        else:
            color = np.array(
                [0.1 + (cidx * 0.30), 0.5 + (cidx * 0.15), 0.9 - (cidx * 0.30)]
            )
        cidx += 1
        ax.bar(
            r,
            scores,
            yerr=error,
            color=color,
            width=bar_width,
            edgecolor="black",
            label=rep,
            capsize=2,
        )
    plt.xticks(
        [r + bar_width for r in range(len(scores))],
        data["Target"].unique(),
        rotation=45,
    )
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    return ax


def individual_formatting(ax, dataset):
    data = load_data(dataset)
    cfg = {
        "mvpa_properties_cls": {
            "xlabel": "Code Properties",
            "ylabel": "Classification Accuracy (%)",
            "ylim": [0, 1],
            "yticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "size": (6, 4),
            "sig_y": 0.025,
            "legend_loc": (0.75, 0.90),
        },
        "mvpa_properties_rgr": {
            "xlabel": "Code Properties",
            "ylabel": "Pearson Correlation (r)",
            "ylim": [-0.1, 0.45],
            "yticks": [0, 0.1, 0.2, 0.3, 0.4],
            "size": (3, 4),
            "sig_y": 0.0125,
            "legend_loc": (0.77, 1.00),
        },
        "mvpa_models": {
            "xlabel": "Code Model",
            "ylabel": "Ranked Accuracy (%)",
            "ylim": [0.45, 0.65],
            "yticks": [0.45, 0.50, 0.55, 0.60, 0.65],
            "size": (8, 4),
            "sig_y": 0.45 + 0.0025,
            "legend_loc": (0.80, 1.03),
        },
    }
    dataset = dataset.replace("_ablation", "")
    dataset_cfg = cfg[dataset]
    plt.xlabel(dataset_cfg["xlabel"], fontweight="bold")
    plt.ylabel(dataset_cfg["ylabel"], fontweight="bold")
    plt.ylim(dataset_cfg["ylim"])
    plt.yticks(dataset_cfg["yticks"], labels=[f"{t}" for t in dataset_cfg["yticks"]])
    plt.legend(loc="center left", bbox_to_anchor=dataset_cfg["legend_loc"])
    for i, target in enumerate(data.Target.unique()):
        for j, baseline in enumerate(data[data.Target == target]["Null Mean"]):
            ax.plot(
                np.array([i - 0.12, i - 0.04]) + (j * 0.16),
                [baseline, baseline],
                "-",
                color="0.25",
            )
    plt.gcf().set_size_inches(*dataset_cfg["size"])
    x_start = -0.12
    for target in data.Target.unique():
        samples = data[data.Target == target]
        sigs = samples["h (corrected)"] == 1
        for i, sig in enumerate(sigs):
            x = x_start + 0.16 * i
            if sig:
                plt.annotate("*", (x, dataset_cfg["sig_y"]))
        x_start += 1
    return ax


def plot_data(dataset):
    data = load_data(dataset)
    data = update_names(data)
    ax = make_base_plot(data, dataset)
    ax = individual_formatting(ax, dataset)
    plt.savefig(f"../plots/{dataset}.png", bbox_inches="tight", dpi=600)
    plt.close()


def filter_data(data):
    data = data[data.Feature.isin(["MD", "Language"])]
    data = data[~(data.Target.isin(["Code vs. Sentence", "Variable Language"]))]
    return data


def make_inline_plot(data, dataset):
    ax = plt.subplot(111, label=dataset)
    for i, network in enumerate(data.Feature.unique()):
        samples = data[data.Feature == network]
        if "model" in dataset:
            basemodel = "Token Projection"
            samples = samples[samples.Target != basemodel]
            score = samples["Score"]
            error = samples["95CI"]
            ylim = [0.49, 0.63]
            xlabel = "Code Model"
            ylabel = "Ranked Accuracy (%)"
            size = [6, 2]
        else:
            score = samples["z"]
            error = np.divide(samples["95CI"], samples["Null SD"])
            ylim = [-1, 12]
            xlabel = "Code Property"
            ylabel = "Decoding Score (z)"
            size = [4, 2]
        c = np.array([0.1 + (i * 0.30), 0.5 + (i * 0.15), 0.9 - (i * 0.30)])
        plt.errorbar(
            samples["Target"],
            score,
            yerr=error,
            fmt="D-",
            color=c,
            markersize=8,
            linewidth=3,
            capsize=5,
        )
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel, fontweight="bold", fontsize=12)
    plt.ylabel(ylabel, fontweight="bold", fontsize=12)
    plt.ylim(ylim)
    plt.legend(
        data.Feature.unique(), loc="center left", bbox_to_anchor=[1, 1], fontsize=12
    )
    for network in data.Feature.unique():
        samples = data[data.Feature == network]
        if "model" in dataset:
            basemodel = "Token Projection"
            baseline = samples[samples.Target == basemodel]["Score"].values
        else:
            baseline = 0
        plt.plot([0, len(samples)], [baseline, baseline], "--", color="0.25")
    if "model" in dataset:
        baseline = data["Null Mean"].mean()
        plt.plot([0, len(samples)], [baseline, baseline], "--", color="0.25")
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    plt.gcf().set_size_inches(size)
    return ax


def make_inline_plots(dataset):
    data = load_data(dataset)
    data = update_names(data)
    data = filter_data(data)
    ax = make_inline_plot(data, dataset)
    plt.savefig(f"../plots/{dataset}_inline.png", bbox_inches="tight", dpi=600)
    plt.close()


def main():
    datasets = [
        "mvpa_properties_cls",
        "mvpa_properties_rgr",
        "mvpa_models",
        "mvpa_properties_cls_ablation",
        "mvpa_properties_rgr_ablation",
        "mvpa_models_ablation",
    ]
    for dataset in datasets:
        plot_data(dataset)
    for dataset in ["mvpa_properties_all", "mvpa_models"]:
        make_inline_plots(dataset)


if __name__ == "__main__":
    main()
