import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "tab:brown",
        "tab:red",
        "tab:orange",
        "tab:olive",
        "tab:green",
        "tab:cyan",
        "tab:blue",
        "tab:purple",
        "tab:pink",
    ]
)


def get_table():
    brain_networks = ["MD", "lang", "vis", "aud"]
    code_models = [
        "projection",
        "bow",
        "tfidf",
        "seq2seq",
        "xlnet",
        "bert",
        "gpt2",
        "transformer",
        "roberta",
    ]
    dims = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    rows = []
    for opt in itertools.product(brain_networks, code_models, dims):
        score = np.load(
            f"../../../../braincode/.cache/scores/mvpa/score_{opt[0]}_{opt[1]}{opt[2]}.npy"
        )
        rows.append([*opt, float(score)])
    return pd.DataFrame(
        rows, columns=["Brain Network", "Code Model", "Dimensionality", "Mapping Score"]
    )


def update_names(df):
    df.loc[df["Brain Network"] == "MD", "Brain Network"] = "MD"
    df.loc[df["Brain Network"] == "lang", "Brain Network"] = "Language"
    df.loc[df["Brain Network"] == "vis", "Brain Network"] = "Visual"
    df.loc[df["Brain Network"] == "aud", "Brain Network"] = "Auditory"
    df.loc[df["Code Model"] == "projection", "Code Model"] = "TokenProjection"
    df.loc[df["Code Model"] == "bow", "Code Model"] = "BagOfWords"
    df.loc[df["Code Model"] == "tfidf", "Code Model"] = "TF-IDF"
    df.loc[df["Code Model"] == "seq2seq", "Code Model"] = "Seq2Seq"
    df.loc[df["Code Model"] == "xlnet", "Code Model"] = "XLNet"
    df.loc[df["Code Model"] == "transformer", "Code Model"] = "CodeTransformer"
    df.loc[df["Code Model"] == "bert", "Code Model"] = "CodeBERT"
    df.loc[df["Code Model"] == "gpt2", "Code Model"] = "CodeGPT"
    df.loc[df["Code Model"] == "roberta", "Code Model"] = "CodeBERTa"
    return df


def plot_data(df):
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
    fig.set_size_inches(16, 4)
    for i, network in enumerate(["MD", "Language", "Visual", "Auditory"]):
        models = df[df["Brain Network"] == network]
        for model in models["Code Model"].unique():
            samples = models[models["Code Model"] == model]
            axes[i].plot(
                samples["Dimensionality"],
                samples["Mapping Score"],
                "D-",
                markersize=8,
                linewidth=3,
            )
        axes[i].set_title(network)
        axes[i].set_xlabel("Embedding Dimensionality")
        if not i:
            axes[i].set_ylabel("Mapping Score (Rank Accuracy)")
        axes[i].set_xscale("log", base=2)
        axes[i].set_xticks([4, 8, 16, 32, 64, 128, 256, 512, 1024])
        axes[i].set_yticks([0.50, 0.52, 0.54, 0.56, 0.58, 0.60])
        axes[i].set_ylim([0.48, 0.62])
        axes[i].yaxis.set_ticks_position("left")
        axes[i].xaxis.set_ticks_position("bottom")
        for spine in ["right", "top"]:
            axes[i].spines[spine].set_visible(False)
    plt.legend(models["Code Model"].unique())
    plt.savefig("model_dim_expt.png", bbox_inches="tight", dpi=600)
    plt.close()


def main():
    df = get_table()
    df = update_names(df)
    df.to_csv("model_dim_expt.csv")
    plot_data(df)


if __name__ == "__main__":
    main()
