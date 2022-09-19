import pandas as pd
from plots import update_names


def load_data(dataset):
    return pd.read_csv(f"../tables/raw/{dataset}.csv")


def add_baselines(data):
    data["Empirical Baseline"] = 0
    for benchmark in data["Target"].unique():
        baseline = data[data["Target"] == benchmark]["Null Mean"].mean()
        data.loc[(data["Target"] == benchmark), "Empirical Baseline"] = baseline
    map = lambda x: f"{x:0.2f}"
    data["Empirical Baseline"] = data["Empirical Baseline"].apply(map)
    data.loc[(data["Empirical Baseline"] == "-0.00"), "Empirical Baseline"] = "0.00"
    return data


def make_pivot_table(data, dataset):
    if "prda" in dataset:
        index = "Feature"
        columns = ["Target", "Empirical Baseline"]
    else:
        index = ["Target", "Empirical Baseline"]
        columns = "Feature"
    return pd.pivot_table(data=data, index=index, columns=columns, values="Score")


def reorder_columns(data, dataset):
    if "prda" in dataset:
        data = data.iloc[[7, 2, 4, 1, 3, 8, 5, 6, 0], :]
    elif "ablation" in dataset:
        pass
    elif "rsa" in dataset:
        data = data.iloc[:, [1, 0]]
    else:
        data = data.iloc[:, [2, 1, 3, 0]]
    if "models" in dataset:
        data = data.iloc[[7, 2, 4, 1, 3, 8, 5, 6, 0], :]
    return data


def format_scores(data, dataset):
    map = lambda x: f"{x:0.2f}"
    for i, row in data.iterrows():
        if (
            "prda" not in dataset
            and "rgr" not in dataset
            and "supplemental" not in dataset
            and "rsa" not in dataset
        ):
            baseline = float(row.name[1])
            diff = row - baseline
            row = row.apply(map).values + " (+" + diff.apply(map).values + ")"
        else:
            row = row.apply(map).values
        row = [s.replace("+-", "-") for s in row]
        data.loc[i] = row
    if "rgr" in dataset or "supplemental" in dataset:
        data = data.reset_index(level=1, drop=True)
    return data


def format_latex(latex, dataset):
    latex = latex.replace("{lllll}", "{l||llll}")
    latex = latex.replace("{llllll}", "{l||l|llll}")
    latex = latex.replace("{llllllllll}", "{l||lllllllll}")
    latex = latex.replace("{lllllllllll}", "{l||l|lllllllll}")
    latex = latex.replace("& Feature", "Feature &")
    if "prda" in dataset:
        latex = latex.replace("Feature", "Model Representation")
    else:
        latex = latex.replace("Feature", "Brain Representation")
    if "model" in dataset and "prda" not in dataset:
        latex = latex.replace("Target", "Code Models")
    else:
        latex = latex.replace("Target", "Code Properties")
    return latex


def make_table(dataset):
    data = load_data(dataset)
    data = update_names(data)
    data = add_baselines(data)
    data = make_pivot_table(data, dataset)
    data = reorder_columns(data, dataset)
    data = format_scores(data, dataset)
    if "ablation" in dataset:
        data = data.loc[:, [col for col in data.columns if "+" in col]]
    latex = format_latex(data.to_latex(), dataset)
    with open(f"../tables/latex/{dataset}.tex", "w") as f:
        f.write(latex)


def make_latex_table(dataset, type):
    dataset = f"{dataset}_{type}stats"
    data = pd.read_csv(f"../stats/raw/{dataset}.csv")
    data["h (corrected)"] = 1 - data["h (corrected)"]
    if "anova" in type:
        priority = ["h (corrected)", "Grouping"]
    else:
        priority = ["h (corrected)", "Grouping", "S1", "S2"]
    data = data.sort_values(by=priority)
    data["h (corrected)"] = 1 - data["h (corrected)"]
    if data.size:
        s = "Brain Region"
        if "crossed" in dataset:
            grouping = "Brain Region"
            if "model" in dataset:
                s = "Code Model"
            else:
                s = "Code Property"
        elif "properties" in dataset:
            grouping = "Code Property"
        elif "model" in dataset:
            grouping = "Code Model"
        data = data.rename(columns={"Grouping": grouping})
        if "anova" not in dataset:
            data = data.rename(columns={"S1": f"{s} A", "S2": f"{s} B"})
        else:
            data = data.rename(columns={"f": "F"})
        data = data.rename(columns={"h (corrected)": "Is Significant?"})
        data = data.set_index(grouping)
        if "anova" in dataset:
            data["F"] = data["F"].apply(lambda x: f"{x:0.2f}")
        else:
            data["t"] = data["t"].apply(lambda x: f"{x:0.2f}")
        data["p"] = data["p"].apply(lambda x: f"{x:0.2e}")
        data["p (corrected)"] = data["p (corrected)"].apply(lambda x: f"{x:0.2e}")
        latex = data.to_latex()
        latex = latex.replace("{llllr}", "{l||lllr}")
        latex = latex.replace("{llllllr}", "{l||ll|lllr}")
        with open(f"../stats/latex/{dataset}.tex", "w") as f:
            f.write(latex)


def main():
    datasets = [
        "mvpa_properties_cls",
        "mvpa_properties_rgr",
        "mvpa_models",
        "mvpa_properties_supplemental",
        "mvpa_properties_cls_ablation",
        "mvpa_properties_rgr_ablation",
        "mvpa_models_ablation",
        "prda_properties",
    ]
    for dataset in datasets:
        try:
            make_table(dataset)
        except:
            print("not making all supplemental tables:", dataset)
    datasets_stats = [
        "mvpa_properties_all_subjects",
        "mvpa_models_subjects",
        "mvpa_models_subjects_crossed",
        "mvpa_properties_all_ablation_subjects",
        "mvpa_models_ablation_subjects",
    ]
    for dataset in datasets_stats:
        try:
            make_latex_table(dataset, "")
            make_latex_table(dataset, "anova_")
        except:
            print("not making all supplemental tables:", dataset)
    make_latex_table("mvpa_properties_rgr_subjects_crossed", "")


if __name__ == "__main__":
    main()
