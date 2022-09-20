import itertools

import numpy as np
import pandas as pd
import scipy.stats as st


def main():
    networks = ["MD", "lang", "vis", "aud"]
    properties = ["lines", "tokens"]
    rows = []
    for n, p in itertools.product(networks, properties):
        score = np.load(
            f"../../../../braincode/.cache/scores/mvpa/score_{n}_{p}_rmse.npy"
        )
        null = np.load(
            f"../../../../braincode/.cache/scores/mvpa/null_{n}_{p}_rmse.npy"
        )
        rows.append([n, p, float(score), float(null.mean()), float(null.std())])
    df = pd.DataFrame(
        rows,
        columns=[
            "Brain Network",
            "Code Property",
            "RMSE Score",
            "Null RMSE Mean",
            "Null RMSE SD",
        ],
    )
    df["z"] = (df["RMSE Score"] - df["Null RMSE Mean"]) / df["Null RMSE SD"]
    df["p (corrected)"] = st.norm.sf(-df["z"]) * df.shape[0]
    df["significant (1/0)"] = df["p (corrected)"] < 0.05
    df.loc[df["Brain Network"] == "MD", "Brain Network"] = "MD"
    df.loc[df["Brain Network"] == "lang", "Brain Network"] = "Language"
    df.loc[df["Brain Network"] == "vis", "Brain Network"] = "Visual"
    df.loc[df["Brain Network"] == "aud", "Brain Network"] = "Auditory"
    df.loc[df["Code Property"] == "lines", "Code Property"] = "Dynamic Analysis"
    df.loc[df["Code Property"] == "tokens", "Code Property"] = "Static Analysis"
    df.to_csv("rmse.csv", index=False)
    rmse = []
    null_rmse = []
    for i, row in df.iterrows():
        rmse.append(f"{row['RMSE Score']:.2f}")
        null_rmse.append(f"{row['Null RMSE Mean']:.2f} ± {row['Null RMSE SD']:.2f}")
    df["RMSE"] = rmse
    df["Null RMSE"] = null_rmse
    df["Is Significant?"] = df["significant (1/0)"].astype(int)
    df = df.iloc[:, [0, 1, 8, 9, 10]]
    latex = df.to_latex(index=False)
    latex = latex.replace("{llllr}", "{ll||llr}")
    print(latex)


if __name__ == "__main__":
    main()
