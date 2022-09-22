import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    names = [
        "Datatype",
        "Conditional",
        "Iteration",
        "Token Count",
        "Node Count",
        "Halstead Difficulty",
        "Cyclomatic Complexity",
        "Runtime Steps",
        "Bytecode Operations",
    ]
    datatype, cond, iter, lines, bytes = [], [], [], [], []
    tokens, nodes, halstead, cyclomatic = [], [], [], []
    for file in Path("../../../../braincode/.cache/profiler").glob("en*.benchmark"):
        with open(file, "r") as f:
            datatype.append(int("math" in file.name))
            cond.append(int("if" in file.name))
            iter.append(int("for" in file.name))
            data = json.loads(f.read())
            tokens.append(data["token_counts"])
            nodes.append(data["ast_node_counts"])
            halstead.append(data["program_difficulty"])
            cyclomatic.append(data["cyclomatic_complexity"])
            lines.append(data["number_of_runtime_steps"])
            bytes.append(data["byte_counts"])
    properties = np.array(
        [datatype, cond, iter, tokens, nodes, halstead, cyclomatic, lines, bytes]
    )
    corrs = np.corrcoef(properties)
    df = pd.DataFrame(corrs, columns=names, index=names)
    map = lambda x: f"{x:0.2f}"
    for i, (idx, row) in enumerate(df.iterrows()):
        row = row.apply(map).values
        row[:i] = "-"
        row = [s.replace("-0.00", "0.00") for s in row]
        df.loc[idx] = row
    latex = df.to_latex().replace("{llllllllll}", "{l|lllllllll}")
    with open(f"code_property_corrs.tex", "w") as f:
        f.write(latex)


if __name__ == "__main__":
    main()
