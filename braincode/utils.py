import json
import os
import sys

import numpy as np
import tqdm

from braincode.benchmarks import ProgramMetrics


def populate_benchmarks(basepath):
    # This method is invoked from a standalone script to
    # run the profiler on all input programs, and cache all results

    def _prepare_src_for_profiler(src):
        indent = "  "
        src = src.replace("\n", "\n" + indent)
        src = indent + src
        src = "@profile\ndef profile_me():\n" + src + "\nprofile_me()"
        return src

    basepath = sys.argv[1]
    inpath = os.path.join(basepath, "inputs", "python_programs")
    outpath = os.path.join(basepath, ".cache", "profiler")
    os.makedirs(outpath, exist_ok=True)

    # For every program in the dataset
    cnt = 0
    for ff in os.listdir(inpath):
        print(ff)
        if ff in ["en", "jap"]:
            for f in tqdm.tqdm(os.listdir(os.path.join(inpath, ff))):
                if ".py" not in f:
                    continue

                with open(os.path.join(inpath, ff, f), "r") as fp:
                    src = fp.read()
                fname = ff + "_" + f
                # Prepare a copy of the src for the profilers
                if not os.path.exists(os.path.join(outpath, fname)):
                    src_profiler = _prepare_src_for_profiler(src)
                    with open(os.path.join(outpath, fname), "w") as fp:
                        fp.write(src_profiler)

                # Run all the profilers on the input program, and save their results as a json
                all_metrics = {}

                metrics = ProgramMetrics(src, fname, basepath)
                all_metrics[
                    "number_of_runtime_steps"
                ] = metrics.get_number_of_runtime_steps()
                all_metrics["ast_node_counts"] = metrics.get_ast_node_counts()
                all_metrics["token_counts"] = metrics.get_token_counts()
                all_metrics[
                    "program_difficulty"
                ] = metrics.get_halstead_complexity_metrics()["program_difficulty"]
                all_metrics[
                    "cyclomatic_complexity"
                ] = metrics.get_halstead_complexity_metrics()["cyclomatic_complexity"]
                all_metrics["byte_counts"] = metrics.get_byte_counts()

                with open(os.path.join(outpath, fname + ".benchmark"), "w") as fp:
                    json.dump(all_metrics, fp)

                cnt += 1

    print(f"Done populating benchmark metrics for {cnt} input files")


def clean_cache(base_pth, choice):
    def _clean_cache(choice):
        if choice == 1:
            folder_name = "scores"
        elif choice == 2:
            folder_name = "representations"
        elif choice == 3:
            folder_name = "profiler"

        pth = os.path.join(base_pth, ".cache", folder_name)
        print(f"Clear path? {pth}")
        inp = input()
        if "y" in inp.lower() or "1" in inp.lower():
            for dirname, _, filenames in os.walk(pth):
                for f in filenames:
                    if folder_name == "scores":
                        condition = "score" in f or ".npy" in f
                    elif folder_name == "representations":
                        condition = "pkl" in f
                    elif folder_name == "profiler":
                        condition = ".lprof" in f or ".py" in f

                    if condition:
                        print(f"Clearing {os.path.join(pth, dirname, f)}")
                        os.remove(os.path.join(pth, dirname, f))

    # clear scores
    if choice == 0:
        for i in [1, 2, 3]:
            _clean_cache(i)
    else:
        _clean_cache(choice)


def print_scores(base_pth):
    pth = os.path.join(base_pth, ".cache", "scores")
    scores = {}
    folders = os.listdir(pth)
    for ff in folders:
        files = os.listdir(os.path.join(pth, ff))
        for f in files:
            if "score" in f and ".npy" in f:
                names = "".join(f.split(".npy")[:-1]).split("_")
                if len(names) == 3:
                    feature = names[1]
                    target = names[2]
                    scores_np = np.load(os.path.join(pth, ff, f))
                    if feature not in scores:
                        scores[feature] = {}
                    scores[feature][target] = scores_np.item()

    print(f"Model-wise scores: \n{json.dumps(scores, indent=2)}")


if __name__ == "__main__":
    pth = sys.argv[1]

    # Choices:
    # 1. Clean cache files
    #    argv[3] choices --
    #    0. Apply all the three choices below
    #    1. Clear scores files
    #    2. Clear representations files
    #    3. Clear profiler files
    # 2: Populate benchmark metrics
    # 3: Pretty print processed scores

    choice = int(sys.argv[2])

    if choice == 1:
        subchoice = int(sys.argv[3])
        clean_cache(pth, subchoice)
    elif choice == 2:
        populate_benchmarks(pth)
    elif choice == 3:
        print_scores(pth)
