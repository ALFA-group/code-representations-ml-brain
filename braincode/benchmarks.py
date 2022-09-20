import ast
import dis
import json
import os
import pickle as pkl
import subprocess
import typing
from io import BytesIO
from pathlib import Path
from tokenize import tok_name, tokenize

import numpy as np


class ProgramBenchmark:
    def __init__(
        self, benchmark: str, basepath: Path, fnames: typing.List[str]
    ) -> None:
        self._benchmark = benchmark
        self._base_path = basepath
        self._fnames = fnames
        self._metrics = self.load_all_benchmarks(self._base_path)

    @staticmethod
    def load_all_benchmarks(basepath: Path) -> dict:
        # Populate all benchmarks needs to be setup before calling this method
        metrics = {}
        inpath = os.path.join(basepath, ".cache", "profiler")
        for f in os.listdir(inpath):
            if ".benchmark" in f:
                with open(os.path.join(inpath, f), "r") as fp:
                    metrics[f.split(".")[0]] = json.load(fp)
        return metrics

    def fit_transform(self, _) -> np.ndarray:
        # Pre-requisite -- the programs list and self._fnames list are sorted in the same order
        # Using results stored in cache instead of processing each program again.
        # Hence, the input `programs` to this function is unused.
        outputs = []
        for f in self._fnames:
            f = "_".join(
                f.split(os.sep)[-2:]
            )  # retain only `en/filename` and rename to `en_filename`
            f = str(f).split(".")[0]  # remove .py
            if self._benchmark == "task-lines":
                metric = self._metrics[f]["number_of_runtime_steps"]
            elif self._benchmark == "task-bytes":
                metric = self._metrics[f]["byte_counts"]
            elif self._benchmark == "task-nodes":
                metric = self._metrics[f]["ast_node_counts"]
            elif self._benchmark == "task-tokens":
                metric = self._metrics[f]["token_counts"]
            elif self._benchmark == "task-halstead":
                metric = self._metrics[f]["program_difficulty"]
            elif self._benchmark == "task-cyclomatic":
                metric = self._metrics[f]["cyclomatic_complexity"]
            elif self._benchmark == "task-bytes":
                metric = self._metrics[f]["byte_counts"]
            else:
                raise ValueError(
                    "Undefined program metric. Make sure to use valid identifier."
                )
            outputs.append(metric)
        return np.array(outputs).reshape([-1, 1])


class ProgramMetrics:
    def __init__(self, program: str, path: str, base_path: Path) -> None:
        self.program = program
        self.fname = "_".join(path.split(os.sep)[-2:])
        self.base_path = base_path
        self.outpath = os.path.join(self.base_path, ".cache", "profiler")

    def get_token_counts(self) -> int:
        exclude_tokens_types = [
            "NEWLINE",
            "NL",
            "INDENT",
            "DEDENT",
            "ENDMARKER",
            "ENCODING",
        ]
        exclude_ops = ["[", "]", "(", ")", ",", ":"]
        token_count = 0
        for res in tokenize(BytesIO(self.program.encode("utf-8")).readline):
            if res and tok_name[res.type] not in exclude_tokens_types:
                if (
                    tok_name[res.type] == "OP" and res.string not in exclude_ops
                ) or tok_name[res.type] != "OP":
                    token_count += 1
        return token_count

    def get_ast_node_counts(self) -> int:
        root = ast.parse(self.program)
        ast_node_count = 0
        for _ in ast.walk(root):
            ast_node_count += 1
        return ast_node_count

    def get_halstead_complexity_metrics(self, sec=30):
        # See https://radon.readthedocs.io/en/latest/intro.html for details on Halstead metrics
        # reported by Radon

        local_fname = os.path.join(self.outpath, self.fname)
        cmd = ["radon", "hal", local_fname, "-j"]
        output = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=sec
        )
        out = output.stdout.decode("utf-8")
        out = json.loads(out)
        if not out:
            err = output.stderr.decode("utf-8")
            print(err)

        metrics = {}
        metrics["number_of_distinct_operators"] = out[local_fname]["total"][0]
        metrics["number_of_distinct_operands"] = out[local_fname]["total"][1]
        metrics["number_of_operators"] = out[local_fname]["total"][2]
        metrics["number_of_operands"] = out[local_fname]["total"][3]
        metrics["program_length"] = out[local_fname]["total"][6]
        metrics["program_difficulty"] = out[local_fname]["total"][7]
        metrics["program_effort"] = out[local_fname]["total"][8]

        cmd = ["radon", "cc", local_fname, "-j"]
        output = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=sec
        )
        out = output.stdout.decode("utf-8")
        out = json.loads(out)
        if not out:
            err = output.stderr.decode("utf-8")
            print(err)

        metrics["cyclomatic_complexity"] = out[local_fname][0]["complexity"]
        # print(json.dumps(metrics, indent=2))
        return metrics

    def get_number_of_runtime_steps(self, sec: int = 30) -> int:
        """
        Requires the package line_profiler to be installed.
        Picks up the # hits for every line from the output of this profiler.
        See https://github.com/rkern/line_profiler

        :param sec: Timeout for subprocess.run
        :return:[# of lines] executed
        """
        if self.fname[-3:] != ".py":
            raise ValueError("Unrecognized file type")

        if not os.path.exists(os.path.join(self.outpath, self.fname + ".lprof")):
            cmd = [
                "kernprof",
                "-o",
                os.path.join(self.outpath, self.fname + ".lprof"),
                "-l",
                os.path.join(self.outpath, self.fname),
            ]
            output = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=sec
            )
            out = output.stdout.decode("utf-8")
            if not out:
                err = output.stderr.decode("utf-8")
                print(err)

        sum_hits = 0
        with open(os.path.join(self.outpath, self.fname + ".lprof"), "rb") as fp:
            obj = pkl.load(fp)
            if len(obj.timings) > 1:
                raise ValueError(
                    "The number of timings cannot be greater than 1 in lprof dump"
                )
            # obj.timings format - {filename: [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]}
            # x1 - line number, y1 - hits, z1 - time spent on the line
            for v in obj.timings.values():
                for i in v:
                    # index 1 contains number of hits.
                    sum_hits += i[1]
        return sum_hits

    def get_byte_counts(self) -> int:
        with open(os.path.join(self.outpath, self.fname)) as fp:
            src = fp.read()
        byte_code = compile(src, os.path.join(self.outpath, self.fname), "exec")
        bc = dis.Bytecode(byte_code)
        bc_profile_me = None
        for b in bc:
            if b.argval.__class__.__name__ == "code":
                bc_profile_me = dis.Bytecode(b.argval)

        if bc_profile_me is None:
            raise ValueError("Disassembler did not find profile_me() method")

        num_bytes = 0
        for _ in bc_profile_me:
            num_bytes += 1

        return num_bytes
