# Convergent Representations of Computer Programs in Human and Artificial Neural Networks

Resources for the paper `Convergent Representations of Computer Programs in Human and Artificial Neural Networks` by Shashank Srikant*, Benjamin Lipkin*, Anna A. Ivanova, Evelina Fedorenko, Una-May O'Reilly.

Published in NeurIPS 2022: [Link]

The labs involved:

https://evlab.mit.edu/

https://alfagroup.csail.mit.edu/

For additional information, contact shash@mit.edu, lipkinb@mit.edu, or unamay@csail.mit.edu, evelina9@mit.edu.

## Details

This pipeline supports several major functions.

-   **MVPA** (multivariate pattern analysis) evaluates decoding of **code properties** or **code model** representations from their respective **brain representations** within a collection of canonical **brain regions**.
-   **PRDA** (program representation decoding analysis) evaluates decoding of **code properties** from **code model** representations.

To run all core experiments from the paper, the following command will suffice after setup:

```bash
python braincode mvpa # runs all core MVPA analyses in parallel
python braincoda prda # runs all supplemental PRDA analyses in parallel
```

To regenerate tables and figures from the paper, run the following after completing the analyses:

```bash
cd paper/scripts
source run.sh # pulls scores, runs stats, generates plots and tables
```

### Supported Brain Regions

-   `brain-MD` (Multiple Demand)
-   `brain-lang` (Language)
-   `brain-vis` (Visual)
-   `brain-aud` (Auditory)

### Supported Code Features

**Code Properties**

-   `test-code` (code vs. sentences)
-   `test-lang` (english vs. japanese)
-   `task-content` (math vs. str) <sup>\*datatype</sup>
-   `task-structure` (seq vs. for vs. if) <sup>\*control flow</sup>
-   `task-tokens` (# of tokens in program) <sup>\*static analysis</sup>
-   `task-lines` (# of runtime steps during execution) <sup>\*dynamic analysis</sup>
-   `task-bytes` (# of bytecode ops executed)
-   `task-nodes` (# of nodes in AST)
-   `task-halstead` (function of tokens, operations, vocabulary)
-   `task-cyclomatic` (function of program control flow graph)

**Code Models**

-   `code-projection` (presence of tokens)
-   `code-bow` (token frequency)
-   `code-tfidf` (token and document frequency)
-   `code-seq2seq`<sup> [1](https://github.com/IBM/pytorch-seq2seq)</sup> (sequence modeling)
-   `code-xlnet`<sup> [2](https://arxiv.org/pdf/1906.08237.pdf)</sup> (autoregressive LM)
-   `code-gpt2`<sup> [4](https://huggingface.co/microsoft/CodeGPT-small-py)</sup> (autoregressive LM)
-   `code-bert`<sup> [5](https://arxiv.org/pdf/2002.08155.pdf)</sup> (masked LM)
-   `code-roberta`<sup> [6](https://huggingface.co/huggingface/CodeBERTa-small-v1)</sup> (masked LM)
-   `code-transformer`<sup> [3](https://arxiv.org/pdf/2103.11318.pdf)</sup> (LM + structure learning)

## Installation

Requirements: [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n braincode python=3.7
source activate braincode
git clone --branch main --depth 1 https://github.com/ALFA-group/code-representations-ml-brain
cd braincode
pip install . # -e for development mode
cd setup
source setup.sh # downloads 'large' files, e.g. datasets, models
```

## Run

```bash
usage: braincode [-h] [-f FEATURE] [-t TARGET] [-m METRIC] [-d CODE_MODEL_DIM]
                 [-p BASE_PATH] [-s] [-b]
                 {mvpa,prda}

run specified analysis type

positional arguments:
  {mvpa,prda}

optional arguments:
  -h, --help            show this help message and exit
  -f FEATURE, --feature FEATURE
  -t TARGET, --target TARGET
  -m METRIC, --metric METRIC
  -d CODE_MODEL_DIM, --code_model_dim CODE_MODEL_DIM
  -p BASE_PATH, --base_path BASE_PATH
  -s, --score_only
  -b, --debug
```

_Note: BASE_PATH must be specified to match setup.sh if changed from default._

**Sample calls**

```bash
# basic examples
python braincode mvpa -f brain-MD -t task-structure # brain -> {task, model}
python braincode prda -f code-bert -t task-tokens # model -> task

# more complex example
python braincode mvpa -f brain-lang+brain-MD -t code-projection -d 64 -m SpearmanRho -p $BASE_PATH --score_only
# note how `+` operator can be used to join multiple representations via concatenation
# additional metrics are available in the `metrics.py` module
```

## Reproducing results from our paper
xx

## Automation

### Make

This package also provides an automated build using [GNU Make](https://www.gnu.org/software/make/manual/make.html). A single pipeline is provided, which starts from an empty environment, and provides ready to use software.

```bash
make setup # see 'make help' for more info
```

### Docker

Build automation can also be containerized in [Docker](https://hub.docker.com/)

```bash
make docker
```

## Citation

If you use this work, please cite
```
[Citation]
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
