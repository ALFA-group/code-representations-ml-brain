SHELL := /usr/bin/env bash
EXEC = python
PACKAGE = braincode
INSTALL = pip install -e .
ACTIVATE = source activate $(PACKAGE)
.DEFAULT_GOAL := help

## help      : print available build commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## update    : update repo with latest version from GitHub.
.PHONY : update
update :
	@git pull origin NeurIPS2022

## env       : setup environment and install dependencies.
.PHONY : env
env : module seq2seq
module: conda $(PACKAGE).egg-info/
seq2seq: conda setup/code_seq2seq.egg-info
$(PACKAGE).egg-info/ : setup.py requirements.txt
	@$(ACTIVATE) ; $(INSTALL)
setup/code_seq2seq.egg-info : setup/setup.py
	@$(ACTIVATE) ; cd $(<D) ; $(INSTALL)
conda :
ifeq "$(shell conda info --envs | grep $(PACKAGE) | wc -l)" "0"
	@conda create -yn $(PACKAGE) $(EXEC)=3.7
endif

## setup     : download prerequisite files, e.g. neural data, models.
.PHONY : setup
setup : inputs benchmarks
inputs : env $(PACKAGE)/inputs/
benchmarks : env $(PACKAGE)/.cache/profiler/
$(PACKAGE)/inputs/ : setup/setup.sh
	@$(ACTIVATE) ; cd $(<D) ; bash $(<F)
$(PACKAGE)/.cache/profiler/ : $(PACKAGE)/utils.py
	@$(ACTIVATE) ; $(EXEC) -m $(PACKAGE).utils $(PACKAGE) 2

## test      : run testing pipeline.
.PHONY : test
test : pylint mypy
pylint : env html/pylint/index.html
mypy : env html/mypy/index.html
html/pylint/index.html : html/pylint/index.json
	@$(ACTIVATE) ; pylint-json2html -o $@ -e utf-8 $<
html/pylint/index.json : $(PACKAGE)/*.py
	@mkdir -p $(@D)
	@$(ACTIVATE) ; pylint $(PACKAGE) --output-format=colorized,json:$@ || pylint-exit $$?
html/mypy/index.html : $(PACKAGE)/*.py
	@$(ACTIVATE) ; mypy --ignore-missing-import -p $(PACKAGE) --html-report $(@D)

## docker    : build docker image and spin up container.
.PHONY : docker
docker :
ifeq "$(shell docker images | grep $(PACKAGE) | wc -l)" "0"
	@docker build -t $(PACKAGE)
endif
	@docker run -it $(PACKAGE)

## analysis  : run core analyses to replicate paper.
.PHONY : analysis
analysis : setup $(PACKAGE)/outputs/
$(PACKAGE)/outputs/ : $(PACKAGE)/*.py
	@$(ACTIVATE) ; $(EXEC) $(PACKAGE) mvpa
	@$(ACTIVATE) ; $(EXEC) $(PACKAGE) prda

## paper     : run scripts to generate final plots and tables.
.PHONY : paper
paper : analysis paper/plots/
paper/plots/ : paper/scripts/*.py
	@$(ACTIVATE) ; cd $(<D) ; bash run.sh
