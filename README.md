# Home Credit Default Risk Challenge: Open Solution

[![Join the chat at https://gitter.im/minerva-ml/open-solution-home-credit](https://badges.gitter.im/minerva-ml/open-solution-home-credit.svg)](https://gitter.im/minerva-ml/open-solution-home-credit?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/minerva-ml/open-solution-home-credit/blob/master/LICENSE)

This is an open solution to the [Home Credit Default Risk challenge](https://www.kaggle.com/c/home-credit-default-risk).

## The purpose of the Open Solution
We are building entirely open solution to this competition. Specifically:
1. Check **live preview of our work** on public projects page: [Home Credit Default Risk](https://app.neptune.ml/neptune-ml/Home-Credit-Default-Risk).
1. Source code and [issues](https://github.com/minerva-ml/open-solution-home-credit/issues) are publicly available.

Rules are simple:
1. Clean code and extensible solution leads to the reproducible experimentations and better control over the improvements.
1. Open solution should establish solid benchmark and give good base for your custom ideas and experiments.

## Disclaimer
In this open source solution you will find references to the neptune.ml. It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :wink:.

## Installation
### Fast Track
1. Clone repository and install requirements (check _requirements.txt_)
1. Register to the [neptune.ml](https://neptune.ml) _(if you wish to use it)_
1. Run experiment based on [LightGBM and random search](https://github.com/minerva-ml/open-solution-home-credit/wiki/LightGBM-and-basic-features):
```bash
neptune run --config neptune_random_search.yaml main.py train_evaluate_predict --pipeline_name lightGBM
```

### Step by step
1. Clone this repository
```bash
git clone https://github.com/minerva-ml/open-solution-home-credit.git
```
2. Install requirements in your Python3 environment
```bash
pip3 install requirements.txt
```
3. Register to the [neptune.ml](https://neptune.ml) _(if you wish to use it)_
4. Update data directories in the [neptune.yaml](https://github.com/minerva-ml/open-solution-home-credit/blob/master/neptune.yaml) configuration file
5. Run experiment based on [LightGBM and random search](https://github.com/minerva-ml/open-solution-home-credit/wiki/LightGBM-and-basic-features):
```bash
neptune login
neptune run --config neptune_random_search.yaml main.py train_evaluate_predict --pipeline_name lightGBM
```
6. collect submit from `experiment_directory` specified in the [neptune.yaml](https://github.com/minerva-ml/open-solution-home-credit/blob/master/neptune.yaml)

## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/minerva-ml/open-solution-home-credit/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in paticular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/home-credit-default-risk/discussion/57175) is our primary way of communication.
1. Read project's [Wiki](https://github.com/minerva-ml/open-solution-home-credit/wiki), where we publish descriptions about the code, pipelines and supporting tools such as [neptune.ml](https://neptune.ml).
1. Submit an [issue]((https://github.com/minerva-ml/open-solution-home-credit/issues)) directly in this repo.
