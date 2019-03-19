# Home Credit Default Risk: Open Solution
[![Join the chat at https://gitter.im/minerva-ml/open-solution-home-credit](https://badges.gitter.im/minerva-ml/open-solution-home-credit.svg)](https://gitter.im/minerva-ml/open-solution-home-credit?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/minerva-ml/open-solution-home-credit/blob/master/LICENSE)

This is an open solution to the [Home Credit Default Risk challenge](https://www.kaggle.com/c/home-credit-default-risk) :house_with_garden:.

## More competitions :sparkler:
Check collection of [public projects :gift:](https://app.neptune.ml/-/explore), where you can find multiple Kaggle competitions with code, experiments and outputs.

## Our goals
We are building entirely open solution to this competition. Specifically:
1. **Learning from the process** - updates about new ideas, code and experiments is the best way to learn data science. Our activity is especially useful for people who wants to enter the competition, but lack appropriate experience.
1. Encourage more Kagglers to start working on this competition.
1. Deliver open source solution with no strings attached. Code is available on our [GitHub repository :computer:](https://github.com/neptune-ml/open-solution-home-credit). This solution should establish solid benchmark, as well as provide good base for your custom ideas and experiments. We care about clean code :smiley:
1. We are opening our experiments as well: everybody can have **live preview** on our experiments, parameters, code, etc. Check: [Home Credit Default Risk :chart_with_upwards_trend:](https://app.neptune.ml/neptune-ml/Home-Credit-Default-Risk) and screens below.

| Train and validation results on folds :bar_chart: | LightGBM learning curves :bar_chart: |
|:---|:---|
|[![train-validation-results-on-folds](https://gist.githubusercontent.com/kamil-kaczmarek/b3b939797fb39752c45fdadfedba3ed9/raw/fbc925f683853fa8af5a95426fcd37fcb3afcf38/hc-1.png)](https://app.neptune.ml/-/dashboard/experiment/bb8d7113-d944-4718-87b5-6aeca4ff85c5)|[![LightGBM-learning-curves](https://gist.githubusercontent.com/kamil-kaczmarek/b3b939797fb39752c45fdadfedba3ed9/raw/fbc925f683853fa8af5a95426fcd37fcb3afcf38/hc-2.png)](https://app.neptune.ml/-/dashboard/experiment/bb8d7113-d944-4718-87b5-6aeca4ff85c5)|

## Disclaimer
In this open source solution you will find references to the [neptune.ml](http://bit.ly/2TkoL9G). It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :snake:.

# How to start?
## Learn about our solutions
1. Check [Kaggle forum](https://www.kaggle.com/c/home-credit-default-risk/discussion/57175) and participate in the discussions.
1. Check our [Wiki pages :house_with_garden:](https://github.com/neptune-ml/open-solution-home-credit/wiki), where we document our work. See solutions below:

| link to code | name | CV | LB | link to description |
|:---:|:---:|:---:|:---:|:---:|
|[solution 1](https://github.com/neptune-ml/open-solution-home-credit/tree/solution-1)|*chestnut* :chestnut:|?|0.742|[LightGBM and basic features](https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-and-basic-features)|
|[solution 2](https://github.com/neptune-ml/open-solution-home-credit/tree/solution-2)|*seedling* :seedling:|?|0.747|[Sklearn and XGBoost algorithms and groupby features](https://github.com/neptune-ml/open-solution-home-credit/wiki/Sklearn-and-XGBoost-algorithms-and-groupby-features)|
|[solution 3](https://github.com/neptune-ml/open-solution-home-credit/tree/solution-3)|*blossom* :blossom:|0.7840|0.790|[LightGBM on selected features](https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-on-selected-features)|
|[solution 4](https://github.com/neptune-ml/open-solution-home-credit/tree/solution-4)|*tulip* :tulip: |0.7905|0.801|[LightGBM with smarter features](https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-with-smarter-features)|
|[solution 5](https://github.com/neptune-ml/open-solution-home-credit/tree/solution-5)|*sunflower* :sunflower:|0.7950|0.804|[LightGBM clean dynamic features](https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-clean-dynamic-features)|
|[solution 6](https://github.com/neptune-ml/open-solution-home-credit/tree/solution-6)|*four leaf clover* :four_leaf_clover:|0.7975|0.806| **priv. LB 0.79804**, [Stacking by feature diversity and model diversity](https://github.com/neptune-ml/open-solution-home-credit/wiki/Stacking-by-feature-diversity-and-model-diversity)|

## Start experimenting with ready-to-use code
You can jump start your participation in the competition by using our starter pack. Installation instruction below will guide you through the setup.

### Installation *(fast track)*
1. Clone repository and **install requirements** (*use Python3.5*)

```bash
pip3 install -r requirements.txt
```

2. Register to the [neptune.ml](http://bit.ly/2TkoL9G) _(if you wish to use it)_
3. Run experiment based on [LightGBM](https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-with-smarter-features):

:trident:
```bash
neptune account login
neptune run --config configs/neptune.yaml main.py train_evaluate_predict_cv --pipeline_name lightGBM
```

:snake:
```bash
python main.py -- train_evaluate_predict_cv --pipeline_name lightGBM
```

### Installation *(step by step)*
[Step by step installation :desktop_computer:](https://github.com/neptune-ml/open-solution-home-credit/wiki/Step-by-step-installation)

### Hyperparameter Tuning
Various options of hyperparameter tuning are available

1. Random Search

    `configs/neptune.yaml`
    ```yaml
      hyperparameter_search__method: random
      hyperparameter_search__runs: 100
    ```
    
    `src/pipeline_config.py`
    ```python
        'tuner': {'light_gbm': {'max_depth': ([2, 4, 6], "list"),
                                'num_leaves': ([2, 100], "choice"),
                                'min_child_samples': ([5, 10, 15 25, 50], "list"),
                                'subsample': ([0.95, 1.0], "uniform"),
                                'colsample_bytree': ([0.3, 1.0], "uniform"),
                                'min_gain_to_split': ([0.0, 1.0], "uniform"),
                                'reg_lambda': ([1e-8, 1000.0], "log-uniform"),
                                },
                  }
    ```

## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/minerva-ml/open-solution-home-credit/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in paticular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/home-credit-default-risk/discussion/57175) is our primary way of communication.
1. Read project's [Wiki](https://github.com/minerva-ml/open-solution-home-credit/wiki), where we publish descriptions about the code, pipelines and supporting tools such as [neptune.ml](http://bit.ly/2TkoL9G).
1. Submit an [issue]((https://github.com/minerva-ml/open-solution-home-credit/issues)) directly in this repo.
