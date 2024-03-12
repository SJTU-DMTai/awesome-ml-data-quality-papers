# Awesome ML Data Quality Papers

This is a list of papers about training data quality management for ML models. 

## Introduction

Data scientists spend ∼80% time on data preparation for an ML pipeline since the data quality issues are unknown beforehand thereby leading to iterative debugging. A good Data Quality Management System for ML (**DQMS for ML**) helps data scientists break free from the arduous process of data selection and debugging, particularly in the current era of big data and large models. Automating the management of training data quality effectively is crucial for improving the efficiency and quality of ML pipelines.

With the emergence and development of "**Data-Centric AI**", there has been increasing research focus on optimizing the quality of training data rather than solely concentrating on model structures and training strategies. This is the motivation behind maintaining this repository.

Before we proceed, let's define **data quality for ML**. In contrast to traditional data cleaning, training data quality for ML refers to the impact of individual or groups of data samples on the behavior of ML models for a given task. It's important to note that the behavior of the model we are concerned with goes beyond performance metrics like accuracy, recall, AUC, MSE, etc. We also consider more generalizable metrics such as model fairness, robustness, and so on.

Considering the following pipeline, DQMS acts as a **middleware** between data, ML model, and user, necessitating interactions with each of them.

<div align=center>
<img src="./framework.png" width = "70%" />
</div>

A DQMS for ML typically consists of three components: **Data Selector**, **Data Attributer**, and **Data Profiler**. To achieve a well-performing ML model, multiple rounds of training are often required. In this process, the DQMS needs to iteratively adjust the training data based on the results of each round of model training. The workflow of DQMS in one round of training is as follows: (a) Data Selector first acquires the training dataset from a data source and trains the ML model with it. (b) After training for one round (several epochs), Data Attributer absorbs feedback from the model and user's task requirements and computes the data quality assessment. (c) Data Profiler then provides a user-friendly summary of the training data. (d) Meanwhile, Data Selector utilizes the data quality assessment as feedback to acquire higher-quality training data, thus initiating a new iteration.

## Paper List

We collect the recent influential papers about DQMS for ML and annotate the relevant DQMS components involved in these papers, where `DS` = Data Selector, `DA` = Data Attributer, and `DP` = Data Profiler.

### 2024

- [ICLR 24] "What Data Benefits My Classifier?" Enhancing Model Performance and Interpretability through Influence-Based Data Selection [[paper](https://openreview.net/pdf?id=HE9eUQlAvo)] [[code](https://github.com/anshuman23/InfDataSel)] `DS` `DA`
- [ICLR 24] Canonpipe: Data Debugging with Shapley Importance over Machine Learning Pipelines [[paper](https://openreview.net/pdf?id=qxGXjWxabq)] [[code](https://github.com/easeml/datascope)] `DS` `DA`

### 2023

- [NIPS 23] Data Selection for Language Models via Importance Resampling [[paper](https://openreview.net/pdf?id=uPSQv0leAu)] [[code](https://github.com/p-lambda/dsir)] `DS`
- [NIPS 23] Sample based Explanations via Generalized Representers [[paper](https://openreview.net/pdf?id=fX64q0SNfL)] `DA`
- [NIPS 23] Model Shapley: Equitable Model Valuation with Black-box Access[[paper](https://openreview.net/pdf?id=Y6IGTNMdLT)] [[code](https://github.com/XinyiYS/ModelShapley)] `DA`
- [NIPS 23] Threshold KNN-Shapley: A Linear-Time and Privacy-Friendly Approach to Data Valuation [[paper](https://openreview.net/pdf?id=FAZ3i0hvm0)] `DA`
- [NIPS 23] GEX: A flexible method for approximating influence via Geometric Ensemble [[paper](https://openreview.net/attachment?id=tz4ECtAu8e&name=pdf)] [[code](https://github.com/sungyubkim/gex)] `DA`
- [ICML 23] Discover and Cure: Concept-aware Mitigation of Spurious Correlation [[paper](https://openreview.net/pdf?id=QDxtrlPmfB)] [[code](https://github.com/Wuyxin/DISC)] `DS` `DA`
- [ICML 23] Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value [[paper](https://proceedings.mlr.press/v202/kwon23e/kwon23e.pdf)] [[code](https://github.com/ykwon0407/dataoob)] `DA`
- [ICLR 23] Data Valuation Without Training of a Model [[paper](https://openreview.net/pdf?id=XIzO8zr-WbM)] [[code](https://github.com/JJchy/CG_score)] `DA`
- [ICLR 23] Distilling Model Failures as Directions in Latent Space [[paper](https://openreview.net/pdf?id=99RpBVpLiX)] [[code](https://github.com/MadryLab/failure-directions)] `DS` `DP`
- [ICLR 23] LAVA: Data Valuation without Pre-Specified Learning Algorithms [[paper](https://openreview.net/pdf?id=JJuP86nBl4q)] [code](https://github.com/ruoxi-jia-group/LAVA)] `DA`
- [VLDB 23] Equitable Data Valuation Meets the Right to Be Forgotten in Model Markets [[paper](https://www.vldb.org/pvldb/vol16/p3349-liu.pdf)] [[code](https://github.com/ZJU-DIVER/ValuationMeetsRTBF)] `DA`
- [VLDB 23] Computing Rule-Based Explanations by Leveraging Counterfactuals [[paper](https://www.vldb.org/pvldb/vol16/p420-geng.pdf)] [[code](https://github.com/GibbsG/GeneticCF)] `DP`
- [AISTATS 23] Data Banzhaf: A Robust Data Valuation Framework for Machine Learning [[paper](https://proceedings.mlr.press/v206/wang23e/wang23e.pdf)] `DA`

### 2022

- [NIPS 22] CS-SHAPLEY: Class-wise Shapley Values for Data Valuation in Classification [[paper](https://openreview.net/pdf?id=KTOcrOR5mQ9)] [[code](https://github.com/stephanieschoch/cs-shapley)] `DA`
- [ICML 22] Measuring the Effect of Training Data on Deep Learning Predictions via Randomized Experiments [[paper](https://proceedings.mlr.press/v162/lin22h/lin22h.pdf)] `DA`
- [ICML 22] Meaningfully Debugging Model Mistakes using Conceptual Counterfactual Explanations [[papeer](https://proceedings.mlr.press/v162/abid22a/abid22a.pdf)] [[code](https://github.com/mertyg/debug-mistakes-cce)] `DS` `DP`
- [ICML 22] Datamodels: Predicting Predictions from Training Data [[paper](https://proceedings.mlr.press/v162/ilyas22a/ilyas22a.pdf)] [[code](https://github.com/MadryLab/datamodels-data)] `DA`
- [ICLR 22] Domino: Discovering systematic errors with cross-modal embeddings [[paper](https://openreview.net/pdf?id=FPCMqjI0jXN)] [[code](https://github.com/HazyResearch/domino)] `DA` `DP`
- [SIGMOD 22] Complaint-Driven Training Data Debugging at Interactive Speeds [[paper](https://dl.acm.org/doi/pdf/10.1145/3514221.3517849)] `DA`
- [SIGMOD 22] Interpretable Data-Based Explanations for Fairness Debugging [[paper](https://arxiv.org/pdf/2112.09745.pdf)] [[video](https://www.youtube.com/watch?v=bt_VL1eSu30)] `DA` `DP`
- [AAAI 22] Scaling Up Influence Functions [[paper](https://arxiv.org/pdf/2112.03052.pdf)] [[code](https://github.com/google-research/jax-influence)] `DA`
- [AISTATS 22] Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning [[paper](https://proceedings.mlr.press/v151/kwon22a/kwon22a.pdf)] [code](https://github.com/ykwon0407/beta_shapley)] `DA`

### 2021 and before

- [NIPS 21] Explaining Latent Representations with a Corpus of Examples [[paper](https://proceedings.neurips.cc/paper/2021/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf)] [[code](https://github.com/JonathanCrabbe/Simplex)] `DA`
- [CHI 21] Data-Centric Explanations: Explaining Training Data of Machine Learning Systems to Promote Transparency [[paper](https://dl.acm.org/doi/10.1145/3411764.3445736)] `DP`
- [NIPS 20] Multi-Stage Influence Function [[paper](https://proceedings.neurips.cc/paper/2020/file/95e62984b87e90645a5cf77037395959-Paper.pdf)] `DA`
- [ICML 20]  On second-order group influence functions for black-box predictions [[paper](https://proceedings.mlr.press/v119/basu20b/basu20b.pdf)] `DA`
- [SIGMOD 20] Complaint Driven Training Data Debugging for Query 2.0 [[paper](https://arxiv.org/pdf/2004.05722.pdf)] [[video](https://www.youtube.com/watch?v=qvgBmM1LP38)] `DA`
- [ICML 19] Data Shapley: Equitable Valuation of Data for Machine Learning [[paper](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf)] [[code](https://github.com/amiratag/DataShapley)] `DA`
- [ICML 17] Understanding Black-box Predictions via Influence Functions [[paper](https://arxiv.org/pdf/1703.04730.pdf)] [[code](https://github.com/kohpangwei/influence-release)] `DA`

### Survey Papers

- [Nature Machine Intelligence 22] Advances, challenges and opportunities in creating data for trustworthy AI [[paper](https://www.nature.com/articles/s42256-022-00516-1)] `DS` `DA`
- [arXiv 23] Data-centric Artificial Intelligence: A Survey [[paper](https://arxiv.org/pdf/2303.10158.pdf)] `DS` `DA` `DP`
- [arXiv 23] Data Management For Large Language Models: A Survey [[paper](https://arxiv.org/pdf/2312.01700.pdf)] [[code](https://github.com/ZigeW/data_management_LLM)] `DS` `DA`
- [arXiv 23] Training Data Influence Analysis and Estimation: A Survey [[paper](https://arxiv.org/pdf/2212.04612.pdf)] [[code](https://github.com/ZaydH/influence_analysis_papers)] `DA`
- [TKDE 22] Data Management for Machine Learning: A Survey [[paper](https://luoyuyu.vip/files/DM4ML_Survey.pdf)] `DS` `DA`
- [IJCAI 21] Data Valuation in Machine Learning: "Ingredients", Strategies, and Open Challenges [[paper](https://www.ijcai.org/proceedings/2022/0782.pdf)] `DA`

### Benchmarks

- [NIPS 23] DataPerf: Benchmarks for Data-Centric AI Development [[paper](https://openreview.net/pdf?id=LaFKTgrZMG)] [[code](https://github.com/MLCommons/dataperf)] [[website](https://dynabench.org/)] `DS` `DA` `DP`
- [NIPS 23] OpenDataVal: a Unified Benchmark for Data Valuation [[paper](https://openreview.net/pdf?id=eEK99egXeB)] [[code](https://opendataval.github.io/)] `DA`
- [DEEM 22] dcbench: A Benchmark for Data-Centric AI Systems [[paper](https://dl.acm.org/doi/pdf/10.1145/3533028.3533310?casa_token=hMwaReODUK0AAAAA:ocILXrfHOqC3QkaUltJRHM54vtiBwwBzZhM7QPBYNF5nIgivFRwTqjc3U7TZAvR5wekIuskjHIEwWQ)] [[code](https://github.com/data-centric-ai/dcbench)] `DS`

### Related Works

- More papers about Data Valuation can be found in [awesome-data-valuation](https://github.com/daviddao/awesome-data-valuation). `DA`

