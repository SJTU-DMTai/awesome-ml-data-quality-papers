# Awesome ML Data Quality Papers

This is a list of papers about training data quality management for ML models. 

Data scientists spend ∼80% time on data preparation for an ML pipeline since the data quality issues are unknown beforehand thereby leading to iterative debugging. A good Data Quality Management System (**DQMS**) for ML helps data scientists break free from the arduous process of data selection and debugging, particularly in the current era of big data and large models. Automating the management of training data quality effectively is crucial for improving the efficiency and quality of ML pipelines.

Fortunately, with the emergence and development of "**Data-Centric AI**", there has been increasing research focus on optimizing the quality of training data rather than solely concentrating on model structures and training strategies. This is the motivation behind maintaining this repository.

Before we proceed, let's define **data quality for ML**. In contrast to traditional data cleaning, training data quality for ML refers to the impact of individual or groups of data samples on the behavior of ML models for a given task. It's important to note that the behavior of the model we are concerned with goes beyond performance metrics like accuracy, recall, and fitting precision. We also consider more generalizable metrics such as model fairness and robustness.

Considering the following pipeline, DQMS acts as a **middleware** between data, ML model, and user, necessitating interactions with each of them.


<div align=center>
<img src="./framework.png" width = "70%" />
</div>
A DQMS typically consists of three components: **Data Selector**, **Data Attributer**, and **Data Profiler**. To achieve a well-performing ML model, multiple rounds of training are often required. In this process, the DQMS needs to iteratively adjust the training data based on the results of each round of model training. The work flow of DQMS in one round of training is as followed: (a) Data Selector first acquires the training dataset from a data source and train the ML model with it. (b) After trained for one round (several epochs), Data Attributer absorbs feedback from the model and user's task specifications, and compute assess the data quality assessment. (c) Data Profiler then provides a user-friendly summary of the training data. (d) Meanwhile, Data Selector utilizes the data quality assessment as feedback to acquire higher-quality training data, thus initiating a new iteration.

We collect the recent papers about DQMS for ML model, and annotate the relevant DQMS components involved in these papers, where ❶ = Data Selector, ❷ = Data Attributer, and ❸ = Data Profiler.

## 2024

- [ICLR 24] "What Data Benefits My Classifier?" Enhancing Model Performance and Interpretability through Influence-Based Data Selection [[paper](https://openreview.net/pdf?id=HE9eUQlAvo)] [[code](https://github.com/anshuman23/InfDataSel)] ❷

## 2023

- [NIPS 23] Data Selection for Language Models via Importance Resampling [[paper](https://openreview.net/pdf?id=uPSQv0leAu)] [[code](https://github.com/p-lambda/dsir)] ❶
- [NIPS 23] Sample based Explanations via Generalized Representers [[paper](https://openreview.net/pdf?id=fX64q0SNfL)] ❷
- [NIPS 23] OpenDataVal: a Unified Benchmark for Data Valuation [[paper]([pdf (openreview.net)](https://openreview.net/pdf?id=eEK99egXeB))] [[code](https://opendataval.github.io/)] ❷
- [SIGMOD 22] Complaint-Driven Training Data Debugging at Interactive Speeds [[paper](https://dl.acm.org/doi/pdf/10.1145/3514221.3517849)] ❷
- [SIGMOD 22] Interpretable Data-Based Explanations for Fairness Debugging [[paper](https://arxiv.org/pdf/2112.09745.pdf)] [[video](https://www.youtube.com/watch?v=bt_VL1eSu30)] ❷❸
- [VLDB 23] Equitable Data Valuation Meets the Right to Be Forgotten in Model Markets [[paper](https://www.vldb.org/pvldb/vol16/p3349-liu.pdf)] [[code](https://github.com/ZJU-DIVER/ValuationMeetsRTBF)] ❷

## 2022



## 2021 and before



## Survey Papers

- [NIPS 23] DataPerf: Benchmarks for Data-Centric AI Development [[paper]([pdf (openreview.net)](https://openreview.net/pdf?id=LaFKTgrZMG))] [[code](https://github.com/MLCommons/dataperf)] ❶❷❸

