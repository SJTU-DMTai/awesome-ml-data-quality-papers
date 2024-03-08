# Awesome Data Quality Papers

This is a list of papers about training data quality management for ML models. 

Data scientists spend ∼80% time on data preparation for an ML pipeline since the data quality issues are unknown beforehand thereby leading to iterative debugging. A good Data Quality Management System (**DQMS**) for ML helps data scientists break free from the arduous process of data selection and debugging, particularly in the current era of big data and large models. Automating the management of training data quality effectively is crucial for improving the efficiency and quality of ML pipelines.

Fortunately, with the emergence and development of "**Data-Centric AI**", there has been increasing research focus on optimizing the quality of training data rather than solely concentrating on model structures and training strategies. This is the motivation behind maintaining this repository.

Before we proceed, let's define **data quality for ML**. In contrast to traditional data cleaning, training data quality for ML refers to the impact of individual or groups of data samples on the behavior of ML models for a given task. It's important to note that the behavior of the model we are concerned with goes beyond performance metrics like accuracy, recall, and fitting precision. We also consider more generalizable metrics such as model fairness and robustness.

Considering the following pipeline, DQMS acts as a **middleware** between data, ML model, and user, necessitating interactions with each of them.

![Framework of DQMS.](./framework.png)

Firstly, DQMS needs to interact with the data sources to acquire the training dataset. Next, it feeds the training dataset into the ML model and obtains model feedback. Then, DQMS combines the model feedback with the user's task requirements to evaluate the data quality of the current training dataset. It provides feedback to the user in a comprehensible format and utilizes the data quality information to select new and improved training data from the data source.

Hence, we can classify existing research work based on the three functional modules of DQMS:

## Data Selector: Interact with Data Source

### 2023

- [NIPS] Data Selection for Language Models via Importance Resampling [[PDF](https://arxiv.org/pdf/2302.03169.pdf)][[Code](https://github.com/p-lambda/dsir)]

## Data Attributer: Interact with ML Model

### 2024

- [ICLR] "What Data Benefits My Classifier?" Enhancing Model Performance and Interpretability through Influence-Based Data Selection [[PDF](https://openreview.net/pdf?id=HE9eUQlAvo)][[Code](https://github.com/anshuman23/InfDataSel)]

## Data Profiler: Interact with User

### 2022

- [SIGMOD] Interpretable Data-Based Explanations for Fairness Debugging [[PDF](https://arxiv.org/pdf/2112.09745.pdf)][[Video](https://www.youtube.com/watch?v=bt_VL1eSu30)]