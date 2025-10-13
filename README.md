# Overton Pluralistic Reinforcement Learning for Large Language Models

This repository provides a complete OP–GRPO implementation pipeline, offering an end-to-end framework for aligning large language models with Overton pluralism. It covers all key stages: data preprocessing with redundancy filtering and perspective augmentation to build high-quality, diverse datasets; training with GRPO, extended by an OP-specific reward system that incorporates perspective coverage, uniqueness, and formatting consistency; and evaluation, using ValuePrism NLI benchmarks, coverage metrics, and token efficiency analysis.

The dataset is available on Hugging Face: [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-OP--V2-yellow.svg)](https://huggingface.co/datasets/Rich740804/OP-V2)

### 📁 **OP-GRPO Folder Structure**

###  `data_preprocess/`
This folder contains all the scripts necessary to process the raw CSV files into the corresponding training datasets in Parquet format. These outputs are tailored for different stages of the Verl training framework, including the Supervised Fine-Tuning (SFT) stage, the Reinforcement Learning (RL) stage, and the extended SFT stage that incorporates summary perspectives.

###  `verl/`
This is the core folder of the project, as it provides the main training framework on which our OP–GRPO implementation is built. More specifically, several files in this folder have been modified to support and adapt the framework for OP–GRPO training:
  
- **verl\workers\reward_manager\batch.py**:
   This is the file responsible for integrating all reward components and adjusting their scaling, ensuring that the final reward signal is properly balanced before being used in training.

###  `train_st/`
This folder contains the training process for the OP-SBERT model. It includes the construction of triplet datasets, hyperparameter optimization, and the final fine-tuning stage. The full pipeline can be executed using the provided `run_trainer.sh` bash script.

###  `benchmark_new/`
This folder contains the test datasets and evaluation tools, including inference scripts for trained models, GPT-4.1 as an LLM-judge, natural language inference benchmarks, SBERT-based similarity checks, and token generation analysis, providing a comprehensive framework for evaluating OP-GRPO performance.


---



## 🚀 Installation and Usage

To install all required packages:

```bash
pip install -r requirements.txt
```

Running Instructions

1. Use the corresponding preprocessing scripts for each dataset to generate the required Parquet training files.
2. Before starting GRPO training, you may fine-tune the SBERT model to improve its performance for the OP-GRPO stage.
3. Run the run_grpo.sh script to launch OP-GRPO training.

## 🚨 Important Notification
1. Please modoify all of the file path in your place, to replace the current path.

2. This project is build on the base of [**Verl**](https://github.com/volcengine/verl): which provides a flexible and efficient foundation for reinforcement learning with large language models. Verl is designed to support diverse RL algorithms such as PPO and GRPO, while offering high-performance GPU utilization and modular extensibility.

3. The Modular Pluralism method inference and evalaution can refer [**Modular Pluralism**](https://github.com/BunsenFeng/modular_pluralism)
