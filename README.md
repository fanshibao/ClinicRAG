# ClinicRAG

**This is the data and code for our paper** `ClinicRAG: A Multi-Stage Diagnostic Framework Enhanced by Multi-Source Heterogeneous Knowledge`.

For reproduction of automatic diagnosis results in our paper, see instructions below.

## Overview

We have modularized and encapsulated the code into a more readable form. In brief, ClinicRAG can be divided into four core modules: symptom extraction, disease retrieval, disease reasoning, and diagnostic inquiry mechanism. It integrates Chain-of-Thought (CoT) and Retrieval-Augmented Generation (RAG) technologies to mitigate LLM hallucinations and improve consultation efficiency.

## Requirements

Make sure your local environment has the following installed:

* `pytorch >= 2.0.0`
* `python >= 3.8`
* `transformers >= 4.35.0`
* `faiss-cpu >= 1.7.4`
* `sentence-transformers >= 2.2.2`
* `peft >= 0.6.2`
* `accelerate >= 0.24.1`
* `numpy >= 1.24.0`
* `scikit-learn >= 1.3.0`
* `tqdm >= 4.66.0`

#### Datasets

We provide the dataset in the [data](data/) folder.

| Data   | Source                                                   | Description                                                  |
| ------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| DXY    | Public medical consultation dataset                       | Contains real-world patient-doctor consultation records for automatic diagnosis evaluation |
| MZ-4   | Multi-turn medical dialogue dataset                       | Includes multi-round symptom inquiry and diagnosis records to test multi-turn decision-making capabilities |

## Documentation

```
--src
  │--README.md
  │--data_loader.py
  │--train_clinicgpt.py
  │--train_retriever.py
  │--clinicrag/
  │   │--__init__.py
  │   │--symptom_extraction.py
  │   │--disease_retrieval.py
  │   │--disease_reasoning.py
  │   │--inquiry_mechanism.py
  │   │--clinicgpt.py
  │--demo.py
  │--util.py
  
--data
  │--dxy/
  │   │--train.json
  │   │--val.json
  │   │--test.json
  │--mz4/
  │   │--train.json
  │   │--val.json
  │   │--test.json
  │--disease_db/
  │   │--disease_symptom_pairs.json
  │--ehr_db/
  │   │--ehr_records.json
  │--processing.py
  
--models
  │--lora_weights/
  │--retrieval_model/
  
--figures
  │--architecture.png
  │--case_study.png
```

## How to Use ClinicRAG

### 1 Install IDE 

Our project is built on PyCharm Community Edition ([click here to get](https://www.jetbrains.com/products/compare/?product=pycharm-ce&product=pycharm)).

### 2 Environment Setting

#### 2.1 Interpreter 

We recommend using `Python 3.8` or higher as the script interpreter. [Click here to get](https://www.python.org/downloads/release/python-380/) `Python 3.8`.

#### 2.2 Packages

First, install [conda](https://www.anaconda.com/)
Then, create the conda environment through yaml file:

```
conda env create -f env.yaml
```

Or install dependencies directly via pip:

```
pip install -r requirements.txt
```

### 3 Start Running

Please follow the steps below:

3.1 Prepare data and process

  In ./data, you can find the well-preprocessed data in JSON form. Also, it's easy to re-generate the data as follows:

  - Download DXY and MZ-4 datasets and put them in ./data/
  - Download disease knowledge base (9,604 diseases) and EHR database, then put them in ./data/disease_db/ and ./data/ehr_db/ respectively
  - Run code ./data/processing.py

  Data information in ./data:
  - **dxy/ & mz4/**: Training/validation/test sets for automatic diagnosis, including patient complaints, symptoms, and target diseases.
  - **disease_db/**: Disease-symptom pairs for disease retrieval module.
  - **ehr_db/**: Real clinical records for EHR-guided reasoning module.

3.2 Train models
  - Train ClinicGPT (LoRA fine-tuning):
    ```bash
    python train_clinicgpt.py \
      --base_model "01-ai/Yi-34B-Base" \
      --train_data "data/dxy/train.json" \
      --lora_rank 8 \
      --batch_size 64 \
      --learning_rate 2e-4 \
      --epochs 3 \
      --output_dir "models/lora_weights"
    ```
  - Train disease retrieval model:
    ```bash
    python train_retriever.py \
      --model_name "all-mpnet-base-v2" \
      --train_data "data/disease_db/disease_symptom_pairs.json" \
      --batch_size 256 \
      --learning_rate 2e-4 \
      --epochs 5 \
      --output_dir "models/retrieval_model"
    ```

3.3 Run demo
  ```bash
  python demo.py
  ```

## Acknowledgement

We sincerely thank these repositories and projects for their valuable support: Yi-34B (https://github.com/01-ai/Yi), FAISS (https://github.com/facebookresearch/faiss), LoRA (https://github.com/microsoft/LoRA), and sentence-transformers (https://github.com/UKPLab/sentence-transformers).

## TODO

To make the experiments more efficient, we developed some experimental scripts (e.g., ablation study scripts, compatibility test scripts) and visualization tools, which will be released along with the paper later. We also plan to expand the disease knowledge base and EHR database to cover more medical scenarios.

