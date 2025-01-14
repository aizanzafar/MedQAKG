# MedQAKG: Exploiting Medical Knowledge Graphs for Medical Question Answering

[![DOI:10.1007/s10489-024-05282-8](https://img.shields.io/badge/DOI-10.1007/s10489--024--05282--8-blue)](https://doi.org/10.1007/s10489-024-05282-8)

## Description

MedQAKG is a knowledge-enabled question-answering (QA) system that leverages large-scale medical knowledge graphs, including PharmKG and UMLS, to enhance the accuracy and relevance of medical answers. This repository provides the codebase, datasets, and instructions to reproduce the results published in:

**Zafar, A., Varshney, D., Kumar Sahoo, S. et al. (2024).**  
*[Are my answers medically accurate? Exploiting medical knowledge graphs for medical question answering](https://doi.org/10.1007/s10489-024-05282-8).*  
Applied Intelligence, 54, 2172–2187.

---

## Features

- Knowledge Graph construction using PharmKG and UMLS.
- Multi-hop reasoning over medical knowledge graphs.
- Contextual representation using transformers (e.g., RoBERTa).
- Benchmarks on MASH-QA and COVID-QA datasets.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aizanzafar/MedQAKG.git
   cd MedQAKG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Preparation

The datasets (`train.json`, `val.json`, and `test.json`) are located in the `Data/` folder. Ensure these files are correctly placed before proceeding.

### 2. Knowledge Graph Construction

#### Using PharmKG
Run the following script to create a Knowledge Graph using PharmKG:
```bash
python pharm_kg_graph.py
```

#### Using UMLS
To create a Knowledge Graph from UMLS, navigate to the `UMLS_KG_Preprocess/` folder and execute:
```bash
python final_preprocess.py
```

---

### 3. Preprocessing

To preprocess the dataset and configure multi-hop relations:
1. Modify the `relation_list` in `load_covidqa_dic.py` to set desired relation types (default: `['uses', 'treats', 'prevents', 'isa', 'diagnoses', 'co-occurs_with', 'associated_with', 'affects']`).
2. Configure the number of hops in `preprocess.py` by setting the `n_hops` parameter in the `sorted_triple` function.

Run preprocessing:
```bash
python preprocess.py
```

---

### 4. Training the Model

To train the QA model, run:
```bash
python qa_st.py
```

#### Default Training Arguments:
```python
train_args = {
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'max_seq_length': 512,
    'doc_stride': 384,
    'output_dir': "roberta_att_2_filter_mashqa/",
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 8,
}
model = QuestionAnsweringModel("roberta", "roberta-large", args=train_args)
```

---

### 5. Evaluation

After training, evaluate the model on the test set:
```bash
python evaluate.py --checkpoint <path_to_model_checkpoint>
```

---

## Datasets

### MASH-QA
- Source: [MASH-QA GitHub](https://github.com/mingzhu0527/MASHQA)
- Description: Consumer healthcare questions from WebMD.

### COVID-QA
- Source: [COVID-QA Dataset](https://github.com/deepset-ai/COVID-QA)
- Description: QA dataset based on academic COVID-19 research.

---

## Reproducibility

To reproduce the results:
1. Follow the installation and data preparation steps.
2. Train the model using the provided scripts.
3. Evaluate using the default configurations or adjust hyperparameters as needed.

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{zafar2024medqakg,
  title={Are my answers medically accurate? Exploiting medical knowledge graphs for medical question answering},
  author={Zafar, Aizan and Varshney, Deeksha and Sahoo, Sovan Kumar and others},
  journal={Applied Intelligence},
  volume={54},
  pages={2172--2187},
  year={2024},
  publisher={Springer}
}
```

---
