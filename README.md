# Evaluating Safety of Large Language Models for Patient-facing Medical Question Answering

This repository contains the code and instructions to reproduce the experiments presented in the paper "Evaluating Safety of Large Language Models for Patient-facing Medical Question Answering."

## Table of Contents
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Code Execution Steps](#code-execution-steps)
  - [Step 1: Download Datasets](#step-1-download-datasets)
  - [Step 2: Rephrase Questions](#step-2-rephrase-questions)
  - [Step 3: Model Answer Generation](#step-3-model-answer-generation)
- [Citation](#citation)

---

## Requirements
To run the code, you will need the following packages:
- Python 3.7 or higher
- pandas
- transformers
- torch
- BERTScore (optional for evaluation purposes)

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Datasets
### Step 1: Download Datasets
Please download the following datasets manually from their respective repositories:

1. **TREC LiveQA Medical Task 2017 Dataset**  
   - GitHub: [LiveQA MedicalTask TREC 2017](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017)

2. **MedQuAD Dataset - CDC QA**  
   - GitHub: [MedQuAD - CDC QA](https://github.com/abachaa/MedQuAD/tree/master/9_CDC_QA)

Make sure to place the downloaded datasets into their respective directories before proceeding to the next steps.

## Code Execution Steps

### Step 2: Rephrase Questions
You will need to run the rephrasing code for each dataset.

- For the TREC LiveQA Medical Task dataset, execute the following script:
    ```bash
    python rephrase_TRECLiveQA.py
    ```

- For the MedQuAD CDC QA dataset, execute the following script:
    ```bash
    python rephrase_MedQuAD.py
    ```

These scripts will preprocess and rephrase the questions from the datasets. The rephrased questions will be stored in the respective directories.

### Step 3: Model Answer Generation
Once the questions have been rephrased, you can run the models to generate answers for each dataset. Instructions for running the models will depend on the specific model used (e.g., Llama 3 or Meditron).

Refer to the relevant scripts and documentation for generating answers with the models.

