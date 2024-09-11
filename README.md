# Evaluating Safety of Large Language Models for Patient-facing Medical Question Answering

This repository contains the code and instructions to reproduce the experiments presented in the paper "Evaluating Safety of Large Language Models for Patient-facing Medical Question Answering."

## Table of Contents
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Code Execution Steps](#code-execution-steps)
  - [Step 1: Download Datasets](#step-1-download-datasets)
  - [Step 2: Rephrase Questions](#step-2-rephrase-questions)
  - [Step 3: Model Answer Generation](#step-3-model-answer-generation)
    - [Running Model Scripts](#running-model-scripts)
- [Benchmark](#benchmark)
  - [Running Benchmark Scripts](#running-benchmark-scripts)

---

## Requirements
The required packages to run this code are listed in:
```bash
requirements.txt
```

## Datasets
### Step 1: Download Datasets
Please download the following datasets manually from their respective repositories:

1. **TREC LiveQA Medical Task 2017 Dataset**  
   - GitHub: [LiveQA MedicalTask TREC 2017](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017)

2. **MedQuAD Dataset - CDC QA**  
   - GitHub: [MedQuAD - CDC QA](https://github.com/abachaa/MedQuAD/tree/master/9_CDC_QA)

Make sure to place the downloaded datasets into their respective directories before proceeding to the next steps:
```bash
/data
```
The column header for the answers should be "answer" and for the questions "question".


## Code Execution Steps

### Step 2: Rephrase Questions
There are two approaches to rephrasing questions in this repository: the **1-prompt approach** and the **5-prompt approach**. 

To run 1P, use the following script:

```bash
python rephrase/1P_rephrase.py <dataset>
```

To run 5P, use the following script:

```bash
python rephrase/5P_rephrase.py <dataset>
```

Where `<dataset>` is:
- `T` for the TREC LiveQA dataset.
- `M` for the MedQuAD dataset.


#### Input Format
The input for both `1P_rephrase.py` and `5P_rephrase.py` should be in CSV format. The input CSV file must have a column labeled `question` containing the questions to be rephrased.

- `original`: The original question.
- `v1`: Rephrased version 1.
- `v2`: Rephrased version 2.
- `v3`: Rephrased version 3.
- `v4`: Rephrased version 4.
- `v5`: Rephrased version 5.

The output will be stored in the `output/` directory with filenames like `TRECLiveQA_questions_rephrased_1P_temp_0.3.csv` or `MedQuAD_questions_rephrased_5P_temp_0.4.csv`, depending on the dataset and temperature.

### Step 3: Model Answer Generation

#### Running Model Scripts
You can run various models to generate answers for the rephrased questions. The scripts for the models accept two arguments:
1. `input_dataset`: This specifies the dataset to use, either `TREC` or `medquad`.
2. `temperature`: This specifies the temperature for text generation (e.g., 0.3, 0.5).

In either case, the model will generate answers based on the temp = 0.1 rephrase dataset of either file following the findings in the paper.

To run the models, use the following commands:

1. **Meta-Llama-3-70B-Instruct**
    ```bash
    python Meta-Llama-3-70B-Instruct.py <input_dataset> <temperature>
    ```

2. **Meta-Llama-3-8B-Instruct**
    ```bash
    python Meta-Llama-3-8B-Instruct.py <input_dataset> <temperature>
    ```

3. **Meditron-70B**
    ```bash
    python Meditron-70B.py <input_dataset> <temperature>
    ```

4. **Meditron-7B**
    ```bash
    python Meditron-7B.py <input_dataset> <temperature>
    ```

5. **PMC_LLaMA_13B**
    ```bash
    python PMC_LLaMA_13B.py <input_dataset> <temperature>
    ```

6. **medalpaca-13b**
    ```bash
    python medalpaca-13b.py <input_dataset> <temperature>
    ```

7. **mellama-13b**
    ```bash
    python mellama-13b.py <input_dataset> <temperature>
    ```

8. **mellama-70b**
    ```bash
    python mellama-70b.py <input_dataset> <temperature>
    ```

### Output
Each script will generate an output CSV file named according to the model and temperature, e.g., `Meta-Llama-3-70B_temp_0.3.csv` or `medalpaca_temp_0.5.csv`. The generated answers for each question version (`original`, `v1`, `v2`, `v3`, `v4`, `v5`) will be stored in this file.

---

## Benchmark

### Running Benchmark Scripts
Two benchmark scripts are provided to calculate and evaluate the similarity between the original and rephrased questions, as well as the model-generated answers. 

**varscores_answers.py**: Used to benchmark answers from the models.

1. **varscores_answers.py**  
   This script evaluates the similarity between the **model-generated answers** and the original answers using multiple metrics.

   #### Usage:
   ```bash
   python benchmark/varscores_answers.py [TREC|MedQuAD] [temperature]
   ```

   - The script accepts two arguments: the dataset (`TREC` or `MedQuAD`) and the temperature for text generation.
   - It processes all the model-generated answers for the given dataset and temperature, and evaluates them against the original answers provided in the corresponding dataset.

   Example for TREC dataset at temperature 0.1:
   ```bash
   python benchmark/varscores_answers.py TREC 0.1
   ```

**varscores_questions.py**: Used to benchmark the rephrases of Meta-Llama-3.

2. **varscores_questions.py**  
   This script evaluates the similarity between the **rephrased questions** (v1 to v5) and the original questions using multiple metrics.

   #### Usage:
   ```bash
   python benchmark/varscores_questions.py [TREC|MedQuAD] [temperature]
   ```

   - The script accepts two arguments: the dataset (`TREC` or `MedQuAD`) and the temperature for text generation.

### Input and Output
Both benchmark scripts expect the input files to be present in the `../output/answer_results` directory, with names formatted as follows:
- For answers: `TREC_{modelname}_temp_{temp}.csv` or `MedQuAD_{modelname}_temp_{temp}.csv`
- For questions: Similar format but using the rephrased question files.

The output of the benchmark scripts will be two CSV files (mean and standard deviation of the metrics) saved in the same directory. Example output files:
- `COMBINED_all_models_metrics_comparisons_mean_temp_0.1.csv`
- `COMBINED_all_models_metrics_comparisons_std_temp_0.1.csv`
