import pandas as pd
import os
import sys
from bert_score import score as bert_score
import numpy as np

def calculate_bertscore(input_file):
    data = pd.read_csv(input_file)

    valid_rows = data.dropna()
    invalid_rows = data[data.isnull().any(axis=1)]
    
    original_questions = valid_rows['original'].tolist()
    v1_questions = valid_rows['v1'].tolist()
    v2_questions = valid_rows['v2'].tolist()
    v3_questions = valid_rows['v3'].tolist()
    v4_questions = valid_rows['v4'].tolist()
    v5_questions = valid_rows['v5'].tolist()

    model_questions = [v1_questions, v2_questions, v3_questions, v4_questions, v5_questions]
    labels = ['v1', 'v2', 'v3', 'v4', 'v5']

    bertscore_results = {}

    origvar_scores = []
    for i, pred_list in enumerate(model_questions):
        results = bert_score(pred_list, original_questions, lang="en")
        f1_scores = results[2].tolist()  
        origvar_scores.append(f1_scores)

    qvar_scores = []
    for i in range(5):
        for j in range(i + 1, 5):
            results = bert_score(model_questions[i], model_questions[j], lang="en")
            f1_scores = results[2].tolist() 
            qvar_scores.append(f1_scores)

    avg_origvar = np.mean([np.mean(score) for score in origvar_scores])
    std_origvar = np.std([np.mean(score) for score in origvar_scores])

    avg_qvar = np.mean([np.mean(score) for score in qvar_scores])
    std_qvar = np.std([np.mean(score) for score in qvar_scores])

    return {
        'avg_origvar_score': avg_origvar,
        'std_origvar_score': std_origvar,
        'avg_qvar_score': avg_qvar,
        'std_qvar_score': std_qvar
    }

def process_files(dataset, temp, directory):
    if dataset == 'TREC':
        filename = f"combined_rephrased_TREC_temp_{temp}.csv"
    elif dataset == 'MedQuAD':
        filename = f"combined_rephrased_MedQuAD_temp_{temp}.csv"
    else:
        print(f"Invalid dataset name: {dataset}. Use 'TREC' or 'MedQuAD'.")
        return

    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        bertscore_results = calculate_bertscore(file_path)
        bertscore_results['temperature'] = temp
        bertscore_results['file_type'] = dataset
        bertscore_results['filename'] = filename

        df_all_results = pd.DataFrame([bertscore_results])
        result_file = f"bertscore_comparison_{dataset}_temp_{temp}.csv"
        df_all_results.to_csv(os.path.join(directory, result_file), index=False)

        print(f"Results saved to {result_file}")
    else:
        print(f"File {filename} does not exist in directory {directory}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python codename.py [TREC|MedQuAD] [temperature]")
        sys.exit(1)

    dataset = sys.argv[1]
    temp = sys.argv[2]

    directory = os.path.join(os.path.dirname(__file__), '../output/rephrase_results')

    process_files(dataset, temp, directory)
