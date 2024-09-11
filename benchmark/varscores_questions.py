import pandas as pd
import os
from bert_score import score as bert_score
import numpy as np

# Load the BERTScore metric
def calculate_bertscore(input_file):
    # Load the questions CSV file
    data = pd.read_csv(input_file)

    # Check for rows with missing text and exclude them
    valid_rows = data.dropna()
    invalid_rows = data[data.isnull().any(axis=1)]
    
    if not invalid_rows.empty:
        print(f"The following rows in {input_file} have missing text and will be excluded from the calculation:")
        print(invalid_rows)

    # Extract the different versions of the questions
    original_questions = valid_rows['original'].tolist()
    v1_questions = valid_rows['v1'].tolist()
    v2_questions = valid_rows['v2'].tolist()
    v3_questions = valid_rows['v3'].tolist()
    v4_questions = valid_rows['v4'].tolist()
    v5_questions = valid_rows['v5'].tolist()

    # Combine all questions for pairwise comparison
    model_questions = [v1_questions, v2_questions, v3_questions, v4_questions, v5_questions]
    labels = ['v1', 'v2', 'v3', 'v4', 'v5']

    # Initialize an empty dictionary to store BERTScore results
    bertscore_results = {}

    # OrigVarScore: Compare original to v1-v5
    origvar_scores = []
    for i, pred_list in enumerate(model_questions):
        print(f"Calculating OrigVarScore for version {labels[i]}...")
        results = bert_score(pred_list, original_questions, lang="en")
        f1_scores = results[2].tolist()  # F1 score is at index 2
        origvar_scores.append(f1_scores)

    # QVarScore: Compare v1 to v5 against each other
    qvar_scores = []
    for i in range(5):
        for j in range(i + 1, 5):
            print(f"Calculating QVarScore for versions {labels[i]} and {labels[j]}...")
            results = bert_score(model_questions[i], model_questions[j], lang="en")
            f1_scores = results[2].tolist()  # F1 score is at index 2
            qvar_scores.append(f1_scores)

    # Calculate averages and standard deviations for OrigVarScore
    avg_origvar = np.mean([np.mean(score) for score in origvar_scores])
    std_origvar = np.std([np.mean(score) for score in origvar_scores])

    # Calculate averages and standard deviations for QVarScore
    avg_qvar = np.mean([np.mean(score) for score in qvar_scores])
    std_qvar = np.std([np.mean(score) for score in qvar_scores])

    return {
        'avg_origvar_score': avg_origvar,
        'std_origvar_score': std_origvar,
        'avg_qvar_score': avg_qvar,
        'std_qvar_score': std_qvar
    }

def process_files_in_directory(directory):
    # Define the temperature range and file patterns
    temp_range = [0.6, 0.3, 0.1]
    base_filenames = [
        "questions_rephrased_temp_{}.csv",
        "alt_questions_rephrased_temp_{}.csv"
    ]

    # Initialize a list to store the results
    all_results = []
    
    # Iterate through the temperatures and file patterns
    for temp in temp_range:
        for base_filename in base_filenames:
            filename = base_filename.format(temp)
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                # Calculate BERTScore and retrieve the results
                bertscore_results = calculate_bertscore(file_path)
                bertscore_results['temperature'] = temp
                if "alt_questions" in filename:
                    bertscore_results['file_type'] = '5P'
                else:
                    bertscore_results['file_type'] = '1P'
                bertscore_results['filename'] = filename
                
                # Append results to the all_results list
                all_results.append(bertscore_results)

                # Print the results
                print(f"Results for {file_path}:")
                print(pd.DataFrame([bertscore_results]).to_string(index=False))
            else:
                print(f"File {filename} does not exist in directory {directory}")

    # Convert the results to a DataFrame
    df_all_results = pd.DataFrame(all_results)

    # Save the results to a single CSV file
    df_all_results.to_csv(os.path.join(directory, 'bertscore_comparison_all_results.csv'), index=False)

# Define the directory containing the CSV files
directory = '/local/scratch/ydiekma/PatientFacingLLM-Eval/MedQUAD_results'

# Process the files in the directory
process_files_in_directory(directory)

