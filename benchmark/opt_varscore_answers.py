import pandas as pd
from bert_score import score as bert_score
import os
import numpy as np

def calculate_bert_scores(input_file, output_dir, model_name, original_answers_file, model_header_prefix):
    headers = [
        f'{model_header_prefix}_answer_original',
        f'{model_header_prefix}_answer_v1',
        f'{model_header_prefix}_answer_v2',
        f'{model_header_prefix}_answer_v3',
        f'{model_header_prefix}_answer_v4',
        f'{model_header_prefix}_answer_v5'
    ]

    print(f"Loading datasets for {model_name}...")
    data = pd.read_csv(input_file)
    original_data = pd.read_csv(original_answers_file)

    print("Checking for and removing rows with missing values...")
    valid_rows = data.dropna(thresh=2)
    invalid_rows = data[data.isnull().any(axis=1)]
    
    if not invalid_rows.empty:
        print(f"The following rows in {input_file} have missing values and will be skipped:")
        print(invalid_rows.index.tolist())

    original_data = original_data.loc[valid_rows.index]

    original_answers = original_data['answer'].tolist()
    model_answers = [valid_rows[header].tolist() for header in headers]  # v1 to v5 only

    qvar_scores = {'bert': []}
    origvar_scores = {'bert': []}
    skipped_rows = []

    # BERTScore metric (batch processing)
    def bert_score_metric_batch(refs, preds):
        try:
            bert_p, _, F1 = bert_score(preds, refs, lang='en')
            return F1.tolist()
        except Exception as e:
            print(f"Error in BERTScore batch calculation: {e}")
            return [np.nan] * len(refs)

    # Calculating OrigVarScore and QVarScore
    for i in range(6):  # v1 to v5 comparisons with original
        bert_scores = bert_score_metric_batch(original_answers, model_answers[i])
        origvar_scores['bert'].extend(bert_scores)

    for i in range(1, 6):
        for j in range(i + 1, 6):
            qvar_scores['bert'].extend(bert_score_metric_batch(model_answers[i], model_answers[j]))

    # Calculating final averages and standard deviations for OrigVarScore and QVarScore
    results_summary_mean = [{
        'Model': model_name,
        'OrigVarScore BERT': np.nanmean(origvar_scores['bert']),
        'QVarScore BERT': np.nanmean(qvar_scores['bert'])
    }]

    results_summary_std = [{
        'Model': model_name,
        'OrigVarScore BERT': np.nanstd(origvar_scores['bert']),
        'QVarScore BERT': np.nanstd(qvar_scores['bert'])
    }]

    results_df_mean = pd.DataFrame(results_summary_mean)
    results_df_std = pd.DataFrame(results_summary_std)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.basename(input_file).replace('.csv', f'_{model_name}_bert_scores')
    output_file_mean = os.path.join(output_dir, base_filename + '_mean.csv')
    output_file_std = os.path.join(output_dir, base_filename + '_std.csv')
    
    results_df_mean.to_csv(output_file_mean, index=False)
    results_df_std.to_csv(output_file_std, index=False)

    print(f"Completed processing for {model_name}. Results saved to {output_file_mean} and {output_file_std}\n")

    return results_df_mean, results_df_std

output_dir = '/local/scratch/ydiekma/PatientFacingLLM-Eval/missing_rows/'
original_answers_file = '/local/scratch/ydiekma/PatientFacingLLM-Eval/missing_rows/TREC-2017-LiveQA_incl_missingrows.csv'

# Iterate over temperatures from 0.2 to 0.6
for temp in [0.2, 0.3, 0.4, 0.5, 0.6]:
    models = [
       # {'file': f'Meta_Llama_3_70B_temp_{temp}.csv', 'header_prefix': 'Meta_Llama_3_70B', 'model_name': 'Meta_Llama_3_70B'},
       # {'file': f'Meta_Llama_3_8B_temp_{temp}.csv', 'header_prefix': 'Meta_Llama_3_8B', 'model_name': 'Meta_Llama_3_8B'},
        {'file': f'PMC_llama13b_temp_{temp}.csv', 'header_prefix': 'PMC-llama13b', 'model_name': 'PMC_llama13b'},
       # {'file': f'medalpaca_temp_{temp}.csv', 'header_prefix': 'medalpaca-13b', 'model_name': 'medalpaca'},
       # {'file': f'meditron70B_temp_{temp}.csv', 'header_prefix': 'meditron_70b', 'model_name': 'meditron70B'},
       # {'file': f'meditron7B_temp_{temp}.csv', 'header_prefix': 'meditron_7b', 'model_name': 'meditron7B'},
       # {'file': f'mellama-70b_temp_{temp}.csv', 'header_prefix': 'mellama-70b', 'model_name': 'mellama-70b'},
       # {'file': f'mellama-13b_temp_{temp}.csv', 'header_prefix': 'mellama-13b', 'model_name': 'mellama-13b'}
    ]

    all_results_df_mean = pd.DataFrame()
    all_results_df_std = pd.DataFrame()

    for model in models:
        print(f"Starting processing for {model['model_name']}...")
        input_file = f'/local/scratch/ydiekma/PatientFacingLLM-Eval/missing_rows/combined_{model["file"]}'
        results_df_mean, results_df_std = calculate_bert_scores(
            input_file, output_dir, model['model_name'], original_answers_file, model['header_prefix']
        )

        all_results_df_mean = pd.concat([all_results_df_mean, results_df_mean], ignore_index=True)
        all_results_df_std = pd.concat([all_results_df_std, results_df_std], ignore_index=True)

    print("\n===== Final DataFrames with BERT Results for All Models =====")
    print("Means:")
    print(all_results_df_mean)
    print("Standard Deviations:")
    print(all_results_df_std)

    final_output_file_mean = os.path.join(output_dir, f'COMBINED_all_models_bert_scores_mean_temp_{temp}_PMC.csv')
    final_output_file_std = os.path.join(output_dir, f'COMBINED_all_models_bert_scores_std_temp_{temp}_PMC.csv')

    all_results_df_mean.to_csv(final_output_file_mean, index=False)
    all_results_df_std.to_csv(final_output_file_std, index=False)

