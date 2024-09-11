import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score
import os
import numpy as np

def calculate_metrics(input_file, output_dir, model_name, original_answers_file, model_header_prefix):
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
    labels = ['original', 'v1', 'v2', 'v3', 'v4', 'v5']

    rouge = Rouge()
    smoothing_function = SmoothingFunction().method1

    # Metrics storage for final averages
    qvar_scores = { 'bleu_1': [], 'bleu_4': [], 'rouge_1': [], 'rouge_l': [], 'bert': [] }
    origvar_scores = { 'bleu_1': [], 'bleu_4': [], 'rouge_1': [], 'rouge_l': [], 'bert': [] }
    maxvar_scores = { 'bleu_1': [], 'bleu_4': [], 'rouge_1': [], 'rouge_l': [], 'bert': [] }

    skipped_rows = []

    # Helper to compute metric averages across all rows for a specific pair (i,j)
    def compute_rowwise_average(metric_func, answers1, answers2):
        try:
            return metric_func(answers1, answers2)
        except Exception as e:
            skipped_rows.append(answers1.index(e))
            return [np.nan] * len(answers1)  # Handle errors and skip rows that fail

    # BLEU metric
    def bleu_1(refs, preds):
        scores = []
        for idx, (ref, pred) in enumerate(zip(refs, preds)):
            try:
                score = sentence_bleu([ref.split()], pred.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
                scores.append(score)
            except Exception as e:
                print(f"Error in BLEU-1 at row {idx}: {e}")
                skipped_rows.append(idx)
                scores.append(np.nan)
        return scores

    def bleu_4(refs, preds):
        scores = []
        for idx, (ref, pred) in enumerate(zip(refs, preds)):
            try:
                score = sentence_bleu([ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
                scores.append(score)
            except Exception as e:
                print(f"Error in BLEU-4 at row {idx}: {e}")
                skipped_rows.append(idx)
                scores.append(np.nan)
        return scores

    # ROUGE metric
    def rouge_1(refs, preds):
        scores = []
        for idx, (ref, pred) in enumerate(zip(refs, preds)):
            try:
                score = rouge.get_scores(pred, ref)[0]['rouge-1']['f']
                scores.append(score)
            except Exception as e:
                print(f"Error in ROUGE-1 at row {idx}: {e}")
                skipped_rows.append(idx)
                scores.append(np.nan)
        return scores

    def rouge_l(refs, preds):
        scores = []
        for idx, (ref, pred) in enumerate(zip(refs, preds)):
            try:
                score = rouge.get_scores(pred, ref)[0]['rouge-l']['f']
                scores.append(score)
            except Exception as e:
                print(f"Error in ROUGE-L at row {idx}: {e}")
                skipped_rows.append(idx)
                scores.append(np.nan)
        return scores

    # BERTScore metric (batch processing)
    def bert_score_metric_batch(refs, preds):
        try:
            bert_p, _, F1 = bert_score(preds, refs, lang='en')
            return F1.tolist()
        except Exception as e:
            print(f"Error in BERTScore batch calculation: {e}")
            return [np.nan] * len(refs)


    # Initialize list to store the row-wise maximum for each metric
    row_max_bleu_1 = []
    row_max_bleu_4 = []
    row_max_rouge_1 = []
    row_max_rouge_l = []
    row_max_bert = []

    # Calculating OrigVarScore and MaxVarScore
    for i in range(6):  # v1 to v5 comparisons with original
        # Calculate OrigVarScore
        bleu_1_scores = bleu_1(original_answers, model_answers[i])
        origvar_scores['bleu_1'].extend(bleu_1_scores)
        row_max_bleu_1.append(bleu_1_scores)
    
        bleu_4_scores = bleu_4(original_answers, model_answers[i])
        origvar_scores['bleu_4'].extend(bleu_4_scores)
        row_max_bleu_4.append(bleu_4_scores)

        rouge_1_scores = rouge_1(original_answers, model_answers[i])
        origvar_scores['rouge_1'].extend(rouge_1_scores)
        row_max_rouge_1.append(rouge_1_scores)

        rouge_l_scores = rouge_l(original_answers, model_answers[i])
        origvar_scores['rouge_l'].extend(rouge_l_scores)
        row_max_rouge_l.append(rouge_l_scores)

        bert_scores = bert_score_metric_batch(original_answers, model_answers[i])
        origvar_scores['bert'].extend(bert_scores)
        row_max_bert.append(bert_scores)
    
    # Calculate maxvarscore for each row by taking the maximum across all versions (v1-v5) for each metric
    for row_idx in range(len(original_answers)):
        maxvar_scores['bleu_1'].append(np.nanmax([row_max_bleu_1[i][row_idx] for i in range(6)]))
        maxvar_scores['bleu_4'].append(np.nanmax([row_max_bleu_4[i][row_idx] for i in range(6)]))
        maxvar_scores['rouge_1'].append(np.nanmax([row_max_rouge_1[i][row_idx] for i in range(6)]))
        maxvar_scores['rouge_l'].append(np.nanmax([row_max_rouge_l[i][row_idx] for i in range(6)]))
        maxvar_scores['bert'].append(np.nanmax([row_max_bert[i][row_idx] for i in range(6)]))

    # Calculating QVarScore (generated answer comparisons with each other)
    for i in range(1, 6):
        for j in range(i + 1, 6):
            qvar_scores['bleu_1'].extend(bleu_1(model_answers[i], model_answers[j]))
            qvar_scores['bleu_4'].extend(bleu_4(model_answers[i], model_answers[j]))
            qvar_scores['rouge_1'].extend(rouge_1(model_answers[i], model_answers[j]))
            qvar_scores['rouge_l'].extend(rouge_l(model_answers[i], model_answers[j]))
            qvar_scores['bert'].extend(bert_score_metric_batch(model_answers[i], model_answers[j]))

    # Calculating final averages and standard deviations for OrigVarScore, QVarScore, and MaxVarScore
    results_summary_mean = [{
        'Model': model_name,
        'OrigVarScore BLEU-1': np.nanmean(origvar_scores['bleu_1']),
        'OrigVarScore BLEU-4': np.nanmean(origvar_scores['bleu_4']),
        'OrigVarScore ROUGE-1': np.nanmean(origvar_scores['rouge_1']),
        'OrigVarScore ROUGE-L': np.nanmean(origvar_scores['rouge_l']),
        'OrigVarScore BERT': np.nanmean(origvar_scores['bert']),
    
        'MaxVarScore BLEU-1': np.nanmean(maxvar_scores['bleu_1']),
        'MaxVarScore BLEU-4': np.nanmean(maxvar_scores['bleu_4']),
        'MaxVarScore ROUGE-1': np.nanmean(maxvar_scores['rouge_1']),
        'MaxVarScore ROUGE-L': np.nanmean(maxvar_scores['rouge_l']),
        'MaxVarScore BERT': np.nanmean(maxvar_scores['bert']),
    
        'QVarScore BLEU-1': np.nanmean(qvar_scores['bleu_1']),
        'QVarScore BLEU-4': np.nanmean(qvar_scores['bleu_4']),
        'QVarScore ROUGE-1': np.nanmean(qvar_scores['rouge_1']),
        'QVarScore ROUGE-L': np.nanmean(qvar_scores['rouge_l']),
        'QVarScore BERT': np.nanmean(qvar_scores['bert'])
    }]

    results_summary_std = [{
        'Model': model_name,
        'OrigVarScore BLEU-1': np.nanstd(origvar_scores['bleu_1']),
        'OrigVarScore BLEU-4': np.nanstd(origvar_scores['bleu_4']),
        'OrigVarScore ROUGE-1': np.nanstd(origvar_scores['rouge_1']),
        'OrigVarScore ROUGE-L': np.nanstd(origvar_scores['rouge_l']),
        'OrigVarScore BERT': np.nanstd(origvar_scores['bert']),
    
        'MaxVarScore BLEU-1': np.nanstd(maxvar_scores['bleu_1']),
        'MaxVarScore BLEU-4': np.nanstd(maxvar_scores['bleu_4']),
        'MaxVarScore ROUGE-1': np.nanstd(maxvar_scores['rouge_1']),
        'MaxVarScore ROUGE-L': np.nanstd(maxvar_scores['rouge_l']),
        'MaxVarScore BERT': np.nanstd(maxvar_scores['bert']),
    
        'QVarScore BLEU-1': np.nanstd(qvar_scores['bleu_1']),
        'QVarScore BLEU-4': np.nanstd(qvar_scores['bleu_4']),
        'QVarScore ROUGE-1': np.nanstd(qvar_scores['rouge_1']),
        'QVarScore ROUGE-L': np.nanstd(qvar_scores['rouge_l']),
        'QVarScore BERT': np.nanstd(qvar_scores['bert'])
    }]

    results_df_mean = pd.DataFrame(results_summary_mean)
    results_df_std = pd.DataFrame(results_summary_std)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.basename(input_file).replace('.csv', f'_{model_name}_metrics_comparisons')
    output_file_mean = os.path.join(output_dir, base_filename + '_mean.csv')
    output_file_std = os.path.join(output_dir, base_filename + '_std.csv')
    
    results_df_mean.to_csv(output_file_mean, index=False)
    results_df_std.to_csv(output_file_std, index=False)

    print(f"Completed processing for {model_name}. Results saved to {output_file_mean} and {output_file_std}\n")
    
    # Print out skipped rows
    if skipped_rows:
        print(f"Skipped rows in {model_name}: {skipped_rows}")
    
    return results_df_mean, results_df_std

output_dir = '/local/scratch/ydiekma/PatientFacingLLM-Eval/missing_rows/'
original_answers_file = '/local/scratch/ydiekma/PatientFacingLLM-Eval/missing_rows/TREC-2017-LiveQA_incl_missingrows.csv'

temp = 0.1

models = [
   # {'file': f'Meta_Llama_3_70B_temp_{temp}.csv', 'header_prefix': 'Meta_Llama_3_70B', 'model_name': 'Meta_Llama_3_70B'},
   # {'file': f'Meta_Llama_3_8B_temp_{temp}.csv', 'header_prefix': 'Meta_Llama_3_8B', 'model_name': 'Meta_Llama_3_8B'},
   # {'file': f'PMC_llama13b_temp_{temp}.csv', 'header_prefix': 'PMC-llama13b', 'model_name': 'PMC_llama13b'},
   # {'file': f'medalpaca_temp_{temp}.csv', 'header_prefix': 'medalpaca-13b', 'model_name': 'medalpaca'},
   # {'file': f'meditron70B_temp_{temp}.csv', 'header_prefix': 'meditron_70b', 'model_name': 'meditron70B'},
   # {'file': f'meditron7B_temp_{temp}.csv', 'header_prefix': 'meditron_7b', 'model_name': 'meditron7B'},
   # {'file': f'mellama-70b_temp_{temp}.csv', 'header_prefix': 'mellama-70b', 'model_name': 'mellama-70b'},
    {'file': f'mellama-13b_temp_{temp}.csv', 'header_prefix': 'mellama-13b', 'model_name': 'mellama-13b'}
]

all_results_df_mean = pd.DataFrame()
all_results_df_std = pd.DataFrame()

for model in models:
    print(f"Starting processing for {model['model_name']}...")
    input_file = f'/local/scratch/ydiekma/PatientFacingLLM-Eval/missing_rows/combined_{model["file"]}'
    results_df_mean, results_df_std = calculate_metrics(
        input_file, output_dir, model['model_name'], original_answers_file, model['header_prefix']
    )

    all_results_df_mean = pd.concat([all_results_df_mean, results_df_mean], ignore_index=True)
    all_results_df_std = pd.concat([all_results_df_std, results_df_std], ignore_index=True)

print("\n===== Final DataFrames with BLEU, ROUGE, and BERT Results for All Models =====")
print("Means:")
print(all_results_df_mean)
print("Standard Deviations:")
print(all_results_df_std)

final_output_file_mean = os.path.join(output_dir, f'COMBINED_all_models_metrics_comparisons_mean_temp_{temp}.csv')
final_output_file_std = os.path.join(output_dir, f'COMBINED_all_models_metrics_comparisons_std_temp_{temp}.csv')

all_results_df_mean.to_csv(final_output_file_mean, index=False)
all_results_df_std.to_csv(final_output_file_std, index=False)

