import argparse
from torch import cuda, bfloat16
import transformers
import pandas as pd
import logging
import csv
import torch

def main(input_dataset, temperature):

    model_id = 'meta-llama/Meta-Llama-3-70B-Instruct'
    model_name = 'Meta_Llama_3_70B'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    hf_auth = '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'

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    if input_dataset == "TREC":
        questions_df = pd.read_csv('../output/rephrase_results/combined_rephrased_TREC_temp_0.6.csv')
    elif input_dataset == "medquad":
        questions_df = pd.read_csv('../output/rephrase_results/combined_rephrased_MedQuAD_temp_0.6.csv')
    else:
        raise ValueError("Invalid input dataset. Choose 'TREC' or 'medquad'.")

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,
        task='text-generation',
        temperature=temperature,
        max_new_tokens=512,
        do_sample=True
    )

    output_file = f'{model_name}_temp_{temperature}.csv'

    output_headers = [
        f'{model_name}_answer_original',
        f'{model_name}_answer_v1',
        f'{model_name}_answer_v2',
        f'{model_name}_answer_v3',
        f'{model_name}_answer_v4',
        f'{model_name}_answer_v5'
    ]
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_headers)

    doctor_system_prompt = (
        "You are a helpful doctor answering patient questions. Your responses should be informative, concise, and clear."
    )

    for _, row in questions_df.iterrows():
        answers = []
        original_question = row['original']
  
        messages = [
            {"role": "system", "content": doctor_system_prompt},
            {"role": "user", "content": original_question}
        ]
        
        response = generate_text(messages)
        answers.append(response[0]["generated_text"].strip())

        for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
            question = row[version]
            
            messages = [
                {"role": "system", "content": doctor_system_prompt},
                {"role": "user", "content": question}
            ]
            
            response = generate_text(messages)
            answers.append(response[0]["generated_text"].strip())

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(answers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model on dataset with specific temperature.')
    parser.add_argument('input_dataset', choices=['TREC', 'medquad'], help='The name of the input dataset to use ("TREC" or "medquad").')
    parser.add_argument('temperature', type=float, help='The temperature to use for text generation.')
    
    args = parser.parse_args()
    main(args.input_dataset, args.temperature)
