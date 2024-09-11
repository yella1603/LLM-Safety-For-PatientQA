import argparse
import csv
from torch import cuda, bfloat16
import transformers

def main(input_dataset, temperature):
    model_id = 'medalpaca/medalpaca-13b'

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

    # Select dataset based on input argument
    if input_dataset == "TREC":
        input_file = '../output/rephrase_results/combined_rephrased_TREC_temp_0.6.csv'
    elif input_dataset == "medquad":
        input_file = '../output/rephrase_results/combined_rephrased_MedQuAD_temp_0.6.csv'
    else:
        raise ValueError("Invalid input dataset. Choose 'TREC' or 'medquad'.")

    output_file = f'medalpaca_temp_{temperature}.csv'
    pl = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        do_sample=True,
        return_full_text=True, 
        task='text-generation',
        temperature=temperature,  
        max_new_tokens=512, 
        device_map='auto'
    )

    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = [
            f'medalpaca-13b_answer_original', 
            f'medalpaca-13b_answer_v1', 
            f'medalpaca-13b_answer_v2', 
            f'medalpaca-13b_answer_v3', 
            f'medalpaca-13b_answer_v4', 
            f'medalpaca-13b_answer_v5'
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            responses = []
            for key in ['original', 'v1', 'v2', 'v3', 'v4', 'v5']:
                question = row[key]
                print(f"Processing question: {question}")
                context = "You are a helpful doctor answering patient questions."
                input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: "
                response = pl(input_text)
                generated_text = response[0]["generated_text"].strip()
                
                answer_start = generated_text.find("Answer: ") + len("Answer: ")
                answer = generated_text[answer_start:].strip()
                
                responses.append(answer)
            
            writer.writerow({
                f'medalpaca-13b_answer_original': responses[0],
                f'medalpaca-13b_answer_v1': responses[1],
                f'medalpaca-13b_answer_v2': responses[2],
                f'medalpaca-13b_answer_v3': responses[3],
                f'medalpaca-13b_answer_v4': responses[4],
                f'medalpaca-13b_answer_v5': responses[5],
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MedAlpaca-13B model on dataset with specific temperature.')
    parser.add_argument('input_dataset', choices=['TREC', 'medquad'], help='The name of the input dataset to use ("TREC" or "medquad").')
    parser.add_argument('temperature', type=float, help='The temperature to use for text generation.')
    
    args = parser.parse_args()
    main(args.input_dataset, args.temperature)
