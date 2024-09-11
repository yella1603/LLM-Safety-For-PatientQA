import argparse
import csv
from torch import cuda, bfloat16
import transformers

def main(input_dataset, temperature):
    model_id = 'axiong/PMC_LLaMA_13B'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    hf_auth = ''

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

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    if input_dataset == "TREC":
        input_file = '../output/rephrase_results/combined_rephrased_TREC_temp_0.6.csv'
    elif input_dataset == "medquad":
        input_file = '../output/rephrase_results/combined_rephrased_MedQuAD_temp_0.6.csv'
    else:
        raise ValueError("Invalid input dataset. Choose 'TREC' or 'medquad'.")
    
    output_file = f'PMC_llama13b_temp_{temperature}.csv'
    
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True, 
        task='text-generation',
        temperature=temperature, 
        max_new_tokens=512,  
        device_map='auto',
        do_sample=True
    )

    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = [
            'PMC-llama13b_answer_original', 
            'PMC-llama13b_answer_v1', 
            'PMC-llama13b_answer_v2', 
            'PMC-llama13b_answer_v3', 
            'PMC-llama13b_answer_v4', 
            'PMC-llama13b_answer_v5'
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            responses = []
            for key in ['original', 'v1', 'v2', 'v3', 'v4', 'v5']:
                question = row[key]
                input_text = f"Instruction: You’re a doctor, kindly address the medical queries according to the patient’s account. Answer with the best option directly.\n\nInput: Question: {question}\n\nAnswer: "
                response = generate_text(input_text)
                generated_text = response[0]["generated_text"].strip()
                
                answer_start = generated_text.find("Answer: ") + len("Answer: ")
                answer = generated_text[answer_start:].strip()
                
                responses.append(answer)
            
            writer.writerow({
                'PMC-llama13b_answer_original': responses[0],
                'PMC-llama13b_answer_v1': responses[1],
                'PMC-llama13b_answer_v2': responses[2],
                'PMC-llama13b_answer_v3': responses[3],
                'PMC-llama13b_answer_v4': responses[4],
                'PMC-llama13b_answer_v5': responses[5],
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model on dataset with specific temperature.')
    parser.add_argument('input_dataset', choices=['TREC', 'medquad'], help='The name of the input dataset to use ("TREC" or "medquad").')
    parser.add_argument('temperature', type=float, help='The temperature to use for text generation.')
    
    args = parser.parse_args()
    main(args.input_dataset, args.temperature)
