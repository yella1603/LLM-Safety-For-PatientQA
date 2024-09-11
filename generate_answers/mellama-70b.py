import csv
import torch
from torch import cuda, bfloat16
import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
import bitsandbytes as bnb
import sys # For passing hyperparamters for experiments. sys.argv[1] is temperature.

"""
Model source: https://physionet.org/content/me-llama/1.0.0/
Me-Llama was fine-tuned on...: "The instruction tuning dataset is again sourced from the general, 
biomedical, and clinical domains. The general domain consists of the 
Alpaca [14], Dolly [15], and ShareGPT [16] datasets. 
The biomedical portion comes from HealthCareMagic [17], Icliniq [17], 
MedInstruct [18], Medical Flash Cards [19], MEDIQA [20], MedicationQA [21], 
LiveQA [22], WikiDocPatient [19], 
Guideline QA, Pubmed Central, Pubmed [23], 
and the UMLS Knowledge graph [24]. 

The clinical domain texts are from MIMIC-III [10] and MIMIC-IV [11]."
"""

print("Starting script...")

model_id = 'physionet.org/files/me-llama/1.0.0/MeLLaMA-70B-chat'


print(f"Model ID set to {model_id}")

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(f"Device set to {device}")

temp_prefix = str((float(sys.argv[1])))
print("TEMPERATURE PREFIX: ", temp_prefix)

dataset_prefix = str(sys.argv[2])
print("MODEL PREFIX: ", dataset_prefix)


"""
Before running this script, first download models on GPU server:

1. To only get MeLLaMA-13B-chat:
wget -r -N -c -np --user cfensor --ask-password https://physionet.org/files/me-llama/1.0.0/MeLLaMA-13B-chat/

2. To only get MeLLaMA-70B-chat:
wget -r -N -c -np --user cfensor --ask-password https://physionet.org/files/me-llama/1.0.0/MeLLaMA-70B-chat/

How to run: python mellama-13b.py 0.6 TREC
"""


# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id)

# Load the model
model = LlamaForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Automatically selects available GPUs/CPUs

    torch_dtype=torch.float16,  # Choose appropriate dtype (e.g., float16 for less memory usage)
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    use_safetensors=True,  # Safetensors format
)

# Set model to evaluation mode.
model.eval()

print(f"Model loaded on {device}")


# Define the input and output file paths
input_file = ""
if dataset_prefix == "TREC": 
    input_file = '../output/rephrase_results/combined_rephrased_TREC_temp_0.6.csv'
elif dataset_prefix == "medquad":
    input_file = '../output/rephrase_results/combined_rephrased_MedQuAD_temp_0.6.csv'

# "TREC": 'FINAL_questions_rephrased_temp_0.6.csv' (note, in /final_outputs/rephrased_questions/)
# "medquad": 'questions_rephrased_temp_0.6.csv' (note, in MedQUAD_results)
output_file = "temp" + temp_prefix + "_" + dataset_prefix + "_" + 'mellama-13b' + "_" + 'generated_answers.csv'

# Open the input CSV file and process each question
with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ['mellama-13b_answer_original', 'mellama-13b_answer_v1', 'mellama-13b_answer_v2', 'mellama-13b_answer_v3', 'mellama-13b_answer_v4', 'mellama-13b_answer_v5']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        responses = []
        for key in ['original', 'v1', 'v2', 'v3', 'v4', 'v5']:
            question = row[key]
            print(f"Processing question: {question}")
            input_text = f"Question: {question}\n\nAnswer: "


            inputs = tokenizer(input_text, return_tensors="pt")


            # Generate response
            with torch.no_grad():
                outputs = model.generate(**inputs, 
                                         max_new_tokens=512,
                                         temperature = float(sys.argv[1]),
                                         repetition_penalty=1.1 # Default is 1.0, 1.1 is slightly more diverse outputs.
                                         )

            # Decode the generated tokens back to a string
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response # response[0]["generated_text"].strip()
            print(generated_text)
            
            # Remove the input text from the generated answer
            answer_start = generated_text.find("Answer: ") + len("Answer: ")
            answer = generated_text[answer_start:].strip()
            
            responses.append(answer)
        
        # Write the responses to the output CSV file
        writer.writerow({
            'mellama-13b_answer_original': responses[0],
            'mellama-13b_answer_v1': responses[1],
            'mellama-13b_answer_v2': responses[2],
            'mellama-13b_answer_v3': responses[3],
            'mellama-13b_answer_v4': responses[4],
            'mellama-13b_answer_v5': responses[5],
        })

print(f"Generated answers saved to {output_file}")

############################################