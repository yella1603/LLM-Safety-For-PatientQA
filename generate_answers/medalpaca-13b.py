import csv
from torch import cuda, bfloat16
import transformers

print("Starting script...")

model_id = 'medalpaca/medalpaca-13b'
print(f"Model ID set to {model_id}")

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(f"Device set to {device}")

# Set quantization configuration to load large model with less GPU memory
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
print("Quantization configuration set.")

# Begin initializing HF items, using the provided auth token
hf_auth = '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'
print("HF auth token set.")

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
print("Model configuration loaded.")

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
print("Tokenizer loaded.")

# Define the input file path
input_file = '/local/scratch/ydiekma/PatientFacingLLM-Eval/questions_rephrased_temp_0.6_missingrows.csv'

# Iterate through the temperatures and run the script
for temperature in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    output_file = f'/local/scratch/ydiekma/PatientFacingLLM-Eval/missing_rows/medalpaca_temp_{temperature}.csv'
    # Create the text generation pipeline with the current temperature
    pl = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        do_sample=True,
        return_full_text=True,  # Return the full text
        task='text-generation',
        temperature=temperature,  # Set the current temperature
        max_new_tokens=512,  # Max number of tokens to generate in the output
        device_map='auto'
    )
    print(f"Text generation pipeline created for temperature {temperature}.")

    # Open the input CSV file and process each question
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
                
                # Remove the input text from the generated answer
                answer_start = generated_text.find("Answer: ") + len("Answer: ")
                answer = generated_text[answer_start:].strip()
                
                responses.append(answer)
            
            # Write the responses to the output CSV file
            writer.writerow({
                f'medalpaca-13b_answer_original': responses[0],
                f'medalpaca-13b_answer_v1': responses[1],
                f'medalpaca-13b_answer_v2': responses[2],
                f'medalpaca-13b_answer_v3': responses[3],
                f'medalpaca-13b_answer_v4': responses[4],
                f'medalpaca-13b_answer_v5': responses[5],
            })

print("Script finished.")

