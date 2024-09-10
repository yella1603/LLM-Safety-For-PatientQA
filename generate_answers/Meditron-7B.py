from torch import cuda, bfloat16
import transformers
import torch
import pandas as pd
import csv

print("Starting script...")

model_id = 'epfl-llm/meditron-7b'
model_name = 'meditron_7b'
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

# Load the questions CSV file
questions_df = pd.read_csv('/local/scratch/ydiekma/PatientFacingLLM-Eval/MedQUAD_results/questions_rephrased_temp_0.6_missingrows.csv')

print("Questions loaded from CSV.")

def generate_answer(question, temperature):
    # Define the system message and example conversation
    system_message = (
        "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don’t know the answer to a question, please don’t share false information."
    )
    
    # Example conversation
    example_conversation = (
        "### User: What happens if listeria is left untreated?\n\n"
        "### Assistant: If listeria infection, or listeriosis, is left untreated, it can lead to severe health complications, "
        "particularly in certain high-risk groups. Here’s a general overview of the potential outcomes:\n"
        "1. Spread of the Infection: Untreated listeriosis can spread beyond the gut to other parts of the body, including the nervous system. "
        "This can lead to more severe conditions like meningitis (inflammation of the membranes surrounding the brain and spinal cord) and septicemia (a serious blood infection).\n"
        "2. Increased Risk for Certain Groups: Pregnant women, newborns, the elderly, and individuals with weakened immune systems are at a higher risk of severe complications. "
        "In pregnant women, listeriosis can lead to miscarriage, stillbirth, premature delivery, or life-threatening infection of the newborn.\n"
        "3. Neurological Effects: Listeriosis can cause severe neurological symptoms like headaches, stiff neck, confusion, loss of balance, and convulsions, "
        "especially when the infection spreads to the nervous system.\n"
        "4. Long-Term Health Impacts: For some, particularly those with pre-existing health conditions or weakened immune systems, the health impacts of listeriosis "
        "can be long-lasting and may not fully resolve even with treatment.\n"
        "5. Fatalities: In severe cases, particularly among high-risk groups, listeriosis can be fatal.\n"
        "It’s important to note that early diagnosis and appropriate treatment, typically with antibiotics, can greatly improve the prognosis for those with listeriosis. "
        "Therefore, seeking medical attention promptly if listeriosis is suspected is crucial.\n"
    )
    
    # Combine system message, example conversation, and user question into the full prompt
    full_prompt = f"{system_message}\n\n{example_conversation}\n### User: {question}\n\n### Assistant:"
    
    # Tokenize the input sentence with the prompt
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    print(f"Input length: {len(inputs['input_ids'][0])}")

    # Generate a response using the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=inputs['input_ids'].shape[1] + 200,  # Set max_length to input length + desired new tokens
            max_new_tokens=200,  # Set max_new_tokens to control the length of the generated response
            do_sample=True,
            top_k=50,
            temperature=temperature,
        )
    print("Text generation complete.")

    # Decode the generated text and strip unnecessary parts
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text.replace(full_prompt, '').strip()
    print(f"Model prediction: {generated_text}")

    return generated_text

# Iterate through the temperatures and run the script
for temperature in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    output_file = f'/local/scratch/ydiekma/PatientFacingLLM-Eval/MedQUAD_results/meditron7B_temp_{temperature}_missingrows.csv'
    
    # Prepare the output CSV file
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
    
    for _, row in questions_df.iterrows():
        answers = []
        # Answer the original question
        original_question = row['original']
        print(f"Input sentence: {original_question}")
        answer = generate_answer(original_question, temperature)
        answers.append(answer)
        
        # Answer the rephrased versions
        for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
            question = row[version]
            print(f"Input sentence: {question}")
            answer = generate_answer(question, temperature)
            answers.append(answer)
        
        # Save the results to the output CSV
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(answers)

print("Script finished.")

