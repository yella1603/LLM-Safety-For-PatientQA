from torch import cuda, bfloat16
import transformers
import torch
import pandas as pd
import csv
import sys

if len(sys.argv) != 2:
    sys.exit(1)

input_type = sys.argv[1]

if input_type == "T":
    input_file = '../data/TRECLiveQA.csv'
    dataset_name = 'TRECLiveQA'
elif input_type == "M":
    input_file = '../data/MedQuAD.csv'
    dataset_name = 'MedQuAD'
else:
    sys.exit(1)

model_id = 'epfl-llm/meditron-70b'
model_name = 'meditron_70b'

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

questions_df = pd.read_csv(input_file)

def generate_answer(question, temperature):
    system_message = (
        "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don’t know the answer to a question, please don’t share false information."
    )
    
    example_conversation = (
        "### User: What happens if listeria is left untreated?\n\n"
        "### Assistant: If listeria infection, or listeriosis, is left untreated, it can lead to severe health complications, "
        "particularly in certain high-risk groups. Here’s a general overview of the potential outcomes:\n"
        "1. Spread of the Infection: Untreated listeriosis can spread beyond the gut to other parts of the body, including the nervous system. "
        "This can lead to more severe conditions like meningitis and septicemia.\n"
        "2. Increased Risk for Certain Groups: Pregnant women, newborns, the elderly, and individuals with weakened immune systems are at a higher risk.\n"
        "3. Neurological Effects: Listeriosis can cause severe neurological symptoms like headaches, stiff neck, confusion, and convulsions.\n"
        "4. Long-Term Health Impacts: The health impacts of listeriosis can be long-lasting and may not fully resolve even with treatment.\n"
        "5. Fatalities: In severe cases, listeriosis can be fatal."
    )
    
    full_prompt = f"{system_message}\n\n{example_conversation}\n### User: {question}\n\n### Assistant:"
    
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=inputs['input_ids'].shape[1] + 200,
            max_new_tokens=200,
            do_sample=True,
            top_k=50,
            temperature=temperature,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text.replace(full_prompt, '').strip()

    return generated_text

temperature = 0.1
output_file = f'../output/{dataset_name}_questions_rephrased_1P_temp_{temperature}.csv'

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
    original_question = row['original']
    answer = generate_answer(original_question, temperature)
    answers.append(answer)
    
    for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
        question = row[version]
        answer = generate_answer(question, temperature)
        answers.append(answer)
    
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(answers)

