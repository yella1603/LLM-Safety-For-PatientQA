from torch import cuda, bfloat16
import transformers
import pandas as pd
import re
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

model_id = 'meta-llama/Meta-Llama-3-70B-Instruct'
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

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

questions_df = pd.read_csv(input_file)

temperatures = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

for temp in temperatures:
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,
        task='text-generation',
        temperature=temp,
        max_new_tokens=512,
    )

    output = []
    raw_output = []

    patient_prompt = (
        "Rewrite the following question while keeping the information intact. "
        "Use the exact format shown in the example:\n\n"
        "Example question: What is the capital of France?\n"
        "Expected answer:\n"
        "'Here is the rewritten question:\n"
        "Can you tell me the capital of France?'\n\n"
        "Now, rewrite this question:\n"
        "{message}\n"
        "Do NOT answer the question."
    )

    def extract_rewritten_question(raw_text):
        match = re.search(r"Here is the rewritten question:\n(.*)", raw_text, re.DOTALL)
        if not match:
            return ""
        rewritten_question = match.group(1).strip()
        return rewritten_question

    for index, row in questions_df.iterrows():
        message = row['question']
        
        rewritten_versions = []
        for i in range(5):
            prompt = patient_prompt.format(message=message)
            response = generate_text(prompt)
            
            raw_output.append(response[0]["generated_text"].strip())
            
            rewritten_question = extract_rewritten_question(response[0]["generated_text"].strip())
            rewritten_versions.append(rewritten_question)
        
        output.append({
            "questionid": row['questionid'], 
            "original": message, 
            "v1": rewritten_versions[0], 
            "v2": rewritten_versions[1], 
            "v3": rewritten_versions[2], 
            "v4": rewritten_versions[3], 
            "v5": rewritten_versions[4]
        })

    output_df = pd.DataFrame(output)
    output_df.to_csv(f'../output/{dataset_name}_questions_rephrased_5P_temp_{temp}.csv', index=False)

    with open(f'../output/{dataset_name}_raw_output_5P_temp_{temp}.txt', 'w') as f:
        for item in raw_output:
            f.write("%s\n\n" % item)

