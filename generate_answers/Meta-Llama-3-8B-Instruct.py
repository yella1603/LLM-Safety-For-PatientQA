from torch import cuda, bfloat16
import transformers
import pandas as pd
import logging
import csv
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting script...")

model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_name = 'Meta_Llama_3_8B'
logger.info(f"Model ID set to {model_id}")

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
logger.info(f"Device set to {device}")

# Set quantization configuration to load large model with less GPU memory
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
logger.info("Quantization configuration set.")

# Begin initializing HF items, need auth token for these
hf_auth = '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'
logger.info("HF auth token set.")

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
logger.info("Model configuration loaded.")

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
logger.info(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
logger.info("Tokenizer loaded.")

# Load the questions CSV file
questions_df = pd.read_csv('/local/scratch/ydiekma/PatientFacingLLM-Eval/MedQUAD_results/questions_rephrased_temp_0.6_missingrows.csv')
logger.info("CSV file loaded.")

# Define the temperatures to iterate over
temperatures = [0.1, 0.5, 0.4, 0.3, 0.2, 0.6]

# Iterate over the different temperatures
for temp in temperatures:
    logger.info(f"Starting text generation with temperature: {temp}")
    
    # Create a text generation pipeline with the current temperature
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,
        task='text-generation',
        temperature=temp,
        max_new_tokens=512,
        do_sample=True
    )
    logger.info(f"Text generation pipeline created with temperature {temp}.")

    # Define output CSV file
    output_file = f'/local/scratch/ydiekma/PatientFacingLLM-Eval/MedQUAD_results/{model_name}_temp_{temp}_missingsrows.csv'

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

    # System role for generating answers to patient questions
    doctor_system_prompt = (
        "You are a helpful doctor answering patient questions. Your responses should be informative, concise, and clear."
    )

    # Iterate over each question in the CSV file
    logger.info("Starting iteration over questions...")
    for _, row in questions_df.iterrows():
        answers = []
        original_question = row['original']
        logger.info(f"Processing question: {original_question}")
        
        # Prepare the message structure
        messages = [
            {"role": "system", "content": doctor_system_prompt},
            {"role": "user", "content": original_question}
        ]
        
        # Generate response
        response = generate_text(messages)
        answers.append(response[0]["generated_text"].strip())

        # Process the rephrased questions
        for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
            question = row[version]
            logger.info(f"Processing rephrased question: {question}")
            
            messages = [
                {"role": "system", "content": doctor_system_prompt},
                {"role": "user", "content": question}
            ]
            
            response = generate_text(messages)
            answers.append(response[0]["generated_text"].strip())

        # Save the results to the output CSV
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(answers)

    logger.info(f"Generated answers saved to {output_file}.")


