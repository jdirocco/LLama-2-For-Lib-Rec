import pandas as pd
import random
from transformers import AutoTokenizer, pipeline
import torch
import ast
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create a new directory with the current date
output_dir = f'./output/{current_date}'
os.makedirs(output_dir, exist_ok=True)

# Load the parsed data
data = pd.read_csv('./data/processed/parsed_data.csv')

# Select a random project from the dataset
random_project_index = random.randint(0, len(data) - 1)
random_project_details = data.iloc[random_project_index]

# Extracting details for the random project
project_name = random_project_details['project']
project_description = random_project_details['readme']
existing_dependencies = random_project_details['dependencies']


# Convert list of dependencies to formatted string as expected in the prompt
formatted_dependencies = ', '.join(ast.literal_eval(existing_dependencies))
# All depencdeices must be from new line
formatted_dependencies = formatted_dependencies.replace(", ", "\n- ")
dynamic_prompt = f"""<s>[INST] <<SYS>>
As an AI specializing in software library recommendations for Java applications, provide recommendations exclusively in the Maven format: groupId:artifactId:version. Here is an example of how recommendations should be formatted:
# Example 1
Project: 'Project A'
Description: 'A web application for managing tasks.'
Existing Dependencies: 
- org.springframework.boot:spring-boot-starter-web:2.5.4
Recommended Libraries:
- org.springframework.boot:spring-boot-starter-data-jpa:2.5.4
- org.springframework.boot:spring-boot-starter-security:2.5.4

# Example 2
Project: 'Project B'
Description: 'A command-line tool for processing text files.'
Existing Dependencies: 
- commons-cli:commons-cli:1.4
Recommended Libraries:
- org.apache.commons:commons-io:2.8.0
- org.apache.commons:commons-lang3:3.12.0

Your tasks are to:
- Provide recommendations exclusively in the Maven format as shown in the example.
- Ensure recommendations are suitable for the project context described. Focus solely on suggesting libraries that could enhance the project's capabilities or performance.
- Consider the project's context and existing dependencies to suggest libraries that complement or enhance the current setup.
- Ignore any code dont write any code.
<</SYS>>

# Project Description
The project, '{project_name}', is described as follows:
{project_description}

# Existing Dependencies
The project's Maven dependencies include:
- {formatted_dependencies}

# Library Recommendation Request 
.
Based on the context of the '{project_name}' project, identify libraries to aid in the development of Java applications that are compatible with the existing dependencies.Please provide a brief explanation for each recommended library, specifying how it could benefit the project in addressing the mentioned needs or enhancing specific functionalities. Recommendations must be in the Maven format and focus on enhancing the project's capabilities or addressing any identified needs. Provide recommendations exclusively in the Maven format. Exmaple: com.fasterxml.jackson.core:jackson-databind:2.9.8
[/INST]</s>
"""
# print(dynamic_prompt)

hf_token = os.getenv("HF_TOKEN")
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device=0 if torch.cuda.is_available() else -1,
    token=hf_token
)

sequences = llama_pipeline(
    dynamic_prompt,
    do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        # max_length=250,  # Increased max_length
        truncation=True,  # Enable truncation
        # Alternatively, specify max_new_tokens to control the length of the generated portion specifically
        max_new_tokens=800  # This would generate up to 150 new tokens beyond the prompt
    )

    # Print the generated recommendations
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

    # Save each generated recommendation to a separate file in the new directory
for seq in sequences:
    current_time = datetime.now().strftime('%H-%M-%S')
    recommendations_file_path = f'{output_dir}/recommendation_{current_time}.txt'
    with open(recommendations_file_path, 'w', encoding='utf-8') as file:
        file.write(seq['generated_text'] + '\n')
