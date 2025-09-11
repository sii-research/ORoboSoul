from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import os
import re

def str_to_list(content):
    content = content.strip()
    if not (content.startswith('[') and content.endswith(']')):
        raise ValueError("Invalid list format: missing brackets")
    
    content = content[1:-1].strip()
    if not content:
        return []
    
    elements = []
    current = 0
    depth = 0
    start = 0
    
    while current < len(content):
        if content[current] == '(':
            depth += 1
        elif content[current] == ')':
            depth -= 1
        elif content[current] == ',' and depth == 0:
            elements.append(content[start:current].strip())
            start = current + 1
        current += 1
    
    elements.append(content[start:current].strip())
    
    elements = [re.sub(r'^[\'"](.*)[\'"]$', r'\1', elem) for elem in elements]
    return elements

def get_plan(text):
    pattern = r'<plan>(.*?)</plan>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            content = match.group(1)
            content_list = str_to_list(content.replace('\n', ''))
            return content_list
        except Exception as e:
            return f"Parsing error: {str(e)}"
    else:
        return "no plan extracted"


def inference_pipeline(model_name_or_path, scene_graph_path, task, prompt_path):
    pipe = pipeline(
        "text-generation",
        model=model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        model_kwargs={"attn_implementation": "flash_attention_2"}
    )
    
    with open(scene_graph_path, 'r') as file:
        scene_graph = json.load(file)
        
    with open(prompt_path, 'r') as file:
        prompt = file.read()
    
    input_content = f"{prompt}\n<Instruction>{task}</Instruction>\n<SceneGraph>{scene_graph}</SceneGraph>"
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': input_content}
    ]
    
    formatted_input = pipe.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    outputs = pipe(
        formatted_input,
        max_new_tokens=5000,
        return_full_text=False 
    )
    
    response = outputs[0]['generated_text']
    plan = get_plan(response)
    return response, plan

if __name__ == "__main__":
    model_name_or_path = 'RoPLSii/decision-making-e2e-0.0'
    scene_graph_path = 'inference_example\scene_graphs\scene_graph_1.json'
    task = 'Pick up a drinking glass from dining_table in kitchen, fill it at sink, and hand it to person in kitchen.'
    prompt_path = 'inference_example\prompt.txt'
    response, plan = inference_pipeline(
        model_name_or_path=model_name_or_path,
        scene_graph_path=scene_graph_path,
        task=task,
        prompt_path=prompt_path,
    )
