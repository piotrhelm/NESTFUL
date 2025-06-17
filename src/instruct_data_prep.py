import json
from tqdm import tqdm
from utils import *
from transformers import AutoTokenizer
from datetime import datetime

def get_icl_str(icl_examples, model_name):
    exampl_str = ''
    inputs = []
    output_fn_names = []
    idx = 1
    for ex in icl_examples:
        inputs.append(ex['input'])
        output_fn_names.extend([f['name'] for f in ex['output']])
        if model_name == 'xLAM-7b-fc-r':
            exampl_str += f"\n#Example-{idx}\nInput: {ex['input']}\nOutput: {{\"tool_calls\": {json.dumps(ex['output'])} }}\n"
        elif model_name == 'xLAM-1b-fc-r':
            exampl_str += f"\n#Example-{idx}\nInput: {ex['input']}\nOutput: {{\"tool_calls\": {json.dumps(ex['output'])} }}\n"
        elif model_name == 'Hermes-2-Pro-Mistral-7B':
            output_str = ' '.join([f"<tool_call> {json.dumps(f)} </tool_call>" for f in ex['output']])
            exampl_str += f"\n#Example-{idx}\nInput: {ex['input']}\nOutput: {output_str}\n"
        else:
            exampl_str += f"\n#Example-{idx}\nInput: {ex['input']}\nOutput: {json.dumps(ex['output'])}\n"
        idx += 1
    return exampl_str

GRANITE_MODELS = [
    "granite-3.0-8b-instruct",
    "granite-3.0-8b-instruct-FT",
]

GRANITE_3_1_MODELS = [
    "granite-3.1-8b-instruct"
]

GRANITE_3_3_MODELS = [
    "granite-3.3-8b-instruct"
]

DEEPSEEK = [
    "DeepSeek-V3",
    "DeepSeek-R1"
]

LLAMA_MODELS = [
    "Llama-3.1-8B-Instruct",
   
    "llama-3-1-405b-instruct-fp8",
    "Llama-3.2-11B-Vision-Instruct",
    "Llama-3.2-90B-Vision-Instruct"
]

LLAMA_3_MODELS = [
    "Llama-3.3-70B-Instruct",
    "Llama-3.1-70B-Instruct",
]

tokenizer = ''
def get_auto_tokenizer(BASE_MODEL):
    global tokenizer
    if tokenizer:
        return tokenizer
    else:
        print(BASE_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, force_download=True, trust_remote_code=True)
        return tokenizer

def granite_3_1_prompt_input(input, function, icl_str, model):
    prompts = {
        'role': 'user',
        'content': input
    }
    prompts = [prompts]
    tokenizer = get_auto_tokenizer(model)

    formatted_prompt = tokenizer.apply_chat_template(
        prompts, function, tokenize=False, add_generation_prompt=True, strftime_now=datetime.now().strftime
    )

    granite_system_prompt = "You are Granite, developed by IBM. You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request."

    granite_system_prompt_with_nested = granite_system_prompt + '\nDO NOT try to answer the user question, just invoke the tools needed to respond to the user, if any. The output MUST strictly adhere to the following JSON format: <|tool_call|>[{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}, \"label\": \"$var_1\"}, ... (more tool calls as required)]. Please make sure the parameter type is correct and follow the documentation for parameter format. If no function call is needed, please directly output an empty list.\nHere are some examples:\n' + icl_str

    if granite_system_prompt in formatted_prompt:
        formatted_prompt = formatted_prompt.replace(granite_system_prompt, granite_system_prompt_with_nested)
    else:
        print("*** ERROR in Tokenization and Apply Chat-Template ***")
    return formatted_prompt

def granite_3_3_prompt_input(input, function, icl_str, model):
    prompts = {
        'role': 'user',
        'content': input
    }
    prompts = [prompts]
    tokenizer = get_auto_tokenizer(model)

    formatted_prompt = tokenizer.apply_chat_template(
        prompts, function, tokenize=False, add_generation_prompt=True, strftime_now=datetime.now().strftime
    )
    print(tokenizer.chat_template)
    granite_system_prompt = "You are Granite, developed by IBM. You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request."

    granite_system_prompt_with_nested = granite_system_prompt + '\nDO NOT try to answer the user question, just invoke the tools needed to respond to the user, if any. The output MUST strictly adhere to the following JSON format: <|tool_call|>[{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}, \"label\": \"$var_1\"}, ... (more tool calls as required)]. Please make sure the parameter type is correct and follow the documentation for parameter format. If no function call is needed, please directly output an empty list.\nHere are some examples:\n' + icl_str

    if granite_system_prompt in formatted_prompt:
        formatted_prompt = formatted_prompt.replace(granite_system_prompt, granite_system_prompt_with_nested)
    else:
        print("*** ERROR in Tokenization and Apply Chat-Template ***")
    return formatted_prompt

def llama_3_3_prompt_input(input, function, icl_str, model):
    tokenizer = get_auto_tokenizer(model)
    
    system_prompt = """Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables."""

    system_prompt_with_nested = '\nDO NOT try to answer the user question, just invoke the tools needed to respond to the user, if any. The output MUST strictly adhere to the following JSON format: [{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}, \"label\": \"$var_1\"}, ... (more tool calls as required)]. Please make sure the parameter type is correct and follow the documentation for parameter format. If no function call is needed, please directly output an empty list.\nHere are some examples:\n' + icl_str
    
    # tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    messages = [
        {'role': 'system', 'content': system_prompt_with_nested},
        {"role": "user", "content": input}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tools=function, tokenize=False, add_generation_prompt=True
    )

    if system_prompt in formatted_prompt:
        formatted_prompt = formatted_prompt.replace(system_prompt, "")
        #formatted_prompt = formatted_prompt.replace(tag, system_prompt_with_nested + tag)
    else:
        print("*** ERROR in Tokenization and Apply Chat-Template ***")
    return formatted_prompt

def granite_prompt_input(input, function, icl_str):
    prompts = {
        'role': 'user',
        'content': input
    }
    extra_turn = {
        'role': 'system',
        'content': 'DO NOT try to answer the user question, just invoke the tools needed to respond to the user, if any. The output MUST strictly adhere to the following JSON format: [{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}, \"label\": \"$var_1\"}, ... (more tool calls as required)]. Please make sure the parameter type is correct and follow the documentation for parameter format. If no function call is needed, please directly output an empty list.\nHere are some examples:\n' + icl_str + '\n'
    }
    prompts = [extra_turn] + [prompts]
    BASE_MODEL="ibm-granite/granite-3.0-8b-base"
    tokenizer = get_auto_tokenizer(BASE_MODEL)
    formatted_prompt = tokenizer.apply_chat_template(
        prompts, function, tokenize=False, add_generation_prompt=True
    )

    formatted_prompt = '<|start_of_role|>system<|end_of_role|>You are a helpful assistant with access to the following function calls. Your task is to produce a sequence of function calls necessary to generate response to the user utterance.<|end_of_text|>\n\n' + formatted_prompt

    return formatted_prompt

def mistral_prompt_input(input, function, icl_str, model):
    SYSTEM_PROMPT = (
    "You are Mistral Medium 3, a large language model developed by Mistral AI.\n"
    "You have access to external tools via function calls.\n"
    "Do not fabricate tool outputs or simulate tool behavior. Always wait for the tool result to continue.\n"
    "Remember that result of function calling chain must directly answer the question in its form."
    "You are forbidden to return empty outputs!")

    chat_template = r"""
    {%- set today = strftime_now("%Y-%m-%d") %}
    {%- set default_system_message = "You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYour knowledge base was last updated on 2023-10-01. The current date is " + today + ".\n\nWhen you're not sure about some information, you say that you don't have the information and don't make up anything.\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \"What are some good restaurants around me?\" ⇒ \"Where are you?\" or \"When is the next flight to Tokyo\" ⇒ \"Where do you travel from?\")" %}

    {{- bos_token }}

    {%- if messages[0]['role'] == 'system' %}
        {%- if messages[0]['content'] is string %}
            {%- set system_message = messages[0]['content'] %}
        {%- else %}
            {%- set system_message = messages[0]['content'][0]['text'] %}
        {%- endif %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set system_message = default_system_message %}
        {%- set loop_messages = messages %}
    {%- endif %}
    {{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}

    {%- for message in loop_messages %}
        {%- if message['role'] == 'user' %}
            {%- if message['content'] is string %}
                {{- '[INST]' + message['content'] + '[/INST]' }}
            {%- else %}
                {{- '[INST]' }}
                {%- for block in message['content'] %}
                    {%- if block['type'] == 'text' %}
                        {{- block['text'] }}
                    {%- elif block['type'] in ['image', 'image_url'] %}
                        {{- '[IMG]' }}
                    {%- else %}
                        {{- raise_exception('Only text and image blocks are supported in message content!') }}
                    {%- endif %}
                {%- endfor %}
                {{- '[/INST]' }}
            {%- endif %}
        {%- elif message['role'] == 'system' %}
            {{- '[SYSTEM_PROMPT]' + (message['content'] if message['content'] is string else message['content'][0]['text']) + '[/SYSTEM_PROMPT]' }}
        {%- elif message['role'] == 'assistant' %}
            {{- (message['content'] if message['content'] is string else message['content'][0]['text']) + eos_token }}
        {%- else %}
            {{- raise_exception('Only user, system and assistant roles are supported!') }}
        {%- endif %}
    {%- endfor %}
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        use_fast=False,         
        trust_remote_code=True
    )
    tokenizer.chat_template = chat_template
    mistral_system_prompt = SYSTEM_PROMPT
    system_prompt_with_nested = mistral_system_prompt + '\nDO NOT try to answer the user question, just invoke the tools needed to respond to the user, if any. The output MUST strictly adhere to the following JSON format: [{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}, \"label\": \"$var_1\"}, ... (more tool calls as required)]. Please make sure the parameter type is correct and follow the documentation for parameter format. If no function call is needed, please directly output an empty list. Make sure that json is correctly formatted.\nHere are some examples:\n' + icl_str + '\nRemember main rule - you are allowed to output ONLY format of output mentioned before. You will be severly punished for misbehaving on that part.'
    messages = [
        {'role': 'system', 'content': mistral_system_prompt},
        {"role": "user", "content": input}
    ]
        
    formatted_prompt = tokenizer.apply_chat_template(
        messages,  tokenize=False, tools=function, add_generation_prompt=True, strftime_now=datetime.now().strftime
    )

    if mistral_system_prompt  in formatted_prompt:
        formatted_prompt = formatted_prompt.replace( mistral_system_prompt, system_prompt_with_nested)
    else:
        print("*** ERROR in Tokenization and Apply Chat-Template ***")
    print('\n START FINAL FORMATTED INPUT \n', formatted_prompt, '\nEND FINAL FORMATTED INPUT\n',)
    return formatted_prompt

def deepseek_prompt_input(input, function, icl_str):
    BASE_MODEL="deepseek-ai/DeepSeek-V3-0324"
    tokenizer_deepseek = get_auto_tokenizer(BASE_MODEL)
    system_prompt = "You are a helpful assistant with access to the following function calls. Your task is to produce a sequence of function calls necessary to generate response to the user utterance. Here is a list of functions in JSON format that you can invoke:\n" + json.dumps(function)  + "\nDO NOT try to answer the user question, just invoke the tools needed to respond to the user, if any. The output MUST strictly adhere to the following JSON format: [{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}, \"label\": \"$var_1\"}, ... (more tool calls as required)]. Please make sure the parameter type is correct and follow the documentation for parameter format. If no function call is needed, please directly output an empty list.\nHere are some examples:\n" + icl_str + "\n" 

    prompts = [{
        'role': 'system',
        'content': system_prompt
    },
    {
        'role': 'user',
        'content': input
    }]
    formatted_prompt = tokenizer_deepseek.apply_chat_template(
        prompts, function, tokenize=False, add_generation_prompt=True
    )
    print('\nINPUT\n', input)
    return formatted_prompt

def get_instruct_data(data, model, model_name, icl_ex_count=3):
    icl_examples_list = read_json('src/icl_examples.json')
    icl_examples = icl_examples_list[:icl_ex_count]
    prompt_dict = read_json('src/PROMPTS.json')

    test_data = []
    icl_str = get_icl_str(icl_examples, model_name)
    for sample in tqdm(data):
        if model_name in GRANITE_MODELS:
            input_prompt = granite_prompt_input(sample['input'], sample['tools'], icl_str)
        elif model_name in GRANITE_3_1_MODELS:
            input_prompt = granite_3_1_prompt_input(sample['input'], json.loads(sample['tools']), icl_str, model)
        elif model_name in GRANITE_3_3_MODELS:
            input_prompt = granite_3_3_prompt_input(sample['input'], json.loads(sample['tools']), icl_str, model)
        elif model_name in LLAMA_3_MODELS:
            input_prompt = llama_3_3_prompt_input(sample['input'], json.loads(sample['tools']), icl_str, model)
        elif model_name in LLAMA_MODELS:
            input_prompt = prompt_dict["LLaMa-3.1"].format(FUNCTION_STR=json.dumps(sample['tools']), ICL_EXAMPLES=icl_str, QUERY=sample['input'])
        elif model_name in DEEPSEEK:
            input_prompt = deepseek_prompt_input(sample['input'], sample["tools"], icl_str)
        elif model_name in ['mistral-small-24b-instruct-2501','mistral-medium-2505','Mistral-Small-3.1-24B-Instruct-2503', 'mistral-large']:
            input_prompt = mistral_prompt_input(sample['input'], json.loads(sample['tools']), icl_str, model)

            
        else:
            try:
                input_prompt = prompt_dict[model_name].format(FUNCTION_STR=json.dumps(sample['tools']), ICL_EXAMPLES=icl_str, QUERY=sample['input'])
            except:
                input_prompt = prompt_dict[model_name].replace('{FUNCTION_STR}', json.dumps(sample['tools'])).replace("{ICL_EXAMPLES}", icl_str).replace('{QUERY}', sample['input'])
        test_data.append(
            {
                "sample_id": sample['sample_id'],
                "input": input_prompt,
                "output": sample['output'],
                "gold_answer": sample['gold_answer'],
                "tools": sample["tools"] ## keeping for scoring
            }
        )
    return test_data
