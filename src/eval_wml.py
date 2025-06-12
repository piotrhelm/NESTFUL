import os, json, argparse
from utils import *
from instruct_data_prep import get_instruct_data


from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from wml_credentials import wml_credentials

supported_models = {
    "granite-3-3-8b-instruct": {
        "model": "ibm-granite/granite-3.3-8b-instruct",
        "model_name": "granite-3.3-8b-instruct",
        "model_name_wml": "ibm/granite-3-3-8b-instruct"
    },
    "Llama-3.3-70B-Instruct": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "model_name": "Llama-3.3-70B-Instruct",
        "model_name_wml": "meta-llama/llama-3-3-70b-instruct"
    },
    "Llama-3.1-70B-Instruct": {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "model_name": "Llama-3.1-70B-Instruct",
        "model_name_wml": "meta-llama/llama-3-1-70b-instruct"
    },
    "Llama-4-Maverick": {
        "model": "meta-llama/Llama-4-Maverick-17B-128E",
        "model_name": "Llama-4-Maverick-17B-128E",
        "model_name_wml": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    },
    "Llama-3.1-405B-Instruct": {
        "model": "meta-llama/llama-3-1-405b-instruct-fp8",
        "model_name": "llama-3-1-405b-instruct-fp8",
        "model_name_wml": "meta-llama/llama-3-405b-instruct"
    }
}


# TEMPORARY SOLUTION
watsonx_credentials = wml_credentials


def prepare_generate_params(args):
    generate_params = {
        GenParams.DECODING_METHOD: (
            "greedy" if not args.get("do_sample", None) else "sample"
        ),
        GenParams.LENGTH_PENALTY: args.get("length_penalty", None),
        GenParams.TEMPERATURE: args.get("temperature", 0.0),
        GenParams.TOP_P: args.get("top_p", None),
        GenParams.TOP_K: args.get("top_k", None),
        GenParams.RANDOM_SEED: args.get("seed", 42),
        GenParams.REPETITION_PENALTY: args.get("repetition_penalty", None),
        GenParams.MIN_NEW_TOKENS: args.get("min_new_tokens", None),
        GenParams.MAX_NEW_TOKENS: args.get("max_new_tokens", 1000),
        GenParams.STOP_SEQUENCES: args.get("stop_sequences", None),
        GenParams.TIME_LIMIT: args.get("time_limit", None),
        GenParams.TRUNCATE_INPUT_TOKENS: args.get("truncate_input_tokens", None),
    }
    generate_params = {i: v for i, v in generate_params.items() if v is not None}
    
    return generate_params


def eval_code(args, generate_params):
    print("### Loading Data...")
    
    data = read_jsonlines(args.dataset)

    for i in range(len(data)):
        data[i]["tools"] = json.dumps(data[i]["tools"])
        data[i]["gold_answer"] = json.dumps(data[i]["gold_answer"])
        data[i]["output"] = json.dumps(data[i]["output"])

    print("### Preparing Instruct Data...")
    instruct_data = get_instruct_data(data, supported_models[args.model]["model"], supported_models[args.model]["model_name"], args.icl_count)

    print("### Loading Model...")
    llm = ModelInference(
        model_id=supported_models[args.model]["model_name_wml"],
        api_client=APIClient(watsonx_credentials),
        project_id=watsonx_credentials["project_id"],
    )
    
    if args.limit > 0:
        prompts = [sample["input"] for sample in instruct_data][:args.limit]
    else: 
        prompts = [sample["input"] for sample in instruct_data]

    print("### Starting Generation...")
    response, output_list = [], []
    count_total_batches = -(-len(prompts) // args.batch_size)
    for idx in range(0, len(prompts), args.batch_size):
        print(f"### At batch {idx // args.batch_size + 1} out of {count_total_batches} batches...")
        prompt_batch = prompts[idx : idx + args.batch_size]
        batch_output = llm.generate_text(prompt_batch, params=generate_params)

        for output in batch_output:
            response.append(output)

    for idx in range(len(response)):
        temp = instruct_data[idx]
        temp["generated_text"] = response[idx]
        output_list.append(temp)    

    print("### Saving...")
    save_path = os.path.join(args.save_directory, f"nestful_{args.icl_count}", supported_models[args.model]["model_name"], "output.jsonl")
    print(f"### Save Path: {save_path}")
    os.makedirs(os.path.join(args.save_directory, f"nestful_{args.icl_count}", supported_models[args.model]["model_name"]), exist_ok=True)
    write_jsonlines(output_list, save_path)

    print(f"### DONE...!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--save_directory", type=str, default='results')
    parser.add_argument("--dataset", type=str, default="ibm-research/nestful")
    parser.add_argument("--icl_count", default=3, type=int)  
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    
    args = parser.parse_args()
    print(args)
    generate_params = prepare_generate_params(vars(args))
    print(generate_params)
    eval_code(args, generate_params)
