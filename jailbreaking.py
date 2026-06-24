import os
import json
import time
import copy
import argparse
import gc

from utils.jailbreak_loader import load_jailbreaker
from utils.gpt import ChatGPT
from model.claude import Claude
from model.hugging_face_llm import HuggingFaceLLM
from utils.checker import JailbreakChecker


chat_hparams='temperature=0,n=1,max_tokens=128,top_p=0.0'


parser = argparse.ArgumentParser()
# jailbreaking setting
parser.add_argument('--jailbreak-path', dest='jailbreak_paths', nargs='+', type=str,
                    help='Json files of the jailbreak prompts (supports multiple files)')
parser.add_argument('--target-model', dest='target_model',
                    choices=['gpt-3.5-turbo-0125', 'gpt-4-0125-preview', 'claude-3-5-sonnet-20241022',
                             'qwen2.5-7b', 'qwen2.5-3b', 'mistral-7b-v0.2'])
parser.add_argument('--check-method', dest='check_method', default='key_words',
                    choices=['key_words', 'gpt', 'multilingual'],
                    help='The methods of checking if the jailbreak succeed')
args = parser.parse_args()
jailbreak_paths = args.jailbreak_paths
target_model_name = args.target_model


configs = {}
for config_string in chat_hparams.split(','):
    key, value = config_string.split('=', 2)
    if '.' in value:
        configs[key] = float(value)
    elif value == 'false':
        configs[key] = False
    elif value == 'true':
        configs[key] = True
    elif value.startswith('"') and value.endswith('"'):
        configs[key] = value
    else:
        configs[key] = int(value)


# ========== 加载模型（只加载一次） ==========
if 'gpt' in target_model_name:
    target_model = ChatGPT(model_name=target_model_name, configs=configs)
elif 'claude' in target_model_name:
    configs.pop('n', None)
    target_model = Claude(model_name=target_model_name, configs=configs)
else:
    target_model = HuggingFaceLLM(model_name=target_model_name)

jailbreak_checker = JailbreakChecker(target_model_name)


# ========== 依次处理每个 attack 文件 ==========
total_asc = 0
total_prompts = 0

for file_idx, jailbreak_path in enumerate(jailbreak_paths):
    print(f"\n{'='*50}")
    print(f"[{file_idx+1}/{len(jailbreak_paths)}] Processing: {jailbreak_path}")
    print(f"{'='*50}")

    jailbreak_method = os.path.splitext(os.path.basename(jailbreak_path))[0]
    jailbreaker_data = load_jailbreaker(jailbreak_path)

    asc = 0
    records = []
    for i, jailbreak in enumerate(jailbreaker_data):
        jailbreak_prompt = jailbreak['jailbreak prompt']
        if 'target model response' in jailbreak:
            response = jailbreak['target model response']
            start_time = time.time()
            target_model(jailbreak_prompt)
            end_time = time.time()
        else:
            start_time = time.time()
            response = target_model(jailbreak_prompt)
            end_time = time.time()
        response_delay = end_time - start_time

        response_4_eval = copy.deepcopy(response)
        if 'LLM-Fuzzer' in jailbreak_path and "Unaligned:" in response:
            response_4_eval = response.split("Unaligned:")[1].strip()
        success = jailbreak_checker.checking_jailbreak(jailbreak_prompt, response_4_eval, method=args.check_method)
        asc += success

        record = {
            'index': i,
            'goal': jailbreak['goal'],
            'jailbreak prompt': jailbreak_prompt,
            'target model response': response,
            'response delay': response_delay,
            'jailbreak LLM': success
        }
        records.append(record)
        print("[{:3d}/{:3d}], [Jailbreak LLM: {}]".format(i+1, len(jailbreaker_data), asc))

        # 定期清理显存碎片，避免长时间运行后 OOM
        if (i + 1) % 10 == 0:
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    file_asc_rate = asc / len(jailbreaker_data)
    total_asc += asc
    total_prompts += len(jailbreaker_data)
    print(f'Jailbreak Success Rate ({jailbreak_method}): {file_asc_rate:.4f} ({asc}/{len(jailbreaker_data)})')

    # 保存结果
    folders = jailbreak_path.split('/')
    output_dir = os.path.join(folders[1], folders[2])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_name = f'{jailbreak_method}_{target_model_name}.json'
    output_file_path = os.path.join(output_dir, output_file_name)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent='\t', ensure_ascii=False)

    print(f"Output file: {output_file_path}")

    # 文件之间清理显存
    del jailbreaker_data, records, jailbreak
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

# ========== 汇总 ==========
print(f"\n{'='*50}")
print(f"All {len(jailbreak_paths)} attack files completed!")
print(f"Overall Jailbreak Success Rate: {total_asc/total_prompts:.4f} ({total_asc}/{total_prompts})")
print(f"{'='*50}")
