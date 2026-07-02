import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class PeftDefense:
    PEFT_MODEL_PATHS = {
        "qwen2.5": {
            "base": "checkpoint/Qwen2.5-7B-Instruct",
            "direct": "checkpoint/qwen2.5-lora-direct",
            "intent": "checkpoint/qwen2.5-lora-intent",
        },
        # legacy Llama-2 paths (kept for backward compatibility)
        "llama2": {
            "base": "checkpoint/Llama-2-7b-hf",
            "direct": "checkpoint/llama-2-7b-lora-direct",
            "intent": "checkpoint/llama-2-7b-lora-intent",
        },
    }

    def __init__(self, defense_prompt="direct", base_model="qwen2.5", device="cuda") -> None:
        paths = self.PEFT_MODEL_PATHS[base_model]

        print(f"[PeftDefense] Loading base model from {paths['base']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(paths['base'])
        base_model = AutoModelForCausalLM.from_pretrained(
            paths['base'],
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True,
        )
        print(f"[PeftDefense] Loading LoRA weights from {paths[defense_prompt]}...")
        self.model = PeftModel.from_pretrained(base_model, paths[defense_prompt])
        self.model.eval()
        print(f"[PeftDefense] Model loaded successfully.")

        self.device = device

    def __call__(self, msg):
        msg = msg.replace('</s>', '')
        inputs = self.tokenizer(msg, return_tensors='pt')
        with torch.no_grad():
            generate_ids = self.model.generate(
                inputs['input_ids'].to(self.device),
                max_new_tokens=128,
            )
            out = self.tokenizer.batch_decode(
                generate_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return out


# Backward-compatible alias
PeftLlama = PeftDefense
