import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HuggingFaceLLM:
    MODEL_PATHS = {
        "qwen2.5-7b": "./checkpoint/Qwen2.5-7B-Instruct",
        "qwen2.5-3b": "./checkpoint/Qwen2.5-3B-Instruct",
        "mistral-7b-v0.2": "./checkpoint/Mistral-7B-Instruct-v0.2",
    }

    def __init__(self, model_name="qwen2.5-7b", device="cuda") -> None:
        self.model_name = model_name
        self.device = device

        print(f"[HuggingFaceLLM] Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATHS[model_name])

        print(f"[HuggingFaceLLM] Loading model to {device} (fp16, low_cpu_mem_usage)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_PATHS[model_name],
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model.eval()
        print(f"[HuggingFaceLLM] Model loaded successfully.")

    def __call__(self, msg):
        inputs = self.tokenizer(msg, return_tensors='pt')
        with torch.no_grad():
            generate_ids = self.model.generate(
                inputs['input_ids'].to(self.device),
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=128,
            )
            out = self.tokenizer.batch_decode(
                generate_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return out
