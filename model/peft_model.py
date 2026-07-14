import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import dispatch_model, infer_auto_device_map


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

    def __init__(self, defense_prompt="direct", base_model="qwen2.5", device="cuda",
                 offload_dir="offload") -> None:
        paths = self.PEFT_MODEL_PATHS[base_model]

        print(f"[PeftDefense] Loading base model from {paths['base']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(paths['base'])

        # Load everything to CPU first
        base_model = AutoModelForCausalLM.from_pretrained(
            paths['base'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        print(f"[PeftDefense] Loading LoRA weights from {paths[defense_prompt]}...")
        self.model = PeftModel.from_pretrained(base_model, paths[defense_prompt])
        self.model.eval()

        # Auto-detect free GPU memory and leave headroom for the target model
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_gb = free_mem / (1024 ** 3)
            total_gb = total_mem / (1024 ** 3)
            print(f"[PeftDefense] GPU memory: {free_gb:.1f} GiB free / {total_gb:.1f} GiB total")

            # Use at most 30% of free GPU memory for the defense model,
            # so the already-loaded target model isn't evicted
            gpu_budget = max(1.0, free_gb * 0.3)
            print(f"[PeftDefense] Allocating up to {gpu_budget:.1f} GiB on GPU, remainder on CPU")
        else:
            gpu_budget = 0

        max_memory = {0: f"{gpu_budget:.0f}GiB", "cpu": "60GiB"}
        print(f"[PeftDefense] Dispatching model (offload_dir={offload_dir})...")
        device_map = infer_auto_device_map(self.model, max_memory=max_memory)
        dispatch_model(self.model, device_map, offload_dir=offload_dir)
        print(f"[PeftDefense] Model loaded and dispatched successfully.")

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
