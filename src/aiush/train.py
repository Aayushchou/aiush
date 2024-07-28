from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
from peft import LoraConfig
from aiush.preprocess import ChatEntry
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, BitsAndBytesConfig
import json
import torch
import wandb
import os
from dotenv import load_dotenv
load_dotenv()

PWD = os.getcwd()

def train(dataset_path: str, model_path: str, output_path: str = "models"):
    
    """set up output path"""    
    output_path = os.path.join(PWD, output_path, model_path)
    os.makedirs(output_path, exist_ok=True)

    """load dataset"""
    dataset = load_dataset("json", data_files=dataset_path, split='train')
    
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    """setup quantization config"""
    bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )

    """load model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

   
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
            target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(output_dir=os.path.join(PWD, output_path),
                           neftune_noise_alpha=5,
                           num_train_epochs=1,
                           logging_steps=100,
                           bf16=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=256,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=sft_config,
        peft_config=peft_config,
        dataset_kwargs = {
            "add_special_tokens": False,
            "append_concat_token": False
        }
    )

    print("training")
    trainer.train()

    trainer.model.save_pretrained(os.path.join(PWD, output_path))

if __name__ == "__main__":
    logging.set_verbosity_error()
    train("data/_rubyChat.jsonl", "meta-llama/Meta-Llama-3.1-8B-Instruct")
