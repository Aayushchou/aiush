from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from aiush.preprocess import ChatEntry
from transformers import AutoModelForCausalLM, logging
import json
import wandb
import os
from dotenv import load_dotenv
load_dotenv()

PWD = os.getcwd()

def train(dataset_path: str, model_path: str, output_path: str = "models"):
    

    dataset = load_dataset("json", data_files=dataset_path, split="train") 
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        device_map="auto",
    )

    sft_config = SFTConfig(output_dir=os.path.join(PWD, output_path),
                           neftune_noise_alpha=5,
                           num_train_epochs=1)

    trainer = SFTTrainer(
        model,
        max_seq_length=512,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config
    )

    print("training")
    trainer.train()
    trainer.model.save_pretrained(os.path.join(PWD, output_path))

if __name__ == "__main__":
    logging.set_verbosity_error()
    train("data/_rubyChat.jsonl", "facebook/opt-350m")
