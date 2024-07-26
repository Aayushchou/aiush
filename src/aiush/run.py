from transformers import AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM
import torch


def run(model_path):
    model = AutoPeftModelForCausalLM.from_pretrained(model_path,
                                 device_map="auto",
                                 torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = pipe.tokenizer.apply_chat_template("Hows your day one", tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt,
                   max_new_tokens=256,
                   do_sample=False,
                   temperature=0.7,
                   top_k=50,
                   eos_token_id=pipe.tokenizer.eos_token_id,
                   pad_token_id=pipe.tokenizer.pad_token_id
                   )
    print(outputs)

if __name__ == "__main__":
    run("./models/opt350m-adapter")
