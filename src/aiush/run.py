from transformers import AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM
import torch


def run(model_path):
    model = AutoPeftModelForCausalLM.from_pretrained(model_path,
                                 device_map="auto",
                                 torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = [
            {
            "role": "system",
            "content": "You are a loving boyfriend of your girlfriend Ruby, respond in a friendly and familiar way."
            },
            {"role": "user",
             "content": "What do you wanna eat poopoo"
             }
            ]
    outputs = pipe(prompt,
                   max_new_tokens=256,
                   do_sample=True,
                   temperature=0.7,
                   top_k=50,
                   eos_token_id=pipe.tokenizer.eos_token_id,
                   pad_token_id=pipe.tokenizer.pad_token_id
                   )
    print(outputs)

if __name__ == "__main__":
    run("./models/opt350m-adapter")
