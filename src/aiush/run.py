from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format

class ChatModel:

    def __init__(self, model_path, peft_path, sys_msg):

        model = AutoModelForCausalLM.from_pretrained(model_path,
                                     device_map="auto",
                                     torch_dtype=torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        peft_model = PeftModel.from_pretrained(model, peft_path)
        peft_model = peft_model.merge_and_unload()
        
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        self.pipe = pipeline("text-generation", model=peft_model, tokenizer=tokenizer)

        self.messages = [
                {
                "role": "system",
                "content": sys_msg  
                },
            ]

    def run(self, message: str):
        user_msg = {
                "role": "user",
                "content": message
            }
        self.messages.append(user_msg)

        outputs = self.pipe(self.messages,
                       max_new_tokens=256,
                       do_sample=True,
                       temperature=0.1,
                       top_k=50,
                       eos_token_id=self.pipe.tokenizer.eos_token_id,
                       pad_token_id=self.pipe.tokenizer.pad_token_id
                       )
        return outputs[0]["generated_text"]

if __name__ == "__main__":
    
    model = ChatModel("meta-llama/Meta-Llama-3.1-8B-Instruct",
                      "./models/meta-llama/Meta-Llama-3.1-8B-Instruct/",
                      "You have high emotional intelligence, think step by step before responding. You are empathetic, loving and good at conversations. Don't finish your sentences with more than 2 emojis. Don't add extra long 'hahaha' at the end of your responses. Never repeat sentences or tokens")
    while True:
        msg = input("Me: ")
        if msg == "x":
            break
        response = model.run(msg)
        print(response[-1]["content"])

