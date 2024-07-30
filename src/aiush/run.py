from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from aiush.preprocess import ConversationDatasetPreprocessor 
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
        self.preprocessor = ConversationDatasetPreprocessor(tokenizer=tokenizer) 
        self.messages = [
                {
                "role": "system",
                "content": sys_msg  
                },
            ]

    def run(self):
        
        outputs = self.pipe(self.messages,
                       max_new_tokens=64,
                       do_sample=True,
                       temperature=0.1,
                       top_k=50,
                       eos_token_id=self.pipe.tokenizer.eos_token_id,
                       pad_token_id=self.pipe.tokenizer.pad_token_id,
                       repetition_penalty=1.2,
                       no_repeat_ngram_size=4)
        return outputs[0]["generated_text"]

if __name__ == "__main__":
   
    prompt = """
    You are a human. Provide realistic and coherent responses. Keep answers concise. No emojis or hahaha"
    """
    model = ChatModel("meta-llama/Meta-Llama-3.1-8B-Instruct",
                      "./models/meta-llama/Meta-Llama-3.1-8B-Instruct/",
                      prompt)
    while True:
        msg = input("Me: ")
        model.messages.append({"role": "user", "content": msg})
        if msg == "x":
            break
        response = model.run()
        model.messages.append({"role": "assistant", "content": response[-1]["content"]})
        print(response[-1]["content"])

