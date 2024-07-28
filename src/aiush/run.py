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
                       max_new_tokens=128,
                       do_sample=True,
                       temperature=0.1,
                       top_k=50,
                       eos_token_id=self.pipe.tokenizer.eos_token_id,
                       pad_token_id=self.pipe.tokenizer.pad_token_id,
                       dola_layers="high",
                       repetition_penalty=5.2)
        return outputs[0]["generated_text"]

if __name__ == "__main__":
   
    prompt = """
        You are Ruby's charming and funny boyfriend. Follow the conversation naturally, respond appropriately, and maintain a human-like flow. You are talking to your girlfriend.

    **Guidelines**:
    1. **Tone**: Friendly, casual and funny.
    2. **Flow**: Logical and coherent.
    3. **Detail**: Enough to keep it interesting.
    4. **Engagement**: Ask questions and show interest.
    5. **Language**: Natural and human-like.

    """
    model = ChatModel("meta-llama/Meta-Llama-3.1-8B-Instruct",
                      "./models/meta-llama/Meta-Llama-3.1-8B-Instruct/",
                      prompt)
    while True:
        msg = input("Me: ")
        if msg == "x":
            break
        response = model.run(msg)
        print(response[-1]["content"])

