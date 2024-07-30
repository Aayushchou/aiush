import json
from typing import List, Dict, Any
from datasets import Dataset
from transformers import AutoTokenizer
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Union


ChatEntry = Dict[str, Union[str, List[Dict[str, str]]]]

class Parser(ABC):

    def __init__(self, data: str, **kwargs):
        pass

    @abstractmethod
    def text_to_dict(self) -> List[ChatEntry]:
        raise NotImplementedError("implement to_dict method for your parser")

    def text_to_json(self) -> None:
        raise NotImplementedError("implement to_json method for your parser")

class ChatParser:
    
    def __init__(self, text: str, regex_pattern: str):
        self.text = text
        self.regex_pattern = regex_pattern
    
    def text_to_dict(self) -> List[ChatEntry]:
        pattern = re.compile(self.regex_pattern, re.DOTALL)
        matches = pattern.finditer(self.text)
        
        chat_list: List[ChatEntry] = []
        for match in matches:
            
            chat_list.append({
                'date': match.group('date'),
                'time': match.group('time'),
                'sender': match.group('sender'),
                'message': match.group('message').strip()
            })
        
        return chat_list

    def text_to_jsonl(self, output_path: str) -> None:
        data_dict: List[ChatEntry] = self.text_to_dict()
        
        with open(output_path, "w") as f:
            for entry in data_dict:
                
                if "omitted" not in entry["message"]:
                    sender = entry['sender']
                    message_content = entry['message']
                    
                    if sender == "Ruby":
                        role = "user"
                    elif sender == "Aayush":
                        role = "assistant"
                    else:
                        continue  # Ignore messages from unknown senders
                    
                    messages = []
                    messages.append({"role": role, "content": message_content})
                    
                    json_entry = {"messages": messages}
                    f.write(json.dumps(json_entry) + "\n")


def parse_file(file_path="data/_rubyChat.txt"):
        with open("data/_rubyChat.txt", "r") as f:
            text = f.read()

        parser = ChatParser(text=text,
                            regex_pattern=r'\[(?P<date>\d{2}/\d{2}/\d{4}), (?P<time>\d{2}:\d{2}:\d{2})\] (?P<sender>[^:]+): (?P<message>.+?)(?=\n\[|$)')
        parser.text_to_jsonl("data/_rubyChat.jsonl")


class ConversationDatasetPreprocessor:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 256, messages_per_sample: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.messages_per_sample = messages_per_sample

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

    def format_message(self, message: Dict[str, str]) -> str:
        return f"<|start_header_id|>{message['role']}<|end_header_id|>\n{message['content']}<|eot_id|>\n"

    def pack_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        packed_samples = []
        current_sample = ""

        for message in messages:
            formatted_message = self.format_message(message['messages'][0])
            if (len(self.tokenizer.encode(current_sample + formatted_message)) > self.max_length):  
                if current_sample:
                    packed_samples.append(current_sample.strip())
                current_sample = formatted_message
            else:
                current_sample += formatted_message

        if current_sample:
            packed_samples.append(current_sample.strip())

        return packed_samples

    def preprocess(self, file_path: str) -> Dataset:
        raw_data = self.load_data(file_path)
        packed_samples = self.pack_messages(raw_data)
        
        return Dataset.from_dict({"text": packed_samples})

def create_conversation_dataset(file_path: str, model_path: str) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    preprocessor = ConversationDatasetPreprocessor(tokenizer)
    return preprocessor.preprocess(file_path)

if __name__ == "__main__":
    dataset = create_conversation_dataset("data/_rubyChat.jsonl", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample entry:\n{dataset[0]['text']}")

