import json
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


if __name__ == "__main__":
    parse_file()
