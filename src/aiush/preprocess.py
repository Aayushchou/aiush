import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict


ChatEntry = Dict[str, str] 

class Parser(ABC):

    def __init__(self, data: str, **kwargs):
        pass

    @abstractmethod
    def text_to_dict(self) -> List[ChatEntry]:
        raise NotImplementedError("implement to_dict method for your parser")

    def text_to_json(self) -> None:
        raise NotImplementedError("implement to_json method for your parser")

class TextParser(Parser):

    def __init__(self,
                 data: str,
                 regex_pattern:str, **kwargs):
        self.data = data
        self.regex_pattern = re.compile(regex_pattern, re.DOTALL) 

    def text_to_dict(self):
        matches = self.regex_pattern.finditer(self.data)

        chat_list = []
        for match in matches:
            chat_list.append({
            'date': match.group('date'),
            'time': match.group('time'),
            'sender': match.group('sender'),
            'message': match.group('message').strip()
        })
        return chat_list

    def text_to_json(self, output_path: str) -> None:
        data_dict: List[ChatEntry] = self.to_dict()
        
        with open(output_path, "w") as f:
            json.dump(data_dict, f)


if __name__ == "__main__":
    with open("data/_rubyChat.txt", "r") as f:
        text = f.read()

    parser = TextParser(data=text,
                        regex_pattern=r'\[(?P<date>\d{2}/\d{2}/\d{4}), (?P<time>\d{2}:\d{2}:\d{2})\] (?P<sender>[^:]+): (?P<message>.+?)(?=\n\[|$)')
    parser.text_to_json("data/_rubyChat.json")

