from abc import ABC, abstractmethod

import yaml

class IRAG(ABC):
    @abstractmethod
    def query(self, prompt: str):
        pass

    def _load_prompt_template(self, prompt_name, template_name = 'template') -> str:
        with open("prompt_template.yaml", 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return data[prompt_name][template_name]
