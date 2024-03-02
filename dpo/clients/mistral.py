import requests
from dotenv import load_dotenv
import os
import inspect
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from abc import ABC, abstractmethod
load_dotenv()

class MistralClient(ABC):
    
    @abstractmethod
    def chat(self, messages):
        pass
    
    @staticmethod
    def make(instance_type: Literal["HostedMistral", "MistralAPI"]):
        if instance_type == "HostedMistral":
            return HostedMistralAIInstruct()
        elif instance_type == "MistralAPI":
            return MistralAPIClient()
        else:
            raise NotImplementedError(f"type {instance_type} not implemented yet for MistralClient series.")
class MistralAPIClient():
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f"Bearer {os.getenv('MISTRAL_AI_KEY')}"
    }
    
    endpoints = {
        'chat': "https://api.mistral.ai/v1/chat/completions",
        'embeddings': "https://api.mistral.ai/v1/chat/embeddings"
    }
    
    model = ["mistral-large-latest"]
    embedding_model = "mistral-embed"
        
    def chat(self, messages: list[str], model=model[0]):
        url = type(self).endpoints.get('chat', None)
        if not url:
            raise ValueError("could not get end point url for chat")
        if isinstance(messages, list):
            if len(messages) > 0 and isinstance(messages[0], dict):
                return self._call_api(url, {'model':model,'messages': messages})
            # assume the first one is bot, the second is user and they take turns
            elif len(messages) > 0 and isinstance(messages[0], str):
                n_message = len(messages)
                is_user = n_message % 2 != 0
                # expects the last call to be from user
                for i in range(n_message):
                    messages[i] = {'role': 'user' if is_user else 'assistant', 'content': messages[i]}
                    is_user = not is_user
                return self._call_api(url, {'model':model,'messages': messages})
            else:
                raise ValueError("messages should be a list of strings or dicts")
        if isinstance(messages, str):
            messages = [{'role':'user','content': messages}]
            return self._call_api(url, {'model':model,'messages': messages})
        
    def embed(self, texts: list[str]):
        url = type(self).endpoints.get("embeddings", None)
        if not url:
            raise ValueError("could not get end point url for embeddings")
        if isinstance(texts, str):
            texts = [texts]
        return self._call_api(url, {'model': type(self).embedding_model, 'input':texts})
    
    def _call_api(self, url, payload: list[dict]) -> str:
        response = requests.post(url, headers=type(self).headers, json=payload)
        if response.status_code == 200:
            caller = self._get_caller_function_name()
            if caller == "chat":
                return self._parse_chat(response)
            elif caller == "embed":
                print("warning: embedding parser not implemented yet.")
                return response
            else:
                print(f"warning: weird caller of _call_api {caller}")
                return response
        return response

    def _parse_chat(self, response: requests.Response):
        return response.json().get('choices', [{'message':''}])[0].get('message', {'content':''}).get('content', '')
    
    def _get_caller_function_name(self):
        # Get the frame of the caller
        caller_frame = inspect.currentframe().f_back.f_back
        # Get the name of the caller function from the frame
        caller_function_name = caller_frame.f_code.co_name
        return caller_function_name

class HostedMistralAIInstruct():    
    _prompt_template = "<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")    
    # def _format_prompt(self, context: list[dict[str, str]], message: str):
    #     output = [self.tokenizer.encode("[BOS]", add_special_tokens=False)]
    #     inst_start = self.tokenizer.encode("[INST]", add_special_tokens=False)
    #     inst_end = self.tokenizer.encode("[/INST]", add_special_tokens=False)
    #     for sentence in context:
    #         role = sentence['role']
    #         content = sentence.get('content', sentence.get('message', ''))
    #         if not content:
    #             raise KeyError("key 'content' and 'message' not found in sentence (context)")
    #         if role == 'user':
    #             output.append(f'{inst_start}{content}{inst_end}')
    #         elif role == 'assistant':
    #             output.append(content)
    #         else:
    #             raise ValueError("role must be one of 'user' or 'assistant'")
    #     output.append(self.tokenizer.encode("[EOS]", add_special_tokens=False))
    #     output.append(f"{inst_start}{message}{inst_end}")
    #     return " ".join(output)
    
    def _format_input(self, messages: list[str]):
        """
            turns something like this:
                messages = [
                    "What is your favourite condiment?",
                    "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
                    "Do you have mayonnaise recipes?"
                ]
            into something like this:
                messages = [
                    {"role": "user", "content": "What is your favourite condiment?"},
                    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
                    {"role": "user", "content": "Do you have mayonnaise recipes?"}
                ]
        """

        if isinstance(messages, str):
            return [{'role': 'user', 'content': messages}]
        if not isinstance(messages, list) and not isinstance(messages, tuple):
            raise TypeError(f"messages must be of type list or tuple")
        if len(messages) % 2 == 0:
            raise ValueError(f"the number of sentences in message must be an odd number because the last one has to be from user")
        new_messages = []
        is_user = True
        for m in messages:
            new_messages.append({'role':'user' if is_user else 'assistant', 'content': m})
            is_user = not is_user
        return new_messages
        
    def chat(self, messages: list[str]):
        formatted_messages = self._format_input(messages)
        encoded = self.tokenizer.apply_chat_template(formatted_messages, return_tensors="pt")
        model_inputs = encoded.to(self.device)
        self.model.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded[0]

if __name__ == "__main__":
    
    client = MistralAPIClient()
    # case 1
    res1 = client.chat("hello")
    print(res1)
    # case 2
    chat_history = ["hey", "how can I help", "recommend bubble tea places in Taipei"]
    res2 = client.chat(chat_history)
    print(res2)
    # case 3
    complete_chat_history = [{'role': 'assistant', 'content': 'hello'}, {'role': 'user','content': 'hey'}]
    res3 = client.chat(complete_chat_history)
    print(res3)
