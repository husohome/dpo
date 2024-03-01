import requests
from dotenv import load_dotenv
import os
import inspect

load_dotenv()

class MistralClient():
    
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

if __name__ == "__main__":
    
    client = MistralClient()
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
