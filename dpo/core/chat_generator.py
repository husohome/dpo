
from dpo.clients import MistralClient
from loguru import logger
import random
from typing import Literal

logger.add("_chat_generator_error.log", rotation="500MB")

class MeetingDialog():
    
    def __init__(
        self,
        num_participants: int = 3,
        talk_turns_before_intervention: int = 6,
        topic="a hackathon project using generative AI",
        client_type: Literal["HostedMistral", "MistralAPI"] = "HostedMistral"
    ):  

        self.participants = [MistralClient.make(client_type) for _ in range(num_participants)]
        self.moderator = MistralClient.make(client_type)
        self.baseline = MistralClient.make(client_type)
            
        self.history = []
        self.talk_turns = talk_turns_before_intervention
        self.topic = topic
    
    def run(self, instruction: str = ""):
        action_candidates = ['ask', 'disagree', 'agree', 'be_rude', 'digress', 'answer']
        weights = [5,7,15,2,6,8]
        actions = random.choices(action_candidates, weights, k=self.talk_turns)
        if instruction:
            self._attach_instruction(self.moderator, instruction)
        self.history = [self.start_conversation(self.moderator, self.topic)]

        for action_name in actions:
            speaker = random.choice(self.participants)
            action = getattr(self, action_name)
            try:
                result = action(speaker, self.history[-1])
                self.history.append(result)
            except:
                logger.error(f"action call failed.")
        try:
            self.intervene(self.moderator, ";".join(self.history))
        except:
            logger.error(f"intervene call failed.")
        return self.history
    
    def _attach_instruction(self, moderator, instruction: str):
        if instruction:
            moderator.instruction = instruction
    
    def start_conversation(self, moderator, topic):
        return moderator.chat(f"{getattr(moderator, 'instruction', '')} you are the moderator of a meeting about {topic}. Now introduce yourself and start the meeting.")
    
    def intervene(self, moderator, conversation):
        return moderator.chat(f"{getattr(moderator, 'instruction', '')} you are the moderator of a meeting: given the previous conversation: {conversation}, reply in a way that's appropriate as a moderator making sure everyone's comfortable.")
    
    def ask(self, client, statement):
        return client.chat(f"ask one question about the statement '{statement}'")
    
    def disagree(self, client, statement):
        return client.chat(f"raise a potential disagreement against the statement '{statement}'")
    
    def agree(self, client, statement):
        return client.chat(f"raise a supporting point to the statement '{statement}'")
    
    def digress(self, client, statement):
        return client.chat(f"respond to topic that's irrelevant to the statement '{statement}' and start a different conversation.")
    
    def be_rude(self, client, statement):
        return client.chat(f"respond to the statement '{statement}' in a rude way.")
    
    def answer(self, client, statement):
        return client.chat(f"say something about the statement or topic: '{statement}'")


if __name__ == "__main__":
    dialog = MeetingDialog()
    history = dialog.run()
    print(history)