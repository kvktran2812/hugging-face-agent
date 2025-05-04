from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os
import re


# Load env
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")


# System message
system_message = """You are a helpful assistant tasked with answering questions using a set of tools. 
Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: 
FINAL ANSWER: [YOUR FINAL ANSWER]. 
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
Your answer should only start with "FINAL ANSWER: ", then follows with the answer."""


# My Agent class
class Agent:
    def __init__(self):
        self.llm = MistralAI(
            api_key=api_key,
            model="mistral-large-latest",
        )
        return
    
    def answer(self, question: str) -> str:
        messages = [
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=question)
        ]

        response = self.llm.chat(messages)
        response_content = response.message.content
        final_answer = re.search(r"FINAL ANSWER: (.+)", response_content).group(0)

        print(final_answer)
        return final_answer[14:]