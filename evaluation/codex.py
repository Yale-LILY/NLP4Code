import openai
import os
from typing import List


def generate(input: str, engine: str, max_tokens: int) -> str:
    completion = openai.Completion.create(engine=engine, prompt=input, max_tokens=max_tokens)
    return completion.choices[0].text


def openai_call(input: List[str], engine: str="code-davinci-001", max_tokens: int=256) -> List[str]:
    
    api_key = os.environ.get('OPENAI_API_KEY')

    if api_key is None:
        raise Exception("Please set your OpenAI API key in the environment variable OPENAI_API_KEY")
    else:
        openai.api_key = api_key

    completions = []

    for prompt in input:
        output = generate(input, engine, max_tokens)
        completions.append(output)

    return completions

def codex(input: List[str], engine: str="code-davinci-001", max_tokens: int=256) -> List[str]:
    return openai_call(input, engine, max_tokens)

def gpt3(input: List[str], engine: str="text-davinci-001", max_tokens: int=256) -> List[str]:
    return openai_call(input, engine, max_tokens)