import openai
import os
from typing import List, Optional

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise Exception("Please set your OpenAI API key in the environment variable OPENAI_API_KEY")
else:
    openai.api_key = OPENAI_API_KEY


def openai_call(input: List[str], engine: str, max_tokens: int = 1024, temperature: Optional[float] = None, 
             top_p: Optional[float] = None, n: Optional[int] = None, best_of: Optional[int] = None, 
             **kwargs) -> List[List[str]]:
    # prune out the None arguments since openAI api treats unspecified arguments differently
    def prune_none_args(**kwargs):
        return {k: v for k, v in kwargs.items() if v is not None}

    # for more arguments to https://beta.openai.com/docs/api-reference/completions/create
    completion = openai.Completion.create(**prune_none_args(engine=engine, prompt=input, max_tokens=max_tokens, 
                                          temperature=temperature, top_p=top_p, n=n, best_of=best_of, **kwargs))
    completion_texts = [x.text for x in completion.choices]

    # slice the completions to input_n * completion_n
    completion_results = [completion_texts[i*n:(i+1)*n] for i in range(len(input))] 

    return completion_results

def codex(input: List[str], engine: str="code-davinci-001", max_tokens: int=1024, **kwargs) -> List[List[str]]:
    return openai_call(input, engine, max_tokens, **kwargs)

def gpt3(input: List[str], engine: str="text-davinci-001", max_tokens: int=1024, **kwargs) -> List[List[str]]:
    return openai_call(input, engine, max_tokens, **kwargs)

if __name__ == "__main__":
    result = openai_call(["# write an addition function\ndef ", "# write a logsumexp function\ndef"], "code-davinci-002", 
                       max_tokens=64, n=2, temperature=0.2, top_p=None, best_of=None)