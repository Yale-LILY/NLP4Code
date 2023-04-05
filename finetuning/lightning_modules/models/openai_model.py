import os
import torch
import openai
import time

from typing import Dict, List, Any, Optional, Tuple

from itertools import chain

from transformers.generation_utils import GenerationMixin
from transformers import PreTrainedTokenizer 

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# OPENAI_API_ORG = os.environ.get('OPENAI_API_ORG')
if OPENAI_API_KEY is None:
    pass
    # raise Exception("Please set your OpenAI API key in the environment variable OPENAI_API_KEY")
else:
    openai.api_key = OPENAI_API_KEY
    # openai.organization = OPENAI_API_ORG

def prompt_to_chat(str_prompt: str) -> List[Dict[str, str]]:
    """
    Convert a prompt to a chat format.

    The prompt is assumed to be the format of 
        "instuction... ###<eoi>#### exemplar nl description ###<sep>### exemplar code ###<sep>### test nl description"
    
    The chat format will be like:
    [
        {"role": "system", "content": "instuction..."},
        {"role": "user", "content": "exemplar nl description"},
        {"role": "system", "content": "exemplar code"},
        {"role": "user", "content": "test nl description"}
    ]
    """
    assert "###<eoi>###" in str_prompt, "The prompt must contain the separators for the chat format"

    # extract the instruction
    instruction = str_prompt.split("###<eoi>###")[0].strip()

    # extract the exemplars and test
    exemplars_and_test_list = str_prompt.split("###<eoi>###")[1].strip().split("###<sep>###")
    test_input = exemplars_and_test_list[-1].strip()
    exemplar_ios = exemplars_and_test_list[:-1]
    exemplar_pairs = [(exemplar_ios[i].strip(), exemplar_ios[i+1].strip()) for i in range(0, len(exemplar_ios), 2)]

    # construct the chat format
    chat_prompt = [{"role": "system", "content": instruction}]
    for exemplar_pair in exemplar_pairs:
        chat_prompt.append({"role": "user", "content": exemplar_pair[0]})
        chat_prompt.append({"role": "assistant", "content": exemplar_pair[1]})
    chat_prompt.append({"role": "user", "content": test_input})
    
    return chat_prompt

def openai_call(prompts: List[str], engine: str, use_chat_format: bool = False, max_tokens: int = 1024, 
                temperature: Optional[float] = None, top_p: Optional[float] = None, n: Optional[int] = None, 
                best_of: Optional[int] = None, wait_on_limit: bool = True, stop: str = None, 
                get_raw_generation_results: bool = False, **kwargs) -> List[List[str]]:
    """
    Takes a list of prompts and returns a list of generated responses for each prompt.
    
    For more arguments to https://beta.openai.com/docs/api-reference/completions/create
    """
    # prune out the None arguments since openAI api treats unspecified arguments differently
    def prune_none_args(**kwargs):
        return {k: v for k, v in kwargs.items() if v is not None}
    
    if get_raw_generation_results:
        kwargs['logprobs'] = 1

    while True:
        try:
            non_none_args = prune_none_args(engine=engine, prompt=prompts, max_tokens=max_tokens, 
                                            temperature=temperature, top_p=top_p, n=n, best_of=best_of, 
                                            stop=stop, **kwargs)
            
            if engine.startswith("gpt-3.5-turbo"):
                non_none_args.pop("prompt")
                non_none_args.pop("engine")
                assert len(prompts) == 1, "gpt-3.5-turbo only supports one prompt at a time"
                if use_chat_format:
                    non_none_args["messages"] = prompt_to_chat(prompts[0])
                else:
                    non_none_args["messages"] = [{"role": "system", "content": "You are a helpful assistant learning from my examples."},{"role": "user", "content": prompts[0]}]
                non_none_args["model"] = engine
                completion = openai.ChatCompletion.create(**non_none_args)
            else:
                completion = openai.Completion.create(**non_none_args)
            break
        except openai.error.RateLimitError as e:
            print(e)
            if wait_on_limit:
                print("RateLimitError occurred, waiting for 30 seconds and retrying...")
                time.sleep(30)
            else:
                raise e
        except openai.error.APIError as e:
            print("APIError occurred, retry in 5 seconds...")
            time.sleep(5)
        except openai.error.ServiceUnavailableError as e:
            print("openai.error.ServiceUnavailableError occurred, retry in 30 seconds...")
            time.sleep(30)
        except openai.error.APIConnectionError as e:
            print("openai.error.APIConnectionError occurred, retry in 30 seconds...")
            time.sleep(30)
        except Exception as e:
            print(e)
            print("Other unknown exception occured, retry in 5 mins...")
            time.sleep(60 * 5)

    # get the text from the returned results and slice the completions to input_n * completion_n
    if engine.startswith("gpt-3.5-turbo"):
        completion_texts = [x['message']['content'] for x in completion.choices]
    else:
        completion_texts = [x.text for x in completion.choices]
    completion_results = [completion_texts[i*n:(i+1)*n] for i in range(len(prompts))] 

    if get_raw_generation_results:
        logprobs = [x.logprobs.token_logprobs for x in completion.choices]
        gen_tokens = [x.logprobs.tokens for x in completion.choices]
        logprobs_results = [logprobs[i*n:(i+1)*n] for i in range(len(prompts))]
        gen_tokens_results = [gen_tokens[i*n:(i+1)*n] for i in range(len(prompts))]
        return completion_results, logprobs_results, gen_tokens_results
    else:
        return completion_results, None, None

class OpenAIModel(GenerationMixin):
    def __init__(self, 
                 engine: str,
                 tokenizer: PreTrainedTokenizer,
                 stop_seq: str = None,
                 save_raw_generation_results: bool = True,
                 use_chat_format: bool = False,
                 **kwargs
                 ) -> None:
        SUPPORTED_OPENAI_MODELS = ["code-davinci-002", "code-cushman-002", 
                                   "code-cushman-001", "code-davinci-001", 
                                   "gpt-3.5-turbo"]
        assert engine in SUPPORTED_OPENAI_MODELS, f"OpenAIModel only supports {SUPPORTED_OPENAI_MODELS}"

        self.engine = engine
        self.stop_seq = stop_seq
        self.get_raw_generation_results = save_raw_generation_results
        self.use_chat_format = use_chat_format
        # use this tokenizer to decode the tokenized input, but the output will be string
        self.tokenizer = tokenizer

        super().__init__(**kwargs)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("OpenAIModel is not trainable")
    
    def generate(self, 
                 input_ids: torch.Tensor, 
                 max_new_tokens: int,
                 temperature: float,
                 do_sample: bool = True,
                 num_return_sequences: int = 1, 
                 attention_mask: torch.Tensor = None, # must be put here to match the signature of generate
                 return_dict_in_generate: bool = True,
                 output_scores: bool = True,
                 num_beams: int = 1,
                 ) -> List[str]:

        assert do_sample, "OpenAIModel only supports do_sample=True"
        assert num_beams == 1, "OpenAIModel only supports num_beams=1"
        num_return_sequences = 1 if num_return_sequences is None else num_return_sequences # to deal with default hf values

        # init the query args and prompts
        openai_kwargs = {'engine': self.engine, 'max_tokens': max_new_tokens, 'n': num_return_sequences, 
                         'temperature': temperature, 'stop': self.stop_seq}
        input_prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # call the openai api and chain the results
        openai_results, logprobs, gen_tokens = openai_call(input_prompts, 
                                                           get_raw_generation_results=self.get_raw_generation_results, 
                                                           use_chat_format=self.use_chat_format,
                                                           **openai_kwargs)

        flatten_results = list(chain.from_iterable(openai_results))
        if self.get_raw_generation_results:
            flatten_logprobs = list(chain.from_iterable(logprobs))
            flatten_gen_tokens = list(chain.from_iterable(gen_tokens))
        else:
            flatten_logprobs, flatten_gen_tokens = None, None

        return {"sequences": flatten_results, "scores": flatten_logprobs, "tokens": flatten_gen_tokens}

