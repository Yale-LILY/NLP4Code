import torch
import io, tokenize, re
import ast, astunparse

from typing import Tuple, Optional, List, Union, Dict, Any

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer, GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPTJForCausalLM
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from transformers import BloomForCausalLM, BartForSequenceClassification
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import CodeGenTokenizer, CodeGenForCausalLM, T5Tokenizer
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import LlamaTokenizer, LlamaForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast

from transformers.generation_utils import GenerationMixin

from finetuning.lightning_modules.models.openai_model import OpenAIModel

def is_model_gpt_style(name: str) -> bool:
    if "t5" in name or "bert" in name or "tapex" in name or "openai" in name:
        return False
    else:
        return True

def is_encoder_only_model(name: str) -> bool:
    if "bert" in name.lower():
        # this covers bert, roberta, deberta
        return True
    elif "bart" in name.lower():
        # we only use the seq classification version of bart
        return True
    else:
        return False

def get_model(model_name: str, 
              tokenizer_only: bool = False,
              gradient_ckpt: bool = False,
              additional_special_tokens: Optional[List[str]] = None,
              additional_init_args: Dict[str, Any] = {}) -> Tuple[GenerationMixin, PreTrainedTokenizer]:
    if additional_special_tokens is None:
        additional_special_tokens = []
    assert len(additional_special_tokens) == 0, f"support for additional tokens has been removed"
    assert not gradient_ckpt, f"gradient checkpointing is not supported"

    if not tokenizer_only:
        print(f"using pretrained model: {model_name}, gradient_ckpt: {gradient_ckpt}")

    if model_name == "microsoft/CodeGPT-small-py":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, additional_special_tokens=additional_special_tokens)
        if not tokenizer_only:
            model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))
    elif model_name == "EleutherAI/gpt-j-6B":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        if not tokenizer_only:
            model = GPTJForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id,
                                                        gradient_checkpointing=gradient_ckpt, use_cache=not gradient_ckpt)
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))
    elif model_name in ["EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-2.7B"]:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, additional_special_tokens=additional_special_tokens)
        tokenizer.pad_token = tokenizer.eos_token

        if not tokenizer_only: 
            model = GPTNeoForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id, 
                                                    gradient_checkpointing=gradient_ckpt, use_cache=not gradient_ckpt)
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))
    elif model_name.startswith("Salesforce/codet5-"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                 additional_special_tokens=additional_special_tokens)

        if not tokenizer_only:
            model = T5ForConditionalGeneration.from_pretrained(model_name, 
                                                               # gradient_checkpointing=gradient_ckpt, 
                                                               use_cache=not gradient_ckpt,
                                                               **additional_init_args)
    elif model_name.startswith("Salesforce/codegen-"):
        tokenizer = CodeGenTokenizer.from_pretrained(model_name,
                                                    additional_special_tokens=additional_special_tokens)
        tokenizer.pad_token = tokenizer.eos_token

        if not tokenizer_only:
            model = CodeGenForCausalLM.from_pretrained(model_name, 
                                                    pad_token_id=tokenizer.eos_token_id, 
                                                    torch_dtype=torch.float16, 
                                                    # device_map="auto",
                                                    use_cache=True)
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))
    elif model_name.startswith("bigscience/bloom-"):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    additional_special_tokens=additional_special_tokens)

        if not tokenizer_only:
            model = BloomForCausalLM.from_pretrained(model_name,
                                                    pad_token_id=tokenizer.eos_token_id,
                                                    use_cache=not gradient_ckpt)
            if gradient_ckpt:
                model._set_gradient_checkpointing(gradient_ckpt)
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))
    elif model_name.startswith("facebook/incoder"):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    additional_special_tokens=additional_special_tokens)
        tokenizer.bos_token_id = 0
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 2

        # tokenizer.decode([0, 1, 2, 56], skip_special_tokens=True)

        if not tokenizer_only:
            if model_name.endswith("6B"):
                model = AutoModelForCausalLM.from_pretrained(model_name, revision="float16", torch_dtype=torch.float16, use_cache=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True)

    elif model_name.startswith("t5-") or model_name.startswith("google/t5-"):
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        if not tokenizer_only:
            model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                            #    gradient_checkpointing=True,
                                                            #    torch_dtype=torch.float16
                                                               )
                                                    
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))
    elif model_name.startswith("facebook/bart-"):
        tokenizer = BartTokenizer.from_pretrained(model_name)

        if not tokenizer_only:
            model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif "llama" in model_name.lower() or "alpaca" in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_name,
                                                    additional_special_tokens=additional_special_tokens)
        tokenizer.pad_token = tokenizer.eos_token

        if not tokenizer_only:
            model = LlamaForCausalLM.from_pretrained(model_name, 
                                                    pad_token_id=tokenizer.eos_token_id, 
                                                    torch_dtype=torch.float16)
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))
    elif "santacoder" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    additional_special_tokens=additional_special_tokens)
        tokenizer.pad_token = tokenizer.eos_token

        if not tokenizer_only:
            model = AutoModelForCausalLM.from_pretrained(model_name,        
                                                        pad_token_id=tokenizer.eos_token_id, 
                                                        torch_dtype=torch.float16,
                                                        trust_remote_code=True)
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))


    elif model_name.startswith("openai/"):
        engine = model_name.split("/")[-1]

        tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # to accomandate the length of openai models and the prompt
        if engine in ["code-davinci-002"]:
            model_length = 8001
        elif engine in ["code-cushman-001", "code-cushman-002"]:
            model_length = 1024
        elif engine in ["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo"]:
            model_length = 4096

        tokenizer.model_max_length = model_length
        tokenizer.max_len_single_sentence = model_length
        tokenizer.max_len_sentences_pair = model_length
        tokenizer.truncation_side = "left"

        if not tokenizer_only: 
            model = OpenAIModel(engine=engine, tokenizer=tokenizer, **additional_init_args)
    else:
        print(f"unknown model: {model_name}")
        raise NotImplementedError

    if tokenizer_only:
        return None, tokenizer
    else:
        return model, tokenizer

def right_pad_sequences(sequences: List[torch.Tensor], batch_first: bool = True, padding_value: Union[int, bool] = 0, 
                       max_len: int = -1, device: torch.device = None) -> torch.Tensor:
    assert all([len(seq.shape) == 1 for seq in sequences])
    max_len = max_len if max_len > 0 else max(len(s) for s in sequences)
    device = device if device is not None else sequences[0].device

    padded_seqs = []
    for seq in sequences:
        # print(padding_value)
        new = torch.full((max_len - seq.shape[0],), padding_value, dtype=torch.long).to(device)
        padded_seqs.append(torch.cat((seq, new)))
    return torch.stack(padded_seqs)

def left_pad_sequences(sequences: List[torch.Tensor], batch_first: bool = True, padding_value: Union[int, bool] = 0, 
                       max_len: int = -1, device: torch.device = None) -> torch.Tensor:
    assert all([len(seq.shape) == 1 for seq in sequences])
    max_len = max_len if max_len > 0 else max(len(s) for s in sequences)
    device = device if device is not None else sequences[0].device

    padded_seqs = []
    for seq in sequences:
        # print(padding_value)
        new = torch.full((max_len - seq.shape[0],), padding_value, dtype=torch.long).to(device)
        padded_seqs.append(torch.cat((new, seq)))
    return torch.stack(padded_seqs)