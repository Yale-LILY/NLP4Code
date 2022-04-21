import torch
import io, tokenize, re
import ast, astunparse

from typing import Tuple, Optional, List, Union

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer, GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPTJForCausalLM
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers.generation_utils import GenerationMixin

# from https://stackoverflow.com/questions/1769332/script-to-remove-python-comments-docstrings
def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out

def post_process_code(code, remove_comments=True, remove_extra_lines=False, ast_back_parse=True):
    """ a series of post-processing steps to clean up the code and avoid duplicated code """

    if remove_comments:
        code = remove_comments_and_docstrings(code)
    
    if ast_back_parse:
        code = astunparse.unparse(ast.parse(code))

    if remove_extra_lines:
        # remove the code after "answer" is generated
        result = []
        for line in code.split("\n"):
            result.append(line)
            if line.startswith("answer"):
                break
        code = "\n".join(result)

    code = code.strip()

    return code

def get_model(model_name: str, 
            tokenizer_only: bool = False,
            gradient_ckpt: bool = False,
            additional_special_tokens: Optional[List[str]] = None) \
        -> Tuple[GenerationMixin, PreTrainedTokenizer]:
    if additional_special_tokens is None:
        additional_special_tokens = []

    if not tokenizer_only:
        print(f"using pretrained model: {model_name}, gradient_ckpt: {gradient_ckpt}")

    if model_name == "microsoft/CodeGPT-small-py":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, additional_special_tokens=additional_special_tokens)
        if not tokenizer_only:
            model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))
    if model_name == "EleutherAI/gpt-j-6B":
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
        tokenizer = RobertaTokenizer.from_pretrained(model_name, 
                                                 additional_special_tokens=additional_special_tokens)
        if not tokenizer_only:
            model = T5ForConditionalGeneration.from_pretrained(model_name, 
                                                    pad_token_id=tokenizer.eos_token_id,
                                                    gradient_checkpointing=gradient_ckpt, 
                                                    use_cache=not gradient_ckpt)
            if len(additional_special_tokens) > 0:
                model.resize_token_embeddings(len(tokenizer))
    else:
        raise NotImplementedError

    if tokenizer_only:
        return None, tokenizer
    else:
        return model, tokenizer

def left_pad_sequences(sequences: List[torch.Tensor], batch_first: bool = True, padding_value: Union[int, bool] = 0, 
                       max_len: int = -1, device: torch.device = None) -> torch.Tensor:
    assert all([len(seq.shape) == 1 for seq in sequences])
    max_len = max_len if max_len > 0 else max(len(s) for s in sequences)
    device = device if device is not None else sequences[0].device

    padded_seqs = []
    for seq in sequences:
        padded_seqs.append(torch.cat((torch.full((max_len - seq.shape[0],), padding_value, dtype=torch.long).to(device), seq)))
    return torch.stack(padded_seqs)

def sanity_check(test_str: str, model, tokenizer):
    print(f"test str is: ###############{test_str}##############")

    input_ids = tokenizer.encode(test_str, add_special_tokens=False, return_tensors="pt").to(model.device)
    attention_mask = torch.where(input_ids == tokenizer.eos_token_id, torch.zeros_like(input_ids), torch.ones_like(input_ids)).to(model.device)

    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=40, num_return_sequences=1)

    output_str = tokenizer.decode(output_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    output_str_no_sp_tokens = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print(f"output str is: ###############{output_str}##############")

    new_test_str = " ".join(output_str_no_sp_tokens.split("\n")[:-1])

    print(f"new test str is: ###############{new_test_str}###############")

    input_ids = tokenizer.encode(new_test_str, add_special_tokens=False, return_tensors="pt").to(model.device)
    attention_mask = torch.where(input_ids == tokenizer.eos_token_id, torch.zeros_like(input_ids), torch.ones_like(input_ids)).to(model.device)

    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=40, num_return_sequences=1)

    output_str = tokenizer.decode(output_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    print(f"new output str is: ###############{output_str}###############")
