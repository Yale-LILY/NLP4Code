import torch
from typing import Tuple, Optional, List, Union

from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration
from transformers import PreTrainedModel, PreTrainedTokenizer, GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPTJForCausalLM

def get_codet5(model_name: str, 
            tokenizer_only: bool = False,
            gradient_ckpt: bool = False,
            additional_special_tokens: Optional[List[str]] = None) \
        -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if additional_special_tokens is None:
        additional_special_tokens = []

    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

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
