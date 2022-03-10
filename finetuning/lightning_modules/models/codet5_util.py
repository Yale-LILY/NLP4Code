import torch
from typing import Tuple, Optional, List, Union

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import PreTrainedModel, PreTrainedTokenizer

def get_codet5(model_name: str = "Salesforce/codet5-base",
            tokenizer_only: bool = False,
            gradient_ckpt: bool = False,
            additional_special_tokens: Optional[List[str]] = None) \
        -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if additional_special_tokens is None:
        additional_special_tokens = []
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name, 
                                                 additional_special_tokens=additional_special_tokens)

    if tokenizer_only:
        return None, tokenizer
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name, 
                                                    pad_token_id=tokenizer.eos_token_id,
                                                    gradient_checkpointing=gradient_ckpt, 
                                                    use_cache=not gradient_ckpt)
        if len(additional_special_tokens) > 0:
            model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer
