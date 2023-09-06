import random

import torch

from pipeline.registry import registry

def random_point_cloud(pc, pc_mask, mask_ratio):
    output_mask = []
    for i in range(len(pc)):
        if pc_mask[i] == 0:
            output_mask.append(0)
        else:
            prob = random.random()
            if prob < mask_ratio:
                output_mask.append(0)
            else:
                output_mask.append(1)
    
    output_mask = torch.tensor(output_mask, dtype=torch.bool)
    return output_mask

def random_caption_word(tokens, tokens_mask, tokenizer, vocab, mask_ratio):
    output_label = []
    output_tokens = tokens.clone()
    for i, token in enumerate(tokens): # 101 cls 102 sep use them as SOS and EOS token
        if tokens_mask[i] == 0 or token == 101:
            output_label.append(-1)
        elif token == 102:
            output_tokens[i] = tokenizer.mask_token_id
            output_label.append(vocab.token_to_id('[EOS]'))
        else:
            prob = random.random()
            # mask token with 15% probability
            if prob < mask_ratio:
                output_tokens[i] = tokenizer.mask_token_id
                output_label.append(vocab.token_to_id(tokenizer.decode([tokens[i]])))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
    output_label = torch.Tensor(output_label).long()
    return output_tokens, output_label