import torch
from transformers import BertTokenizer

def create_new_segment(input_tokens, tokenizer):
    tokens = []
    cand_indexes = []
    for index, input_token in enumerate(input_tokens):
        ts = tokenizer.tokenize(input_token)
        tokens.extend(ts)
        cand_indexes.extend([index] * len(ts))
    return tokens, cand_indexes

def create_token_masks(batch_inputs, tokenizer, args):
    inputs = batch_inputs['token']
    cand_indexes = batch_inputs['masked_indice']
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    wwm_masked_indices = []
    for cand_index, maskeds in zip(cand_indexes.tolist(), masked_indices.tolist()):
        cand_index = torch.tensor(cand_index)
        masked_indice = torch.tensor(maskeds)
        for i, masked in enumerate(maskeds):
            if masked == True:
                cand_ind = cand_index[i]
                find_positions = torch.where(cand_index==cand_ind)[0]
                masked_indice[find_positions] = True
        wwm_masked_indices.append(masked_indice.tolist())
    wwm_masked_indices = torch.tensor(wwm_masked_indices)
    labels[~wwm_masked_indices] = -100

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & wwm_masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & wwm_masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    return inputs, labels


if __name__ == "__main__":
    test_text = "使用 语言 模型 来 预测 下 一个 词 的 probability 。"
    seq_cws = [test_text.split()]
    # tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # batch_encoding = tokenizer([test_text], add_special_tokens=True, truncation=True, max_length=512)
    # print(batch_encoding)
    block_size = 5
    for i in range(0, len(seq_cws[0]), block_size):
        blocked_tokenized_text = seq_cws[0][i : i + block_size]
        print(i, blocked_tokenized_text)