import os
import random
import torch
from d2l import torch as d2l
from datasets import load_dataset
import re
from collections import Counter


import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')  # 下载 punkt_tab 资源，解决 missing resource 错误


def _read_wiki(data_dir, subset_ratio=0.01):
    """使用 Hugging Face wikitext 数据集的子集"""
    # 加载 Arrow 格式的数据集
    dataset = load_dataset('arrow', data_files={
        'train': f"{data_dir}/wikitext-train-00000-of-00002.arrow",
        'validation': f"{data_dir}/wikitext-validation.arrow"
    })
    
    print(f"Total training samples: {len(dataset['train'])}")
    dataset = dataset['train'].select(range(int(len(dataset['train']) * subset_ratio)))
    print(f"Selected subset samples: {len(dataset)}")

    paragraphs = []
    for text in dataset['text']:
        # 使用 nltk 进行句子拆分
        sentences = sent_tokenize(text)  # 使用 NLTK 的句子分割器
        # 将每个句子转换为小写，并且确保每个句子至少包含 2 个单词
        sentences = [sentence.strip().lower() for sentence in sentences if len(sentence.split()) >= 2]
        
        if len(sentences) >= 1:  # 确保每个段落至少包含 1 个有效句子
            paragraphs.append(sentences)
    
    print(f"Processed paragraphs: {len(paragraphs)}")
    #print(paragraphs[2])  # 打印第一个段落，查看处理后的数据格式
    #print(sentences)  # 打印最后处理的句子，查看处理后的数据格式
    # 打印第二个段落中的每个句子
    #for sentence in paragraphs[1]:
    #    print(sentence)

    return paragraphs

def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph)- 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        
        # 打印每对句子的长度和是否满足条件
        #print(f"tokens_a length: {len(tokens_a)}, tokens_b length: {len(tokens_b)}")
        
        # 调整 max_len 过滤条件，确保不丢弃数据
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            #print(f"Skipping sentence pair: {tokens_a} + {tokens_b} exceeds max_len")
            continue
        
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    
    #print(f"Generated {len(nsp_data_from_paragraph)} NSP examples for the paragraph.")
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    
    #print(f"Tokens: {tokens}")
    #print(f"Candidate prediction positions: {candidate_pred_positions}")
    #print(f"Number of tokens to predict (MLM): {num_mlm_preds}")
    
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    
    #print(f"Generated MLM examples: {len(pred_positions)}")
    
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []

    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        #valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        valid_lens.append(torch.tensor([len(token_ids)] * max_len, dtype=torch.float32))  # 为每个 token 填充有效长度
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

    return all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels

class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        print(f"Total paragraphs received: {len(paragraphs)}")
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        

        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])

        #sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        print(f"Total sentences after tokenization: {len(sentences)}")
        #words = [word for paragraph in paragraphs for sentence in paragraph for word in sentence.split()]
        # 使用 Counter 来计算每个单词的频率
        #word_counts = Counter(words)
        #print(f"Word counts: {word_counts.most_common(10)}")  # 打印出现频率最高的 10 个单词

        #self.vocab = d2l.Vocab(words, min_freq=3, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])

        examples = []
        
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
        
        print(f"Total examples for NSP: {len(examples)}")
        
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)) for tokens, segments, is_next in examples]
        
        print(f"Total examples for MLM: {len(examples)}")
        
        self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels, self.nsp_labels = _pad_bert_inputs(
            examples, max_len, self.vocab)
        
        print(f"Total token_ids: {len(self.all_token_ids)}")

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx], self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len, data_dir=r'C:\Users\Administrator\.cache\huggingface\datasets\wikitext\wikitext-103-raw-v1\0.0.0\b08601e04326c79dfdd32d625aee71d232d685c3', subset_ratio=0.01):
    paragraphs = _read_wiki(data_dir, subset_ratio=subset_ratio)
    
    if not paragraphs:
        raise ValueError("No paragraphs found in the dataset after loading and processing.")
    
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
    words = [word for paragraph in paragraphs for sentence in paragraph for word in sentence.split()]
    # 使用 Counter 来计算每个单词的频率
    word_counts = Counter(words)
    print(f"Word counts: {word_counts.most_common(10)}")  # 打印出现频率最高的 10 个单词

    vocab = d2l.Vocab(words, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
    print(f"len(vocab): {len(vocab)}")  # 打印词汇表的大小

    train_set = _WikiTextDataset(paragraphs, max_len)
    print(f"Total samples in train_set: {len(train_set)}")

    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    
    return train_iter, vocab


batch_size, max_len = 64, 64
train_iter, vocab = load_data_wiki(batch_size, max_len, subset_ratio=0.01)



for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y) in train_iter:
     key_padding_mask = valid_lens_x.transpose(0, 1) != 0
     print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,key_padding_mask.shape,
           pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape, nsp_y.shape)
     break
