import torch
from torch import nn
from d2l import torch as d2l

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取标记和段落标签"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 段落 A 的标签为 0
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        # 段落 B 的标签为 1
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, 
                 ffn_num_hiddens, num_heads, dropout, use_bias=True):
        super(EncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=num_hiddens, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(ffn_num_input, ffn_num_hiddens),
            nn.ReLU(),
            nn.Linear(ffn_num_hiddens, num_hiddens)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(norm_shape)
        self.layer_norm2 = nn.LayerNorm(norm_shape)

    def forward(self, X, valid_lens):
        # Self-attention
        attn_output, _ = self.attention(X, X, X, key_padding_mask=valid_lens)
        attn_output = self.dropout(attn_output)
        X = self.layer_norm1(attn_output + X)  # Residual connection + Layer normalization

        # Feed-forward network
        ffn_output = self.ffn(X)
        ffn_output = self.dropout(ffn_output)
        X = self.layer_norm2(ffn_output + X)  # Residual connection + Layer normalization
        
        return X
    
class BERTEncoder(nn.Module):
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dorpout,
                max_len=1000,key_size=768,query_size=768,value_size=768,**kwargs):
        
        super(BERTEncoder,self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size,num_hiddens)
        self.segment_embedding = nn.Embedding(2,num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),EncoderBlock(
                key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,
                ffn_num_hiddens,num_heads,dorpout,True))
        self.pos_embedding = nn.Parameter(torch.randn(1,max_len,num_hiddens))
    
    def forward(self,tokens,segments,valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X,valid_lens)
        return X

class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                nn.ReLU(),nn.LayerNorm(num_hiddens),
                                nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
         # 扩展 batch_idx，使其在每个 batch 内重复 num_pred_positions 次
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
         # 使用 batch_idx 和 pred_positions 来从 X 中提取被掩蔽的词
        masked_X = X[batch_idx, pred_positions]
        # 将 masked_X 重塑为 batch_size x num_pred_positions x num_inputs
        masked_X = masked_X.reshape((batch_size, num_pred_positions,-1))
        # 使用 MLP 对掩蔽的词进行预测
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
    
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)
        
    def forward(self, X):
    # X的形状：(batchsize,num_hiddens)
        return self.output(X)
    
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                                   dropout, max_len=max_len, key_size=key_size,
                                   query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)
 
    def forward(self, tokens, segments, valid_lens=None,pred_positions=None,key_padding_mask=None):

        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
         # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat