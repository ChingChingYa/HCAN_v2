import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import copy

# 自身MultiheadAttention
class ConvMultiheadAttention(Module):
    # 初始化參數
    # input_dim 輸入資料 
    # multihead_cnt    句子包含單詞數? 30一個句子最多30單詞l
    # conv_cnt    cnn數量 6
    # kernel_dim    filter大小? 
    def __init__(self, input_dim, kernel_dim, multihead_cnt, conv_cnt):
        super(ConvMultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.multihead_cnt = multihead_cnt

        self.convs = nn.ModuleList([nn.Conv1d(input_dim, input_dim, kernel_dim, stride=1,padding=1)
                                    for _ in range(conv_cnt)])
        for w in self.convs:
            nn.init.xavier_normal_(w.weight)

    # 加入self-attention Attention(Q, K, V ) = softmax(QKT√d) = attention score
    # softmax(QKT√d)V = attention vector
    # bmm   矩阵乘法  
    # div   除法   
    # permute   将tensor的维度换位
    def attention(self, q, k, v):
        attention_score = torch.softmax(torch.div(
                torch.bmm(q.permute(0, 2, 1), k), np.sqrt(self.input_dim)),
                2)
        attention_vector = attention_score.bmm(v.permute(0, 2, 1)).permute(0, 2, 1)     
        # return torch.softmax(torch.div(
        #         torch.bmm(q.permute(0, 2, 1), k), np.sqrt(self.input_dim)),
        #         2).bmm(v.permute(0, 2, 1)).permute(0, 2, 1)
        return attention_vector

    # 加入 MultiHead(Q, K, V ) = 
    # [head1, ..., headh]where headi = Attention(Qi, Ki, Vi)
    # multihead_cnt 投影次數
    # chunk 分割
    # cat 串接在一起
    def multihead(self, hiddens):
        hiddens = [torch.chunk(hidden, self.multihead_cnt, 1)
                   for hidden in hiddens]
        hiddens = torch.cat([self.attention(hiddens[0][i], hiddens[1][i],
                                            hiddens[2][i])
                            for i in range(self.multihead_cnt)], 1)

        return hiddens





# 兩個做哈達瑪乘積 Parallel(E) = MultiHead(Qa, Ka, Va) MultiHead(Qb, Kb, Vb)
class ConvMultiheadSelfAttWord(ConvMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=6):
        super(ConvMultiheadSelfAttWord, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)

    def forward(self, input):
        hiddens = [F.elu(conv(input)) for conv in self.convs[:-1]]
        hiddens.append(torch.tanh(self.convs[-1](input)))
        # [:3]  擷取前3的字串
        # [3:]  擷取後3的字串
        elu_hid = self.multihead(hiddens[:3])
        tanh_hid = self.multihead(hiddens[3:])
        # F.layer_norm  layer正規化
        # mul   哈達瑪乘積
        output = F.layer_norm(torch.mul(elu_hid, tanh_hid), elu_hid.size()[1:])
        # shape为[sent_in_doc, embedding_size, 1]
        # shape为[80, embedding_size, 30]
        return output


# 加入T MultiheadTargetAttention
class ConvMultiheadTargetAttnWord(ConvMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=6):
        super(ConvMultiheadTargetAttnWord, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)
        self.target = nn.Parameter(torch.randn(input_dim, 1))
        stdv = 1. / math.sqrt(self.target.size(1))
        self.target.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.size(0)
        hiddens = [F.elu(conv(input)) for conv in self.convs]
        sent_vec = self.multihead([self.target.expand
                                (batch_size, self.input_dim, 1)]+hiddens)
        # shape为[batch_size, embedding_size, 1]
        return sent_vec


# 兩個做哈達瑪乘積 Parallel(E) = MultiHead(Qa, Ka, Va) MultiHead(Qb, Kb, Vb)
class ConvMultiheadSelfAttSent(ConvMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=6):
        super(ConvMultiheadSelfAttSent, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)

    def forward(self, input):
        hiddens = [F.elu(conv(input)) for conv in self.convs[:-1]]
        hiddens.append(torch.tanh(self.convs[-1](input)))
        # [:3]  擷取前3的字串
        # [3:]  擷取後3的字串
        elu_hid = self.multihead(hiddens[:3])
        tanh_hid = self.multihead(hiddens[3:])
        # F.layer_norm  layer正規化
        # mul   哈達瑪乘積
        output = F.layer_norm(torch.mul(elu_hid, tanh_hid), elu_hid.size()[1:])
        # shape为[sent_in_doc, embedding_size, 1]
        # shape为[80, embedding_size, 30]
        return output


# 加入T MultiheadTargetAttention
class ConvMultiheadTargetAttSent(ConvMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=6):
        super(ConvMultiheadTargetAttSent, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)
        # self.target = [(nn.Parameter(torch.randn(input_dim, 6)) * 4) for i in range(4)]
        self.target = nn.Parameter(torch.randn(input_dim, 4))
        stdv = 1. / math.sqrt(self.target.size(1))
        self.target.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.size(0)
        hiddens = [F.elu(conv(input)) for conv in self.convs]
        sentiment_vec = self.multihead([self.target.expand(batch_size, self.input_dim, 4)]+hiddens)
        # sentiment_vec = torch.gather(sentiment_vec, 2, torch.LongTensor(batch_size, self.input_dim, 1))

        aspect_vec = self.target.permute(1, 0)
        # a = torch.bmm(aspect_vec, hiddens)
        # sentiment_vec = torch.reshape(batch_size, self.input_dim, 1)

        # output = self.multihead([self.target.expand(batch_size, self.input_dim, 1)]+hiddens)      
        # shape为[batch_size, embedding_size, 1]
        return sentiment_vec, aspect_vec
