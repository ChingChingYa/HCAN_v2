import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
# import torch.nn.utils as utils
# import torch.nn.init as init
from layers import ConvMultiheadSelfAttWord as CMSA_Word
from layers import ConvMultiheadTargetAttnWord as CMTA_Word
from layers import ConvMultiheadSelfAttSent as CMSA_Sent
from layers import ConvMultiheadTargetAttSent as CMTA_Sent


class Proto(nn.Module):
    def __init__(self, num_emb, input_dim, pretrained_weight):
        super(Proto, self).__init__()
        # num_emb   單詞數量
        # input_dim   維度300
        self.id2vec = nn.Embedding(num_emb, input_dim, padding_idx=1)
        # unk, pad, ..., keywords
        self.id2vec.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.id2vec.requires_grad = False

    # rating結果作比較
    def accuracy(self, rating_vec, rating_batch):
        predict = [torch.argmax(i, 1) for i in rating_vec]
        # label = [torch.argmax(i, 0) for i in rating_batch]
        equals = [torch.eq(p, l) for p, l in zip(predict, rating_batch)]
        # acc = torch.mean(torch.FloatTensor(equals))
        acc = torch.sum(torch.FloatTensor(equals))
        return acc


    def loss(self, sentiment_batch, rating_batch, pmf_vec, rating_vec):
        loss_fn = torch.nn.MSELoss(reduction='mean')
        entroy = nn.CrossEntropyLoss()
        loss_sentiment = 0
        loss_rating = 0


        # 計算UV loss (MSE)
        for x, y in zip(pmf_vec, sentiment_batch):
            loss_sentiment += loss_fn(x, y)
            # loss_sentiment.append(loss_fn(x, y))

        # 計算評分 loss (CrossEntropy)
        for x, y in zip(rating_vec, rating_batch):
            a = entroy(x, y-1)
            loss_rating += torch.mean(a)
        loss_all = loss_sentiment + loss_rating
        return loss_all


    
    def forward(self, input_batch):
        pmf_vec, rating_vec, aspect_vec = self.predict(input_batch) 
        return pmf_vec, rating_vec, aspect_vec


class Proto_CNN(Proto):
    def __init__(self, input_dim, hidden_dim, kernel_dim,
                 sent_maxlen, word_maxlen, num_classes,
                 dropout_rate, num_emb, pretrained_weight):
        super(Proto_CNN, self).__init__(num_emb, input_dim, pretrained_weight)
        # 加入位置向量
        # sent_maxlen   句子長度
        self.positions_word = nn.Parameter(torch.randn(sent_maxlen, word_maxlen, input_dim))
        stdv = 1. / self.positions_word.size(1) ** 0.5
        self.positions_word.data.uniform_(-stdv, stdv)

        self.positions_sent = nn.Parameter(torch.randn(sent_maxlen, input_dim))
        stdv = 1. / self.positions_sent.size(1) ** 0.5
        self.positions_sent.data.uniform_(-stdv, stdv)

        self.cmsa_word = CMSA_Word(input_dim, kernel_dim[0])
        self.cmta_word = CMTA_Word(input_dim, kernel_dim[1])
        self.cmsa_sent = CMSA_Sent(input_dim, kernel_dim[0])
        self.cmta_sent = CMTA_Sent(input_dim, kernel_dim[1])

        self.dropout = nn.Dropout(dropout_rate)
        self.cls1 = nn.Linear(input_dim, 1)
        self.cls2 = nn.Linear(input_dim, 5)
        self.cls3 = nn.Linear(4, 1)
        # nn.Linear是用于设置网络中的全连接层的
        nn.init.xavier_normal_(self.cls1.weight)
        nn.init.xavier_normal_(self.cls2.weight)
        nn.init.xavier_normal_(self.cls3.weight)
        # nn.init.xavier_uniform_(self.cls.weight)

    def predict(self, x):
        input = self.id2vec(x)
        # input [batch_size, sent_in_doc, word_in_sent, embedding_size]
        # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
        # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量
        # shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
        input = self.dropout(input + self.positions_word)
        new = torch.reshape(input, [-1, 29, 300])
        # new [sent_in_doc, word_in_sent, embedding_size]
        hidden = self.cmsa_word(new.permute(0, 2, 1))
        # new.permute(0, 2, 1) 
        # [sent_in_doc, embedding_size, word_in_sent]
        # hidden [sent_in_doc, embedding_size, word_in_sent]

        hidden = self.cmta_word(hidden)
        # hidden [sent_in_doc, embedding_size, 1]
        # sent_vec = self.cls(hidden.squeeze(-1))
        sent_vec = torch.reshape(hidden.squeeze(-1), [-1, 6, 300])
        # squeeze降维
        # logits [batch_size, hidden_dim]

        # =====================Sent to doc=========================
        
        sent_vec = self.dropout(sent_vec + self.positions_sent)
        hidden = self.cmsa_sent(sent_vec.permute(0, 2, 1))
        sentiment_vec, aspect_vec = self.cmta_sent(hidden)
        sentiment_vec = torch.abs(sentiment_vec)
        sentiment_vec1 = self.cls1(sentiment_vec.permute(0, 2, 1)) 
        logits = (sentiment_vec1.squeeze(2))

        sentiment_vec2 = self.cls2(sentiment_vec.permute(0, 2, 1)) 
        rating_vec = (self.cls3(sentiment_vec2.permute(0, 2, 1))).squeeze(-1)
        

        pmf_vec = logits
        rating_vec = [F.softmax(i, 0) for i in rating_vec]
        
        # rating_vec = [torch.flip(i, [0]) for i in rating_vec]


        # rating_vec = [torch.softmax(i, 0) for i in logits]
        rating_vec = [i.unsqueeze(0) for i in rating_vec]
        
        # rating = torch.softmax(doc_vec, 1)
        # pmf_vec = [i.squeeze(0) for i in pmf_vec]
        return pmf_vec, rating_vec, aspect_vec
 

