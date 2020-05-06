# coding=utf-8
import json
import pickle
import nltk
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from gensim.models import word2vec
import xlsxwriter
from load_data import load_data3


# 使用nltk分词分句器
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

workbook = xlsxwriter.Workbook('./output/data_output.xlsx')
worksheet = workbook.add_worksheet('rating')
bold = workbook.add_format({'bold': True})
format = workbook.add_format({'text_wrap': True})


def SetUp():
    worksheet.write('A1', 'user', bold)
    worksheet.write('B1', 'prouduct', bold)
    worksheet.write('C1', 'rating', bold)
    worksheet.write('D1', 'review', bold)
    worksheet.write('E1', 'review_embed', bold)
    worksheet.set_column('D:E', 30)
   

def LoadFileSingle(filename, field):
    label = []
    with open(filename, 'rb') as f:
        for line in f:
            review = json.loads(line)
            label.append((review[field]))
    return label


def ConvertToList(filename, words, sents, model, all_lens, maxlen_sentences, maxlen_words):
    # 将所有的评论文件都转化为每篇都有maxlen_sentences个句子，
    # 每个句子有maxlen_words个单词，每個單詞300維
    # 不够的补零，多余的删除，并保存到最终的数据集文件之中
    vocab_word = []
    # review_comment_data = []
    prouduct_rating_data = []
    dict_reviewerID = {}
    dict_asin = {}
    for word in model.wv.index2word:
        vocab_word.append(word)
    overall_list = (LoadFileSingle('./dataset/'+filename+'.json', 'overall'))
    reviewerID_list = (LoadFileSingle('./dataset/'+filename+'.json', 'reviewerID'))
    asin_list = (LoadFileSingle('./dataset/'+filename+'.json', 'asin'))
    data_x = []
    data_save = []
    data_y = []
    data_y1 = []
    data_x1 = []
    for l, sent in enumerate(words):
        doc2 = [0] * all_lens
        for i, s in enumerate(sent):
            if i < all_lens:
                if s in vocab_word:
                    doc2[i] = vocab_word.index(s)

        data_y.append([int(overall_list[l])-1])
        data_x1.append(doc2)
        prouduct_rating_data.append([reviewerID_list[l],
                                    asin_list[l], int(overall_list[l])-1])

    for i, sent in enumerate(sents):
        doc = np.zeros((maxlen_sentences, maxlen_words), dtype=np.int32)
        for k, s in enumerate(sent):
            if k < maxlen_sentences:
                for j, word in enumerate(s):
                    if j < maxlen_words:
                        if word in vocab_word:
                            doc[k][j] = vocab_word.index(word)
                      


        data_x.append(doc)
        data_save.append(doc.tolist())
        # review_comment_data.append([doc, int(overall_list[l])])

    for i, t in enumerate(prouduct_rating_data):
        if len(dict_asin) == 0:
            dict_asin[t[1]] = i
        if t[1] not in dict_asin:
            dict_asin[t[1]] = len(dict_asin)
        tmp = prouduct_rating_data[i][1]
        prouduct_rating_data[i][1] = dict_asin.get(tmp)

        if len(dict_reviewerID) == 0:
            dict_reviewerID[t[0]] = i
        if t[0] not in dict_reviewerID:
            dict_reviewerID[t[0]] = len(dict_reviewerID)
        tmp = prouduct_rating_data[i][0]
        prouduct_rating_data[i][0] = dict_reviewerID.get(tmp)

    for i, t in enumerate(prouduct_rating_data):
        worksheet.write(i+1, 0, t[0])
        worksheet.write(i+1, 1, t[1])
        worksheet.write(i+1, 2, t[2])
        worksheet.write(i+1, 3, str(sents[i]), format)
        worksheet.write(i+1, 4, str(data_save[i]), format)

    workbook.close() 
    pickle.dump((data_x, data_y),
                open('./data/review_comment_data_'+filename+'.pickle', 'wb'))   
    pickle.dump((data_x1, data_y),
                open('./data/review_comment_bywords_'+filename+'.pickle', 'wb'))            
    pickle.dump((prouduct_rating_data),
                open('./data/prouduct_rating_data_'+filename+'.pickle', 'wb'))


class FileData():
    def __init__(self, words, len_words, sents, len_sents, alllens):
        self.sents = sents
        self.len_sents = len_sents
        self.words = words
        self.len_words = len_words
        self.alllens = alllens
        

if __name__ == '__main__':
    SetUp()
    model = word2vec.Word2Vec.load('./voc/word2vec.pickle')
    # ConvertToList(filedata.words, model)
    
    print("load_data:filedata1")
    filedata1, data1_all, data1_sents, data1_words = load_data3('./data/file_len1.pickle')
    print("ConvertToList:filedata1")
    ConvertToList('1', filedata1.words, filedata1.sents, model, data1_all, data1_sents, data1_words)

    print("load_data:filedata2")
    filedata2, data2_all, data2_sents, data2_words = load_data3('./data/file_len2.pickle')
    print("ConvertToList:filedata2")
    ConvertToList('2', filedata2.words, filedata2.sents, model, data2_all, data2_sents, data2_words)


    print("load_data:filedata3")
    filedata3, data3_all, data3_sents, data3_words = load_data3('./data/file_len3.pickle')
    print("ConvertToList:filedata3")
    ConvertToList('3', filedata3.words, filedata3.sents, model, data3_all, data3_sents, data3_words)
    print("Save success")