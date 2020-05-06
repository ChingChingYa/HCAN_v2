
import json
import pickle
import nltk
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from gensim.models import word2vec
import xlsxwriter
from nltk.corpus import stopwords

# 使用nltk分词分句器
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

workbook = xlsxwriter.Workbook('./output/data_len_output.xlsx')
worksheet_table = workbook.add_worksheet('table')
bold = workbook.add_format({'bold': True})
format = workbook.add_format({'text_wrap': True})


class FileData():
    def __init__(self, words, len_words, sents, len_sents, alllens):
        self.sents = sents
        self.len_sents = len_sents
        self.words = words
        self.len_words = len_words
        self.alllens = alllens


def SetUp():
    worksheet_table.set_column('B:D', 20)
    worksheet_table.write('A1', 'dataset1', bold)
    worksheet_table.write('B1', 'maxlen_all', bold)
    worksheet_table.write('C1', 'maxlen_sentences', bold)
    worksheet_table.write('D1', 'maxlen_words', bold)

    worksheet_table.write('A6', 'dataset2', bold)
    worksheet_table.write('B6', 'maxlen_all', bold)
    worksheet_table.write('C6', 'maxlen_sentences', bold)
    worksheet_table.write('D6', 'maxlen_words', bold)

    worksheet_table.write('A11', 'dataset3', bold)
    worksheet_table.write('B11', 'maxlen_all', bold)
    worksheet_table.write('C11', 'maxlen_sentences', bold)
    worksheet_table.write('D11', 'maxlen_words', bold)

    worksheet_table.write('A2', '100%', bold)
    worksheet_table.write('A3', '90%', bold)
    worksheet_table.write('A4', '80%', bold)

    worksheet_table.write('A7', '100%', bold)
    worksheet_table.write('A8', '90%', bold)
    worksheet_table.write('A9', '80%', bold)

    worksheet_table.write('A12', '100%', bold)
    worksheet_table.write('A13', '90%', bold)
    worksheet_table.write('A14', '80%', bold)


def Word2VecFun(sents):
    model = gensim.models.Word2Vec(sents, size=300, min_count=5)
    model.save('./voc/word2vec.pickle')
    model = word2vec.Word2Vec.load('./voc/word2vec.pickle')
    model.wv.save_word2vec_format('./voc/word2vec2.pickle', binary=False)


def LoadFile(filename, field):
    result_sents = []
    len_sent = []
    len_word = []
    result_words = []
    all_len = []
    stop_words = set(stopwords.words('english'))
    with open(filename, 'rb') as f:
        for line in f:
            word1 = []
            word2 = []
            review = json.loads(line)
            sents = nltk.sent_tokenize(review[field])
            len_sent.append(len(sents))
            for sentence in sents:
                newword = []
                sentence = word_tokenizer.tokenize(sentence)
                len_word.append(len(sentence))
                # 變成小寫
                for i, w in enumerate(sentence):
                    if w not in stop_words:
                        w = w.lower()
                        newword.append(lemmatizer.lemmatize(''.join(w), pos='v'))
       
                word1.extend(newword)
                word2.append(newword)
                # word.extend(newword) 全部串連在一起
                # word.append(newword) 分成二維
                # result_sents.append(newword)
 
            result_words.append(word1)
            all_len.append(len(word1))
            result_sents.append(word2)
        f.close()
    filedata = FileData(result_words, len_word, result_sents, len_sent, all_len)
    return filedata


def CutSents(index, words, sents, len_all, len_sent, len_word):

    data_x1 = []
    sorted_all = sorted(len_all)
    sorted_sents = sorted(len_sent)
    sorted_words = sorted(len_word)

    maxlen_all = sorted_all[-1]
    maxlen_sentences = sorted_sents[-1]
    maxlen_words = sorted_words[-1]

    maxlen_all_90 = sorted_all[int(len(sorted_all) * 0.9)]
    maxlen_sentences_90 = sorted_sents[int(len(sorted_sents) * 0.9)]
    maxlen_words_90 = sorted_words[int(len(sorted_words) * 0.9)]

    maxlen_all_80 = sorted_all[int(len(sorted_all) * 0.8)]
    maxlen_sentences_80 = sorted_sents[int(len(sorted_sents) * 0.8)]
    maxlen_words_80 = sorted_words[int(len(sorted_words) * 0.8)]

    for i, sent in enumerate(sents):
        # doc = [['' for i in range(maxlen_words)] for j in range(maxlen_sentences)]
        doc1 = []
        
        for k, s in enumerate(sent):
            if k < maxlen_sentences:
                t = []
                for j, word in enumerate(s):
                    if j < maxlen_words:
                        t.append(word)

                doc1.append(t)
        data_x1.append(doc1)
       

    pickle.dump((data_x1),
                open('./data/filedata'+str(index)+'.pickle', 'wb'))

    if(index == 1):
        worksheet_table.write('B2', maxlen_all, format)
        worksheet_table.write('C2', maxlen_sentences, format)
        worksheet_table.write('D2', maxlen_words, format)
        worksheet_table.write('B3', maxlen_all_90, format)
        worksheet_table.write('C3', maxlen_sentences_90, format)
        worksheet_table.write('D3', maxlen_words_90, format)
        worksheet_table.write('B4', maxlen_all_80, format)
        worksheet_table.write('C4', maxlen_sentences_80, format)
        worksheet_table.write('D4', maxlen_words_80, format)
    elif(index == 2):
        worksheet_table.write('B7', maxlen_all, format)
        worksheet_table.write('C7', maxlen_sentences, format)
        worksheet_table.write('D7', maxlen_words, format)
        worksheet_table.write('B8', maxlen_all_90, format)
        worksheet_table.write('C8', maxlen_sentences_90, format)
        worksheet_table.write('D8', maxlen_words_90, format)
        worksheet_table.write('B9', maxlen_all_80, format)
        worksheet_table.write('C9', maxlen_sentences_80, format)
        worksheet_table.write('D9', maxlen_words_80, format)
    else:
        worksheet_table.write('B12', maxlen_all, format)
        worksheet_table.write('C12', maxlen_sentences, format)
        worksheet_table.write('D12', maxlen_words, format)
        worksheet_table.write('B13', maxlen_all_90, format)
        worksheet_table.write('C13', maxlen_sentences_90, format)
        worksheet_table.write('D13', maxlen_words_90, format)
        worksheet_table.write('B14', maxlen_all_80, format)
        worksheet_table.write('C14', maxlen_sentences_80, format)
        worksheet_table.write('D14', maxlen_words_80, format)
            

    return maxlen_all_90, maxlen_sentences_90, maxlen_words_90


if __name__ == '__main__':

    SetUp()
    print("LoadFile:1.json")
    filedata_x = ((LoadFile('./dataset/1.json', 'reviewText')))
    print("LoadFile:2.json")
    filedata_y = ((LoadFile('./dataset/2.json', 'reviewText')))
    print("LoadFile:3.json")
    filedata_z = ((LoadFile('./dataset/3.json', 'reviewText')))

    print("==========================================")
    print("將words切成固定長度")
    maxlen_all_x, maxlen_sentences_x, maxlen_words_x = CutSents(1, filedata_x.words, filedata_x.sents, filedata_x.alllens, filedata_x.len_sents, filedata_x.len_words)
    maxlen_all_y, maxlen_sentences_y, maxlen_words_y = CutSents(2, filedata_y.words, filedata_y.sents, filedata_y.alllens, filedata_y.len_sents, filedata_y.len_words)
    maxlen_all_z, maxlen_sentences_z, maxlen_words_z = CutSents(3, filedata_z.words, filedata_z.sents, filedata_z.alllens, filedata_z.len_sents, filedata_z.len_words)
    
    pickle.dump((filedata_x, maxlen_all_x, maxlen_sentences_x, maxlen_words_x),
                open('./data/file_len1.pickle', 'wb'))
    pickle.dump((filedata_y, maxlen_all_y, maxlen_sentences_y, maxlen_words_y),
                open('./data/file_len2.pickle', 'wb'))
    pickle.dump((filedata_z, maxlen_all_z, maxlen_sentences_z, maxlen_words_z),
                open('./data/file_len3.pickle', 'wb'))

    print("Save success")   
    print("==========================================")

    print("Word2Vec")
    Word2VecFun(filedata_x.words+filedata_y.words+filedata_z.words)
    workbook.close()