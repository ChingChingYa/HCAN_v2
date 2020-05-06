# https://blog.csdn.net/u012052268/article/details/79560768  
# coding:utf-8  

import os  
import sys  
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from load_data import read_dataset
from load_data import cut_dataset
from operator import itemgetter, attrgetter
import xlsxwriter


if __name__ == "__main__":
    cut_dataset('./data/filedata1.pickle')
    train, test = read_dataset('./data/filedata11.pickle')

    workbook = xlsxwriter.Workbook('./output/tfidf_output.xlsx') #打开一个xlsx文件（如果打开的文件存在 ，则清空该文件，如果文件不存在，则新建）
    worksheet = workbook.add_worksheet() #新建一个Sheet（名字缺省的话，默认从Sheet1开始，可以添加自己的sheet名字
    bold = workbook.add_format({'bold': True})
    format = workbook.add_format({'text_wrap': True})
    worksheet.write('A1', 'word', bold)
    worksheet.write('B1', 'weight', bold)
    worksheet.set_column('A:B', 20)
    
    data_x = []
    for i, sent in enumerate(train):
        doc = []
        for k, s in enumerate(sent):
            t = ''
            for j, word in enumerate(s):
                    if (k == 0) & (j == 0):
                        t += word
                    else:
                        t += ' '+word

            doc.append(t)
        # data_x.append(doc)
        data_x.extend(doc)
    
    corpus = data_x
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    x = vectorizer.fit_transform(corpus)
    tfidf = transformer.fit_transform(x)
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    # print(weight)
    cnt = 0
    dict_all = []
    all_doc = []
    weight_vec = []
    all_weight = []
    a_dict = {}

    for i in range(len(weight)):
        for j in range(len(word)):
            a_dict.setdefault(word[j], weight[i][j])
   

    all_weight = sorted(set(a_dict.values()), reverse=True)
    # print(all_weight[0])
    for i, a in enumerate(all_weight):
        all_doc.append(filter(lambda x:all_weight[i]== x[1], a_dict.items())) 
    
    for i, a in enumerate(all_doc):    
        if(cnt < 5):
            for (key, value) in a:
                print('%s : %s' % (key, value))
                cnt = cnt+1
                worksheet.write(cnt, 0, key, format)
                worksheet.write(cnt, 1, value, format)
        else:
            break

    workbook.close()

