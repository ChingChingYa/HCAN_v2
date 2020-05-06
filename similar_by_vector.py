import json
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from gensim.models import word2vec
from load_data import load_data
import csv
import xlsxwriter


if __name__ == '__main__':
    model = []
    similar_by_vector = []
    # model.append(load_data('./voc/aspect_vec0.pickle'))
    for i in range(4):
        model.append(load_data('./data/aspect_vec'+str(i)+'.pickle'))

    wvmodel = KeyedVectors.load_word2vec_format('./voc/word2vec2.pickle')
    # 添加用于突出显示单元格的粗体格式。
    
    workbook = xlsxwriter.Workbook('./output/aspect_output.xlsx') #打开一个xlsx文件（如果打开的文件存在 ，则清空该文件，如果文件不存在，则新建）
    worksheet = workbook.add_worksheet() #新建一个Sheet（名字缺省的话，默认从Sheet1开始，可以添加自己的sheet名字
    bold = workbook.add_format({'bold': True})
    format = workbook.add_format({'text_wrap': True})
    worksheet.write('A1', 'user', bold)
    worksheet.write('B1', 'aspect1', bold)
    worksheet.write('C1', 'aspect2', bold)
    worksheet.write('D1', 'aspect3', bold)
    worksheet.write('E1', 'aspect4', bold)
    worksheet.write('F1', 'aspect5', bold)
    worksheet.write('G1', 'aspect6', bold)
    worksheet.set_column('B:G', 25)

    aspects = []
    for index, m in enumerate(model):
        a = (wvmodel.similar_by_vector(m[0], topn=5))
        for i, word in enumerate(a):
            worksheet.write(index+1, 0, index)
            worksheet.write(index+1, i+1, str(word), format)

    
    workbook.close()#最后关闭文件              