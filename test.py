from models import Proto_CNN as HCAN
from load_data import load_data, read_dataset2
from data_clean import LoadFileSingle
from gensim.models import KeyedVectors
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn import metrics
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data
from torchsummary import summary
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchviz import make_dot       
from graphviz import Digraph
# https://blog.csdn.net/sinat_34604992/article/details/103078205


class Config(object):
    def __init__(self):
        self.model_name = 'HCAN'
        self.dataset = './data/review_comment_data_11.pickle'
        # 训练集
        self.num_classes = 4 
        self.vocab_path = './voc/word2vec2.pickle'
        # 词表
        self.save_path = './run/' + self.model_name + '.ckpt'
        # 模型训练结果
        self.log_path = './log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 设备
        self.dropout = 0.4                                              
        self.require_improvement = 1000
        # 若超过1000batch效果还没提升，则提前结束训练            
        self.n_vocab = 216
        # 词表大小，在运行时赋值
        self.num_epochs = 10                                            
        self.batch_size = 32                                           
        # 每句话处理成的长度，截长、补短
        # self.learning_rate = 1e-5 
        self.learning_rate = 0.01                         
        self.embed_dim = 300
        # 字向量维度
        self.filter_sizes = [3, 3]
        # 卷积核尺寸
        self.hidden_dim = 80
        self.sent_maxlen = 6
        self.word_maxlen = 29

        print("Loading data...")
        alldata = load_data('./data/review_comment_data_1.pickle')
        self.x_train_data, self.x_test_data = train_test_split(alldata[0], test_size=0.2, random_state=1)
        self.x_test_data, self.x_val_data = train_test_split(self.x_test_data, test_size=0.2, random_state=1)
        self.y_train_data, self.y_test_data = train_test_split(alldata[1], test_size=0.2, random_state=1)
        self.y_test_data, self.y_val_data = train_test_split(self.y_test_data, test_size=0.2, random_state=1)
        # 先转换成 torch 能识别的 Dataset
        # # 设备配置
        # torch.cuda.set_device(0) # 这句用来设置pytorch在哪块GPU上运行
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test(self, model, test, rating, pmf_muti):
        input_batch = Variable(torch.LongTensor(test)).to(self.device)
        rating_batch = Variable(torch.LongTensor(rating)).to(self.device)
        sentiment_batch = Variable(torch.FloatTensor(pmf_muti)).to(self.device)
        loss = 0
        acc = 0
        torch_dataset = Data.TensorDataset(input_batch, rating_batch, sentiment_batch)
        test_loader = Data.DataLoader(dataset=torch_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)

        for i, (batch_x, batch_y, batch_z) in enumerate(test_loader):
            # batch_x, batch_y, batch_z = batch_x.to(device), batch_y.to(device), batch_z.to(device)
            pmf_vec, rating_vec, aspect_vec = model(batch_x)
            loss = model.loss(batch_z, batch_y, pmf_vec, rating_vec)
            acc = model.accuracy(rating_vec, batch_y)
            # pmf_vec = [i.detach().cpu().numpy() for i in pmf_vec]
            # aspect_vec = [i.detach().cpu().numpy() for i in aspect_vec]
            loss += loss.item()
            acc += acc.item()

        return loss / len(test), acc / len(test)
    

    def train(self, model, tain, rating, pmf_muti):
        input_batch = Variable(torch.LongTensor(tain)).to(self.device)
        rating_batch = Variable(torch.LongTensor(rating)).to(self.device)
        sentiment_batch = Variable(torch.FloatTensor(pmf_muti)).to(self.device)

        torch_dataset = Data.TensorDataset(input_batch, rating_batch, sentiment_batch)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # Train the model
        train_loss = 0
        train_acc = 0
        pmf = []
        train_loader = Data.DataLoader(dataset=torch_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)

        for i, (batch_x, batch_y, batch_z) in enumerate(train_loader):
            optimizer.zero_grad()
            # batch_x, batch_y, batch_z = batch_x.to(device), batch_y.to(device), batch_z.to(device)
            pmf_vec, rating_vec, aspect_vec = model(batch_x)
            pmf.append(pmf_vec)
            loss = model.loss(batch_z, batch_y, pmf_vec, rating_vec)
            acc = model.accuracy(rating_vec, batch_y)
            # pmf_vec = [i.detach().cpu().numpy() for i in pmf_vec]
            # aspect_vec = [i.detach().cpu().numpy() for i in aspect_vec]
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += acc.item()

        # Adjust the learning rate
        scheduler.step()

        return (train_loss / len(tain), train_acc / len(tain), pmf, aspect_vec)

    def Setup(self, pmf_muti_train, pmf_muti_val, pmf_muti_test):
        pmf_muti_train = [i.astype(np.float32) for i in pmf_muti_train]
        pmf_muti_val = [i.astype(np.float32) for i in pmf_muti_val]
        pmf_muti_test = [i.astype(np.float32) for i in pmf_muti_test]

        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    
        wvmodel = KeyedVectors.load_word2vec_format(self.vocab_path)
        # a = torch.LongTensor(wvmodel.vectors)
        model = HCAN(self.embed_dim, self.hidden_dim, self.filter_sizes,
                     self.sent_maxlen, self.word_maxlen, self.num_classes,
                     self.dropout, self.n_vocab, wvmodel.vectors).to(self.device)
                     
        # print(model)
        # print(summary(model, (10, 30)))             

        
        # 记录训练的日志
        # writer = SummaryWriter(log_dir=self.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        for epoch in range(self.num_epochs):
            start_time = time.time()
            train_loss, train_acc, pmf_vec, aspect_vec = self.train(model, self.x_train_data, self.y_train_data, pmf_muti_train)

            valid_loss, valid_acc = self.test(model, self.x_val_data, self.y_val_data, pmf_muti_val)



            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
            
            # if epoch == 49:
            #     vec.append(pmf_vec)
            #     aspects.append(aspect_vec)

        # writer.close()
        # plt.figure()
        # plt.plot(np.array(acc), label="acc",color="red")
        # plt.show()
        print('Checking the results of test dataset...')
        test_loss, test_acc = self.test(model, self.x_test_data, self.y_test_data, pmf_muti_test)
        print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

        pmf_vec = [i.detach().cpu().numpy() for i in pmf_vec]
        aspect_vec = [i.detach().cpu().numpy() for i in aspect_vec]
        return pmf_vec, aspect_vec