import mnist
import numpy as np
import cv2
import pandas as pd
import xlrd
import xlwt


class Conv3_3:
    """
    卷积层，卷积核尺度3x3
    """    
    def __init__(self, num_filters, initmethod='create', filters='./filters.xls'):
        """
        卷积层初始化。

        Args:
            num_filters (int): 卷积核个数
            initmethod (str, optional): 初始化方法，create随机化创建，load从文件读取。Defaults to 'create'.
            filters (str, optional): 卷积核储存文件名。Defaults to './filters.xls'.
        """            
        if initmethod == 'create':
            self.num_filters = num_filters  #定义卷积核的个数
            self.filters = np.random.randn(num_filters,3,3) / 9  #除以9保证初始化的值不太大也不太小
        elif initmethod == 'load':
            self.num_filters = num_filters  #定义卷积核的个数
            self.filters = np.zeros((num_filters,3,3))
            filters_get = np.array(excel2matrix(filters))
            for i in range(num_filters):
                self.filters[i,:,:] = np.reshape(filters_get[i,:],(3,3))
            self.filters = np.squeeze(self.filters)
        
    
    def iterator_regions(self, image):
        h, w = np.shape(image)
        # 取出与卷积核大小一致的图像块
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i+3),j:(j+3)]
                yield im_region, i, j
    
    def forward(self, input):
        self.last_input = input

        h, w = np.shape(input)
        output = np.zeros((h-2, w-2, self.num_filters))
        
        # 图像与卷积核做卷积
        for im_region, i, j in self.iterator_regions(input):
            output[i,j] = np.sum(im_region * self.filters, axis = (1,2))
        
        return output
    
    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        # 卷积层反向传播，卷积核梯度=后一层梯度*输入
        for im_region, i, j in self.iterator_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i,j,f] * im_region
             
        # 梯度下降，卷积核更新值=学习率*梯度   
        self.filters -= learn_rate * d_L_d_filters

    def get_filter(self):
        return self.filters

##池化层
class MaxPool2:
    def iterate_regions(self, image):
        h,w,_ = image.shape
        new_h = h // 2
        new_w = w // 2
        
        # 将输入分割为2x2的图像块
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2),(j*2):(j*2+2)]
                yield im_region, i, j
        
    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))

        # 图像块内最大值作为该处输出值，完成最大池化
        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region, axis = (0,1))
        
        return output

    def backprop(self,d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        
        # 池化层反向传播，最大值处梯度等于后一层该最大化池梯度，其余位置梯度为0
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0,1))
        
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2,j2,f2] == amax[f2]:
                            d_L_d_input[i+i2,j+j2,f2] = d_L_d_out[i,j,f2]
        
        return d_L_d_input
    
##全连接层（用softmax进行归一化）
class Softmax:
    def __init__(self,input_len,nodes, initmethod='create', weights='./weights.xls', biases='./biases.xls'):
        if initmethod == 'create':
            self.weights = np.random.randn(input_len,nodes) / input_len
            self.biases = np.zeros(nodes)
        elif initmethod == 'load':
            self.weights = np.array(excel2matrix(weights))
            self.biases = np.array(excel2matrix(biases))            
            self.weights = np.squeeze(self.weights)
            self.biases = np.squeeze(self.biases)

    def forward(self, input):
        self.last_input_shape = input.shape

        input = input.flatten()

        self.last_input = input

        input_len, nodes = self.weights.shape

        # 首先将输出变成一维向量，与权重做矩阵乘、加上偏置
        totals = np.dot(input,self.weights) + self.biases

        # print(np.shape(input))
        # print("**************")
        # print(np.shape(self.weights))
        # print("**************")
        # print(np.shape(self.biases))
        # print("**************")

        self.last_totals =totals

        # Softmax激活函数得到输出值
        exp = np.exp(totals)
        return exp/np.sum(exp,axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
        
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            
            # 对Softmax函数求梯度：
            # c==k时，dout_c/dt_k = exp(t_k)*exp(t_c)/(Σ(exp(t)))^2
            # c!=k时，dout_c/dt_k = exp(t_k)*(Σ(exp(t))exp(t_k))/(Σ(exp(t)))^2
            #先全部赋值为c!=k的时候的值，再单独修改c==k的时候的值
            d_out_d_t = -t_exp[i]*t_exp/(S**2)
            d_out_d_t[i] = t_exp[i]*(S-t_exp[i])/(S**2)
            
            # 对t求梯度：
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            
            # 用后一层梯度求本层梯度
            d_L_d_t = gradient*d_out_d_t
            
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]   #向量转化为矩阵
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t
            
            # 梯度下降更新权重和偏置
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            
            return d_L_d_inputs.reshape(self.last_input_shape)
    
    def get_w_b(self):
        return self.weights, self.biases
        
def excel2matrix(path, sheet_name = 'page_1'):
    data = xlrd.open_workbook(path,formatting_info = True)
    sheet = data.sheet_by_name(sheet_name)
    data_list = []
    for rows in range(1,sheet.nrows):
        templist = []
        for cols in range(1, sheet.ncols):
            if cols == 0:
                templist.append(rows)
            else:
                templist.append(sheet.cell_value(rows, cols))
        data_list.append(templist)
    
    return data_list