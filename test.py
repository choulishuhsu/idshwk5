from sklearn.svm import SVC
import math
from sklearn.preprocessing import StandardScaler
p={}
#构造训练数据集
def splitset(txt):
    #分割domain和label
    data=[];labelmat=[]
    f1=open(txt)
    for line in f1.readlines():
        curline=[]
        line=line.split(",")
        data.append(line[0])
        labelmat.append(line[1].strip('\n'))
    return data,labelmat
#获得字符的熵
def createntropy(txt):
    data,labelmat=splitset(txt)
    f1 = open(txt)
    al=""
    for i in range(0-9):
        p[i]=0
    for line in f1.readlines():
        line = line.split(".")
        al=al+line[0]
    for i in al:
        p[i]=al.count(i)
    for i in p:
        p[i]=p[i]/len(al)
#获得域名的长度
def getlength(url):
    num=0
    for i in url:
        if i !='.':
            num+=1
        else:
            break
    return num
#获得域名数字的个数
def getnum(url):
    num=0
    for i in url:
        if i.isdigit():
            num+=1
    return num
#获得每个域名的熵
def getentropy(url,p):
    num=0
    for i in url:
        if i !='.':
          tem = math.log(p[i])
          num-=p[i]*tem
        else:
            break
    return num
#获得segment的个数
def getsegment(url):
    return url.count('.')+1
#获得训练集的特征矩阵
def gettrain(data):
     datamat=[]
     for i in data:
         cur=[]
         cur.append(getlength(i))
         cur.append(getnum(i))
         cur.append(getentropy(i,p))
         cur.append(getsegment(i))
         datamat.append(cur)

     return datamat
#构造测试集的特征矩阵
def gettest(txt):
    test=[]
    testmat=[]
    f1 = open(txt)
    for line in f1.readlines():
        test.append(line.strip('\n'))
    for i in test:
        cur = []
        cur.append(getlength(i))
#        print(cur)
        cur.append(getnum(i))
#        print(cur)
        cur.append(getentropy(i, p))
#        print(cur)
        cur.append(getsegment(i))
#        print(cur)
        testmat.append(cur)

    return testmat
#main
#获得特征向量
data,labelmat=splitset("train.txt")
createntropy("train.txt")
train_data=gettrain(data)
test_data=gettest("test.txt")
#数据处理
std_train=StandardScaler().fit(train_data)
data_stdtrain=std_train.transform(train_data)
data_stdtest=std_train.transform(test_data)
#print(data_stdtrain)
#print(data_stdtest)
#训练
svm=SVC().fit(data_stdtrain,labelmat)
#print('建立的SVM模型为：\n',svm)
#预测
data_pred=svm.predict(data_stdtest)

#输出
f=open('result.txt',mode="w")
g=open('test.txt',mode="r")
num=0
for line in g:
    f.write(line.strip('\n')+","+data_pred[num]+'\n')
    num+=1
print("finish")
#print('预测前3个结果为：\n',data_pred[:3])