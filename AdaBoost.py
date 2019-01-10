
"""
机器学习-AdaBoost算法
姓名：pcb
日期：2019.1.2
"""
from numpy import *
import matplotlib.pyplot as plt
#创建一个简单的数据集
def loadSimpData():
    datMat=matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classLabel=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabel


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=ones((shape(dataMatrix)[0],1))                    #将数组中的元素全部设为1
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0          #将所有不满足不等式要求的的元素设置为-1
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

"""
程序的伪代码：
    将最小的错误率minError设为+无穷大
    对数据集中的每一个特征（第一层循环）：
        对每个步长（第二层循环）：
            对每个不等号（第三层循环）：
                建立一棵单层决策树，并利用加权数据集对它进行测试
                如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
    返回最佳单层决策树
"""
#将遍历stumpClassify函数中所有可能的输入值，找到数据集上最佳的单层决策树
def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr);labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0                                            #用于在特征所有可能值上遍历
    bestStump={}                                             #空字典，用于存储给定权重的向量D时所得到最佳单层
    bestClasEst=mat(zeros((m,1)))                            #
    minError=inf                                             #用于寻找可能的最小的错误率
    for i in range(n):                                       #在数据集上的所有特征上遍历
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps                #通过计算最小值和最大值计算需要多长的步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:                      #最后一个循环在在大于和小于之间切换不等式
                thresVal=(rangeMin+float(j)*stepSize)
                predictedVal=stumpClassify(dataMatrix,i,thresVal,inequal)
                errArr=mat(ones((m,1)))                      #构建一个列向量，判断
                errArr[predictedVal==labelMat]=0
                weightedError=D.T*errArr
                #print('split:dim %d,thresh %.2f,thresh ineqal:%s,the weighted error is %.3f'
                      #%(i,thresVal,inequal,weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVal.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=thresVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst

#基于单层决策树的AdaBoost训练过程
def adaBoostTrianDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)                  #D是一个概率分布向量，所有元素之和是1.0
    aggClassEst=mat(zeros((m,1)))         #记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D) #返回则是利用D而得到的具有最小错误率的单层决策树
        #print("D:",D.T)
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        #print("classEst:",classEst.T)
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        #print("aggClassEst:",aggClassEst.T)
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        print("total error:",errorRate,"\n")

        if errorRate==0.0:
            break
    return weakClassArr,aggClassEst

#基于adaBoost分类算法的测试函数
def adaclassify(datToClass,classifierArr):
    dataMatrix=mat(datToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)


#加载数据
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#1.ROC(接收者操作特征，receiver operating characteristic)曲线的绘制
#2.AUC(曲线下的面积，Area Unser the Curve,AUC)计算函数
#3.分类器需要提供每个样例被判为阳性或者阴性的可信程度值
def plotROC(predStrengths,classLabels):
    cur=(1.0,1.0)                                #构建一个浮点数的二维元组，该元组保留的是绘制光标的位置
    ySum=0.0                                     #用于计算AUC的值
    numPosClas=sum(array(classLabels)==1.0)      #通过数组过滤方式计算正例的数目，并将该值赋给numPosClas
    yStep=1/float(numPosClas)                    #确定坐标轴上的步长
    xStep=1/float(len(classLabels)-numPosClas)
    sortedIndicies=predStrengths.argsort()       #对分类器预测的强度进行排序(从小到大)得到排序的索引,

    #构建画笔
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:     #调用tolist,产生一个表进行迭代循环
      if classLabels[index]==1.0:                #每得到一个标签为1.0的类，则沿着y轴的方向下降一个步长，即不断降低真阳率
          delX=0
          delY=yStep
      else:
          delX=xStep                             #对于其他类别则需要在x轴上倒退一个步长
          delY=0
          ySum+=cur[1]                           #所有的高度和（ySum）随着x轴的每次移动而增加，计算总的AUC=ySum*xStep

      #一旦决定了在X轴或者Y轴方向进行移动，则可以在当前点和新点之间画一条线段，然后跟新当前点
      ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='r')
      cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')
    ax.axis([0,1,0,1])
    plt.savefig('AdaBoost马疝病检测系统的ROC曲线.png')
    plt.show()
    print('the Area Under the curve is :',ySum*xStep)




def main():

# #1.-----------测试算法------------------------------------------
#     D=mat(ones((5,1))/5)
#     datMat,classLabels=loadSimpData()
#     #buildStump(datMat,classLabels,D)
#     classifierArr=adaBoostTrianDS(datMat,classLabels,9)
#     classifierResult=adaclassify([[0,0],[5,5]],classifierArr)
#     print(classifierResult)

# #2.----------在大的数据集上使用AdaBoost算法-----------------------
#     dataArr,labelArr=loadDataSet('horseColicTraining2.txt')
#     classifierArry,aggClassEst=adaBoostTrianDS(dataArr,labelArr,50)
#     testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
#     prediction10=adaclassify(testArr,classifierArry)
#     errArr=mat(ones((67,1)))
#     errCount=errArr[prediction10!=mat(testLabelArr).T].sum()
#     print(errCount/67)

#3.绘制ROC曲线以及AUC计算函数

    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArry,aggClassEst=adaBoostTrianDS(dataArr,labelArr,10)
    plotROC(aggClassEst.T,labelArr)



if __name__=='__main__':
    main()