import pandas as pd
from TreeNode import *
import math
import copy
#参数
TRESHOLD = 0.05
ALPHA = 0.8
#全局变量
root = None
max_deep = 0
leaf_number = 0
toCutLeaf = None
toCutRootEntroSum =0
toCutRootLeafSum =0
D_concat = None

def init():
    #print("start")
    data_source = pd.read_csv('./titanic.txt')
    D = data_source[['pclass','age','sex','embarked','survived']]
    #空缺数据填充
    D['age'].fillna(D['age'].mean(),inplace=True)
    D['embarked'].fillna('null',inplace=True)
    D = D.sample(frac=1.0)  # 打乱数据
    cut_index = int(round(0.2 * D.shape[0]))
    D_test = D.iloc[:cut_index]
    D = D.iloc[cut_index:]
    #创建根节点
    root = TreeNode(D)
    Xcolumns = D.columns.values.tolist()
    Xcolumns.remove('survived')
    root.attributes = copy.deepcopy(Xcolumns)
    #建树
    bulidTree(root)
    print("决策树构建完毕")
    Test(root, D)
    #剪枝
    leafCut_init(root)
    print("剪枝完毕")
    #测试
    Test(root,D_test)
    print('end')


#计算某个数据集的经验熵
def getEntro(D):
    D_size = D.shape[0] + 0.0
    groups = D.groupby('survived')
    entro = 0
    for name,group in groups:
        Di_size = group.shape[0]
        p = Di_size/D_size
        entro += p*math.log(p,2)
    #print("计算出此分组信息熵为: %s"%(-1*entro))
    return -1*entro

#得到信息增益最大的属性
def getBestAttribute(attributes,D):
    D_size = D.shape[0] + 0.0
    D_entro = getEntro(D)
    max_gain = 0
    best_attr = None
    for attribute in attributes:
        #按每个属性进行分组
        groups = D.groupby(attribute)
        #遍历每一组,计算经验熵
        sum = 0 #定义经验条件熵
        for name,group in  groups:
            Di = group
            Di_size = Di.shape[0]
            Di_entro  = getEntro(Di)
            sum += (Di_size/D_size)*Di_entro
        gain = D_entro - sum
        if(gain>max_gain):
            max_gain = gain
            best_attr = attribute
    if(max_gain>TRESHOLD):
        #print("该节点的信息增益为%s "%max_gain)
        return best_attr
    else :
        #print("该节点的信息增益小于阈值%s "%TRESHOLD)
        return  None

#构建决策树
def bulidTree(root):
    attributes = copy.deepcopy(root.attributes)
    #print("此节点分类属性为: ")
    #print(attributes)
    D = root.data
    #case1 如果所有实例属于同一类,该类作为标记,返回叶节点
    if D.groupby('survived').ngroups==1:
        root.classification = D['survived'].values.tolist()[0]
        root.size = D.shape[0]
        root.entro = 0
        #print("此节点的类标记为: %s"%root.classification)
        return
    #case2 如果只有一个属性,取实例数最多的分类为结点的分类
    if len(attributes)==1:
        leaf_groups = D.groupby(attributes[0])
        max_count = 0
        classfy = None
        for name,group in leaf_groups:
            count = group['survived'].values.tolist()[0]
            if count>=max_count:
                max_count = count
                classfy = name
        root.classification = classfy
        root.size = D.shape[0]
        root.entro = getEntro(D)
        return
    bestAttribute = getBestAttribute(attributes, D)
    #print("找到最佳属性为 %s"%bestAttribute)
    #case3-1 如果信息增益小于阈值,置为叶节点
    if bestAttribute is None:
        leaf_groups = D.groupby('survived')
        max_count = 0
        classfy = None
        for name, group in leaf_groups:
            count = group['survived'].values.tolist()[0]
            if count >= max_count:
                max_count = count
                classfy = name
        root.classification = classfy
        root.size = D.shape[0]
        root.entro = getEntro(D)
        return
    else:
    #case 3-2 如果信息增益大于阈值,分裂节点,递归构造树
        #移除此属性
        attributes.remove(bestAttribute)
        #按最佳属性划分数据集
        groups = D.groupby(D[bestAttribute])
        for name,group in groups:
            childNode = TreeNode(group,father=root,
                                 classify_attr=bestAttribute, classify_attr_value=name)
            childNode.attributes = attributes
            root.childs.append(childNode)
    for child in root.childs:
        bulidTree(child)

    return

#遍历决策树,得到最大深度和叶节点个数
def getMaxDeep(root,deep):
    global max_deep
    global leaf_number
    deep += 1
    if (root.classification != None):
        leaf_number += 1
        if deep > max_deep:
            max_deep = deep
        return
    if(len(root.childs)!=0):
        for child in root.childs:
            getMaxDeep(child, deep)
    return

#得到待剪枝的叶节点(满足层高==max_deep 且 其父节点未被标记不剪枝),否则toCutLeaf=None
def getleaf(root,deep):
    global toCutLeaf
    deep += 1
    if(root.classification!=None):
        if deep == max_deep and root.father.notCut is False:
            toCutLeaf = root
        return
    for child in root.childs:
        if(child.notCut is True):
            continue
        if(toCutLeaf is not None):
            break
        getleaf(child,deep)
    return

#计算该节点的子树的经验熵之和以及叶子节点个数
def getEntroAndLeafSum(root):
    global toCutRootEntroSum
    global toCutRootLeafSum
    if(root.classification!=None):
        toCutRootLeafSum += 1
        toCutRootEntroSum+=(root.entro*root.size)
    else:
        for child in root.childs:
            getEntroAndLeafSum(child)
    return

#将该节点子树的数据集合并
def dataConcat(root):
    global D_concat
    if(root.classification!=None):
        if(D_concat is None):
            D_concat = root.data
        else:
            D_concat = pd.concat([D_concat, root.data])
        return
    else:
        for child in root.childs:
            dataConcat(child)

#对待剪枝的子树进行处理, 根据考虑决策树复杂度的损失函数在剪枝后是否变小决定是否剪枝
def leafCut(toCutRoot):
    if(toCutRoot is None):
        return
    #计算toCutRoot子树的经验熵之和
    #计算toCutRoot子树的叶子节点个数
    global toCutRootEntroSum
    global toCutRootLeafSum
    toCutRootEntroSum = 0
    toCutRootLeafSum =0
    getEntroAndLeafSum(toCutRoot)
    #计算剪枝后,回缩的叶节点的经验熵
    global D_concat
    D_concat = None
    dataConcat(toCutRoot)
    D_concat_size = D_concat.shape[0]
    Entro_concat = getEntro(D_concat)
    #判断是否需要剪枝
    CostAfter_plus_CostBefore = D_concat_size*Entro_concat-toCutRootEntroSum+ALPHA*(1-toCutRootLeafSum)
    if CostAfter_plus_CostBefore<=0:
        #剪枝
        toCutRoot.childs = []
        toCutRoot.data = D_concat
        #得到数量最多的实例类作为标记类
        leaf_groups = D_concat.groupby('survived')
        max_count = 0
        classfy = None
        for name, group in leaf_groups:
            count = group['survived'].values.tolist()[0]
            if count >= max_count:
                max_count = count
                classfy = name
        toCutRoot.classification = classfy
        toCutRoot.size = D_concat.shape[0]
        toCutRoot.entro = getEntro(D_concat)
    else:
        #不剪枝
        #子树根节点标记为不剪枝
        toCutRoot.notCut = True
    return

def leafCut_init(root):
    global max_deep
    global toCutLeaf
    #获取剪枝需要的基础参数, 节点最大深度和叶节点个数
    getMaxDeep(root, 0)
    #若根节点为None或者为单节点,不剪枝直接返回
    if root==None or len(root.childs)==0 :
        return
    #外层循环不断的从叶到根向上推
    while max_deep >=2  :
        #内层循环找到深度最大的待剪枝结点
        toCutLeaf = None
        while max_deep >= 2:
            getleaf(root, 0)
            if toCutLeaf is None:
                max_deep -= 1
            else:
                break
        if toCutLeaf is None:
            break
        toCutRoot = toCutLeaf.father  #定位到叶节点的父节点
        #进行剪枝
        leafCut(toCutRoot)
    return

def calculate(root,row):
    result = None
    while(root.classification is None):
        find = False
        for child in root.childs:
            if row[child.classify_attr] == child.classify_attr_value:
                root = child
                find = True
                break
        #如果row中属性值不在树中,取第一个
        if(find is False):
            root = root.childs[0]
    return root.classification

def Test(root, D_test):
    yes = 0
    no = 0
    for index,row in D_test.iterrows():
        result = calculate(root,row)
        if result==row['survived']:
            yes += 1
        else:
            no += 1
    accuracy = round((yes+0.0)/(yes+no),4)
    print("决策树预测准确率为: %s%%"%(accuracy*100))

if __name__ == '__main__':
    init()