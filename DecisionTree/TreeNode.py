
class TreeNode:
    def __init__(self, data, classification=None, father=None,
                    classify_attr=None, classify_attr_value=None,
                    entro = None, size = None):
        self.data = data #节点数据集
        self.childs = [] #子节点,叶节点为空
        self.classification = classification  #分类标签,非叶节点为空
        self.attributes = [] #可用属性集,叶节点为空
        self.father = father #父节点
        self.classify_attr = classify_attr #分支判断属性
        self.classify_attr_value = classify_attr_value #分支判断属性值
        self.entro = entro #叶节点经验熵,非叶节点为空
        self.size = size #叶节点数据集大小
        self.notCut = False #不剪枝标记