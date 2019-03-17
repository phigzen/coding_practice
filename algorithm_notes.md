1. 二维数组中的查找：在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

```python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        for ii in array:
            for jj in ii:
                if target == jj:
                    return True
        return False     
```
2. 替换空格：请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        raw = ''
        for i in list(s):
            if i ==' ':
                raw+= '%20'
            else:
                raw+=i
        return raw
```

3. 从尾到头打印链表：输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        l=[]
        while listNode:
            l.append(listNode.val)
            listNode=listNode.next
        return l[::-1]
```