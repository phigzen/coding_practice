## Python first！！！

## THEN CPP ！！！

## Focus on ！！！

# 1. 二维数组中的查找

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

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
# 2. 替换空格
请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

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

# 3. 从尾到头打印链表
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

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

# 4. 重建二叉树
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

解释：

- **前序遍历(Preorder Traversal )**：NLR（根左右），访问根结点的操作发生在遍历其左右子树之前。
- **中序遍历(Inorder Traversal)**：LNR（左根右），访问根结点的操作发生在遍历其左右子树之中（间）。
- **后序遍历(Postorder Traversal)**：LRN（左右根），访问根结点的操作发生在遍历其左右子树之后。

![img](pics/ab103822e75b5b15c615b68560cb2416.jpg)

注意：由于被访问的结点必是某子树的根，所以**N(Node）**、**L(Left subtree）**和**R(Right subtree）**又可解释为根、根的左子树和根的右子树。NLR、LNR和LRN分别又称为先根遍历、中根遍历和后根遍历。


```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if not pre or not tin:
            return None
        root = TreeNode(pre.pop(0))
        index = tin.index(root.val)
        root.left = self.reConstructBinaryTree(pre, tin[:index])
        root.right = self.reConstructBinaryTree(pre, tin[index + 1:])
        return root

```

# 5. 用两个栈实现队列

用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

解释：
1.栈(stacks)是一种只能通过访问其一端来实现数据存储与检索的线性数据结构，具有后进先出(last in first out，LIFO)的特征。python列表方法使得列表作为堆栈非常容易，最后一个插入，最先取出（“后进先出”）。要添加一个元素到堆栈的顶端，使用`append()` 。要从堆栈顶部取出一个元素，使用 `pop()` ，不用指定索引。

```python
>>> stack = [3, 4, 5]
>>> stack.append(6)
>>> stack.append(7)
>>> stack
[3, 4, 5, 6, 7]
>>> stack.pop()
7
>>> stack
[3, 4, 5, 6]
>>> stack.pop()
6
>>> stack.pop()
5
>>> stack
[3, 4]
```
2.队列(queue)是一种具有先进先出特征的线性数据结构，元素的增加只能在一端进行，元素的删除只能在另一端进行。能够增加元素的队列一端称为队尾，可以删除元素的队列一端则称为队首。

列表也可以用作队列，其中先添加的元素被最先取出 (“先进先出”)；然而列表用作这个目的相当低效。因为在列表的末尾添加和弹出元素非常快，但是在列表的开头插入或弹出元素却很慢 (因为所有的其他元素都必须移动一位)。

若要实现一个队列， `collections.deque` 被设计用于快速地从两端操作。

```python
>>> from collections import deque
>>> queue = deque(["Eric", "John", "Michael"])
>>> queue.append("Terry")           # Terry arrives
>>> queue.append("Graham")          # Graham arrives
>>> queue.popleft()                 # The first to arrive now leaves
'Eric'
>>> queue.popleft()                 # The second to arrive now leaves
'John'
>>> queue                           # Remaining queue in order of arrival
deque(['Michael', 'Terry', 'Graham'])
```

```python
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        self.stack1.append(node)
    def pop(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
```

# 6. 旋转数组的最小数字
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

```python
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        if not rotateArray:
            return 0
        else:
            max_a = max(rotateArray)
            index_a = rotateArray.index(max_a)
            for i in range(index_a+1,len(rotateArray)):
                if rotateArray[i]==max_a:
                    continue
                else:
                    return rotateArray[i]
```

```python
# 二分排序解法：
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        if len(rotateArray) == 0:
            return 0
        left = 0
        right = len(rotateArray) - 1
        mid = 0
        while rotateArray[left] >= rotateArray[right]:
            if right - left == 1:
                mid = right
                break
            mid = left + (right - left) // 2
            if rotateArray[left] == rotateArray[mid] and rotateArray[mid] == rotateArray[right]:
                return self.minInorder(rotateArray, left, right)
            if rotateArray[mid] >= rotateArray[left]:
                left = mid
            else:
                right = mid
        return rotateArray[mid]
    
    def minInorder(self, array, left, right):
        result = array[left]
        for i in range(left+1, right+1):
            if array[i] < result:
                result = array[i]
        return result
```
# 7. 斐波那契数列

大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0），n<=39

思路：用循环不用递归，很容易Stack Overflow.
- Python
```python
class Solution:
    def Fibonacci(self, n):
        res=[0,1,1,2]
        while len(res)<=n:
            res.append(res[-1]+res[-2])
        return res[n]
#########################################
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n <= 1:
            return n
        first, second, third = 0, 1, 0
        for i in range(2, n+1):
            third = first + second
            first = second
            second = third
        return third
```
- C++
```cpp
class Solution {
public:
    int Fibonacci(int n) {
        int f = 0, g = 1;
        while(n--) {
            g += f;
            f = g - f;
        }
        return f;
    }
};
/////////////////////////////////////////////
class Solution {
public:
    int Fibonacci(int n) {
        if(n <= 0)
            return 0;
        if(n == 1)
            return 1;
        int first = 0, second = 1, third = 0;
        for (int i = 2; i <= n; i++) {
            third = first + second;
            first = second;
            second = third;
        }
        return third;
    }
};
```
# 8. 跳台阶
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

思路：假设有n个台阶，跳法为f(n)。n=1时，f(n)=1，n=2时，f(n)=1，n>2时，如果第一次跳1级，剩余的有f(n-1)种跳法，如果第一次跳2级，剩余的有f(n-2)种跳法，可见这就是个斐波那契问题。
- Python
```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        a,b=0,1
        while number > 0:
            c=a+b 
            a,b=b,c 
            number-=1
        return c
```
- C++
```cpp
class Solution {
public:
    int jumpFloor(int n) {
        int f=1,g=2;
        n--;
        while(n--)
        {
            g+=f;
            f=g-f;
        }
        return f;
    }
};
//////////////////////////////
class Solution {
public:
    int jumpFloor(int number) {
        if(number<4){
            return number;
        }
        int a=2, b=3, c=0;
        for(int i=4;i<=number;i++){
            c=a+b;
            a=b; 
            b=c; 
        }
        return c;
    }
};

```
# 9. 变态跳台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

思路：
- f(0) = 0
- f(1) = 1
- f(2) = f(2-1) + f(2-2)         # f(2-2) 表示2阶一次跳2阶的次数。
- f(3) = f(3-1) + f(3-2) + f(3-3) 
- ...
- f(n-1) = f(n-2) + f(n-3) + ... + f(n-(n-1)) + f(n-n) 
- f(n) = f(n-1) + f(n-2) + f(n-3) + ... + f(n-(n-1)) + f(n-n) 

f(n)表示n级台阶的跳法，如果第一次跳1级，剩下则有f(n-1)种跳法，如果第一次跳2级，则有f(n-2)种跳法，... ，如果第一次跳n级，就跳完了。
归纳可得：

 f(n) = 2*f(n-1) = 2^(n-1)

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        return 2**(number-1)
```

# 10. 矩形覆盖

我们可以用2\*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

思路：同样也是斐波那契数列问题，第一个位置可以横着放，也可以竖着放，对应剩下的位置分别有f(n-1)和f(n-2)种放法。则f(n)=f(n-1)+f(n-2)

```cpp
class Solution {
public:
    int rectCover(int number) {
        if(number<=3){
            return number;
        }
        int fn_2=2,fn_1=3,fn=0;
        while(number>3){
            fn=fn_1+fn_2;
            fn_2=fn_1;
            fn_1=fn;
            number--;
        }
        return fn;
    }
};
```

# 11. 二进制中1的个数
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

如果一个整数不为0，那么这个整数至少有一位是1。如果我们把这个整数减1，那么原来处在整数最右边的1就会变为0，原来在1后面的所有的0都会变成1(如果最右边的1后面还有0的话)。其余所有位将不会受到影响。

举个例子：一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，它后面的两位0变成了1，而前面的1保持不变，因此得到的结果是1011.我们发现减1的结果是把最右边的一个1开始的所有位都取反了。这个时候如果我们再把原来的整数和减去1之后的结果做与运算，从原来整数最右边一个1那一位开始所有位都会变成0。如1100&1011=1000.也就是说，把一个整数减去1，再和原整数做与运算，会把该整数最右边一个1变成0.那么一个整数的二进制有多少个1，就可以进行多少次这样的操作。

在Python中，由于负数使用补码表示的，对于负数，最高位为1，而负数在计算机是以补码存在的，往右移，符号位不变，符号位1往右移，最终可能会出现全1的情况，导致死循环。与0xffffffff相与，就可以消除负数的影响。

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        if n<0:
            n = n & 0xffffffff
        while n:
            count += 1
            n = n & (n-1)
        return count
```

# 12. 数值的整数次方
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

需要对base和exponent分情况讨论：
如果base为0，exponent<=0会报错；
其他情况需要讨论exponent的正负。

由于计算机中表示小数（float和double）都会有误差，我们不能直接用==来判断它们是否相等，可以通过比较它们之间的差的绝对值是否为一个极小值（如0.0000001）来判断它们是否相等。但是python中的等于没有误差，可以直接使用：
```python
class Solution:
    def Power(self, base, exponent):
        flag = 0
        result = 1
        if (base == 0) & (exponent<=0) :
            return False
        if exponent < 0:
            flag = 1
        for i in range(abs(exponent)):
            result *= base
        if flag == 1:
            result = 1 / result
        return result
```

# 13. 调整数组顺序使奇数位于偶数前面

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

python常规解法1，利用两个列表来存储奇数和偶数，然后再拼接：
```python
class Solution:
    def reOrderArray(self, array):
        odd_lst=[]
        even_lst=[]
        for ele in array:
            if ele%2==0:
                even_lst.append(ele)
            else:
                odd_lst.append(ele)
        final_arr = odd_lst+even_lst
        return final_arr

# 简洁版：
def reOrderArray(self, array):
        # write code here
        odd,even=[],[]
        for i in array:
            odd.append(i) if i%2==1 else even.append(i)
        return odd+even
```
python常规解法2，从后往前遍历奇数，从前往后遍历偶数，然后利用双向队列来存储：
```python
from collections import deque
class Solution:
    def reOrderArray(self, array):
        odd = deque()
        l = len(array)
        for i in range(l):
            if array[-i-1] % 2 != 0:
                odd.appendleft(array[-i-1])
            if array[i] % 2 == 0:
                odd.append(array[i])
        return list(odd)
```
当然也可以用列表来存储，利用insert方法将后向遍历的奇数插到列表的开头。
```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        res = []
        l = len(array)
        for i in range(l):
            if array[-i-1] % 2 != 0:
                res.insert(0,array[-i-1])
            if array[i] % 2 == 0:
                res.append(array[i])
        return res
```
python解法3，不开辟新空间：
```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        boarder = -1
        for idx in range(len(array)):
            if array[idx] % 2:
                boarder += 1
                array.insert(boarder, array.pop(idx))
        return array

# 用sorted()结合key参数：
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        return sorted(array,key=lambda c:c%2,reverse=True)
```

# 14. 链表中倒数第k个结点
输入一个链表，输出该链表中倒数第k个结点。

思路：利用两个指针，让它们之间的距离为k，则当前面的指针到达末尾时候，另一个指针对应的即为倒数第k个节点。但是注意要考虑好各种边界条件。

```python
class Solution:
    def FindKthToTail(self, head, k):
        if (head is None) or (k == 0):
            return None
        a = head
        b = head
        for i in range(k-1):
            if a.next is None:
                return None
            else:
                a = a.next
        while (a.next is not None):
            a = a.next
            b = b.next
        return b
```
另外一种用列表开辟新空间的做法：
```python
class Solution:
    def FindKthToTail(self, head, k):
        res=[]
        while head:
            res.append(head)
            head=head.next
        if k>len(res) or k<1:
            return
        return res[-k]
```


# 15. 反转链表

输入一个链表，反转链表后，输出新链表的表头。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def ReverseList(self, pHead):
        if (pHead is None) or (pHead.next is None):
            return pHead
        p_1 = None
        p0 = pHead
        while (p0 is not None):
            tmp = p0.next
            p0.next = p_1
            p_1 = p0
            p0 = tmp
        return p_1
```

![示意图](pics/image-20190522002319171.png)

# 16. 合并两个排序的链表

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

递归版本：

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        if pHead1 is None:
            return pHead2
        if pHead2 is None:
            return pHead1
        pMerged = None
        if pHead1.val < pHead2.val:
            pMerged = pHead1
            pMerged.next = self.Merge(pHead1.next, pHead2)
        else:
            pMerged = pHead2
            pMerged.next = self.Merge(pHead1, pHead2.next)
        return pMerged
```

非递归版本：

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        pMerged = ListNode(0)
        pHead = pMerged
        while (pHead1 is not None) and (pHead2 is not None):
            if pHead1.val >= pHead2.val:
                pMerged.next = pHead2
                pHead2 = pHead2.next
            else:
                pMerged.next = pHead1
                pHead1 = pHead1.next
            pMerged = pMerged.next
        if (pHead1 is not None):
            pMerged.next = pHead1
        else:
            pMerged.next = pHead2 
        return pHead.next
```

# 17. 树的子结构

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

思路：可以分为两个步骤，首先在A树中寻找有无与B树根节点值相等的节点，如果有，则判断A树中以该节点为根节点的树是不是包含和B树一样的结构。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        if (pRoot1 is None) or (pRoot2 is None):
            return False
        return self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2) or self.is_subtree(pRoot1, pRoot2)
    def is_subtree(self, A, B):
        if (B is None):
            return True
        if (A is None) or (A.val != B.val):
            return False
        return self.is_subtree(A.left, B.left) and self.is_subtree(A.right, B.right)
```

# 18. 二叉树的镜像

操作给定的二叉树，将其变换为源二叉树的镜像。

思路：还是用递归，依次对每个非叶节点进行翻转。

```python
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        if (root is None or (root.left is None and root.right is None)):
            return None
        temp = root.left
        root.left = root.right
        root.right = temp
        if root.left is not None:
            self.Mirror(root.left)
        if root.right is not None:
            self.Mirror(root.right)
```

# 19. 顺时针打印矩阵

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

```python
class Solution:

    def printMatrix(self, matrix):
        res = []
        while matrix:
            res += matrix.pop(0)
            if matrix and matrix[0]: # 一定要记得判断 matrix[0]，否则在第二轮循环的时候会漏判[[],[]]
                for row in matrix:
                    res.append(row.pop())
            if matrix:
                res += matrix.pop()[::-1]
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    res.append(row.pop(0))
        return res
```

也可以循环遍历：

```python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        rows = len(matrix)
        cols = len(matrix[0])
        result = []
        if rows == 0 and cols == 0:
            return result
        left, right, top, buttom = 0, cols - 1, 0, rows - 1
        while left <= right and top <= buttom:
            for i in range(left, right+1):
                result.append(matrix[top][i])
            for i in range(top+1, buttom+1):
                result.append(matrix[i][right])
            if top != buttom:
                for i in range(left, right)[::-1]:
                    result.append(matrix[buttom][i])
            if left != right:
                for i in range(top+1, buttom)[::-1]:
                    result.append(matrix[i][left])
            left += 1
            top += 1
            right -= 1
            buttom -= 1
        return result
```

# 20. 包含min函数的栈
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

思路：定义两个栈stack和min_stack，一个存储所有元素，一个存储最小值。
```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []
        self.min_stack = []
         
    def push(self, node):
        min = self.min()
        if not min or node < min:
            self.min_stack.append(node)
        else:
            self.min_stack.append(min)
        self.stack.append(node)
         
    def pop(self):
        if self.stack:
            self.min_stack.pop()
            return self.stack.pop()
     
    def top(self):
        # write code here
        if self.stack:
            return self.stack[-1]
         
    def min(self):
        # write code here
        if self.min_stack:
            return self.min_stack[-1]

```

# 21. 栈的压入、弹出序列
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
```python
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        stack_ = []
        for i in pushV:
            stack_.append(i)
            while stack_ and stack_[-1] == popV[0]:
                stack_.pop()
                popV.pop(0)
        if stack_:
            return False
        return True
```

![image-20190602192939687](pics/image-20190602192939687.png)

![image-20190602192951826](pics/image-20190602192951826.png)

```python
# -*- coding:utf-8 -*-
class Solution:
 
    def IsPopOrder(self, pushV, popV):
        # stack中存入pushV中取出的数据
        stack=[]
        while popV:
            # 如果第一个元素相等，直接都弹出，根本不用压入stack
            if pushV and popV[0]==pushV[0]:
                popV.pop(0)
                pushV.pop(0)
            #如果stack的最后一个元素与popV中第一个元素相等，将两个元素都弹出
            elif stack and stack[-1]==popV[0]:
                stack.pop()
                popV.pop(0)
            # 如果pushV中有数据，压入stack
            elif pushV:
                stack.append(pushV.pop(0))
            # 上面情况都不满足，直接返回false。
            else:
                return False
        return True
```

# 22. 从上往下打印二叉树

从上往下打印出二叉树的每个节点，同层节点从左至右打印。

思路：其实就是广度优先遍历二叉树。可以借助于一个队列实现。每次打印一个节点，并判断该节点的左右节点是否存在，如果存在则将它们依次放入队列的尾部，并从队列的头部开始重复上述操作。

不论是广度优先遍历一个有向图还是一棵树，都要用到队列。首先把起始节点(根节点)放入队列，接下来每次从队列的头部取出一个节点，遍历该节点后把这个节点都能到达的节点（子节点）依次放入队列，重复该过程直到队列中的所有节点都被遍历为止。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        result = []
        if root is None:
            return result
        queue = [root]
        while queue:
            cur = queue.pop(0)
            result.append(cur.val)
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
        return result
    
```

# 23. 二叉搜索树的后序遍历序列

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

思路：

> 二叉查找树（Binary Search Tree），（又：二叉搜索树，二叉排序树）它或者是一棵空树，或者是具有下列性质的二叉树： 若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值； 若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值； 它的左、右子树也分别为二叉排序树。          
>
> --baidu百科

二叉搜索树的后序遍历序列中，最后一个数字是树的根节点 ，数组中前面的数字可以分为两部分：第一部分是左子树节点 的值，都比根节点的值小；第二部分是右子树节点的值，都比根节点的值大，可用递归分别判断前后两部分是否符合以上原则。

```python

# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not len(sequence):
            return False
        if len(sequence) == 1:
            return True
        length = len(sequence)
        # 最后一个是根节点
        root = sequence[-1]
        i = 0
        # 遍历序列，寻找左右子树分界点
        while sequence[i] < root:
            i = i + 1
        k = i # 右树的起点
        # 根节点右侧子树应该都比根节点值大，如果小则返回False
        for j in range(i, length-1):
            if sequence[j] < root: 
                return False
        left_tree_s = sequence[:k] # 左树序列
        right_tree_s = sequence[k:length-1] # 右树序列
        left, right = True, True
        if len(left_tree_s) > 0:
            left = self.VerifySquenceOfBST(left_tree_s)
        if len(right_tree_s) > 0:
            right = self.VerifySquenceOfBST(right_tree_s)
        return left and right

```

```python
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if sequence is None or len(sequence)==0:
            return False
        length=len(sequence)
        root=sequence[-1]
        # 在二叉搜索 树中 左子树节点小于根节点
        for i in range(length):
            if sequence[i]>root:
                break
        # 二叉搜索树中右子树的节点都大于根节点
        for j  in range(i,length):
            if sequence[j]<root:
                return False
        # 判断左子树是否为二叉树
        left=True
        if  i>0:
            left=self.VerifySquenceOfBST(sequence[0:i])
        # 判断 右子树是否为二叉树
        right=True
        if i<length-1:
            right=self.VerifySquenceOfBST(sequence[i:-1])
        return left and right
```

# ?24. 二叉树中和为某一值的路径

输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if root is None:
            return []
        if (root.left is None) and (root.right is None) and (expectNumber == root.val):
            return [[root.val]]
        result = []
        left = self.FindPath(root.left, expectNumber-root.val)
        right = self.FindPath(root.right, expectNumber-root.val)
        for i in left+right:
            result.append([root.val]+i)
        return result
```

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if not root:
            return []
        result = []
        def FindPathMain(root, path, currentSum):
            currentSum += root.val
            path.append(root)
            isLeaf = (root.left is None) and (root.right is None)
            if (currentSum == expectNumber) and isLeaf:
                onePath = []
                for node in path:
                    onePath.append(node.val)
                result.append(onePath)
            if currentSum < expectNumber:
                if root.left:
                    FindPathMain(root.left, path, currentSum)
                if root.right:
                    FindPathMain(root.right, path, currentSum)
            path.pop()
        FindPathMain(root, [], 0)
        return result
```

# ?25. 复杂链表的复制

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

第一步，复制原始链表。![image-20190604094825820](pics/image-20190604094825820.png)

![image-20190604093417728](pics/image-20190604093417728.png)

```python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        if pHead is None:
            return None
        temp = pHead
        # first step: copy nodes except random nodes
        while temp:
            tempnext = temp.next
            copynode = RandomListNode(temp.label)
            copynode.next = tempnext
            temp.next = copynode
            temp = tempnext
        temp = pHead # back the Head
        # second step: copy random
        while temp:
            temprandom = temp.random
            copynode = temp.next
            if temprandom:
                copynode.random = temprandom.next
            temp = copynode.next
        # third step: split
        temp = pHead # back the Head
        copyHead = pHead.next
        while temp:
            copyNode = temp.next
            tempnext = copyNode.next
            temp.next = tempnext
            if tempnext:
                copyNode.next = tempnext.next
            else:
                copyNode.next = None
            temp = tempnext
        return copyHead
```

递归法：

```python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    def Clone(self, head):
        if head is None: 
          return None
        newNode = RandomListNode(head.label)
        newNode.random = head.random
        newNode.next = self.Clone(head.next)
        return newNode
```

哈希表：

```python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None

class Solution:
    def Clone(self, head):
        nodeList = []     # 存放各个节点
        randomList = []   # 存放各个节点指向的random节点。没有则为None
        labelList = []    # 存放各个节点的值
        while head: # 将链表转换为列表
            randomList.append(head.random)
            nodeList.append(head)
            labelList.append(head.label)
            head = head.next
        # 获取random节点的索引，如果没有则为-1   
        labelIndexList = map(lambda c: nodeList.index(c) if c else -1, randomList)
        dummy = RandomListNode(0)
        pre = dummy
        # 节点列表，只要把这些节点的random设置好，顺序串起来就ok了。
        nodeList=map(lambda c:RandomListNode(c),labelList)
        #把每个节点的random绑定好，根据对应的index来绑定
        for i in range(len(nodeList)):
            if labelIndexList[i]!=-1:
                nodeList[i].random=nodeList[labelIndexList[i]]
        for i in nodeList:
            pre.next=i
            pre=pre.next
        return dummy.next
```



# ?26. 二叉搜索树与双向列表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        if pRootOfTree is None:
            return None
        p = pRootOfTree
        stack = []
        resStack = []
        while p or stack:
            if p:
                stack.append(p)
                p = p.left
            else:
                node = stack.pop()
                resStack.append(node)
                p = node.right
        resP = resStack[0]
        while resStack:
            top = resStack.pop(0)
            if resStack:
                top.right = resStack[0]
                resStack[0].left = top
        return resP
```

```python
class Solution:
    def Convert(self, pRootOfTree):
        # write code here
        if not pRootOfTree:return
        self.arr = []
        self.midTraversal(pRootOfTree)
        for i,v in enumerate(self.arr[:-1]):
            v.right = self.arr[i+1]
            self.arr[i + 1].left = v
        return self.arr[0]
    
    def midTraversal(self, root):
        if root is None: 
            return None
        self.midTraversal(root.left)
        self.arr.append(root)
        self.midTraversal(root.right)
```

递归解法，比较清晰：

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, root):
        if root is None:
            return None
        if (root.left is None) and (root.right is None):
            return root
        # 将左子树构建成双链表，返回链表头：
        left = self.Convert(root.left)
        p = left
        # 定位至左子树的最右的一个结点:
        while left and p.right:
            p = p.right
        # 如果左子树不为空，将当前root节点加入左子树链表：
        if left is not None:
            p.right = root
            root.left = p
        # 将右子树构造成双链表，返回链表头：
        right = self.Convert(root.right)
        # 如果右子树不为空，将该链表追加到root结点之后：
        if right is not None:
            right.left = root
            root.right = right
        if left is not None:
            return left
        else:
            return root
```

# ??27. 字符串的排列

题目描述：输入一个字符串，按字典序打印出该字符串中字符的所有排列。例如输入字符串abc, 则打印出由字符a, b, c所能排列出来的所有字符串abc, acb, bac, bca, cab和cba。

输入描述：输入一个字符串,长度不超过9(可能有字符重复), 字符只包括大小写字母。

```python
class Solution:
    def Permutation(self, ss):
        # write code here
        result = []
        if len(ss) <= 1:
            return ss
        for i in range(len(ss)):
            for n in map(lambda x: x+ss[i], self.Permutation(ss[:i]+ss[i+1:])):
                if n not in result:
                    result.append(n)
        return sorted(result)
```

![image-20190605085920705](pics/image-20190605085920705.png)

```python
# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        # write code here
        if not ss:
            return []
        if len(ss) <= 1:
            return [ss]
        result = []
        # 遍历字符串，固定第一个元素，然后递归求解
        for i in range(len(ss)):
            for j in self.Permutation(ss[:i]+ss[i+1:]):
                result.append(ss[i]+j)
        # 通过set进行去重，sorted进行重新排序
        return sorted(list(set(result))) 
```

# 28. 数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

常规写法：对于列表中的元素取不重复的集合，然后遍历统计，如果某个元素超过了列表长度的一半则返回。

```python
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        ident = list(set(numbers))
        length = len(numbers)
        half_threshold = length//2+1
        for ident_i in ident:
            count = 0
            for num_i in numbers:
                if num_i == ident_i:
                    count += 1
            if count >= half_threshold:
                return ident_i
        return 0
```

思路：

如果是排序好的数组，如果这个数存在，则中间那个数一定是我们要找的那个数，也是统计学上的中位数！

# 29. 最小的K个数

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

思路：本题涉及各种排序，可以先对序列排序然后选出前K个数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if k > len(tinput):
            return []
        tinput.sort()
        return tinput[:k]
```

用快速排序：

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        def quick_sort(lst):
            if not lst:
                return []
            pivot = lst[0]
            left = quick_sort([x for x in lst[1: ] if x < pivot])
            right = quick_sort([x for x in lst[1: ] if x >= pivot])
            return left + [pivot] + right
        if tinput == [] or k > len(tinput):
            return []
        tinput = quick_sort(tinput)
        return tinput[: k]
```

# 30. 连续子数组的最大和

HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

思路：如果枚举所有的可能性，则含有n个元素的数组的子数组有$\frac{n(n+1)}{2} $ 种可能性。所以计算出所有子数组的和最快也要$O(n^2)$的时间。

通过分析我们发现，累加的子数组和，如果大于零，那么我们继续累加就行；否则，则需要剔除原来的累加和重新开始。如下图：

![image-20190606093616835](pics/image-20190606093616835.png)

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if not array:
            return 0
        max_ = array[0] # 记录当前最大子数组和
        marker_ = array[0] # 记录当前的子数组和
        
        for ele in array[1:]:
            if marker_ < 0 :
                marker_ = ele
            else:
                marker_ += ele
            if marker_ >= max_:
                max_ = marker_
        return max_
```

动态规划：
$$
f(i)=\left\{\begin{array}{ll}{\text { pData }[i]} & {i=0\ 或者\ f(i-1)\leq0} \\ {f(i-1)+\text { pData }[i]} & {i \neq 0\ 或者\ f(i-1)>0}\end{array}\right.
$$
$f(i)$表示以第$i$个数字结尾的子数组的最大和，那我们需要求出$max(f(i))$。

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        if not array:
            return 0
        max_ = [array[0]]
        for i,num in enumerate(array[1:],1):
            if max_[i - 1] <= 0:
                max_.append(num)
            else:
                max_.append(max_[i - 1] + num)
        return max(max_)
```

# 31. 整数中1出现的次数(从1到n整数中1出现的次数)

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        n_str = [p for i in range(1,n+1) for p in str(i)]
        n_str_filter = [i for i in n_str if i == '1' ]
        return len(n_str_filter)
```

？更好的解法：

```python
# -*- coding:utf-8 -*-
class Solution:

    def NumberOf1Between1AndN_Solution(self, n):
        """
        :type n: int
        :rtype: int

        例：对于824883294，先求0－800000000之间（不包括800000000）的，再求0－24883294之间的。
        如果等于1，如1244444，先求0－1000000之间，再求1000000－1244444，那么只需要加上244444＋1，再求0－244444之间的1
        如果大于1，例：0－800000000之间1的个数为8个100000000的1的个数加上100000000，因为从1000000000－200000000共有1000000000个数且最高位都为1。
        对于最后一位数，如果大于1，直接加上1即可。
        """
        result = 0
        if n < 0:
            return 0
        length = len(str(n))
        listN = list(str(n))
        for i, v in enumerate(listN):
            a = length - i - 1  # a为10的幂
            if i==length-1 and int(v)>=1:
                result+=1
                break
            if int(v) > 1:
                result += int(10 ** a * a / 10) * int(v) + 10**a
            if int(v) == 1:
                result += (int(10 ** a * a / 10) + int("".join(listN[i+1:])) + 1)
        return result
```



# 32. 把数组排成最小的数

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

思路：最常规的做法是把所有的可能性都列出来。$n$个数会有$n!$种组合，算法的时间复杂度较高，为O(n!)。

```python
class Solution:
    def PrintMinNumber(self, numbers):
        if not numbers: return ""
        numbers = list(map(str, numbers))
        numbers.sort(cmp=lambda x, y: cmp(x + y, y + x))
        return "".join(numbers).lstrip('0') or'0'
```

```python
class Solution:
    def compare(self,num1,num2):
        t = str(num1)+str(num2)
        s = str(num2)+str(num1)
        if t>s:
            return 1
        elif t<s:
            return -1
        else:
            return 0
    def PrintMinNumber(self, numbers):
        # write code here
        if numbers is None:
            return ""
        lens = len(numbers)
        if lens ==0:
            return ""
        tmpNumbers = sorted(numbers,cmp=self.compare)
        return int(''.join(str(x)for x in tmpNumbers))

```

注意：`cmp`参数在python3中已经被取消了，换成了`key`，`cmp(f(a), f(b))`需要被改成`f(item)`的形式。也可以用[`functools.cmp_to_key`](https://docs.python.org/3/library/functools.html#functools.cmp_to_key)来替代。

```python
actors = [Person('Eric', 'Idle'),
          Person('John', 'Cleese'),
          Person('Michael', 'Palin'),
          Person('Terry', 'Gilliam'),
          Person('Terry', 'Jones')]
# Python 2
def cmp_last_name(a, b):
    """ Compare names by last name"""
    return cmp(a.last, b.last)
  
sorted(actors, cmp=cmp_last_name)
# ['John Cleese', 'Terry Gilliam', 'Eric Idle', 'Terry Jones', 'Michael Palin']

# Python 3
def keyfunction(item):
    """Key for comparison by last name"""
    return item.last

sorted(actors, key=keyfunction)
# ['John Cleese', 'Terry Gilliam', 'Eric Idle', 'Terry Jones', 'Michael Palin']
```

所以在python3中，对于本题只需要修改一行代码：

```python
from functools import cmp_to_key
class Solution:
    def compare(self,num1,num2):
        t = str(num1)+str(num2)
        s = str(num2)+str(num1)
        if t>s:
            return 1
        elif t<s:
            return -1
        else:
            return 0
    def PrintMinNumber(self, numbers):
        # write code here
        if numbers is None:
            return ""
        lens = len(numbers)
        if lens ==0 :
            return ""
        tmpNumbers = sorted(numbers,key=cmp_to_key(self.compare))
        return int(''.join(str(x)for x in tmpNumbers))
```

# 33. 丑数

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

常规写法（因为时间复杂度高无法AC，因为对于每一个整数都得进行取余数和除法计算）：思路是写一个判断是否为丑数的子函数，然后对进行遍历。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if (index<=0) or (not isinstance(index,int)):
            return None
        i = 0 # to mark the ugly number
        num = 1 # to iterate
        while True:
            if self.isUgly(num):
                i+=1
                if i == index:
                    return num
            num+=1
             
    def isUgly(self, number):
        while number%2 == 0:
            number/=2
        while number%3 == 0:
            number/=3
        while number%5 == 0:
            number/=5
        return number==1
```

更好的做法：创建数组保存已找到的丑数，用空间换时间。

根据丑数的定义，每一个丑数都应该是前面的丑数乘以2、3或者5得到的结果。关键问题在于如何保证数组里面的丑数是排好序的。对乘以2而言，肯定存在某一个丑数T2，排在它之前的每一个丑数乘以2得到的结果都会小于已有最大的丑数，在它之后的每一个丑数乘以乘以2得到的结果都会太大。我们只需要记下这个丑数的位置，同时每次生成新的丑数的时候，去更新这个T2。对乘以3和5而言，也存在着同样的T3和T5。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index < 7:
            return index
        result = [1, 2, 3, 4, 5, 6]
        t2, t3, t5 = 3, 2, 1 # mark the index
        # result[t2]*2 > max(result)
        # result[t3]*3 > max(result)
        # result[t5]*5 > max(result)
        for i in range(6, index):
            result.append(min(result[t2] * 2, result[t3] * 3, result[t5] * 5)) # M2, M3, M5中的最小者
            while result[t2] * 2 <= result[i]:
                t2 += 1
            while result[t3] * 3 <= result[i]:
                t3 += 1
            while result[t5] * 5 <= result[i]:
                t5 += 1
        return result[index - 1]
```

# 34. 第一个只出现一次的字符

在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）

常规思路：每次拿到一个字符串，和后面的每个字符串进行比较，如果没有相同的则输出该字符串。但时间复杂度是$O(n^{2})$。

可以借助于哈希表，用空间换时间：

```python
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        length = len(s)
        if length == 0:
            return -1
        dict_s = {}
        for i in range(length):
            if s[i] not in dict_s.keys():
                dict_s[s[i]] = 1
            else:
                dict_s[s[i]] += 1
        for i in range(length):
            if dict_s[s[i]] == 1:
                return i
        return -1
```

python更简洁的写法：

```python
class Solution:
    def FirstNotRepeatingChar(self, s):
        return s.index(list(filter(lambda c:s.count(c)==1,s))[0]) if s else -1

```

首先用字符串的count方法统计其中某个字符出现的次数，然后过滤出count结果为1的字符，返回第一个的索引。完整点的写法：

```python
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        if len(s)<0:
            return -1
        for i in s:
            if s.count(i)==1:
                return s.index(i)
                break
        return -1
```

对于原书中提到的建立哈希表的方法：

```python
class Solution:
    def FirstNotRepeatingChar(self, s):
        # 建立哈希表,字符长度为8的数据类型,共有256种可能,于是创建一个长度为256的列表
        ls=[0]*256
        # 遍历字符串,下标为ASCII值,值为次数
        for i in s:
            ls[ord(i)]+=1
        # 遍历列表,找到出现次数为1的字符并输出位置
        for j in s:
            if ls[ord(j)]==1:
                return s.index(j)
                break
        return -1
```

# ?35. 数组中的逆序对

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

* 常规思路：暴力解法，顺序扫描整个数组，每扫描到一个数字的时候，逐个比较该数字和它后面的数字的大小。如果后面的数字比它小，则这两个数字就组成一个逆序对。假设数组中含有$n$个数字，由于每个数字都要和$O(n)$个数字作比较，因此这个算法的时间复杂度是$O(n^2)$。

* 分治，归并排序：

![image-20190607223904387](pics/image-20190607223904387.png)

```python
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
        if not data:
            return 0
        temp = [i for i in data]
        return self.mergeSort(temp, data, 0, len(data)-1) % 1000000007
    def mergeSort(self, temp, data, low, high):
        if low >= high:
            temp[low] = data[low]
            return 0
        mid = (low + high) / 2
        left = self.mergeSort(data, temp, low, mid)
        right = self.mergeSort(data, temp, mid+1, high)
        count = 0
        i = low
        j = mid+1
        index = low
        while i <= mid and j <= high:
            if data[i] <= data[j]:
                temp[index] = data[i]
                i += 1
            else:
                temp[index] = data[j]
                count += mid-i+1
                j += 1
            index += 1
        while i <= mid:
            temp[index] = data[i]
            i += 1
            index += 1
        while j <= high:
            temp[index] = data[j]
            j += 1
            index += 1
        return count + left + right
```

# 36. 两个链表的第一个公共结点

输入两个链表，找出它们的第一个公共结点。

常规解法：

* 思路一：分别遍历两个链表，每次进行比较，如果两个链表的长度分别为$m$和$n$，则时间复杂度为$O(mn)$。蛮力的方法不是最好的选择。

进一步思考：

![image-20190608110551376](pics/image-20190608110551376.png)

* 思路二：因为这两个链表是单向链表，则如果两个链表上有公共节点，那么这两个链表从某一节点开始，它们的下一个节点都指向同一个节点，并且唯一！因此从第一个公共节点开始，之后它们的所有节点都是重合的，不可能再出现分叉。其拓扑结构是Y而非X。因此我们可以从链表的尾部出发，寻找它们的最后一个公共节点。但是单向链表中我们只能从头结点开始顺序遍历，最后才能到达尾节点，但尾节点却要先用于比较。这是典型的“后进先出”：用栈！可以借助两个辅助栈存储节点，用空间换时间。
* 思路三：首先遍历两个链表得到它们的长度$m$, $n$, 然后第二次遍历在较长的链表上先走$|m-n|$步，接着同时在两个链表上遍历，找到第一个相同的节点就是它们的第一个公共节点。

思路3的python实现：

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        len1 = self.get_length(pHead1)
        len2 = self.get_length(pHead2)
        if len1>len2:
            step0 = len1-len2
            for i in range(step0):
                pHead1 = pHead1.next
        elif len1<len2:
            step0 = len2-len1
            for i in range(step0):
                pHead2 = pHead2.next
        else:
            pass
        final_head = None
        while pHead1 is not None:
            if pHead1 == pHead2:
                final_head = pHead1
                break
            else:
                pHead1 = pHead1.next
                pHead2 = pHead2.next
        return final_head
    def get_length(self,head):
        i=0
        while head is not None:
            i+=1
            head = head.next
        return i
```

思路二：

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        if not pHead1 or not pHead2:
            return None
        stack1 = []
        stack2 = []
        while pHead1:
            stack1.append(pHead1)
            pHead1 = pHead1.next
        while pHead2:
            stack2.append(pHead2)
            pHead2 = pHead2.next
        final_head = None
        while stack1 and stack2:
            top1 = stack1.pop()
            top2 = stack2.pop()
            if top1 is top2:
                final_head = top1
            else:
                break
        return final_head
```

# 37. 数字在排序数组中出现的次数

统计一个数字在排序数组中出现的次数。

常规做法：顺序扫描，时间复杂度为$O(n)$

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        if not data:
            return 0
        count=0
        for i in data:
            if i == k:
                count+=1
        return count
```

python 调用内置函数：

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        return data.count(k)
```

更加通用的做法：

既然是已经排序好的数组，那么第一个想到的就是二分查找法。做法就是使用二分法找到数字在数组中出现的第一个位置，再利用二分法找到数字在数组中出现的第二个位置。时间复杂度为O(logn + logn)，最终的时间复杂度为O(logn)。

具体过程：

**如何使用二分查找算法在数组中找到第一个k：**先拿数组中间的数字与k进行比较，分三种情况，（1）如果中间数字比k大，那么k只可能出现在数组的前半段，下一轮就在前半段内找k。（2）若中间数字比k小，则k只可能出现在数组后半段，下一轮只需在后半段找k。（3）如果中间数字与k相等，我们需要判断这是否为第一个k，如果中间数字的前一个数字不是k，则中间数字正好是第一个k；如果中间数字的前一个数字是k，则第一个k只可能在前半段中，下一轮就在前半段内找k。具体过程我们可以用递归或者循环实现。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        if not data:
            return 0
        if self.GetLastK(data, k) == -1 and self.GetFirstK(data, k) == -1:
            return 0
        return self.GetLastK(data, k) - self.GetFirstK(data, k) + 1
    
    def GetFirstK(self, data, k):
        low = 0
        high = len(data) - 1
        while low <= high:
            mid = (low + high) // 2
            if data[mid] < k:
                low = mid + 1
            elif data[mid] > k:
                high = mid - 1
            else:
                if mid == low or data[mid - 1] != k: # 当到list[0]或不为k的时候跳出函数
                    return mid
                else:
                    high = mid - 1
        return -1

    def GetLastK(self, data, k):
        low = 0
        high = len(data) - 1
        while low <= high:
            mid = (low + high) // 2
            if data[mid] > k:
                high = mid - 1
            elif data[mid] < k:
                low = mid + 1
            else:
                if mid == high or data[mid + 1] != k:
                    return mid
                else:
                    low = mid + 1
        return -1
```

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        l = 0
        r = len(data)-1
        firstIndex  = self.getFirstIndex(data, k, l, r)
        lastIndex = self.getLastIndex(data, k, l, r)
        return lastIndex - firstIndex + 1
    def getFirstIndex(self, data, k, l, r):
        if l > r:
            return -1
        mid = int((r+l)/2)
        if data[mid] == k and (mid==0 or data[mid-1] != k):
            return mid
        else:
            if data[mid]>=k:
                return self.getFirstIndex(data, k, l, mid-1)
            else:
                return self.getFirstIndex(data, k, mid+1, r)
    def getLastIndex(self, data, k, l, r):
        while l<=r:
            mid = int((l+r)/2)
            if data[mid] == k and (mid==len(data)-1 or data[mid+1] != k):
                return mid
            else:
                if data[mid] >k:
                    r = mid -1
                else:
                    l = mid+1
        return -2 # to make (lastIndex - firstIndex + 1) equal 0 when no k in data.
```

# 38. 二叉树的深度

输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

这里的深度定义为：最大的节点数，而非边数。

<img src='pics/image-20190608145024014.png' style="zoom:66%" />

思路：可用递归实现，属于DFS（深度优先搜索）；另一种方法是按照层次遍历，属于BFS（广度优先搜索）。

```python
# DFS
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot is None:
            return 0
        left_depth = self.TreeDepth(pRoot.left)
        right_depth = self.TreeDepth(pRoot.right)
        return max(left_depth, right_depth)+1
```

```python
# BFS
class Solution:
    # 层次遍历
    def levelOrder(self, root):
        # write your code here
        # 存储最后层次遍历的结果
        res = []
        # 层数
        count = 0
        # 如果根节点为空，则返回空列表
        if root is None:
            return count
        # 模拟一个队列储存节点
        q = []
        # 首先将根节点入队
        q.append(root)
        # 列表为空时，循环终止
        while len(q) != 0:
            # 使用列表存储同层节点
            tmp = []
            # 记录同层节点的个数
            length = len(q)
            for i in range(length):
                # 将同层节点依次出队
                r = q.pop(0)
                if r.left is not None:
                    # 非空左孩子入队
                    q.append(r.left)
                if r.right is not None:
                    # 非空右孩子入队
                    q.append(r.right)
                tmp.append(r.val)
            if tmp:
                count += 1  # 统计层数
            res.append(tmp)
        return count
    
    def TreeDepth(self, pRoot):
        # write code here
        # 使用层次遍历
        # 当树为空直接返回0
        if pRoot is None:
            return 0
        count = self.levelOrder(pRoot)
        return count
```

# ?39. 平衡二叉树

输入一棵二叉树，判断该二叉树是否是平衡二叉树。(任意节点的左右子树的深度相差不超过1)

常规做法：借助于38题中的TreeDepth，可以得到二叉树左右子树的深度，然后求差，判断是否符合条件，对所有的节点进行这样的递归判断。但是这样会重复遍历一些节点，影响性能。

<img src='pics/image-20190608145024014.png' style="zoom:66%" />

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if pRoot is None:
            return True
        depth_l = self.TreeDepth(pRoot.left)
        depth_r = self.TreeDepth(pRoot.right)
        delta = abs(depth_l-depth_r)
        if delta<= 1:
            return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
          
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot is None:
            return 0
        left_depth = self.TreeDepth(pRoot.left)
        right_depth = self.TreeDepth(pRoot.right)
        return max(left_depth, right_depth)+1
```

??每个节点只遍历一次的解法：

如果我们用**后序遍历**的方式遍历二叉树的每一个结点，在遍历到一个结点之前我们就已经遍历了它的左右子树。只要在遍历每个结点的时候记录它的深度（某一结点的深度等于它到叶结点的路径的长度），我们就可以一边遍历一边判断每个结点是不是平衡的。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, p):
        return self.dfs(p) != -1
    def dfs(self, p):
        if p is None:
            return 0
        left = self.dfs(p.left)
        if left == -1:
            return -1
        right = self.dfs(p.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return max(left, right) + 1
```

# 40. 数组中只出现一次的数字

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

常规解法：借助于哈希表，两次遍历

```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        dict_count={}
        for i in array:
            if i in dict_count:
                dict_count[i]+=1
            else:
                dict_count[i]=1
        result=[]
        for k,v in dict_count.items():
            if v == 1:
                result.append(k)
        return result
```

如果要求时间复杂度是$O(n)$，空间复杂度是$O(1)$

???借助于异或运算：任何一个数字异或它自己都等于0

```python
class Solution:
    def FindNumsAppearOnce(self, array):
        if not array:
            return []
        # 对array中的数字进行异或运算
        tmp = 0
        for i in array:
            tmp ^= i
        # 获取tmp中最低位1的位置
        idx = 0
        while (tmp & 1) == 0:
            tmp >>= 1
            idx += 1
        a = b = 0
        for i in array:
            if self.isBit(i, idx):
                a ^= i
            else:
                b ^= i
        return [a, b]

    def isBit(self, num, idx):
        """
        判断num的二进制从低到高idx位是不是1
        :param num: 数字
        :param idx: 二进制从低到高位置
        :return: num的idx位是否为1
        """
        num = num >> idx
        return num & 1
```

```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        if len(array) <= 0:
            return []
        resultExclusiveOR = 0
        length = len(array)
        for i in array:
            resultExclusiveOR ^= i
        firstBitIs1 = self.FindFisrtBitIs1(resultExclusiveOR)
        num1, num2 = 0, 0
        for i in array:
            if self.BitIs1(i, firstBitIs1):
                num1 ^= i
            else:
                num2 ^= i
        return num1, num2
        
    def FindFisrtBitIs1(self, num):
        indexBit = 0
        while num & 1 == 0 and indexBit <= 32:
            indexBit += 1
            num = num >> 1
        return indexBit
    
    def BitIs1(self, num, indexBit):
        num = num >> indexBit
        return num & 1
```

# 41. 和为S的连续正数序列

小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序。

我们可以考虑用两个数small和big分别表示序列中的最小值和最大值。首先初始化：small=1，big=2，如果从small到big的序列和大于s，则可以从序列中去掉较小的值，也就是增大small的值。如果从small到big的序列和小于s，则可以增大big，让这个序列包含更多的数字。因为这个序列至少要有两个数字，我们一直增加small到$\frac{1+s}{2}$为止。
$$
\frac{1+s}{2}-1 + \frac{1+s}{2} = s
\\
\frac{1+s}{2} + \frac{1+s}{2} +1 = s+2>s
$$
![image-20190608222258987](pics/image-20190608222258987.png)

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        result = []
        low, high = 1, 2
        while low < high:
            curSum = (low + high) * (high - low + 1) / 2
            if curSum == tsum:
                temp = []
                for i in range(low, high+1):
                    temp.append(i)
                result.append(temp)
                low += 1
            elif curSum < tsum:
                high += 1
            else:
                low += 1
        return result
```

# 42. 和为S的两个数字

输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

对应每个测试案例，输出两个数，小的先输出。

最常规的做法：先在数组中固定一个数字，再依次判断数组中其余的$n-1$个数字与它的和是不是等于S，时间复杂度为$O(n^2)$

更好的做法：对于一个数组，我们可以定义两个指针，一个从左往右遍历（pleft），另一个从右往左遍历（pright）。首先，我们比较第一个数字和最后一个数字的和curSum与给定数字sum，如果curSum < sum，那么我们就要加大输入值，所以，pleft向右移动一位，重复之前的计算；如果curSum > sum，那么我们就要减小输入值，所以，pright向左移动一位，重复之前的计算；如果相等，那么这两个数字就是我们要找的数字，直接输出即可。这么做的好处是，也保证了乘积最小。

![image-20190608180946504](pics/image-20190608180946504.png)

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        if len(array) <= 1:
            return []
        left, right = 0, len(array)-1
        while left < right:
            if array[left] + array[right] == tsum:
                return array[left], array[right]
            elif array[left] + array[right] < tsum:
                left += 1
            else:
                right -= 1
        return []
```

# 43. 左旋转字符串

汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！





# 44.

# 45.

# 46.

# 47.

# 48. 

# 49.

# 50.

# 51.

# 52.

# 53.

# 54. (06-08)

# 55.

# 56.

# 57.

# 58. 

# 59.

# 60.

# 61.

# 62.

# 63.

# 64.

# 65.

# 66.  (06-09)





