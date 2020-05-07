"""
102. Binary Tree Level Order Traversal
https://leetcode.com/problems/binary-tree-level-order-traversal/

Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
"""

import collections


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution1:
    """
    广度优先搜索
    """
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return None
        res = []
        q = collections.deque()  # 双端队列
        q.append(root)
        # visited = set(root) # 图遍历的时候需要加上
        while q:
            level_size = len(
                q
            )  # 在下个循环里面 q是不断动态更新的，所以在开始的时候记录一个开始的length会比较好，下层循环直接循环这么多次即为一层
            cur_level = []
            for _ in range(level_size):
                node = q.popleft()
                cur_level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(cur_level)

        return res


class Solution2:
    """
    广度优先搜索2
    """
    def levelOrder(self, root):
        ans, level = [], [root]
        while root and level:
            ans.append([node.val for node in level])
            LRpair = [(node.left, node.right) for node in level]
            level = [leaf for LR in LRpair for leaf in LR if leaf]
        return ans

    def levelOrder2(self, root):
        """
        类似上面的解，但更加简洁
        """
        ans, level = [], [root]
        while root and level:
            ans.append([node.val for node in level])
            level = [kid for n in level for kid in (n.left, n.right) if kid]
        return ans

    def levelOrder3(self, root):
        """
        类似上面的解，但更加易理解。
        符合原始思路。
        """
        if not root:
            return []
        ans, level = [], [root]
        while level:
            ans.append([node.val for node in level])
            temp = []
            for node in level:
                temp.extend([node.left, node.right])
            level = [leaf for leaf in temp if leaf]
        return ans
