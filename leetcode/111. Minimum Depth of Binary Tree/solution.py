"""
111. Minimum Depth of Binary Tree
https://leetcode.com/problems/minimum-depth-of-binary-tree/

Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its minimum depth = 2.
"""

# 思路类似于，104题，最大深度。
# 只是要注意左右子树为空的情况。易出bug。

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        if not root.left:
            return 1 + self.minDepth(root.right)
        if not root.right:
            return 1 + self.minDepth(root.left)

        lmin = self.minDepth(root.left)
        rmin = self.minDepth(root.right)
        return min(lmin, rmin) + 1


class Solution2:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        lmin = self.minDepth(root.left)
        rmin = self.minDepth(root.right)
        if (lmin == 0) or (rmin == 0):
            return lmin + rmin + 1
        else:
            return min(lmin, rmin) + 1
