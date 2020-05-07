"""
22. Generate Parentheses
https://leetcode.com/problems/generate-parentheses/

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

"""


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        self.list = []
        self._gen(0, 0, n, "")  # 递归
        return self.list

    def _gen(self, l, r, n, res):  # l,r分别是左右括号用的次数
        if l == n and r == n:  # 递归终止条件
            self.list.append(res)
            return
        if l < n:  # 左括号可以直接加
            self._gen(l + 1, r, n, res + "(")

        if l > r and r < n:  # 但右括号一定少于左括号
            self._gen(l, r + 1, n, res + ")")
