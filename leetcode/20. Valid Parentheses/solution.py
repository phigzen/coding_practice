"""
20. Valid Parentheses

Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.

Example 1:

Input: "()"
Output: true
Example 2:

Input: "()[]{}"
Output: true
Example 3:

Input: "(]"
Output: false
Example 4:

Input: "([)]"
Output: false
Example 5:

Input: "{[]}"
Output: true

"""


class Solution1:
    """
    依次把(), {}, []消除
    """
    def isValid(self, s: str) -> bool:
        # if len(s)%2 !=0:
        # return False
        if s is None:
            return None
        while True:
            raw_len = len(s)
            s = s.replace("[]", "").replace("{}", "").replace("()", "")
            after_len = len(s)
            if s == "":
                return True
            if raw_len == after_len:
                return False


class Solution2:
    """
    上述解法更简洁的写法。
    """
    def isValid(self, s):
        while "()" in s or "{}" in s or '[]' in s:
            s = s.replace("()", "").replace('{}', "").replace('[]', "")
        return s == ''


class Solution3:
    """
    利用一个栈的数据结构。把每次匹配的消除，判断最后是不是一一匹配完成。
    """
    def isValid(self, s: str) -> bool:
        if s is None:
            return False
        if len(s) % 2 != 0:
            return False
        stack = []
        match_map = {']': '[', ')': '(', '}': '{'}
        for si in s:
            if si not in match_map:
                stack.append(si)
            else:
                if len(stack) == 0:
                    return False
                if stack.pop() != match_map[si]:
                    return False
        if len(stack) == 0:
            return True
        else:
            return False


class Solution4:
    """
    上述解法更简洁的写法。
    """
    def isValid(self, s: str) -> bool:
        stack = []
        match_map = {']': '[', ')': '(', '}': '{'}
        for si in s:
            if si not in match_map:
                stack.append(si)
            elif not stack or stack.pop() != match_map[si]:
                return False
        return not stack
