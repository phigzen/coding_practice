"""
242. Valid Anagram
https://leetcode.com/problems/valid-anagram/

Given two strings s and t , write a function to determine if t is an anagram of s.

Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false

Note:
You may assume the string contains only lowercase alphabets.
"""


# 进行排序后比较，时间复杂度：O(nlogn)
class Solution1:
    def isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)


# 哈希表存每个字符的出现次数，如果完全一致，则表示一样，时间复杂度：O(n)
class Solution2:
    def isAnagram(self, s: str, t: str) -> bool:
        dict1, dict2 = {}, {}
        for item in s:
            # dict.get: 取键为item的值，如果不存在，给默认值0
            dict1[item] = dict1.get(item, 0) + 1
        for item in t:
            dict2[item] = dict2.get(item, 0) + 1
        return dict1 == dict2


# 用列表来代替哈希表实现类似功能
class Solution3:
    def isAnagram(self, s: str, t: str) -> bool:
        dict1, dict2 = [0] * 26, [0] * 26
        for item in s:
            dict1[ord(item) - ord('a')] += 1
        for item in t:
            dict2[ord(item) - ord('a')] += 1
        return dict1 == dict2