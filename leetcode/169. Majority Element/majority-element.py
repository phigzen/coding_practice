"""
169. Majority Element
https://leetcode.com/problems/majority-element/

Given an array of size n, find the majority element. 
The majority element is the element that appears more than ⌊ n/2 ⌋ times.

You may assume that the array is non-empty and the majority element always exist in the array.

Example 1:

Input: [3,2,3]
Output: 3
Example 2:

Input: [2,2,1,1,1,2,2]
Output: 2
"""


class Solution1:
    def majorityElement(self, nums: List[int]) -> int:
        res = {}
        for i in nums:
            res[i] = res.get(i, 0) + 1
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return res[0][0]


class Solution2:
    def majorityElement(self, nums: List[int]) -> int:
        dict_ = {}
        for i in nums:
            dict_[i] = dict_.get(i, 0) + 1
            if dict_.get(i, 0) > len(nums) // 2:
                return i


# 更简洁的写法：
import collections


class Solution3:
    """
    根据value对key排序：
    a = {1:5,2:6,3:1}
    max(a.keys(), key=a.get) # 2
    min(a.keys(), key=a.get) # 3
    或者：
    a = sorted(a.items(), key=lambda x: x[1], reverse=True)
    """
    def majorityElement(self, nums):
        counts = collections.Counter(nums)
        return max(counts.keys(), key=counts.get)


# 递归
class Solution4:
    def majorityElement(self, nums, lo=0, hi=None):
        def majority_element_rec(lo, hi):
            # base case; the only element in an array of size 1 is the majority
            # element.
            if lo == hi:
                return nums[lo]

            # recurse on left and right halves of this slice.
            mid = (hi - lo) // 2 + lo
            left = majority_element_rec(lo, mid)
            right = majority_element_rec(mid + 1, hi)

            # if the two halves agree on the majority element, return it.
            if left == right:
                return left

            # otherwise, count each element and return the "winner".
            left_count = sum(1 for i in range(lo, hi + 1) if nums[i] == left)
            right_count = sum(1 for i in range(lo, hi + 1) if nums[i] == right)

            return left if left_count > right_count else right

        return majority_element_rec(0, len(nums) - 1)
