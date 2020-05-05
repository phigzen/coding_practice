"""
1. Two Sum
https://leetcode.com/problems/two-sum/

Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
"""


class Solution1:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, num1 in enumerate(nums[:-1]):
            num2 = target - num1
            if (num2 in nums[i + 1:]):
                if (num2 != num1):
                    return i, nums.index(num2)
                else:
                    return i, nums[i + 1:].index(num2) + i + 1
        return None


# 可以用下面这个优化：nums.index(target - x, i + 1)
class Solution2:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        if (not nums) or (len(nums) <= 1):
            return []
        for i, x in enumerate(nums):
            if target - x in nums[i + 1:]:
                return [i, nums.index(target - x, i + 1)]
        return []


class Solution3:
    def twoSum(self, nums, target):
        hash_map = dict()
        for i, x in enumerate(nums):
            if target - x in hash_map:
                return [i, hash_map[target - x]]
            hash_map[x] = i


class Solution4:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, x in enumerate(nums[:-1]):
            try:
                return i, nums.index(target - x, i + 1)
            except:
                pass
        return None
