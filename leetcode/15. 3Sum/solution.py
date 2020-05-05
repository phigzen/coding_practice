"""
https://leetcode.com/problems/3sum/

Given an array nums of n integers, 
are there elements a, b, c in nums such that a + b + c = 0? 
Find all unique triplets in the array which gives the sum of zero.

Note:

The solution set must not contain duplicate triplets.

Example:

Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
"""

# 1. 暴力循环
# 2. 循环两次，维护一个set用作查询
# 3. 循环一次，然后两端夹逼


class Solution1:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []
        nums.sort()
        res = set()
        # 遍历到nums[:-2]即可
        for i, v in enumerate(nums[:-2]):
            # 去重
            if i >= 1 and v == nums[i - 1]:
                continue

            d = {}
            # 第二层循环
            for x in nums[i + 1:]:
                # x 与 -v-x 配对，表示等着后面可能出现的-v-x和现在的x配对。
                if x not in d:
                    d[-v - x] = 1
                else:
                    # 此时的-v-x是之前循环过的x
                    # 用set可以去重，按照(v, -v - x, x)顺序添加进set即可
                    res.add((v, -v - x, x))
        return list(map(list, res))


class Solution2:
    """
    双指针。
    """
    def threeSum(self, nums):
        res = []
        nums.sort()  # 从小到大排序，方便去重等后续操作
        length = len(nums)
        for i in range(length - 2):  #[8] 循环nums[:-2]就可以了
            #[7] 序列是从小到大排列的，如果num[:-2]有出现大于0的，那后面的都比0大，可以跳过
            if nums[i] > 0:
                break
            #[1] 如果当前元素=之前元素，说明已经判断过了，跳过以去重
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            #[2] 从i+1开始第二层循环，双指针夹逼
            l, r = i + 1, length - 1
            while l < r:
                total = nums[i] + nums[l] + nums[r]
                #[3] 判断当前三数和的状态，小于0则左边指针右移
                if total < 0:
                    l += 1
                #[4] 判断当前三数和的状态，大于0则右边指针左移
                elif total > 0:
                    r -= 1
                #[5] 等于0则说明找到了一组答案
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    #[6] 结果去重，也可以考虑用set存结果
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
        return res
