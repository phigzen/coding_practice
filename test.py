# # -*- coding:utf-8 -*-
# def replaceSpace(s):
#     # write code here
#     for i in range(len(s)):
#         if s[i] ==' ':
#             s[i] = '%20'
#     return s

# print(replaceSpace('hellow world!'))

class Solution:
 
    def printMatrix(self, matrix):
        res = []
        i = 0
        while matrix:
            i+=1
            print(f"this is {i}")
            res += matrix.pop(0)
            if matrix: #  and matrix[0]
                for row in matrix:
                    res.append(row.pop())
            if matrix:
                res += matrix.pop()[::-1]
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    res.append(row.pop(0))
        return res


if __name__ == '__main__':
    # print(Solution().printMatrix([[1],[2],[3],[4],[5]]))
    a = [[], []]
    if a:
        print('aaa')