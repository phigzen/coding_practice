# -*- coding:utf-8 -*-
def replaceSpace(s):
    # write code here
    for i in range(len(s)):
        if s[i] ==' ':
            s[i] = '%20'
    return s

print(replaceSpace('hellow world!'))