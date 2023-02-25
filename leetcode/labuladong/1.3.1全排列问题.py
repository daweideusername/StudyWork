"""
回溯 -- 穷举
"""

#其中 n 表示外层列表的长度，_ 是 Python 中的一个占位符，表示不需要使用该变量。这行代码创建了一个长度为 n 的列表，其中每个元素都是一个空列表。你可以将整数添加到其中的某个空列表中。
# res = [[] for _ in range(5)]

res = []
# print(res)
from collections import deque
def permute(nums:[]):
    track = deque() #记录每条路线
    backtrack(nums,track)
    return res

def backtrack(nums:[],track:deque):
    #触发结束条件
    if(len(track)==len(nums)):
        res.append(list(track)) #把队列track,转化为列表数据,添加进结果中,否则结果只有同一个列表(索引)
        return
    for i in range(len(nums)):
        #排除不合法的选择
        if(nums[i] in track):
            continue
        #做出"选择"
        track.append(nums[i])
        #进入下一层决策树
        backtrack(nums,track)
        #撤销"选择"
        track.pop()


a = [1,2,3,4]
print(permute(a))






