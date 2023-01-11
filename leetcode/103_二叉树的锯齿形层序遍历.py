# Definition for a binary tree node.
import collections
from typing import List


class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:  # 如果树的节点是空的，返回[]即可
            return []
        res = []  # 初始化一个返回列表
        q = collections.deque()  # 初始化一个双边队列
        q.append(root)  # 首先将根节点压入双边队列中
        while q:  # 因为我们的目的就是将所有的节点遍历到，所以结束条件就是队列为空
            res_tmp = []  # 因为题目要求返回的是按照层划分的数组，所以这边需要一个过渡数组
            n = len(q)  # 首先记录一下该层的长度，因为是每一层单边遍历记录，所以需要知道该层的长度
            # 因为队列还在不停的往里面压入这些节点的子孩子，所以q的长度是变化的，所以这边需要提前记录一下
            for i in range(n):  # 将每行所有的节点全部遍历到
                tmp = q.popleft()  # 首先将最左边的队列元素弹出
                res_tmp.append(tmp.val)  # 将弹出的节点值压入每行的过渡数组中
                if tmp.left:  # 如果当前节点存在左子树
                    q.append(tmp.left)  # 就把相应的左子树放入到队列中作为下一层待遍历的节点
                if tmp.right:  # 如果当前节点存在右子树
                    q.append(tmp.right)  # 就把相应的右子树放入到队列中作为下一层待遍历的节点
            res.append(res_tmp)  # 每层遍历结束，就将该层的遍历结果放入到最终的输出数组中
        for j in range(len(res)):  # 因为要求是锯齿遍历，所以要把偶数行给翻转一下
            if j % 2 == 1:  # 因为索引是从0开始的，所以这边判断余数是否为1
                res[j].reverse()  # 如果是第偶数行，翻转，reverse是一个自带的反转函数，蛮好用的

        return res  # 返回最终值即可

