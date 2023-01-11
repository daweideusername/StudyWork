# Definition for a binary tree node.
from typing import Optional

class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # 节点为空，高度为 0
        if root == None:
            return 0

        # 递归计算左子树的最大深度
        leftHeight = self.maxDepth(root.left)
        # 递归计算右子树的最大深度
        rightHeight = self.maxDepth(root.right)

        # 二叉树的最大深度 = 子树的最大深度 + 1（1 是根节点）
        return max(leftHeight, rightHeight) + 1
