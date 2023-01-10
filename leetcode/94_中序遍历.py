from typing import  List

class TreeNode:
	def __init__(self, key):
		self.left = None
		self.right = None
		self.val = key

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]
        while stack:
            color, node = stack.pop()
            if node is None: continue
            if color == WHITE:
                stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                stack.append((WHITE, node.left))
            else:
                res.append(node.val)
        return res

root = TreeNode(6)
# root.val = 9
r = Solution()
print(r.inorderTraversal(root))

