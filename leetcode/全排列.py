class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(first = 0):
            # 所有数都填完了
            if first == n:  
                res.append(nums[:])#nums 为什么呢??? --- 感觉应该和绝对引用有关系
                #用':',进行切片操作,使变成 值传递
            for i in range(first, n):
                #print(first)
                #if first != i & nums[first] == nums[i]:
                #    continue

                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1)
                if nums[first] == nums[i]:
                    continue
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first]
        
        n = len(nums)
        res = []
        backtrack()
        print("*************")
        print(res)
        return res

a = Solution()
l = [1,1,2,3]
a.permute(l)
'''                print("first =",first)
                print('i = ',i)
                print("nums = ",nums)
                print("res = ",res)
'''
'''
                if first != i & nums[first] == nums[i]:
                    continue
'''
