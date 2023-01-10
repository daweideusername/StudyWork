class Solution:
    def permuteUnique(self, nums):
    #def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # res用来存放结果
        #if not nums: return []
        
        res = [] # 最终返回的列表结果,所有排列方法
        #列表元素是列表path[],记录的是一种排列
        used = [0] * len(nums) #给列表中的每个元素做一个标记,0/1,是否调用过
        #用布尔值来代替0 1 也一样,0 - False, 1 = True
        #如果这个两个相同的元素都调用过,那就可能重复了
        
        #print('used = ',used)
        def backtracking(nums, used, path):#path 路径,回溯
            # 终止条件
            if len(path) == len(nums):
                res.append(path.copy())#copy()完成值传递
                return 
            for i in range(len(nums)):
                #print('used[',i,']','',not used[i])
                #每个元素只能使用一次 --- 标记的作用
                if not used[i]:#not used[i] --- 0/1 --- True/False
                    
                    #used[] 记录每个元素使用的次数,0次时,正确,使用过1次时不再排列-减枝。
                    #if i>0 and nums[i] == nums[i-1] and not used[i-1]:
                    if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                        '''精髓 : 回溯
                        i>0 : i=0 时是第一个元素,第一个元素肯定没用过;
                        #貌似去了也行 --- 去了不行  对于 nums[i-1] 有影响???

                           nums[i] == nums[i-1] : 当前元素如果和前一个元素相同时,就重复了,没必要了
                           not used[i-1] : 前一个元素没用过的话

                           三个条件都满足时,就是重复
                        '''
                        continue
                    used[i] = 1 #做标记,used

                    path.append(nums[i]) #把当前元素,排列到nums[]中
                    backtracking(nums, used, path)#继续排下一个元素
                    path.pop() #排完后要清除这次的排列
                    used[i] = 0 #而且要把元素重新标记为 not used
        # 记得给nums排序
        backtracking(sorted(nums),used,[])
        
        return res

a = Solution()
l = [7,7,8,9]
print(a.permuteUnique(l))
