"""
k种面值,分别为:c1,c2,c3,...,ck
凑出金额 -- amount
求最少需要硬币数量
"""

#暴力递归
def coinChange(coins:[],amount:int):
    #定义:要凑出目标金额:n,至少需要dp(n)枚硬币
    def dp(n):
        #base case
        if n==0:return 0
        if n<0: return -1

        res = float('INF') # 正无穷大的缩写,一个概念, 它用于表示一个大于任何有限浮点数的数值。
        #做选择:选择需要硬币数最少的那个结果
        for coin in coins:
            subproblem = dp(n-coin)
            #子问题无解,跳过
            if subproblem == -1 :continue
            res = min(res, 1+dp(n-coin)) #每选择一枚硬币,就相当于减少了目标金额
        return res

    return dp(amount)


a = [1,2,3,5,7,11]
print(coinChange(a,2))

#带"备忘录"的递归
def coinChange1(coins:[],amount:int):
    #"备忘录"
    memo = dict()
    def dp(n):
        #查"备忘录",避免重复
        if n in memo : return memo[n]
        #base case
        if n==0 : return 0
        if n<0  : return -1
        res = float('INF')
        for coin in coins:
            subproblem = dp(n-coin)
            if subproblem == -1 :continue
            res = min(res,1+subproblem)
        #记入"备忘录"
        memo[n] = res if res != float('INF') else -1 #无解,返回-1
        return memo[n]
    return dp(amount)

print(coinChange1(a,-10))

#dp数组迭代解法
#dp数组:当目标金额为i时,至少需要dp[i]个硬币
def coinChange2(coins:[],amount:int):
    dp = [amount+1]*(amount+1) #初始化数组,值取amount+1,相当于最多
    #base case
    dp[0] = 0
    #外层for循环,遍历所有状态的所有取值
    for i in range(amount):
        #内层for循环,求所有选择的最小值
        for coin in coins:
            #子问题无解,则跳过
            if (i-coin<0):continue
            dp[i] = min(dp[i],1+dp[i-coin])
    if dp[amount]==amount+1:
        return -1
    else:
        return dp[amount]

print(coinChange1(a,123))











