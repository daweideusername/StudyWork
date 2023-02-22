

#暴力递归
def fib(n):
    if(n==0) :
        return 0
    if(n==1 or n==2) :
        return 1
    return fib(n-1)+fib(n-2)
print(fib(0),fib(1),fib(2),fib(3),fib(4))

#带备忘录
memo = []
def fib1(n):
    if(n==0):
        return 0
    memo = [0]*(n+1) #创建一个大小为 n+1 的空数组
    return helper(memo,n)

def helper(memo,n):
    if (n==1 or n==2):
        return 1
    if(memo[n] != 0):
        return memo[n]
    memo[n] = helper(memo,n-1) + helper(memo,n-2)
    return memo[n]

print(fib1(3))

#"自底向上"的备忘录
def fib2(n):
    if(n==0):
        return 0
    if(n==1 or n==2):
        return 1
    dp = [0]*(n+1)
    dp[1] = dp[2] = 1
    for i in range (2,n+1): #从 2 ~ n+1 遍历(从0开始)
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
print(fib2(3))

#状态转移
def fib3(n):
    if(n==0):
        return 0
    if(n==1 or n==2):
        return  1
    prev = 1
    curr = 1
    for i in range (3,n+1):
        sum = prev + curr
        prev = curr
        curr = sum
    return curr

print(fib3(3))











