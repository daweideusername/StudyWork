"""
N皇后
"""

res = []
def solveNQueens(n):
    board =[['.' for j in range(n)] for i in range(n)] #for循环嵌套,创建二维棋盘
    backtrack(board,0) #从第0行开始放皇后
    return res
def backtrack(board:[[]],row:int):
    #触发结束条件
    if(row == len(board)):
        res.append(board.copy()) #没能把结果保存
        # print(board)
        return
    n = len(board[row])
    for col in range(n):
        #排除不合法的
        if(not isValid(board,row,col)):
            continue
        #做"选择"
        board[row][col] = 'Q'
        #进入下一行的决策层
        backtrack(board,row+1)
        #撤销选择
        board[row][col] = '.'

#是否可以在board[row][col]放置一个皇后
def isValid(board:[[]],row:int,col:int):
    n = len(board)
    # 检查,列(每一行)是否有皇后冲突
    for i in range(row):
        if (board[i][col] == 'Q'):
            return False

    # # 检查,右上是否有皇后冲突
    # for i in range(n):
    #     if (board[row-1][col+1] == 'Q'):
    #         return False
    # # # 检查,左上是否有皇后冲突
    # # for i in range(n):
    # #     if (board[row-1][col-1] == 'Q'):
    # #         return False

    return True




print(solveNQueens(4))
#算了吧,,,没那么好看
# def printSolution(board):
#     N=len(board[0][0])
#     for i in range(N):
#         for j in range(N):
#             print(board[i][j], end=" ")
#         print()
#
#
# print(printSolution(solveNQueens(4)))






