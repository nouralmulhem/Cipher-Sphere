def solve2(x, y):
    dp = [[0] * y for _ in range(x)]

    for i in range(x):
        dp[i][y - 1] = 1

    for i in range(y):
        dp[x - 1][i] = 1

    for i in range(x - 2, -1, -1):
        for j in range(y - 2, -1, -1):
            dp[i][j] = dp[i + 1][j] + dp[i][j + 1]

    return dp[0][0]

def nCr(n, k):
    if k > n - k:
        k = n - k
    ans = 1
    j = 1
    for j in range(1, k + 1):
        if n % j == 0:
            ans *= n // j
        elif ans % j == 0:
            ans = ans // j * n
        else:
            ans = (ans * n) // j
        n -= 1
    return ans
# olve is the most optimal solution for the given problem O(n)
def solve(x, y):
    down, right = x - 1, y - 1
    return nCr(down + right, min(right, down))

if __name__ == "__main__":
    x, y = 7, 6
    out = solve(x, y)
    print(out)
