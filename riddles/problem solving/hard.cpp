#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
using namespace std;

long long solve2(long long x, long long y)
{

    vector<vector<long long>> dp(x, vector<long long>(y, 0));
    for (int i = 0; i < x; i++)
    {
        dp[i][y - 1] = 1;
    }
    for (int i = 0; i < y; i++)
    {
        dp[x - 1][i] = 1;
    }
    for (int i = x - 2; i >= 0; i--)
    {
        for (int j = y - 2; j >= 0; j--)
        {
            dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
        }
    }
    return dp[0][0];
}

long long nCr(long long n, long long k)
{
    if (k > n - k)
    {
        k = n - k;
    }
    long long ans = 1;
    int j = 1;
    for (; j <= k; j++, n--)
    {
        if (n % j == 0)
        {
            ans *= n / j;
        }
        else if (ans % j == 0)
        {
            ans = ans / j * n;
        }
        else
        {
            ans = (ans * n) / j;
        }
    }
    return ans;
}
// solve is the most optimal solution for the given problem O(n)
long long solve(long long x, long long y)
{
    long long down = x - 1, right = y - 1;
    return nCr(down + right, min(right, down));
}

int main()
{
    long long x = 30, y = 20;
    long long out = solve(x, y);
    cout << out << endl;
    return 0;
}