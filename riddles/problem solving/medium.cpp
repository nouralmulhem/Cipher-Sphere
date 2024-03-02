#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
using namespace std;

string solve(string &s)
{
    vector<int> num;
    vector<string> str;
    string temp = "";
    for (int i = 0; i < (int)s.size(); i++)
    {
        if (s[i] >= '0' && s[i] <= '9')
        {
            int n = 0;
            while (s[i] >= '0' && s[i] <= '9')
            {
                n = n * 10 + (s[i] - '0');
                i++;
            }
            i--;
            num.push_back(n);
        }
        else if (s[i] == '[')
        {
            str.push_back(temp);
            temp = "";
        }
        else if (s[i] == ']')
        {
            string t = str.back();
            str.pop_back();
            int n = num.back();
            num.pop_back();
            for (int j = 0; j < n; j++)
            {
                t += temp;
            }
            temp = t;
        }
        else
        {
            temp += s[i];
        }
    }
    return temp;
}
int main()
{
    string s = "3[a4[b]3[c]]2[a]"; // out delldelldell
    string out = solve(s);
    cout << out << endl;
    return 0;
}