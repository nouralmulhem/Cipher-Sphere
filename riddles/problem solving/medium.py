def solve(s):
    num = []
    str_list = []
    temp = ""
    
    i = 0
    while i < len(s):
        if '0' <= s[i] <= '9':
            n = 0
            while '0' <= s[i] <= '9':
                n = n * 10 + int(s[i])
                i += 1
            i -= 1
            num.append(n)
        elif s[i] == '[':
            str_list.append(temp)
            temp = ""
        elif s[i] == ']':
            t = str_list.pop()
            n = num.pop()
            temp = t + temp * n
        else:
            temp += s[i]
        i += 1
    
    return temp

if __name__ == "__main__":
    s = "3[a4[b]3[c]]2[a]"  # out delldelldell
    out = solve(s)
    print(out)
