
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the repeatedString function below.
def repeatedString(s, n):
    length = len(s)
    times = int(n/length)
    rest = n%length
    count = 0
    countr = 0
    for i in range(length):
        if s[i]=='a':
            count+=1
    for j in range(rest):
        if s[j]=='a':
            countr+=1
    return times*count + countr

    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    n = int(input())

    result = repeatedString(s, n)

    fptr.write(str(result) + '\n')

    fptr.close()

