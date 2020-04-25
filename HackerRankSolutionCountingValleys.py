#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the countingValleys function below.
def countingValleys(n, s):
    sea_level = 0
    count = 0
    for i in range(len(s)):
        if s[i]=='D':
            sea_level-=1
            i +=1
        elif s[i]=='U':
            sea_level +=1
            if sea_level == 0:
                count+=1
            i+=1
    return count


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    s = input()

    result = countingValleys(n, s)

    fptr.write(str(result) + '\n')

    fptr.close()

