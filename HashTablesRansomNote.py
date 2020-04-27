#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the checkMagazine function below.
def checkMagazineComplexityON2(magazine, note):
    for word in note:
        if word in magazine:
            magazine.remove(word)
        else:
            return False
    return True
#A better approach would be to use hash tables linear complexity O(N)

def checkMagazineHT(magazine, note):
    magazine_hash={}

    for word in magazine:
        if word in magazine_hash:
            magazine_hash[word]+=1
        else:
            magazine_hash[word] = 1

    for word in note:
        if word in magazine_hash and magazine_hash[word]> 0:
            magazine_hash[word]-=1
        else:
            return False
    return True


if __name__ == '__main__':
    mn = input().split()

    m = int(mn[0])

    n = int(mn[1])

    magazine = input().rstrip().split()

    note = input().rstrip().split()

    answer = checkMagazineHT(magazine, note)

    if answer:
        print('Yes')
    else:
        print('No')

