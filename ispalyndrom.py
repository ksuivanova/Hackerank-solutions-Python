def IsPalyndrom1(s):
    n = len(s)
    print(n)
    i=0
    j=n-1
    while j > i  :
        while i < n and s[i].isalpha()==False :
            i+=1
        while j > 0 and s[j].isalpha()==False:
            j-=1
        if j < i:
            return True
        if s[i].lower()==s[j].lower():
            i+=1
            j-=1 
        else:
            return False               
    return True

def IsPalyndrom2(s):
    s1="".join([ch for ch in s.lower() if ch.isalpha()])
    return s1==s1[::-1]

def IsPalyndrom3(s):
    s=[ch for ch in s.lower() if ch.isalpha()]
    for i in range(len(s)//2):
        if s[i]!=s[-i-1]:
            return False
    return True 


#main
if __name__ == '__main__':

    s = "No lemon, no melon"
    print(IsPalyndrom3(',;:  '))