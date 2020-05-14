
def polyndrom(s):
    n = len(s)
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
