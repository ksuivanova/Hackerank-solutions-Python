from pylab import *
import numpy as np

def bisec(a,b,f,tol): #Bisection method
    t = abs(a-b)
    m=(a+b)/2
    while t>=tol:
        m=(a+b)/2
        if f(m) == 0:
            x=m
        if sign(f(m)*f(a))<0:
            b = m
        else:
            a=m
        t = abs(a-b)
        x=m
    return x
   
def sign_count(L):  #Counts number of sign changes in a list L
    c=0
    s=sign(L[0])
    for i in L[1:]:
        if s*i<0:
        	c=c+1
        	s=sign(i)
    return c

def legendre_polynomial(n,x):
    for k in range(n):
        L= [1,x]
        for j in range(1,n):
        	t=x*L[j]-(j)**2/float(4*(j)**2-1)*L[j-1]
        	L.append(t)
    return L
def Ln(n,x):
    if (n==0):
        return 1
    elif(n==1):
        return x
    else:
        return x*Ln(n-1,x)-(n-1)**2/float(4*(n-1)**2-1)*Ln(n-2,x)

def leg_root2(i,n):     #i'th  root of n'th legendre polynomial using Bisection method
    a = -1
    b = 1
    tol = 10**(-7)
    Leg = lambda x: Ln(n,x)
    
    if i>n:
        print('error, i has to be <=n')
    if n==0:
        print('n should be positive')
    elif n==1:
        x = 0
    elif i == n:
        a = leg_root2(i-1,n-1)
        b =2*a+1
        x = bisec(a,b,Leg,tol)
    elif i==1:
        b = leg_root2(i,n-1)
        a = 2*b-1
        x = bisec(a,b,Leg,tol)
    else:
        a = leg_root2(i-1,n-1)
        b= leg_root2(i,n-1)
        x = bisec(a,b,Leg,tol)
    return x
   
def leg_root1(i,n):    #i'th  root of n'th legendre polynomial using the sign changes
    
    i=n-i+1
    a = -1
    b = 1
    tol = 10**(-7)
    while abs(b-a)>  tol:
        x = (a+b)/2.
        L=legendre_polynomial(n,x)
        
        if sign_count(L) > i-1:
            a = x
        else:
            b = x
    return x
    
print('using the sign changes',leg_root1(8,10))
print('using Bisection',leg_root2(8,10))
