import numpy as np
import math
from matplotlib import pyplot as plt
from numpy import linalg as LA
import timeit
import cmath
import matplotlib.image as mpimg
from PIL import Image

def DFT1(f):
    """Discrete Fourier transform, complexity O(N*N)"""

    if type(f).__module__ != np.__name__:
        print("The input argument F must be an instance of NUMPY.NDARRAY.")
        return
    elif f.ndim>1:
        print("The input argument F must be an array of rank 1.")
        return
    else:
        N = f.size

    ##############################

    co = 2*np.pi/N # 2*pi/(b-a)*dt
    fhat = np.dot(np.exp(-1j*co*\
                               np.outer(np.arange(N),np.arange(N))),\
                     f)
    fhat = fhat/N

    Nm1d2 = int(np.floor((N-1)/2))
    if N & 1: # odd
        Ahat = np.zeros(1+Nm1d2)
    else: # even
        Ahat = np.zeros(1+Nm1d2+1)
        Ahat[-1] = np.real(fhat[int(N/2)])
    Ahat[0] = np.real(fhat[0])
    Ahat[1:(Nm1d2+1)] = +2*np.real(fhat[1:(Nm1d2+1)])

    Bhat = -2*np.imag(fhat[1:(Nm1d2+1)])

    ##############################
    N= 2**9
    t = np.arange(N)/N
    a = 0
    b = 1
  

    ft = evalT(Ahat, Bhat, a, b, t)
    fig = plt.plot(t, ft, label='n = ' + str(f.size))
    ##############################
    plt.legend(['DFT']) 
    return fhat, Ahat, Bhat, fig, ft

##################################################

def evalT(Ahat, Bhat, a, b, t):
    """Evaluate the trigonometric polynomial T(t)"""

    if type(t).__module__ != np.__name__:
        print("The input argument T must be an instance of NUMPY.NDARRAY.")
        return
    elif t.ndim>1:
        print("The input argument T must be an array of rank 1.")
        return

    ##############################

    if (Ahat.size - Bhat.size) == 1: # N is odd
        N = 2*Bhat.size + 1
    elif (Ahat.size - Bhat.size) == 2: # N is even
        N = 2*Bhat.size + 1 + 1
    else:
        print("The numbers of elements in the AHAT and BHAT arrays mismatch.")
        return

    Nm1d2 = Bhat.size

    ##############################

    co = 2*np.pi/(b-a)*np.outer(t, np.arange(1,Nm1d2+1))
    Tt = np.squeeze(Ahat[0] + \
                       np.dot(np.cos(co), Ahat[1:(Nm1d2+1)]) + \
                       np.dot(np.sin(co), Bhat))

    if not(N & 1): # N is even
        Tt = Tt + Ahat[int(N/2)]*np.cos(np.pi*N*t/(b-a))
    ##############################

    return Tt


def DFT2D(matrix):
    """Compute the 2D discrete Fourier Transform using 1D DFT implemented below"""
    m=np.size(matrix, 0)
    n=np.size(matrix, 1)
    matrix1 = np.zeros(matrix.shape, dtype=complex)

    for col in range(matrix.shape[0]):
        matrix1[:, col] = DFT(matrix[:, col]) 
    for row in range(matrix.shape[1]):
        matrix1[row, :] = DFT(matrix1[row, :])

    return  matrix1

def invDFT2D(matrix):
    """Compute the inverse 2D discrete Fourier Transform using 1D DFT implemented below"""
    m=np.size(matrix, 0)
    n=np.size(matrix, 1)
    matrix1 = np.zeros(matrix.shape, dtype=complex)

    for col in range(matrix.shape[0]):
        matrix1[:, col] = invDFT(matrix[:, col])
    for row in range(matrix1.shape[1]):
        matrix1[row, :]= invDFT(matrix1[row, :]) 

    return  matrix1

def DFT(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2*cmath.sqrt(-1) * cmath.pi * k * n / N)
    return np.dot(M, x)

def invDFT(X):
    """Compute the inverse discrete Fourier Transform of the 1D array x"""
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    k = np.arange(N)
    n = k.reshape((N, 1))
    M = np.exp(2*cmath.sqrt(-1) * cmath.pi * k * n / N)
    return np.dot(M, X)/N


    
def FFT_vec(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x,dtype=complex)
    N = x.shape[0]
    
    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    Nm1d2 = int(np.floor((N-1)/2))
    if N & 1: # odd
        Ahat = np.zeros(1+Nm1d2)
    else: # even
        Ahat = np.zeros(1+Nm1d2+1)
        
    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2*cmath.sqrt(-1) * cmath.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    Y =   X.ravel()
    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1]/ 2)]
        X_odd = X[:,int( X.shape[1] / 2):]
        factor = np.exp(-1*cmath.sqrt(-1) * cmath.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])
    Y =   X.ravel()/N
        

    Ahat[-1] = np.real(Y[int(N/2)])
    Ahat[0] = np.real(Y[0])
    Ahat[1:(Nm1d2+1)] = +2*np.real(Y[1:(Nm1d2+1)])

    Bhat = -2*np.imag(Y[1:(Nm1d2+1)])

##############################
    N= 2**9
    t = np.arange(N)/N
    a = 0
    b = 1
  

    ft = evalT(Ahat, Bhat, a, b, t)
    fig =  plt.plot(t, ft, label='n = ' + str(x.size))
    plt.legend(['FFTvec'])
    
    ##############################

    return X.ravel(), Ahat, Bhat, fig, ft


    
def invFFT_vec(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x,dtype=complex)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(2*cmath.sqrt(-1) * cmath.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))/N

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        factor = np.exp(1*cmath.sqrt(-1) * cmath.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()

if (__name__=="__main__"):
    
    # Exercise 1, HW5
    timeFFTvec = []
    timeDFT = []
    xs = [2,4,8,16,32,64,128, 256, 512]
 

    
    N = 2**9           # Number of samplepoints
    Fs = 800.0
    T = 1.0 / Fs      # N_samps*T (#samples x sample period) is the sample spacing.
    N_fft = 80        # Number of bins 
    x = np.linspace(0, N*T, N)     # the interval
    sig = np.zeros(x.size)
    for i in range(x.size):
        sig[i]= np.sin(50.0 * 2.0*np.pi*x[i]) + 0.5*np.sin(80.0 * 2.0*np.pi*x[i])   # the signal
   
    lSet = np.arange(1, 10)
    N = sig.size
    print('N =', N)
   
   
    err1 = np.zeros(lSet.shape)
    ft1 = np.zeros((N, lSet.size))
    err2 = np.zeros(lSet.shape)
    ft2 = np.zeros((N, lSet.size))
    for i in range(lSet.size):
       
        l = lSet[i]
        idx = np.arange(2**l)*2**(9-l)
        start= timeit.default_timer()
        fhatdft, Ahat, Bhat, figi, ft1[:,i] = DFT1(sig[idx])
        stop=timeit.default_timer()
        timeDFT.append(stop - start)
        print('time dft vectorized', stop - start)
    
        err1[i] = max(abs(ft1[:, i] - sig))
        
    print('\n')
    print('error DFT:\n' + str(err1))

    plt.show()
    
    
    for i in range(lSet.size):
       
        l = lSet[i]
        idx = np.arange(2**l)*2**(9-l)
       
        start= timeit.default_timer()
        fhatfft, Ahat, Bhat, figi, ft2[:,i] = FFT_vec(sig[idx])
        stop=timeit.default_timer()
        timeFFTvec.append(stop - start)
        print('time fft vectorized', stop - start)
    
        err2[i] = max(abs(ft2[:, i] - sig))

    print('\n')
    print('error FFT:\n' + str(err2))
    plt.show()
    plt.plot(xs, timeFFTvec, 'b')
    plt.plot(xs, timeDFT, 'g')
    plt.plot(0,0,'ok') #<-- plot a black point at the origin
    plt.legend(['FFTvec', 'DFT']) #<-- give a legend
    plt.grid(b=True, which='major') #<-- plot grid lines
    plt.show()
    

 # Exercice 3, HW5
    img_c1=mpimg.imread('imageHW5_grey.png', 0)
    img_c2 = np.fft.fft2(img_c1)
    img_c3 = np.fft.fftshift(img_c2)
    img_c4 = np.fft.ifftshift(img_c3)
    img_c5 = np.fft.ifft2(img_c4)
 #First, let s see what we will get using builtin functions
    plt.subplot(151), plt.imshow(img_c1, "gray"), plt.title("Original Image")
    plt.subplot(152), plt.imshow(np.log(1+np.abs(img_c2/255)), "gray"), plt.title("Spectrum")
    plt.subplot(153), plt.imshow(np.log(1+np.abs(img_c3/255)), "gray"), plt.title("Centered Spectrum")
    plt.subplot(154), plt.imshow(np.log(1+np.abs(img_c4/255)), "gray"), plt.title("Decentralized")
    plt.subplot(155), plt.imshow(np.abs(img_c5/255), "gray"), plt.title("Processed Image")

    plt.show()

# # You can also do this: Recreate input image using my 2D DFT results to compare to the input image
#     image1 =invDFT2D(DFT2D(img_c1))
# #   image1.save("mytestafterDFT.png", "PNG")
#     plt.imshow(np.abs(image1/255), "gray"), plt.title("Processed Image")
#     plt.show()
  
 # read in the image and convert it to a black-and-white image
    image1 = Image.open('imageHW5_grey.png').convert('L')
    
    # store the converted image as an array
    image1 = np.array(image1)
    image1 = np.asarray(image1, dtype=complex)
    
    # compute the 2D DFT of the image
    num = len(image1)
    
    # transform each row of the image
    
    for i in range(num):
        image1[i] = np.fft.fft(image1[i])
    # transpose the intermediate matrix
    image1 = image1.T
    # transform each row of the transposed intermediate matrix
    for i in range(num):
        image1[i] = np.fft.fft(image1[i])
    # transpose the resulting matrix
    image1 = image1.T
    
    # zero out the set of (high) frequency components of the transform
    zerout = np.asarray(np.zeros(num), dtype=complex)
    for i in range((num // 2) + 1, num, 1):
        image1[i] = zerout
    image1 = image1.T
    for i in range((num // 2) + 1, num, 1):
        image1[i] = zerout
    image1 = image1.T
    
    # inverse-transform the columns and then the rows
    
    image1 = image1.T
    # transform each row of the transposed matrix
    for i in range(num):
        image1[i] = np.fft.ifft(image1[i])
    # transpose the intermediate matrix
    image1 = image1.T
    # transform each row of the matrix im
    for i in range(num):
        image1[i] = np.fft.ifft(image1[i])

    # take the modulus of each element
    image1 = np.absolute(image1)
    # restrict the final value to the range of legal pixel values
    image1 = np.asarray(image1, dtype=np.uint8)
    
    # write out the filtered image as a .PNG file with the name "ProcessedImage.png"
    Image.fromarray(image1).save("ProcessedImage.png")


    plt.imshow(np.abs(image1), "gray"), plt.title("Processed Image")

    plt.show()
  














