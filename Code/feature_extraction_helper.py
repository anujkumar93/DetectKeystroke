import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from scipy.fftpack import rfft
from scipy.optimize import curve_fit

def get_analytical_features(input_matrix):
    is_vector=False
    if input_matrix.ndim==1:
        is_vector=True
        input_matrix=input_matrix[:,np.newaxis]
    result=np.zeros((16,input_matrix.shape[1]))
    for j in range(input_matrix.shape[1]):
        curr_array=input_matrix[:,j]
        if 0 in curr_array.shape:
            print "0 sized array"
        result[0, j] = np.mean(curr_array)
        result[1, j] = np.std(curr_array)
        result[2, j] = result[1, j] / (float(result[0, j])*1000+1e-8)
        result[3, j] = np.max(curr_array) - np.min(curr_array)
        result[4, j] = np.percentile(curr_array, 10)
        result[5, j] = np.percentile(curr_array, 25)
        result[6, j] = np.percentile(curr_array, 50)
        result[7, j] = np.percentile(curr_array, 75)
        result[8, j] = np.percentile(curr_array, 90)
        result[9, j] = np.percentile(curr_array, 75) - np.percentile(curr_array, 25)
        result[10, j] = lag_one_autocorrelation(curr_array)
        result[11, j] = skewness(curr_array)
        result[12, j] = kurtosis(curr_array)
        result[13, j] = ((np.linalg.norm(curr_array))**2)/1000
        result[13, j] = log_energy(curr_array)
        result[14, j] = num_zero_crossings(curr_array)
        result[15, j] = get_straight_line_gradient(curr_array)
    if is_vector:
        result=np.ravel(result)
    else:
        result=np.vstack((result,get_adjacent_correlations(input_matrix)))

    for i in range(result.shape[0]):
        result[i,:]/=(np.linalg.norm(result[i,:])+1e-8)
    return result

def get_adjacent_correlations(input_matrix):
    result=np.zeros(input_matrix.shape[1])
    for j in range(input_matrix.shape[1]):
        curr_array=input_matrix[:,j]
        curr_array2=input_matrix[:,(j+1)%input_matrix.shape[1]]
        result[j]=correlation(curr_array,curr_array2)
    return result

def straight_line_func(x, A, B):
    return A*x + B

def get_straight_line_gradient(input_array):
    slope=None
    if len(input_array)>1:
        slope,_ = curve_fit(straight_line_func, np.arange(input_array.shape[0]), input_array)[0]
    else:
        slope=1e-8
    if slope>1000:
        print "very large value in get_straight_line_gradient:", slope
    return slope

def get_fft_features(input_matrix):
    result=rfft(input_matrix,axis=0) #gets real fft features of the same size as input matrix
    return result

def lag_one_autocorrelation(arra):
    denom=((np.std(arra))**2)*arra.size
    summ=0
    mean=np.mean(arra)
    for i in range(arra.size-1):
        summ+=(arra[i]-mean)*(arra[i+1]-mean)
    if denom==0:
        return 0
    result = summ/(float(denom)+1e-8)
    if result>1000:
        print "very large value in lag_one_autocorrelation:", result
    return result

def skewness(arra):
    denom=(np.std(arra))**3
    num=0
    mean=np.mean(arra)
    for i in range(arra.size):
        num+=(arra[i]-mean)**3
    num=num/float(arra.size)
    if denom==0:
        return 0
    result=num/(denom+1e-8)
    if result>1000:
        print "very large value in skewness:", result
    return result

def kurtosis(arra):
    denom=(np.std(arra))**6
    num=0
    mean=np.mean(arra)
    for i in range(arra.size):
        num+=(arra[i]-mean)**4
    num=num/float(arra.size)
    if denom==0:
        return 0
    result=num/(float(denom*1000)+1e-8)
    if result>10000:
        print "very large value in kurtosis:", result
    return result-3

def log_energy(arra):
    summ=0
    for i in range(arra.size):
        if (arra[i]==0):
            pass
#             summ+=np.log(0.01)
        else:
            summ+=np.log(arra[i]**2)
    summ/=100
    if summ>1000:
        print "very large value in log_energy:", summ
    return summ

def num_zero_crossings(arra):
    arra2=np.copy(arra)
    arra2=arra2-np.mean(arra2)
    return (np.where(np.diff(np.sign(arra2)))[0]).size

def correlation(arra, arra2):
    mean1=np.mean(arra)
    mean2=np.mean(arra2)
    num=0
    denom1=((np.std(arra))**2)*arra.size
    denom2=((np.std(arra2))**2)*arra2.size
    for i in range(arra.size):
        num+=(arra[i]-mean1)*(arra2[i]-mean2)
    if denom1==0 or denom2==0:
        return 0
    return float(num)/(denom1*denom2+1e-8)