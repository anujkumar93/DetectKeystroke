import numpy as np
import os
import pickle
import csv
import datetime as dt
import feature_extraction_helper as feh
from scipy.signal import medfilt
from data_processing import *

def process_data_for_rnn(log_keystrokes,lefthand_accel,lefthand_gyro,lefthand_gravity,righthand_accel,righthand_gyro,righthand_gravity,\
            char_set,test_ratio=0.2,val_ratio=0.2,data_sampling='continuous',window_time=dt.timedelta(seconds=1)):
    """
    Arguments:
    first 7: whatever preprocess_data returns
    window_time: window duration around whole word. Typically around 0.5 seconds
    char_set: the original char set that needs to be considered
    test_ratio: fraction of data to be taken into test, in order
    val_ratio: fraction of data to be taken for validation, in order
    
    Returns:
    Training, validation and test features and ground truth labels
    char_to_indx: char to index dictionary
    indx_to_char: index to char dictionary
    """
    char_to_indx=dict()
    indx_to_char=dict()
    char_indx=0
    for char in char_set:
        char_to_indx[char]=char_indx
        indx_to_char[char_indx]=char
        char_indx+=1
    char_to_indx["<NULL>"]=char_indx
    indx_to_char[char_indx]="<NULL>"
    char_indx+=1
    char_to_indx["<START>"]=char_indx
    indx_to_char[char_indx]="<START>"
    char_indx+=1
    char_to_indx["<END>"]=char_indx
    indx_to_char[char_indx]="<END>"
    data_features=[]
    data_labels=[]
    num_keystrokes=log_keystrokes.shape[0]
    i=0
    while i<num_keystrokes:
    	if log_keystrokes[i][-1]!="spc":
	    	curr_label=[char_to_indx["<START>"],char_to_indx[log_keystrokes[i][-1].lower()]]
	        time_start_keystroke=log_keystrokes[i][0]
	        time_end_keystroke=time_start_keystroke
	        i+=1
	        while i<num_keystrokes and log_keystrokes[i][-1]!="spc" and len(curr_label)<16:
	        	time_end_keystroke=log_keystrokes[i][0]
	        	curr_label.append(char_to_indx[log_keystrokes[i][-1].lower()])
	        	i+=1
	        curr_label.append(char_to_indx["<END>"])
	        while len(curr_label)<17:
	        	curr_label.append(char_to_indx["<NULL>"])
	        lh_accel_input=get_data_process(lefthand_accel,time_start_keystroke-window_time/2,time_end_keystroke+window_time/2,window_time)
	        lh_gyro_input=get_data_process(lefthand_gyro,time_start_keystroke-window_time/2,time_end_keystroke+window_time/2,window_time)
	        lh_gravity_input=get_data_process(lefthand_gravity,time_start_keystroke-window_time/2,time_end_keystroke+window_time/2,window_time)
	        rh_accel_input=get_data_process(righthand_accel,time_start_keystroke-window_time/2,time_end_keystroke+window_time/2,window_time)
	        rh_gyro_input=get_data_process(righthand_gyro,time_start_keystroke-window_time/2,time_end_keystroke+window_time/2,window_time)
	        rh_gravity_input=get_data_process(righthand_gravity,time_start_keystroke-window_time/2,time_end_keystroke+window_time/2,window_time)
	        if 0 in lh_accel_input.shape or 0 in lh_gyro_input.shape or 0 in lh_gravity_input.shape\
	             or 0 in rh_accel_input.shape or 0 in rh_gyro_input.shape or 0 in rh_gravity_input.shape:
	            print "omitting keylog instance:", [indx_to_char[g] for g in curr_label]
	            continue
	        lh_accel_input=medfilt(lh_accel_input,[7,1])
	        lh_gyro_input=medfilt(lh_gyro_input,[7,1])
	        lh_gravity_input=medfilt(lh_gravity_input,[7,1])
	        rh_accel_input=medfilt(rh_accel_input,[7,1])
	        rh_gyro_input=medfilt(rh_gyro_input,[7,1])
	        rh_gravity_input=medfilt(rh_gravity_input,[7,1])
	        curr_features=feh.get_analytical_features(lh_accel_input)
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(lh_gyro_input)))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(lh_gravity_input)))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(rh_accel_input)))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(rh_gyro_input)))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(rh_gravity_input)))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(feh.get_fft_features(lh_accel_input))))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(feh.get_fft_features(lh_gyro_input))))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(feh.get_fft_features(lh_gravity_input))))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(feh.get_fft_features(rh_accel_input))))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(feh.get_fft_features(rh_gyro_input))))
	        curr_features=np.vstack((curr_features,feh.get_analytical_features(feh.get_fft_features(rh_gravity_input))))
	        data_features.append(curr_features)
	        data_labels.append(curr_label)
        i+=1
        if i%200==0:
            print "Completed iteration ",i,"/",num_keystrokes
    assert len(data_labels)==len(data_features)
    data_labels=np.array(data_labels)
    data_features=np.array(data_features)
    print "data_features.shape:",data_features.shape
    print "data_labels.shape:",data_labels.shape
    num_samples=data_features.shape[0]
    if (data_sampling=='random'):
        mask=np.random.choice(num_samples,int(round(test_ratio*num_samples)),replace=False)
    elif (data_sampling=='continuous'):
        mask=np.arange(int(round(num_samples-test_ratio*num_samples)),num_samples)
    else:
        print "data_sampling is invalid"
    boolean_mask=np.zeros(num_samples, dtype=bool)
    boolean_mask[mask]=True
    Xte=data_features[boolean_mask,:,:] #test features
    yte=data_labels[boolean_mask]
    Xtr=data_features[~boolean_mask,:,:] #train features
    ytr=data_labels[~boolean_mask]

    if (data_sampling=='random'):
        mask=np.random.choice(Xtr.shape[0],int(round(val_ratio*Xtr.shape[0])),replace=False)
    elif (data_sampling=='continuous'):
        mask=np.arange(int(round(Xtr.shape[0]-val_ratio*Xtr.shape[0])),Xtr.shape[0])
    else:
        print "data_sampling is invalid"
    boolean_mask=np.zeros(Xtr.shape[0], dtype=bool)
    boolean_mask[mask]=True
    Xva=Xtr[boolean_mask,:,:] #val features
    yva=ytr[boolean_mask]
    Xtr=Xtr[~boolean_mask,:,:] #train features
    ytr=ytr[~boolean_mask]
    return Xtr, ytr, Xva, yva, Xte, yte, char_to_indx, indx_to_char