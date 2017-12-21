import numpy as np
import os
import pickle
import csv
import datetime as dt
import feature_extraction_helper as feh
from scipy.signal import medfilt

def preprocess_data(char_set,input_case=["all"],log_buffer=dt.timedelta(seconds=1)):
    """
    Arguments:
    char_set: set of characters for which input has to be considered
    input_case: format of ['room'], ['room','library'] or ['all']
    log_buffer: time duration of type timedelta giving the buffer allowed between two successive keyfreq readings
    
    Returns:
    Log_data from keyfreq
    Corresponding lefthand and righthand acceleration, gyro and gravity vectors
    """
    if input_case[0]=="all":
        input_case=["room","room2","room3","library"]
    final_log_keystrokes=[]
    final_lefthand_gyro=[]
    final_lefthand_accel=[]
    final_lefthand_gravity=[]
    final_righthand_gyro=[]
    final_righthand_accel=[]
    final_righthand_gravity=[]
    keyfreq_folder='../Data/keyfreq/'
    righthand_folder='../Data/righthand/'
    lefthand_folder='../Data/lefthand/'
    for folder in input_case:
        log_files=sorted(os.listdir(keyfreq_folder+folder))
        righthand_files=sorted(os.listdir(righthand_folder+folder))
        lefthand_files=sorted(os.listdir(lefthand_folder+folder))
        log_files=[os.path.join(keyfreq_folder,folder,f) for f in log_files]
        lefthand_files=[os.path.join(lefthand_folder,folder,f) for f in lefthand_files]
        righthand_files=[os.path.join(righthand_folder,folder,f) for f in righthand_files]
        print "total number of files in ", folder,":", len(log_files)
        #going over keylogged files first
        for file_num,log_file in enumerate(log_files):
            curr_log_keystrokes=[]
            curr_lefthand_gyro=[]
            curr_lefthand_accel=[]
            curr_lefthand_gravity=[]
            curr_righthand_gyro=[]
            curr_righthand_accel=[]
            curr_righthand_gravity=[]
            with open(log_file) as curr_log_f:
                line0 = curr_log_f.readline()
                line0_split=line0.split()
                time0_split=line0_split[0].split(':')
                line0_time=dt.datetime.fromtimestamp(float(time0_split[0]))
                line0_time=line0_time+dt.timedelta(microseconds=float(time0_split[1]))
                lefthand_file,righthand_file=None,None
                for rh_file in righthand_files:
                    file_time_str=rh_file[-19:-4]
                    file_time=dt.datetime.strptime(file_time_str,'%Y%m%d-%H%M%S')
                    if file_time>line0_time:
                        righthand_file=rh_file
                        break
                for lh_file in lefthand_files:
                    file_time_str=lh_file[-19:-4]
                    file_time=dt.datetime.strptime(file_time_str,'%Y%m%d-%H%M%S')
                    if file_time>line0_time:
                        lefthand_file=lh_file
                        break
                curr_log_f.seek(0)
                if lefthand_file is None or righthand_file is None:
                    continue
                lh_accel0,lh_gyro0,lh_gravity0=find_starting_line(lefthand_file)
                rh_accel0,rh_gyro0,rh_gravity0=find_starting_line(righthand_file)
                # print lh_accel0,lh_gyro0,lh_gravity0
                # print rh_accel0,rh_gyro0,rh_gravity0
                curr_log_start_time,curr_log_end_time=None,None
                for line in curr_log_f:
                    line_split=line.split()
                    if check_in_charset(line_split,char_set):
                        time_split=line_split[0].split(':')
                        curr_log_start_time=dt.datetime.fromtimestamp(float(time_split[0]))
                        curr_log_start_time=curr_log_start_time+dt.timedelta(microseconds=float(time_split[1]))
                        line_split[0]=curr_log_start_time
                        if ']' in line_split[-1] and '[' in line_split[-1]:
                            line_split.append('spc')
                        curr_log_keystrokes=[line_split]
                        prev_end_time=curr_log_start_time
                        curr_end_time=curr_log_start_time
                        for end_line in curr_log_f:
                            end_line_split=end_line.split()
                            if check_in_charset(end_line_split,char_set):
                                end_time_split=end_line_split[0].split(':')
                                curr_end_time=dt.datetime.fromtimestamp(float(end_time_split[0]))
                                curr_end_time=curr_end_time+dt.timedelta(microseconds=float(end_time_split[1]))
                                prev_end_time=curr_end_time
                                end_line_split[0]=curr_end_time
                                if ']' in end_line_split[-1] and '[' in end_line_split[-1]:
                                    end_line_split.append('spc')
                                curr_log_keystrokes.append(end_line_split)
                            if (curr_end_time-prev_end_time)>log_buffer and not check_in_charset(end_line_split, char_set):
                                curr_log_end_time=prev_end_time
                                break
                        else: #handles EOF case
                            curr_log_end_time=prev_end_time
                        curr_lefthand_gyro=get_chunk_data(lh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,lh_gyro0,log_buffer)
                        curr_lefthand_accel=get_chunk_data(lh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,lh_accel0,log_buffer)
                        curr_lefthand_gravity=get_chunk_data(lh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,lh_gravity0,log_buffer)
                        curr_righthand_gyro=get_chunk_data(rh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,rh_gyro0,log_buffer)
                        curr_righthand_accel=get_chunk_data(rh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,rh_accel0,log_buffer)
                        curr_righthand_gravity=get_chunk_data(rh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,rh_gravity0,log_buffer)
                        if curr_log_keystrokes and curr_lefthand_gyro and curr_lefthand_accel and curr_lefthand_gravity\
                             and curr_righthand_gyro and curr_righthand_accel and curr_righthand_gravity:
                            final_log_keystrokes.extend(curr_log_keystrokes)
                            final_lefthand_gyro.extend(curr_lefthand_gyro)
                            final_lefthand_accel.extend(curr_lefthand_accel)
                            final_lefthand_gravity.extend(curr_lefthand_gravity)
                            final_righthand_gyro.extend(curr_righthand_gyro)
                            final_righthand_accel.extend(curr_righthand_accel)
                            final_righthand_gravity.extend(curr_righthand_gravity)
            print "Successfully completed",str(file_num+1),"files"
    final_log_keystrokes=np.array(final_log_keystrokes)
    final_lefthand_accel=np.array(final_lefthand_accel)
    final_lefthand_gyro=np.array(final_lefthand_gyro)
    final_lefthand_gravity=np.array(final_lefthand_gravity)
    final_righthand_accel=np.array(final_righthand_accel)
    final_righthand_gyro=np.array(final_righthand_gyro)
    final_righthand_gravity=np.array(final_righthand_gravity)
    return final_log_keystrokes,final_lefthand_accel,final_lefthand_gyro,final_lefthand_gravity,\
                final_righthand_accel,final_righthand_gyro,final_righthand_gravity

def check_in_charset(line_split,char_set):
    if (line_split[-1].lower() in char_set) or ("spc" in char_set and ']' in line_split[-1] and '[' in line_split[-1]):
        return True
    return False

def find_starting_line(filename):
    accel_start,gyro_start,gravity_start=None,None,None
    accel_set,gyro_set,gravity_set=False,False,False
    i=0
    with open(filename,'rb') as lh_f:
        curr_f=csv.reader(lh_f)
        for row in curr_f:
            if not accel_set and "accel" in row[0].lower():
                accel_start=i
                accel_set=True
            if not gyro_set and "gyro" in row[0].lower():
                gyro_start=i
                gyro_set=True
            if not gravity_set and "grav" in row[0].lower():
                gravity_start=i
                gravity_set=True
            i+=1
    return accel_start,gyro_start,gravity_start

def get_chunk_data(filename, start_time, end_time, offset, log_buffer):
    result_set=[]
    with open(filename,'rb') as f:
        reader=csv.reader(f)
        lines=list(reader)
        for i in range(offset,len(lines)):
            curr_time_str=lines[i][1]+" "+lines[i][2]
            curr_time=dt.datetime.strptime(curr_time_str,' %b %d %Y %H:%M:%S.%f')
            if curr_time>start_time and curr_time<end_time:
                curr_set=[curr_time]
                curr_set.extend(lines[i][3:])
                result_set.append(curr_set)
            if curr_time>end_time+2*log_buffer: #due to discrepancy in data of lower values coming after higher values
                break
    return result_set

def get_data_process(lines, start_time, end_time,window_time,dtype=np.float32):
    result_set=[]
    for i in range(lines.shape[0]):
        curr_time=lines[i][0]
        if curr_time>=start_time and curr_time<=end_time:
            curr_set=lines[i][1:]
            result_set.append(curr_set)
        if curr_time>end_time+2*window_time: #due to discrepancy in data of lower values coming after higher values
            break
    return np.array(result_set).astype(dtype)

def process_data(log_keystrokes,lefthand_accel,lefthand_gyro,lefthand_gravity,righthand_accel,righthand_gyro,righthand_gravity,\
            window_time,char_set,test_ratio,val_ratio,data_sampling='continuous'):
    """
    Arguments:
    first 7: whatever preprocess_data returns
    window_time: window duration around each keystroke that needs to be considered. Typically around 0.5 seconds
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
    data_features=[]
    data_labels=[]
    for i,l_d in enumerate(log_keystrokes):
        time_keystroke=l_d[0]
        lh_accel_input=get_data_process(lefthand_accel,time_keystroke-window_time/2,time_keystroke+window_time/2,window_time)
        lh_gyro_input=get_data_process(lefthand_gyro,time_keystroke-window_time/2,time_keystroke+window_time/2,window_time)
        lh_gravity_input=get_data_process(lefthand_gravity,time_keystroke-window_time/2,time_keystroke+window_time/2,window_time)
        rh_accel_input=get_data_process(righthand_accel,time_keystroke-window_time/2,time_keystroke+window_time/2,window_time)
        rh_gyro_input=get_data_process(righthand_gyro,time_keystroke-window_time/2,time_keystroke+window_time/2,window_time)
        rh_gravity_input=get_data_process(righthand_gravity,time_keystroke-window_time/2,time_keystroke+window_time/2,window_time)
        if 0 in l_d.shape or 0 in lh_accel_input.shape or 0 in lh_gyro_input.shape or 0 in lh_gravity_input.shape\
             or 0 in rh_accel_input.shape or 0 in rh_gyro_input.shape or 0 in rh_gravity_input.shape:
            print "omitting keylog instance:", l_d
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
        data_labels.append(char_to_indx[l_d[-1].lower()])
        if i%200==0:
            print "Completed iteration ",i,"/",log_keystrokes.shape[0]
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