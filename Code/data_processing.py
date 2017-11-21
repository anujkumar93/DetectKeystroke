import numpy as np
import os
import pickle
import csv
import datetime as dt

def preprocess_data(char_set,input_case=["all"],log_buffer=dt.timedelta(seconds=1)):
    """
    char_set: set of characters for which input has to be considered
    input_case: one of ['room'], ['library'] or 'all'
    log_buffer: time duration of type timedelta giving the buffer allowed between two successive keyfreq readings
    """
    if input_case=="all":
        input_case=["room","library"]
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
                lh_accel0,lh_gyro0,lh_gravity0=find_starting_line(lh_file)
                rh_accel0,rh_gyro0,rh_gravity0=find_starting_line(rh_file)
                # print lh_accel0,lh_gyro0,lh_gravity0
                # print rh_accel0,rh_gyro0,rh_gravity0
                curr_log_start_time,curr_log_end_time=None,None
                for line in curr_log_f:
                    line_split=line.split()
                    if line_split[-1] in char_set:
                        time_split=line_split[0].split(':')
                        curr_log_start_time=dt.datetime.fromtimestamp(float(time_split[0]))
                        curr_log_start_time=curr_log_start_time+dt.timedelta(microseconds=float(time_split[1]))
                        line_split[0]=time_split[0]+time_split[1]
                        curr_log_keystrokes=[line_split]
                        prev_end_time=curr_log_start_time
                        for end_line in curr_log_f:
                            end_line_split=end_line.split()
                            end_time_split=end_line_split[0].split(':')
                            curr_end_time=dt.datetime.fromtimestamp(float(end_time_split[0]))
                            curr_end_time=curr_end_time+dt.timedelta(microseconds=float(end_time_split[1]))
                            if (curr_end_time-prev_end_time)>log_buffer and end_line_split[-1] not in char_set:
                                curr_log_end_time=prev_end_time
                                break
                            prev_end_time=curr_end_time
                            end_line_split[0]=end_time_split[0]+end_time_split[1]
                            curr_log_keystrokes.append(line_split)
                        curr_lefthand_gyro.append(get_chunk_data(lh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,lh_gyro0))
                        curr_lefthand_accel.append(get_chunk_data(lh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,lh_accel0))
                        curr_lefthand_gravity.append(get_chunk_data(lh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,lh_gravity0))
                        curr_righthand_gyro.append(get_chunk_data(rh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,rh_gyro0))
                        curr_righthand_accel.append(get_chunk_data(rh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,rh_accel0))
                        curr_righthand_gravity.append(get_chunk_data(rh_file,curr_log_start_time-log_buffer/2,curr_log_end_time+log_buffer/2,rh_gravity0))
            final_log_keystrokes.append(curr_log_keystrokes)
            final_lefthand_gyro.append(curr_lefthand_gyro)
            final_lefthand_accel.append(curr_lefthand_accel)
            final_lefthand_gravity.append(curr_lefthand_gravity)
            final_righthand_gyro.append(curr_righthand_gyro)
            final_righthand_accel.append(curr_righthand_accel)
            final_righthand_gravity.append(curr_righthand_gravity)
            print "Successfully completed",str(file_num+1),"files"
    return final_log_keystrokes,final_lefthand_accel,final_lefthand_gyro,final_lefthand_gravity,\
                final_righthand_accel,final_righthand_gyro,final_righthand_gravity

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

def get_chunk_data(filename, start_time, end_time, offset):
    result_set=[]
    with open(filename,'rb') as f:
        reader=csv.reader(f)
        lines=list(reader)
        for i in range(offset,len(lines)):
            curr_time_str=lines[i][1]+" "+lines[i][2]
            curr_time=dt.datetime.strptime(curr_time_str,' %b %d %Y %H:%M:%S.%f')
            if curr_time>start_time and curr_time<end_time:
                curr_set=[dt.datetime.strftime(curr_time,'%s')+str(curr_time.microsecond)]
                curr_set.extend(lines[i][3:])
                result_set.append(curr_set)
            if curr_time>end_time:
                break
    return result_set