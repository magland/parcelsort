import numpy as np

def detect_on_channel(data,*,detect_threshold,detect_interval,detect_sign,margin=0):
    # Adjust the data to accommodate the detect_sign
    # After this adjustment, we only need to look for positive peaks
    if detect_sign<0:
        data=data*(-1)
    elif detect_sign==0:
        data=np.abs(data)
    elif detect_sign>0:
        pass

    data=data.ravel()
        
    #An event at timepoint t is flagged if the following two criteria are met:
    # 1. The value at t is greater than the detection threshold (detect_threshold)
    # 2. The value at t is greater than the value at any other timepoint within plus or minus <detect_interval> samples
    
    # First split the data into segments of size detect_interval (don't worry about timepoints left over, we assume we have padding)
    N=len(data)
    S2=int(np.floor(N/detect_interval))
    N2=S2*detect_interval
    data2=np.reshape(data[0:N2],(S2,detect_interval))
    
    # Find the maximum on each segment (these are the initial candidates)
    max_inds2=np.argmax(data2,axis=1)
    max_inds=max_inds2+detect_interval*np.arange(0,S2)
    max_vals=data[max_inds]
    
    # The following two tests compare the values of the candidates with the values of the neighbor candidates
    # If they are too close together, then discard the one that is smaller by setting its value to -1
    # Actually, this doesn't strictly satisfy the above criteria but it is close
    # TODO: fix the subtlety
    max_vals[ np.where((max_inds[0:-1]>=max_inds[1:]-detect_interval) & (max_vals[0:-1]<max_vals[1:]))[0] ]=-1
    max_vals[1+np.array( np.where((max_inds[1:]<=max_inds[0:-1]+detect_interval) & (max_vals[1:]<=max_vals[0:-1]))[0] )]=-1
    
    # Finally we use only the candidates that satisfy the detect_threshold condition
    times=max_inds[ np.where(max_vals>=detect_threshold)[0] ]
    if margin>0:
        times=times[np.where((times>=margin)&(times<N-margin))[0]]

    return times
