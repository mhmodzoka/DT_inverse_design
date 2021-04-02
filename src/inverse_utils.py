# -*- coding: utf-8 -*-

from decimal import Decimal

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold

# DT_compression_Elzouka.py
import matplotlib.pyplot as plt
from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import pi

from other_utils import df_to_csv


spherical_harmonics = ['L0','L1','L2','L3','L4','L5','L6','L7','L8']


def random_draw_my_number_list(X, feature_name, n):
    avg_diff_in_spacing = np.diff(np.sort(X[feature_name])).mean()
    feature_values = np.random.choice(X[feature_name],size=n) + np.random.normal(scale=avg_diff_in_spacing/2,size=n)
    feature_values[feature_values<X[feature_name].min()] = X[feature_name].min()
    feature_values[feature_values>X[feature_name].max()] = X[feature_name].max()    
    return feature_values


### --- Utils for Inverse Design --- ###

#utils
#initialize limits on feature space
def Spherical_Harmonic_init(x):
    min_L = x.min()
    max_L = x.max()
    if min_L<0:
        min_lim = min_L*1.05
    else:
        min_lim = min_L*0.95
    if max_L<0:
        max_lim = max_L*0.95
    else:
        max_lim = max_L*1.05
    return [min_lim,max_lim]

def init_sample_lims(X):
    """
    This function get the limits (low and high bounds) for each of the features
    """
    print("Initial limits, based on the training data range:")
    
    feature_names = list(X.columns)
    min_max_dict = {}
    if 'ShortestDim' in feature_names:
        min_max_dict['ShortestDim'] = [X['ShortestDim'].min()*0.95,X['ShortestDim'].max()*1.05]
        print("ShortestDim Initial Range: {0}".format(min_max_dict['ShortestDim']))
        
    if 'MiddleDim' in feature_names:
        min_max_dict['MiddleDim'] = [X['MiddleDim'].min()*0.95,X['MiddleDim'].max()*1.05]
        print("MiddleDim Initial Range: {0}".format(min_max_dict['MiddleDim']))
        
    if 'LongDim' in feature_names:
        min_max_dict['LongDim'] = [X['LongDim'].min()*0.95,X['LongDim'].max()*1.05]
        print("LongDim Initial Range: {0}".format(min_max_dict['LongDim']))
        
    if 'ShortToLong' in feature_names:
        min_max_dict['ShortToLong'] = [0.0,1.0]
        print("ShortToLong Initial Range: {0}".format(min_max_dict['ShortToLong']))
    if 'MidToLong' in feature_names:
        min_max_dict['MidToLong'] = [0.0,1.0]
        print("MidToLong Initial Range: {0}".format(min_max_dict['MidToLong']))
    
    if 'Area/Vol' in feature_names:
        min_max_dict['Area/Vol'] = [X['Area/Vol'].min()*0.95,X['Area/Vol'].max()*1.05]
        print("Area/Vol Initial Range: {0}".format(min_max_dict['Area/Vol']))
    else:
        if 'log Area/Vol' in feature_names:
            min_max_dict['log Area/Vol'] = [X['log Area/Vol'].min()*0.95,X['log Area/Vol'].max()*1.05]
            print("'log Area/Vol' Initial Range: {0}".format(min_max_dict['log Area/Vol']))
            
            # I will define Area/Vol here, because the entire code depend on it
            min_max_dict['Area/Vol'] = np.exp(np.array(min_max_dict['log Area/Vol']))
            print("Area/Vol Initial Range: {0}".format(min_max_dict['Area/Vol']))
    
        
    # for the OHE features
    features_OHE = ['Geometry', 'Material']
    for OHE_here in features_OHE:
        feature_names_OHE = [x for x in feature_names if OHE_here in x]
        for gg in feature_names_OHE:
            min_max_dict[gg] = [0.0,1.0]
    
    #check for Spherical harmonics
    for l in spherical_harmonics:
        if l in feature_names:
            min_max_dict[l] = Spherical_Harmonic_init(X[l])
            print(l+" Initial Range: {0}".format(min_max_dict[l]))
    
    print("")
    
    return min_max_dict

#does unit conversion and determines values for area/vol
def get_areavol_helper(min_max_dict,sample_value,threshold_value,feature_name):
    """
    We only care for 'Area/Vol', which means we return design rules for only Area/Vol, not 'log Area/Vol' or any other derivatives
    """
    #convert to same units
    if feature_name == 'Vol/Area':
        threshold_value = 1000.0 / float(threshold_value) #difference in scaling of values
        sample_value = 1000.0 / float(sample_value)
    elif feature_name == 'log Vol/Area':
        threshold_value = 1000.0 / np.exp(threshold_value)
        sample_value = 1000.0 / np.exp(sample_value)
    elif feature_name == 'log Area/Vol':
        threshold_value = np.exp(threshold_value)
        sample_value = np.exp(sample_value)
    
    #update
    feature_names_all = list(min_max_dict.keys())
    if 'Area/Vol' in feature_names_all:
        if sample_value<=threshold_value:
            if min_max_dict['Area/Vol'][1]>threshold_value:
                min_max_dict['Area/Vol'][1] = threshold_value
        else:
            if min_max_dict['Area/Vol'][0]<threshold_value:
                min_max_dict['Area/Vol'][0] = threshold_value 
                
        if 'log Area/Vol' in feature_names_all:
            min_max_dict['log Area/Vol'][0] = np.log(min_max_dict['Area/Vol'][0])
            min_max_dict['log Area/Vol'][1] = np.log(min_max_dict['Area/Vol'][1])
    
    return min_max_dict

#updates the limits dictionary for each node it reaches
def update_sample_lims(min_max_dict,sample_value,threshold_value,feature_name, round_prec_MidShrtLong_dims = 3):
    if sample_value<=threshold_value:
        if 'Area' in feature_name: # for anything that has 'Area'
            min_max_dict = get_areavol_helper(min_max_dict,sample_value,threshold_value,feature_name)
            
        elif ('Geometry' in feature_name) or ("Material" in feature_name): # for OHE features ...
            min_max_dict[feature_name][1] = 0 # for binary values
        
        elif 'ShortestDim' in feature_name or 'MiddleDim' in feature_name or 'LongDim' in feature_name: # for all other features ...
            min_max_dict[feature_name][1] = np.round(threshold_value, round_prec_MidShrtLong_dims)
            
        else: # for all other features ...
            min_max_dict[feature_name][1] = threshold_value

    else:
        if 'Area' in feature_name:
            min_max_dict = get_areavol_helper(min_max_dict,sample_value,threshold_value,feature_name)
        
        elif ('Geometry' in feature_name) or ("Material" in feature_name):
            feature_type = feature_name.split('_')[0]
            print('this is the winning: {0} -------------------------------------------------'.format(feature_type))
            #we've committed to a given material/geometry
            #zero everything out
            for key,val in min_max_dict.items():
                if feature_type in key:
                    min_max_dict[key] = [0.0,0.0]
            #set the given one to 1
            min_max_dict[feature_name] = [1.0,1.0]
            
        elif 'ShortestDim' in feature_name or 'MiddleDim' in feature_name or 'LongDim' in feature_name: # for all other features ...
            min_max_dict[feature_name][0] = np.round(threshold_value, round_prec_MidShrtLong_dims)

        else:
            min_max_dict[feature_name][0] = threshold_value

    #check if lower>upper,

    return min_max_dict

#finds keys in min_max_dict with value [1,1] i.e. allowed binary_values
def find_allowed_binaries(min_max_dict):
    '''
    Finds keys in min_max_dict with value [1,1] i.e. allowed binary_values.
    '''
    allowed_binaries = []
    for key,val in min_max_dict.items():
        if (val[0]==1) and (val[1]==1):
            allowed_binaries.append(key)
        elif (val[0]==0) and (val[1]==1):
            allowed_binaries.append(key)
    return allowed_binaries


def find_allowed_binaries_Elzouka(min_max_dict):
    allowed_binaries = []
    allowed_binaries_fully = []
    for key,val in min_max_dict.items():
        if (val[0]==1) and (val[1]==1):
            allowed_binaries.append(key)
            allowed_binaries_fully.append(key)            
        elif (val[0]==0) and (val[1]==1):
            allowed_binaries.append(key)
    return allowed_binaries,allowed_binaries_fully



#does sanity checks
def check_gen_samples(gen_samples_dict,estimator,min_max_dict):
    #check that every key in gen_samples_dict is in min_max_dict
    feature_keys = min_max_dict.keys() #only check area/vol, not its derivatives
    for key in feature_keys:
        gen_features = gen_samples_dict[key]
        bottom,top = min_max_dict[key]
        try:
            assert bottom<=top #check that limits make sense
            assert isinstance(gen_features,np.ndarray)
            assert sum((gen_features>=bottom)&(gen_features<=top)) == len(gen_features)
        except AssertionError:
            print(bottom)
            print(top)
            print(key)
    #check shorttolong<midtolong
    if 'ShortToLong' in feature_keys:
        short,mid = gen_samples_dict['ShortToLong'],gen_samples_dict['MidToLong']
        assert sum(short<=mid) == len(short)

### --- Generates Samples based on given decision tree and desired leaf --- ###


#given key, value, min_max_dict, return true if value and key are inside bound
def check_val_in_bound(min_max_dict,key,value):
    bottom,top = min_max_dict[key]
    if (value>=bottom) and (value<=top):
        return True
    else:
        return False


#check final df results are physical and fit geometric constraint
def check_physical_bounds(df,min_max_dict, verbosity = 'quite'):
    #check geometric constraint on dimensions
    feature_names = list(set(min_max_dict.keys()) & set(df.columns)) #df.columns
    """"
    if 'ShortToLong' in feature_names:
        keep_idx_1 = (df['ShortToLong']>0.01)&(df['MidToLong']>0.01)
        df = df[keep_idx_1]
    
    if 'Area/Vol' in feature_names:
        keep_idx_2 = (df['Area/Vol']>0.1) & (df['Area/Vol']<10**3)
        df = df[keep_idx_2]
    """
    
    
    #check geometric constraint on wire
    drop_idx = []
    for index, row in df.iterrows():
        if row['Geometry_wire']:
            if ('ShortToLong' in feature_names) & ('MidToLong' in feature_names):
                if row['ShortToLong'] != row['MidToLong']:
                    if check_val_in_bound(min_max_dict,'ShortToLong',row['MidToLong']):
                        row['ShortToLong'] = row['MidToLong']
                    elif check_val_in_bound(min_max_dict,'MidToLong',row['ShortToLong']):
                        row['MidToLong'] = row['ShortToLong']
                    else:
                        drop_idx.append(index)    
    if len(drop_idx)>0:
        df = df.drop(drop_idx,axis=0)    
    print('now we have {0} samples'.format(len(df)))
    
    keep_idx_3 = []
    if 'ShortestDim' in feature_names and 'MiddleDim' in feature_names:        
        ShortestDim_here = df['ShortestDim'].values
        MiddleDim_here = df['MiddleDim'].values        
        keep_idx_3 = np.greater_equal(MiddleDim_here, ShortestDim_here)
        print("we eliminated {0} samples, which has short > middle".format(sum(~keep_idx_3))) if verbosity == 'verbose' else 0
        #display(df[~keep_idx_3])
        df = df[keep_idx_3]    
    print('now we have {0} samples'.format(len(df)))  if verbosity == 'verbose'  else 0
    
    keep_idx_4 = []
    if 'LongDim' in feature_names and 'MiddleDim' in feature_names:        
        LongDim_here = df['LongDim'].values
        MiddleDim_here = df['MiddleDim'].values        
        keep_idx_4 = np.greater_equal(LongDim_here, MiddleDim_here)
        print("we eliminated {0} samples, which has middle > long".format(sum(~keep_idx_4))) if verbosity == 'verbose' else 0
        #display(df[~keep_idx_4])
        df = df[keep_idx_4]     
    print('now we have {0} samples'.format(len(df)))  if verbosity == 'verbose' else 0
    
    # check all features are within min_max_dict bounds
    drop_idx = []    
    for index, row in df.iterrows():         
        for ff in feature_names:
            if ff in min_max_dict.keys():                        
                if not (row[ff] >= min_max_dict[ff][0] and row[ff] <= min_max_dict[ff][1]):
                    print('WARNING: row {0} and feature {1} are out of range {2}, {3}'.format(index, ff,row[ff],min_max_dict[ff]))                    
                    drop_idx.append(index)
                    
    if len(drop_idx)>0:
        df = df.drop(drop_idx,axis=0)
    else:               
        print('Check! all inverse design features are within min_max_dict bounds')  if verbosity == 'verbose' else 0
    
    
        
        
    
    return df


#gives statistics on how many samples are inside min_max_dict for given dict
def find_num_samples_in_lim(X,min_max_dict):
    X = X.reset_index()
    print("Total Number of Samples: {0}".format(len(X)))
    valid_indices = range(len(X))
    drop_indices =[]
    for k,vals in min_max_dict.items():
        bottom,top = vals[0],vals[1]
        sub_df = X[(X[k]<bottom)|(X[k]>top)].index
        drop_indices.extend(list(sub_df))
    keep_indices = [x for x in valid_indices if x not in drop_indices]
    print("Total Valid Samples: {0}".format(len(keep_indices)))
    print("Percent of Valid Samples: {0}".format(float(len(keep_indices))/float(len(valid_indices))))
    
## pass in pca transformed vector coord of desired spectral curve. We find leaf on estimator tree that is closest to 
## the desired curve and return samples that fit under that leaf
def DoInverseDesign_SaveCSV(estimator,X_train,y_train,vector, scaling_factors, \
                                    gen_n=500, search_method='min_sum_square_error', \
                                    plotting=False, frequency = [], n_best_candidates = 5, \
                                    bestdesign_method = 'AbsoluteBest', round_prec_MidShrtLong_dims = 3, verbosity = 'verbose', \
                                    InverDesignSaveFilename_CSV = 'InverseDesign', \
                                    n_best_leaves_force = -1):    
    """
    Inputs ===============================
    
    estimator : DT or RF object    
    ---------
    
    X_train,y_train : training data used for "estimator". This represent the "leaves"
    ---------------
    
    vector : list or array that represent the target emissivity.    
    ------
        For scalar emissivity, vector is a float.
        For spectral emissivity, vector is either an array with the same shape as the "frequency" or a list that determine the ranges where we need to minimize/maximize emissivity.
        
    gen_n : number of inverse designs suggested
    -----
    
    search_method
    -------------
    This function return samples that satisfies a given "vector". There are two methods for search:
        'min_sum_square_error'
        ---------------------- find the spectrum which minimizes the Where 'sum_square_error' between the target "vector" is defining a specific shape of the spectrum VS frequency. 
        If we need to consider only a portion of the frequency, then we set the indices to be ignored to Nan.
        
        'scalar_target'
        --------------- find the minimum square error
        'max_integrated_contrast'
        ------------------------------------ find the maximum contrast
        'max_integrated_contrast_high_emiss'
        ------------------------------------ find the maximum contrast and the highest emissivity
        
    bestdesign_method
    -----------------
    'AbsoluteBest': return the best design, regardless of material and geometry
    'all_comb_mat_geom': for each unique combination of material and geometry, return the best design
    'MorethanOneMatGeom': the inverse design path return more than one material and one geometry
    
    round_prec_MidShrtLong_dims: assuming the scaled dimensions are in microns, we need to round to the closest nm
    
    
    n_best_leaves_force
    ------------------- number of best leaves that I have to return
    ---------------------------
    
    """
    
    assert isinstance(y_train,np.ndarray), "convert to np array so no indexing error"
    
    if len(y_train.shape) == 1: # if y is 1D "i.e., scalar emissivity"
        print('-------------------------- Inverse design for scalar emissivity')
        search_method = 'scalar_target'
    else:
        print('-------------------------- Inverse design for spectral emissivity')
    
    # to add the "diff"; the error from target
    X_train_copy = X_train.copy()
    X_train_with_diff = X_train.copy()
    
    if len(frequency) == 0:
        frequency = np.arange(np.shape(y_train)[1]) # if there is no frequency in the input, then assume spectral emissivity is at equally-spaced frequency points
    
    min_max_dict = init_sample_lims(X_train) #get initialized dict of min,max values for each feature    
    
    feature_names = list(X_train.columns) #list of strings of feature names
    X_train = X_train.values # convert to numpy to prevent indexerror
    
    ### Get info about branch
    n_nodes,children_left,children_right,feature,threshold = get_dt_attributes(estimator)
    
    InvDesign_start_time = time()
    # find the closest spectrum in the traning dataset -------------------------------------------------------------------------------------    
    if search_method == 'scalar_target' : 
        vector_arr = np.ones(len(y_train)) * vector
        diff = np.abs(np.subtract(y_train,vector_arr)) / vector #get minimum relative error w.r.t desired sample
        diff_threshould = 0.1 # this is the maximum error, if InverseDesign is tracking a leave beyond that, it should be rejected
        
    
    elif search_method == 'min_sum_square_error' :
        assert vector.shape[0] == 1, "Vector needs to be (1,n) shaped"
        
        # in the comparison, we need to ignore indices which have "Nan" in the desired "vector"    
        idx_nonNan = np.array(np.where(~np.isnan(vector[0])))[0]        
        vector_ = vector[:,idx_nonNan]
        y_train_ = y_train[:,idx_nonNan]   
        
        # instead of SUM, we are using TRAPZ (i.e., numerical integration)
        area_under_orig_spectrum = np.trapz(vector_)
        diff = np.trapz(np.abs(np.subtract(y_train_,vector_)),axis=1) / area_under_orig_spectrum #get minimum error w.r.t desired sample
        diff_threshould = 0.5 # this is the maximum error, if InverseDesign is tracking a leave beyond that, it should be rejected
    
    elif search_method == 'max_integrated_contrast' :        
        freq_low_high_bounds = vector
        [freq_low_bounds_idx, freq_high_bounds_idx] = return_frequency_idx_containing_ranges(frequency, freq_low_high_bounds, plotting = False)           
        
        # calculated the integral for all frequency ranges where emissivity needs to be LOW
        diff_low = 0     
        for i in freq_low_bounds_idx:            
            i = i[0]            
            diff_low = diff_low + np.trapz(y_train[:,i], frequency[i],axis=1)
        
        # calculated the integral for all frequency ranges where emissivity needs to be HIGH
        diff_high = 0
        for i in freq_high_bounds_idx:
            i = i[0]
            diff_high = diff_high + np.trapz(y_train[:,i], frequency[i],axis=1)
        
        # the contrast, that has to be minimized
        diff = diff_low/diff_high    
        diff_threshould = 100 # this is the maximum error, if InverseDesign is tracking a leave beyond that, it should be rejected
    
    elif search_method == 'max_integrated_contrast_high_emiss' :
        freq_low_high_bounds = vector
        [freq_low_bounds_idx, freq_high_bounds_idx] = return_frequency_idx_containing_ranges(frequency, freq_low_high_bounds, plotting = False)           
        
        # calculated the integral for all frequency ranges where emissivity needs to be LOW
        diff_low = 0     
        for i in freq_low_bounds_idx:            
            i = i[0]            
            diff_low = diff_low + np.trapz(y_train[:,i], frequency[i],axis=1)
        
        # calculated the integral for all frequency ranges where emissivity needs to be HIGH
        diff_high = 0
        for i in freq_high_bounds_idx:
            i = i[0]
            diff_high = diff_high + np.trapz(y_train[:,i], frequency[i],axis=1)
        
        # the contrast, that has to be minimized
        diff = 1/ (diff_high/diff_low + 0.1*diff_high)
        diff_threshould = 100 # this is the maximum error, if InverseDesign is tracking a leave beyond that, it should be rejected
    
    # plot the closest sample VS our target spectrum -------------------------------------------------------------------------------------
    if plotting:        
        if search_method != 'scalar_target' :  
            plt.figure()

            if search_method == 'min_sum_square_error' :        
                plt.plot(frequency, vector[0], label='Target');

            elif search_method == 'max_integrated_contrast' or search_method == 'max_integrated_contrast_high_emiss':
                low_high_emissivity_values = [[0],[1]]
                spectrum_here = draw_target_spectrum_from_low_high_ranges(frequency, freq_low_high_bounds, low_high_emissivity_values, plotting = False)
                plt.plot(frequency, spectrum_here[0], label='Target'); 

            sample_id_top_candid = diff.argsort()[:n_best_candidates]
            for mm in sample_id_top_candid:
                plt.plot(frequency, y_train[mm], label='Closest candidate #'); 

            #plt.legend()
            plt.xlabel('index')
            plt.ylabel('emissivity')        
    
    
    # finding the design rules ------------------------------------------------
    TrackNextBestLeave = True    
    
    rank_min_diff = 0 #this is the index in the sorted "diff", the lower this number, the closer the sample to the target "vector"
    
    _,allowed_binaries_fully = find_allowed_binaries_Elzouka(min_max_dict)
    print('============================================')
    print('------ Here we have allowed -----------------')
    print(allowed_binaries_fully)
    print('============================================')        
    
    
    
    gen_samples_df_all_over_many_leaves = pd.DataFrame(columns=feature_names)
    gen_samples_df_all                  = pd.DataFrame(columns=feature_names)        
    
    min_max_dict = {}
    min_max_dict_orig   = init_sample_lims(X_train_copy) #get initialized dict of min,max values for each feature
            
    while TrackNextBestLeave and rank_min_diff < 10 and diff[diff.argsort()[rank_min_diff]] <= diff_threshould:
        
        gen_samples_df_all = pd.DataFrame(columns=feature_names)        
        
        print('I am tracking back the {0}th best leaf'.format(rank_min_diff))
        
        # to make sure I am initializing "min_max_dict_orig" at every run
        min_max_dict        = init_sample_lims(X_train_copy)
            
        sample_id = diff.argsort()[rank_min_diff]        
        
        leave_id = estimator.apply(X_train[sample_id].reshape(1,-1)) # get the leave_id the corresponds to a given training sample 'X_train[sample_id]'

        node_indicator = estimator.decision_path(X_train[sample_id].reshape(1,-1))
        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[1]]

        features = []
        print('Rules used to predict sample %s: ' % sample_id)    
        print('node_index is: {0}'.format(node_index))
        print('leave_id is: {0}'.format(leave_id))
        for node_id in node_index:
            if leave_id[0] == node_id:                
                continue

            feature_idx = feature[node_id] #int, feature index
            feature_name = feature_names[feature_idx] #string, feature name
            sample_value = X_train[sample_id,feature_idx]
            threshold_value = threshold[node_id]
            min_max_dict = update_sample_lims(min_max_dict,sample_value,threshold_value,feature_name, round_prec_MidShrtLong_dims = round_prec_MidShrtLong_dims)                

            if (sample_value <= threshold_value):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
                  % (node_id,
                     sample_id,
                     feature_name,
                     sample_value,
                     threshold_sign,
                     threshold_value))
            
            features.append(feature_idx)                        
        
        InvDesign_end_time = time()
        print('==================================================================================')
        print('==================================================================================')
        print('==================================================================================')
        print('==================================================================================')
        print('time for inverse design to trace back one leaf = {0}'.format(InvDesign_end_time - InvDesign_start_time))
        print('==================================================================================')
        print('==================================================================================')
        print('==================================================================================')
        print('==================================================================================')
        
        # check if we get what we want
        if rank_min_diff < n_best_leaves_force:
            TrackNextBestLeave = True
        
        else:
            if bestdesign_method == 'AbsoluteBest':
                TrackNextBestLeave = False
                
            elif bestdesign_method == 'MorethanOneMatGeom':            
                # find number of allowed materials and geometries
                allowed_binaries = find_allowed_binaries(min_max_dict)
                n_geomm = len([x for x in allowed_binaries if 'Geometry' in x ])
                n_matt = len([x for x in allowed_binaries if 'Material' in x ])
                
                if n_matt > 1 or n_geomm > 1:
                    TrackNextBestLeave = False
                else:
                    TrackNextBestLeave = True                
                    print("Here, my inverse design suggested {0} materials and {1} geometries, but I need more than 1 for either ************************************************** Try the next best leaf ".format(n_matt, n_geomm))
                    #print(min_max_dict_orig)            
            
            else:
                TrackNextBestLeave = False        
    
        
        if ~TrackNextBestLeave:
            gen_more = True            
            gen_n_new = gen_n
            
            # to make sure we have generated enough samples
            gen_n_new = gen_n_new/2
            while gen_more:
                gen_n_new = int(gen_n_new*2)
                print('\nUsing the best {0}th leaf, I am generating extra {1} features'.format(rank_min_diff, gen_n_new))                
                
                sample_dict = generate_samples_from_lims_Elzouka_2(min_max_dict, gen_n_new, feature_names, round_prec_MidShrtLong_dims)
                                
                if isinstance(sample_dict, pd.DataFrame):
                    gen_samples_df = sample_dict[feature_names]
                else:
                    check_gen_samples(sample_dict,estimator,min_max_dict)        
                    new_samples = []        
                    for feat in feature_names:
                        new_samples.append(sample_dict[feat].reshape(-1,1))
                    gen_samples_df = pd.DataFrame(np.concatenate(new_samples,axis=1),columns=feature_names)
                    
                
                
                gen_samples_df['diff'] = diff[diff.argsort()[rank_min_diff]]*np.ones(len(gen_samples_df))
                gen_samples_df['leaf_id'] = leave_id*np.ones(len(gen_samples_df)) # leaf_ID used for inverse design
                gen_samples_df_all = gen_samples_df_all.append(gen_samples_df)
                
                gen_samples_df_all = gen_samples_df_all[~gen_samples_df_all[feature_names].duplicated()] # remove any duplicates
                
                # add what we have here to what we have from other leaves
                gen_samples_df_all_over_many_leaves = gen_samples_df_all_over_many_leaves.append(gen_samples_df_all)
                
                
                # find the number of Geometry and Material produced                
                all_feat = gen_samples_df_all_over_many_leaves.columns
                n_geom_all = 0; n_mat_all = 0
                for ff in all_feat:
                    if 'Geometry' in ff:
                        if sum(gen_samples_df_all_over_many_leaves[ff]) > 0:
                            n_geom_all = n_geom_all+1
                    elif 'Material' in ff:
                        if sum(gen_samples_df_all_over_many_leaves[ff]) > 0:
                            n_mat_all = n_mat_all+1
                                 
                We_have_enough_geom_mat = (bestdesign_method != 'MorethanOneMatGeom' or n_geom_all>1 or n_mat_all>1)
                We_have_generated_enough_features = len(gen_samples_df_all_over_many_leaves) >= gen_n
                We_have_achieved_target = We_have_enough_geom_mat and We_have_generated_enough_features
                We_have_generated_aLot_still_not_enough = gen_n_new > 1*gen_n
                
                if rank_min_diff < n_best_leaves_force:
                    TrackNextBestLeave = True
                    gen_more = False
        
                else:                                
                    if We_have_achieved_target or We_have_generated_aLot_still_not_enough:
                        gen_more = False
                    else:
                        gen_more = True
                    
                    if We_have_achieved_target:
                        TrackNextBestLeave = False
                    else:
                        TrackNextBestLeave = True
                    
                            
            
            # add short/long and Mid/long if they don't exist            
            if not('ShortToLong' in feature_names):
                if ('ShortestDim' in feature_names) and('LongDim' in feature_names):
                    gen_samples_df_all['ShortToLong'] = gen_samples_df_all['ShortestDim'].values / gen_samples_df_all['LongDim'].values
            if not('MidToLong' in feature_names):
                if ('MiddleDim' in feature_names) and('LongDim' in feature_names):
                    gen_samples_df_all['MidToLong'] = gen_samples_df_all['MiddleDim'].values / gen_samples_df_all['LongDim'].values
            
            # reduce number of gen_samples to desired    
            if len(gen_samples_df_all) > gen_n:
                #features_all = 
                feat_mat_geom = [k for k in feature_names if ('Material' in k) or ('Geometry' in k)]        
                
                if n_geom_all>1 or n_mat_all>1:
                    test_size = gen_n / len(gen_samples_df_all)
                    try:
                        _,_,gen_samples_df_all,_ = train_test_split(gen_samples_df_all,gen_samples_df_all,test_size=test_size,stratify=gen_samples_df_all[feat_mat_geom]) # pass by reference? # Charles?
                    except:
                        _,_,gen_samples_df_all,_ = train_test_split(gen_samples_df_all,gen_samples_df_all,test_size=test_size) # pass by reference? # Charles?
                else:
                    aa = gen_samples_df_all.sample(n = gen_n)
                    gen_samples_df_all = aa.copy()
            
            # Check physical bounds            
            n_before = len(gen_samples_df_all)
            gen_samples_df_all = check_physical_bounds(gen_samples_df_all,min_max_dict)            
            print("Out of {0} samples, {1} passed physicality bounds check".format(n_before,len(gen_samples_df_all)))    
            
            # save to CSV
            if len(gen_samples_df_all) > 0: 
                bestleaf_title = '_BestLeaf_{0}'.format(rank_min_diff)
                csv_filename = InverDesignSaveFilename_CSV+bestleaf_title
                df_to_csv(gen_samples_df_all, csv_filename+'.csv', scaling_factors)
        
                # save design rules
                csv_filename = InverDesignSaveFilename_CSV+'_DesignRules'+bestleaf_title
                df_to_csv(pd.DataFrame.from_dict(min_max_dict), csv_filename+'.csv', scaling_factors)
        
                # save DataLim
                csv_filename = InverDesignSaveFilename_CSV+'_DataLim'+bestleaf_title
                df_to_csv(pd.DataFrame.from_dict(min_max_dict_orig), csv_filename+'.csv', scaling_factors)
                
            
            
            if TrackNextBestLeave:
                rank_min_diff = rank_min_diff + 1 # use the next best leaf
                    
                               
    '''  
    gen_samples_df = gen_samples_df_all.copy()   
    gen_samples_df = gen_samples_df[~gen_samples_df[feature_names].duplicated()] # remove any duplicates
    
    # reduce number of gen_samples to desired    
    if len(gen_samples_df) > gen_n:
        #features_all = 
        feat_mat_geom = [k for k in feature_names if ('Material' in k) or ('Geometry' in k)]        
        
        gen_samples_df_matgeom_only = gen_samples_df[feat_mat_geom].copy()
        n_unique_matgeom = len(gen_samples_df_matgeom_only.drop_duplicates())
        if n_unique_matgeom > 1:
            test_size = gen_n / len(gen_samples_df)
            try:
                _,_,gen_samples_df,_ = train_test_split(gen_samples_df,gen_samples_df,test_size=test_size,stratify=gen_samples_df[feat_mat_geom]) # pass by reference? # Charles?
            except:
                _,_,gen_samples_df,_ = train_test_split(gen_samples_df,gen_samples_df,test_size=test_size) # pass by reference? # Charles?
        else:
            aa = gen_samples_df.sample(n = gen_n)
            gen_samples_df = aa.copy()
    
    # since we possible have multiple leaves, we wil; have multiple 'min_max_dict'. So, this check will can't hold
    n_reduced = len(gen_samples_df)
    gen_samples_df = check_physical_bounds(gen_samples_df,min_max_dict)
    print("Out of {0} samples, {1} passed physicality bounds check".format(n_reduced,len(gen_samples_df)))    
    
    
    # add short/long and Mid/long if they don't exist
    feature_names = list(gen_samples_df.columns)
    if not('ShortToLong' in feature_names):
        if ('ShortestDim' in feature_names) and('LongDim' in feature_names):
            gen_samples_df['ShortToLong'] = gen_samples_df['ShortestDim'].values / gen_samples_df['LongDim'].values
    if not('MidToLong' in feature_names):
        if ('MiddleDim' in feature_names) and('LongDim' in feature_names):
            gen_samples_df['MidToLong'] = gen_samples_df['MiddleDim'].values / gen_samples_df['LongDim'].values
    
    do_check = False
    if do_check: ######################################################## WHY  WHY  WHY  WHY  WHY  # Charles?
        #check only one leaf returned and it is the same leaf as sample_id
        target_leaf_id = estimator.apply(X_train[sample_id].reshape(1,-1))[0]
        try:
            assert len(np.unique(estimator.apply(gen_samples_df))) == 1, "Generated Samples don't map to single leaf"
        except AssertionError:
            print(np.unique(estimator.apply(gen_samples_df)))
        assert np.unique(estimator.apply(gen_samples_df))[0] == target_leaf_id, "Generated Samples don't map to target leaf"    
    '''
    
    return gen_samples_df_all, min_max_dict,min_max_dict_orig # WARNING! this return only the results from the last traversed leaf. This is because it is not convenient for me to return multiple dictionaries for 'min_max_dict'
    



def ignore():
        ### Get info about branch
    n_nodes,children_left,children_right,feature,threshold = get_dt_attributes(estimator)
    #get leaves
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    closest_idx = -1
    closest_mse = 10**6
    for value,is_leaf,idx in zip(estimator.tree_.value,is_leaves,range(n_nodes)):
        if not is_leaf:
            continue
        else:
            print(value.shape)
            diff = np.sum(np.square(np.subtract(y_train,value)),axis=1)
            if diff<closest_mse:
                closest_mse = diff
                closest_idx = idx
    
    sample_id = 0
    node_indicator = estimator.decision_path(X_train[closest_idx].reshape(1,-1))
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]


## creating tailored target spectrum =============================================================START===================================
def return_frequency_idx_containing_ranges(frequency, freq_low_high_bounds, plotting = False):
    """
    Inputs:
    -------
    frequency : 1D array for frequency points
    freq_low_high_bounds: list containing two lists:
        The first describes the start and end of frequency intervals describing the low emissivity.
        The second list is the same but for the high emissivity
    plotting : if True, plot the low emissivity as 0 and high emissivity as 1 VS indices
    
    Outputs:
    --------
    list containing two lists, the first one contains lists describing all the indices in "frequency" that represent the interval
    """
    freq_low_bounds  = freq_low_high_bounds[0]
    freq_high_bounds = freq_low_high_bounds[1]
        
    freq_low_bounds_idx = [np.where( (i[0] <= frequency) & (i[1] >= frequency)) for i in freq_low_bounds]
    freq_high_bounds_idx = [np.where( (i[0] <= frequency) & (i[1] >= frequency)) for i in freq_high_bounds]
    
    if plotting:
        plt.figure()
        for i in freq_low_bounds_idx:
            plt.plot(i[0], np.zeros(np.shape(i[0])))                    
        for i in freq_high_bounds_idx:
            plt.plot(i[0], np.ones (np.shape(i[0])))
        plt.xlabel('indices')
        plt.ylabel('high or low')
    
    return [freq_low_bounds_idx, freq_high_bounds_idx]

def draw_target_spectrum_from_low_high_ranges(frequency, freq_low_high_bounds, low_high_emissivity_values, plotting = False):
    low_emissivity_values  = low_high_emissivity_values[0]
    high_emissivity_values = low_high_emissivity_values[1]
    
    target_vector = np.nan*frequency # initialize with Nan
    
    [freq_low_bounds_idx, freq_high_bounds_idx] = return_frequency_idx_containing_ranges(frequency, freq_low_high_bounds, plotting = True)    
    
    for i in np.arange(len(freq_low_bounds_idx)):        
        target_vector[freq_low_bounds_idx [i]] = low_emissivity_values [np.min([i, len(low_emissivity_values)-1])]
    
    for i in np.arange(len(freq_high_bounds_idx)):
        target_vector[freq_high_bounds_idx[i]] = high_emissivity_values[np.min([i, len(high_emissivity_values)-1])]
        
    if plotting:
        plt.figure()
        plt.plot(frequency, target_vector)  
        plt.xlabel('frequency')
        plt.ylabel('emissivity')
        
    target_vector = np.reshape(target_vector, [1, len(target_vector)])
    
    return target_vector

## creating tailored target spectrum =============================================================END===================================



def fill_features_by_geom(gen_features_sample,n):
    
    feature_names = gen_features_sample.columns        
    
    for index, row in gen_features_sample.iterrows():
        if row['row_defined']==False and row['valid_geom']==False:
            if ~np.isnan(row['ShortestDim']) and ~np.isnan(row['LongDim']):
                if np.abs(row['ShortestDim'] - row['LongDim']) < 1e-8:
                    gen_features_sample.at[index, 'MiddleDim']   = row['ShortestDim']        

            if row['Geometry_sphere']:                
                if ~np.isnan(row['Area/Vol']):                    
                    Shrt_here = 6/row['Area/Vol']
                    if np.isnan(row['ShortestDim']):
                        gen_features_sample.at[index, 'ShortestDim'] = Shrt_here
                    else:
                        if np.abs(row['ShortestDim'] - Shrt_here)/Shrt_here > 0.05:
                            gen_features_sample.at[index, 'valid_geom'] = False
                            
                        
                    gen_features_sample.at[index, 'MiddleDim']   = gen_features_sample.at[index, 'ShortestDim']
                    gen_features_sample.at[index, 'LongDim']     = gen_features_sample.at[index, 'ShortestDim']
                    gen_features_sample.at[index, 'row_defined'] = True
                elif ~np.isnan(row['ShortestDim']):                    
                    gen_features_sample.at[index, 'MiddleDim']   = row['ShortestDim']
                    gen_features_sample.at[index, 'LongDim']     = row['ShortestDim']
                    gen_features_sample.at[index, 'Area/Vol']    = 6/row['ShortestDim']
                    gen_features_sample.at[index, 'row_defined'] = True
                elif ~np.isnan(row['MiddleDim']):                    
                    gen_features_sample.at[index, 'ShortestDim'] = row['MiddleDim']
                    gen_features_sample.at[index, 'LongDim']     = row['MiddleDim']
                    gen_features_sample.at[index, 'Area/Vol']    = 6/row['MiddleDim']
                    gen_features_sample.at[index, 'row_defined'] = True
                elif ~np.isnan(row['LongDim']):                    
                    gen_features_sample.at[index, 'ShortestDim'] = row['LongDim']
                    gen_features_sample.at[index, 'MiddleDim']   = row['LongDim']
                    gen_features_sample.at[index, 'Area/Vol']    = 6/row['LongDim']
                    gen_features_sample.at[index, 'row_defined'] = True
                

            elif row['Geometry_wire']:
                if ~np.isnan(row['MiddleDim']):
                    D = row['MiddleDim']
                    if (~np.isnan(row['Area/Vol'])):
                        L = 2/(row['Area/Vol'] - 4/D) # A/V = 2/L + 4/D
                        ss_wire = np.sort([D,D,L])
                        gen_features_sample.at[index, 'ShortestDim'] = ss_wire[0]
                        gen_features_sample.at[index, 'MiddleDim']   = ss_wire[1]
                        gen_features_sample.at[index, 'LongDim']     = ss_wire[2]
                        gen_features_sample.at[index, 'row_defined'] = True
                    elif (~np.isnan(row['ShortestDim'])) and (~np.isnan(row['LongDim'])):
                        if np.abs(row['MiddleDim'] - row['ShortestDim']) < 1e-8: # if MiddleDim = ShortestDim
                            L = row['LongDim']
                        elif np.abs(row['MiddleDim'] - row['LongDim']) < 1e-8: # if MiddleDim = LongDim
                            L = row['ShortestDim'] 
                        else: # if non of the three Dims are equal
                            if np.abs(row['MiddleDim'] - row['ShortestDim']) < np.abs(row['MiddleDim'] - row['LongDim']):
                                L = row['LongDim']
                                row['MiddleDim'] = row['ShortestDim']
                            else:
                                L = row['ShortestDim']
                                row['MiddleDim'] = row['LongDim']                        
                        gen_features_sample.at[index, 'Area/Vol']    = 2/L + 4/D
                        ss_wire = np.sort([D,D,L])
                        gen_features_sample.at[index, 'ShortestDim'] = ss_wire[0]
                        gen_features_sample.at[index, 'MiddleDim']   = ss_wire[1]
                        gen_features_sample.at[index, 'LongDim']     = ss_wire[2]
                        gen_features_sample.at[index, 'row_defined'] = True
                    else:
                        print("this one needs attention")
                        print(row)
                        
                        
                elif (~np.isnan(row['Area/Vol'])) and (~np.isnan(row['ShortestDim'])) and (~np.isnan(row['LongDim'])):
                    # assume L = row['ShortestDim']
                    L = row['ShortestDim']
                    D = row['LongDim']
                    AV_hyp1 = 2/L + 4/D
                    
                    D = row['ShortestDim']
                    L = row['LongDim']
                    AV_hyp2 = 2/L + 4/D
                    
                    if (abs(row['Area/Vol'] - AV_hyp1)) < (abs(row['Area/Vol'] - AV_hyp2)):
                        gen_features_sample.at[index, 'MiddleDim'] = row['LongDim']
                    else:
                        gen_features_sample.at[index, 'MiddleDim'] = row['ShortestDim']                    
                    
                    ss_wire = np.sort([D,D,L])
                    gen_features_sample.at[index, 'ShortestDim'] = ss_wire[0]
                    gen_features_sample.at[index, 'MiddleDim']   = ss_wire[1]
                    gen_features_sample.at[index, 'LongDim']     = ss_wire[2]
                    gen_features_sample.at[index, 'row_defined'] = True
                
                else:
                    print("this one needs attention")
                    print(row)               
                    

            elif row['Geometry_parallelepiped']:
                if ~np.isnan(row['ShortestDim']) and ~np.isnan(row['MiddleDim']) and ~np.isnan(row['LongDim']):
                    w = row['ShortestDim']
                    h = row['MiddleDim']
                    Lx = row['LongDim']
                    Area=2*(w+h)*Lx + 2*w*h
                    Volume=w*h*Lx
                    gen_features_sample.at[index, 'Area/Vol']    = Area/Volume
                    gen_features_sample.at[index, 'row_defined'] = True

            elif row['Geometry_TriangPrismIsosc']:
                if ~np.isnan(row['ShortestDim']) and ~np.isnan(row['MiddleDim']) and ~np.isnan(row['LongDim']):
                    h = row['ShortestDim']
                    L = row['MiddleDim']
                    Lx = row['LongDim']
                    Area=2*(0.5*L*h) + Lx*L + 2*Lx*np.sqrt((L/2)**2+h**2)
                    Volume=0.5*L*h*Lx 
                    gen_features_sample.at[index, 'Area/Vol']    = Area/Volume
                    gen_features_sample.at[index, 'row_defined'] = True    
        
    return gen_features_sample



##########################################################
def z_range_intrsct_two_ranges(rng_1, rng_2):
    """
    Find the range of the intersection between rng_1 and rng_2
    """
    rng_intrsct = []
    if len(rng_1) == 2 and len(rng_2) == 2:
        min_intrsct = np.max([rng_1[0], rng_2[0]])
        max_intrsct = np.min([rng_1[1], rng_2[1]])
        if min_intrsct < max_intrsct:
            rng_intrsct = [min_intrsct, max_intrsct]        
    
    return rng_intrsct
    

def z_range_intrsct_ranges(rngss):
    rng_intrsct = rngss[0]
    for rng in rngss:
        rng_intrsct = z_range_intrsct_two_ranges(rng, rng_intrsct)
    
    return rng_intrsct

def generate_samples_from_lims_Elzouka_2(min_max_dict,n,X_test_feature_names, round_prec_MidShrtLong_dims):
    """
    Generate design parameters within the design rules, and consistent with goemetry.

    Parameters:
    ===========
    min_max_dict                : dictionary, represetning inverse design rules. It contains bounds (minimum and maximum) dictary for each feature

    round_prec_MidShrtLong_dims : int, number of decimal points for rounding Short, Mid and Long dimensions

    Return:
    =======
    
    """
    feature_names = list(min_max_dict.keys())    
    gen_features_sample = pd.DataFrame(columns= feature_names )
    
    #gen_features_sample = pd.DataFrame(columns= feature_names+['row_defined'] )   
    #gen_features_sample['row_defined'] = [False for i in np.arange(n)]    
    #gen_features_sample['valid_geom'] = [False for i in np.arange(n)]    
    allowed_binaries = find_allowed_binaries(min_max_dict)
    
    # setting the material and geometry ----------------------------------------------------------
    #GEOMETRY
    if any(['Geometry' in x for x in feature_names]):
        #we've committed to a given material/geometry
        #zero everything out
        #figure out how many allowed geometries
        num_allowed_geoms = 0
        single_geom_flag = False
        allowed_geom_list = []
        for key,val in min_max_dict.items():
            if "Geometry" in key:
                #low=min_max_dict[key][0]
                high=min_max_dict[key][1]
                if high == 1:
                    allowed_geom_list.append(key)        
        
    if any(['Material' in x for x in feature_names]):
        #we've committed to a given material/geometry
        #zero everything out
        #figure out how many allowed geometries
        num_allowed_geoms = 0
        single_geom_flag = False
        allowed_mat_list = []
        for key,val in min_max_dict.items():
            if "Material" in key:
                #low=min_max_dict[key][0]
                high=min_max_dict[key][1]
                if high == 1:
                    allowed_mat_list.append(key)                   
          
        print("Allowed Materials: {0}, allowed Geometries: {1}".format(allowed_mat_list, allowed_geom_list))
        
        
    # based on ranges in min_max_dict, check possibility of creating a valid geometry, for every allowed geometry
    valid_geom_list = []
    Shrt = min_max_dict['ShortestDim']
    Mdl = min_max_dict['MiddleDim']
    Lng = min_max_dict['LongDim']
    AV = min_max_dict['Area/Vol']
    for geom_here in allowed_geom_list:
        if "sphere" in geom_here:            
            D_range = z_range_intrsct_ranges((Shrt,Mdl, Lng))            
            if len(D_range) > 0: # if there is a common intersection between all the Short, Mid, Long                
                D_range_from_AV = np.sort(6/np.array(AV))
                D_range_final = z_range_intrsct_ranges((D_range,D_range_from_AV))
                if len(D_range_final) > 0:                    
                    # generate spheres from the D_range_final
                    D_gen = np.random.uniform(low=D_range_final[0],high=D_range_final[1],size=n)
                    D_gen = np.round(D_gen, round_prec_MidShrtLong_dims)
                    AV_gen = 6/D_gen
                    
                    gen_features_sample_here = pd.DataFrame(columns=feature_names)
                    gen_features_sample_here['ShortestDim'] = D_gen
                    gen_features_sample_here['MiddleDim']   = D_gen
                    gen_features_sample_here['LongDim']     = D_gen
                    gen_features_sample_here['Area/Vol']    = AV_gen
                    
                    # setting the geometry and appending to previous dataframe
                    for mm in feature_names:
                        if 'Geometry' in mm:
                            if geom_here in mm:
                                gen_features_sample_here[mm] = 1
                            else:
                                gen_features_sample_here[mm] = 0                                
                    for mattt in allowed_mat_list:
                        for mm in feature_names:
                            if 'Material' in mm:
                                if mattt in mm:
                                    gen_features_sample_here[mm] = 1
                                else:
                                    gen_features_sample_here[mm] = 0
                        gen_features_sample = gen_features_sample.append(gen_features_sample_here)                    
                    
        elif "wire" in geom_here:            
            # range where Mdl = Shrt
            D_range_long_wire = z_range_intrsct_ranges((Shrt,Mdl))
            L_range_long_wire = Lng
            # generate valid combinations (L>D), calculate AV, then check if it is within the range of AV
            D_,Lx_= z_generate_random_while_keep_small_large_2rnges(D_range_long_wire, L_range_long_wire, n)      
            
            
            # range where Mdl = Lng
            D_range_flat_disc = z_range_intrsct_ranges((Lng,Mdl))
            L_range_flat_disc = Shrt
            # generate valid combinations (L<D), calculate AV, then check if it is within the range of AV            
            Lx__,D__ = z_generate_random_while_keep_small_large_2rnges(L_range_flat_disc, D_range_flat_disc, n)
            
            Lx = np.append(Lx_,Lx__)
            D = np.append(D_, D__)
            
            Lx = np.round(Lx, round_prec_MidShrtLong_dims)
            D = np.round(D, round_prec_MidShrtLong_dims)
            
            Area   = 2*(0.25*pi*D**2) + Lx*pi*D
            Volume = 0.25*pi* D**2 * Lx
            
            AV_gen = Area/Volume            
            # find AV_gen that is within the given range of AV
            idx_AV_valid = np.intersect1d(np.where(AV_gen>=AV[0]), np.where(AV_gen<=AV[1]))
            if len(idx_AV_valid) > 0:
                gen_features_sample_here = pd.DataFrame(columns=feature_names)
                AV_gen_ok = AV_gen[idx_AV_valid]
                D_ok = D[idx_AV_valid]
                Lx_ok = Lx[idx_AV_valid]
                
                ss_wire = np.sort([D_ok,D_ok,Lx_ok])
                
                param_wire = np.column_stack((D_ok,D_ok,Lx_ok))                        
                param_wire_sorted = param_wire
                param_wire_sorted.sort(axis=1)
                
                gen_features_sample_here["ShortestDim"] = param_wire_sorted[:,0]
                gen_features_sample_here["MiddleDim"]   = param_wire_sorted[:,1]
                gen_features_sample_here["LongDim"]     = param_wire_sorted[:,2]          
                gen_features_sample_here["Area/Vol"]    = AV_gen_ok        
                # setting the geometry and appending to previous dataframe
                for mm in feature_names:
                    if 'Geometry' in mm:
                        if geom_here in mm:
                            gen_features_sample_here[mm] = 1
                        else:
                            gen_features_sample_here[mm] = 0                    
                for mattt in allowed_mat_list:
                    for mm in feature_names:
                        if 'Material' in mm:
                            if mattt in mm:
                                gen_features_sample_here[mm] = 1
                            else:
                                gen_features_sample_here[mm] = 0
                    gen_features_sample = gen_features_sample.append(gen_features_sample_here)
            
            
        elif "parallelepiped" in geom_here:            
            w,h,Lx = z_generate_random_while_keep_small_large_3rnges(Shrt, Mdl, Lng, n)
            w = np.round(w, round_prec_MidShrtLong_dims)
            h = np.round(h, round_prec_MidShrtLong_dims)
            Lx = np.round(Lx, round_prec_MidShrtLong_dims)
            
            # calculate AV, check if it is within the range of given AV
            Area=2*(w+h)*Lx + 2*w*h
            Volume=w*h*Lx
            
            AV_gen = Area/Volume                        
            idx_AV_valid = np.intersect1d(np.where(AV_gen>=AV[0]), np.where(AV_gen<=AV[1]))
            if len(idx_AV_valid) > 0:
                gen_features_sample_here = pd.DataFrame(columns=feature_names)
                AV_gen_ok = AV_gen[idx_AV_valid]
                w_ok = w[idx_AV_valid]
                h_ok = h[idx_AV_valid]
                Lx_ok = Lx[idx_AV_valid]
                gen_features_sample_here["ShortestDim"] = w_ok
                gen_features_sample_here["MiddleDim"]   = h_ok
                gen_features_sample_here["LongDim"]     = Lx_ok
                gen_features_sample_here["Area/Vol"]    = AV_gen_ok        
                # setting the geometry and appending to previous dataframe
                for mm in feature_names:
                    if 'Geometry' in mm:
                        if geom_here in mm:
                            gen_features_sample_here[mm] = 1
                        else:
                            gen_features_sample_here[mm] = 0                    
                for mattt in allowed_mat_list:
                    for mm in feature_names:
                        if 'Material' in mm:
                            if mattt in mm:
                                gen_features_sample_here[mm] = 1
                            else:
                                gen_features_sample_here[mm] = 0
                    gen_features_sample = gen_features_sample.append(gen_features_sample_here)
            
        elif "TriangPrismIsosc" in geom_here:            
            h,L,Lx = z_generate_random_while_keep_small_large_3rnges(Shrt, Mdl, Lng, n)            
            
            h = np.round(h, round_prec_MidShrtLong_dims)
            L = np.round(L, round_prec_MidShrtLong_dims)
            Lx = np.round(Lx, round_prec_MidShrtLong_dims)
            
            # calculate AV, check if it is within the range of given AV
            Area=2*(0.5*L*h) + Lx*L + 2*Lx*np.sqrt((L/2)**2+h**2)
            Volume=0.5*L*h*Lx 
            
            AV_gen = Area/Volume            
            idx_AV_valid = np.intersect1d(np.where(AV_gen>=AV[0]), np.where(AV_gen<=AV[1]))            
            if len(idx_AV_valid) > 0:
                gen_features_sample_here = pd.DataFrame(columns=feature_names)
                AV_gen_ok = AV_gen[idx_AV_valid]                
                h_ok = h[idx_AV_valid]
                L_ok = L[idx_AV_valid]
                Lx_ok = Lx[idx_AV_valid]
                gen_features_sample_here["ShortestDim"] = h_ok
                gen_features_sample_here["MiddleDim"]   = L_ok
                gen_features_sample_here["LongDim"]     = Lx_ok
                gen_features_sample_here["Area/Vol"]    = AV_gen_ok        
                # setting the geometry and appending to previous dataframe
                for mm in feature_names:
                    if 'Geometry' in mm:
                        if geom_here in mm:
                            gen_features_sample_here[mm] = 1
                        else:
                            gen_features_sample_here[mm] = 0                    
                for mattt in allowed_mat_list:
                    for mm in feature_names:
                        if 'Material' in mm:
                            if mattt in mm:
                                gen_features_sample_here[mm] = 1
                            else:
                                gen_features_sample_here[mm] = 0
                    gen_features_sample = gen_features_sample.append(gen_features_sample_here)
    
    gen_features_sample = gen_features_sample.astype(np.float)
    if 'log Area/Vol' in feature_names and 'Area/Vol' in gen_features_sample.columns:        
        gen_features_sample['log Area/Vol'] = np.log(gen_features_sample['Area/Vol'])        
   
    return gen_features_sample


def z_generate_random_while_keep_small_large_2rnges(lim_smaller, lim_larger, n):
    """
    generate random numbers, while making sure that always the random generated from "lim_smaller" is smaller than that is generated from "lim_larger"
    """
    rnd_smaller = np.array([])
    rnd_larger = np.array([])
    if len(lim_smaller) > 0 and len(lim_larger) > 0:
        rnd_smaller = np.random.uniform(low=lim_smaller[0],high=lim_smaller[1],size=n)        
        for ss in rnd_smaller:
            lb = np.max([lim_larger[0], ss])
            ub = lim_larger[1]
            vv = np.random.uniform(low=lb,high=ub,size=1)[0]
            rnd_larger = np.append(rnd_larger, vv)
            
    
    return rnd_smaller,rnd_larger

def z_generate_random_while_keep_small_large_3rnges(lim_smaller, lim_middle, lim_larger, n):
    """
    generate random numbers, while making sure that always the random generated from "lim_smaller" is smaller than that is generated from "lim_larger"
    """
    rnd_smaller = np.array([])
    rnd_middle = np.array([])
    rnd_larger = np.array([])
    
    if len(lim_smaller) > 0 and len(lim_middle) > 0 and len(lim_larger) > 0:
        rnd_smaller = np.random.uniform(low=lim_smaller[0],high=lim_smaller[1],size=n)        
        for ss in rnd_smaller:
            lb = np.max([lim_middle[0], ss])
            ub = lim_middle[1]
            mdl = np.random.uniform(low=lb,high=ub,size=1)[0]
            rnd_middle = np.append(rnd_middle, mdl)
            
            lb = np.max([lim_larger[0], mdl])
            ub = lim_larger[1]
            lrg = np.random.uniform(low=lb,high=ub,size=1)[0]
            rnd_larger = np.append(rnd_larger, lrg)
            
    
    return rnd_smaller,rnd_middle,rnd_larger
    
    
def plot_spectral_target_VS_spectral_predicted_using_InverseDesignFeatures(my_x, target_vector, estimator_here, features_InverseDesign_here, feature_names, use_log_emissivity, title_here):
    
    # predict the spectral emissivity
    if use_log_emissivity:
        y_pred = np.exp(estimator_here.predict(features_InverseDesign_here[feature_names]))    
    else:
        y_pred = estimator_here.predict(features_InverseDesign_here[feature_names])    
    
    
    
    plt.figure()
    #plt.plot(my_x, target_vector.T, label='target',    c='b')
    plt.plot(my_x, y_pred.T       , label='predicted', c='g', alpha=0.2)
    plt.xlabel("Angular frequency (rad/s)")
    plt.ylabel("Emissivity")
    plt.title(title_here)
    
    
    
    
    
    
    



## Random Forest Utils Wrapper around DT utils
def get_rf_num_nodes(rf):
    trees = rf.estimators_ 
    n_nodes = [tree.tree_.node_count for tree in trees]
    return n_nodes
def get_rf_num_leaves(rf):
    trees = rf.estimators_
    n_leaves = [get_dt_num_leaves(tree) for tree in trees]
    return n_leaves

#adopted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

def get_dt_attributes(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    return n_nodes,children_left,children_right,feature,threshold

#adopted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
def get_dt_num_leaves(estimator):
    n_nodes,children_left,children_right,feature,threshold = get_dt_attributes(estimator)
    num_leaves = 0
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
            num_leaves +=1 
    return num_leaves
def print_decision_tree_structure(estimator):

    n_nodes,children_left,children_right,feature,threshold = get_dt_attributes(estimator)

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))

def getpath(estimator,X_test,sample_id=0,feature_names=None):
    
    
    n_nodes,children_left,children_right,feature,threshold = get_dt_attributes(estimator)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.
    node_indicator = estimator.decision_path(X_test)
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    features = []
    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        if feature_names:
            print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
                  % (node_id,
                     sample_id,
                     feature_names[feature[node_id]],
                     X_test[sample_id, feature[node_id]],
                     threshold_sign,
                     threshold[node_id]))
        else:
            print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
                  % (node_id,
                     sample_id,
                     feature[node_id],
                     X_test[sample_id, feature[node_id]],
                     threshold_sign,
                     threshold[node_id]))
        features.append(feature[node_id])
    return features
def get_common_nodes(estimator,X_test,sample_ids=[0]):
    n_nodes,children_left,children_right,feature,threshold = get_dt_attributes(estimator)
    node_indicator = estimator.decision_path(X_test)
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                    len(sample_ids))

    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree"
          % (sample_ids, common_node_id))
    print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
    