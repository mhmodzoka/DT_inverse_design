# %% loading ===================================================================
# misc
import json
from IPython.display import Image
from sklearn.externals.six import StringIO
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from forward_utils import load_spectrum_param_data_mat, calc_RMSE_MAE_MSE_Erel, z_RF_DT_DTGEN_error_folds, spectra_prediction_corrector, feature_importance_by_material, get_iloc_for_test_on_edges_of_column, error_integ_by_spectrum_integ, plt2matlab,get_all_children, get_confidence_interval_for_random_forest, gen_data_P1_P2_P3_Elzouka
import warnings
import joblib
import sys
import os
from time import strftime, time
from math import pi

# data processing
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat

# ML tools, sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


# to ignore warnings displayed in notebook
warnings.filterwarnings('ignore')



# %% inputs ====================================================================
n_estimators = 200
test_size = 0.2
n_cpus = 10
num_folds_training_for_errors = 2  # 100
n_gen_to_data_ratio = 1  # 180 # the ratio between n_gen to the data used for ML

# the fraction of original data to be used for ML.
train_datasize_fraction_scalar = 0.5
# the fraction of original data to be used for ML.
train_datasize_fraction_spectral = 0.5
n_data_desired = {'Geometry_sphere': 500, 'Geometry_wire': 800,
                  'Geometry_parallelepiped': 2000, 'Geometry_TriangPrismIsosc': 2000}

# True: use log emissivity as input to ML, this will make the training target to monimize relative error (i.e., MINIMIZE( log(y_pred) - log(y_test) ) is equivalent to MINIMIZE( log(y_pred / y_test) ))
use_log_emissivity = True

Study_performance_VS_datasize = False
data_size_fraction_ = np.array(
    [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
data_size_fraction_ = np.array([0.5])
#data_size_fraction_ = np.array([0.01,0.02,0.05])

Models_to_Study_performanceVSsize = ['DTGEN', 'DT']
Models_to_Study_performanceVSsize = ['RF']

num_folds_repeat_DataReduction = 1

# list that contains either 'scalar' or 'spectral' or both
spectral_or_scalar_calc_all = ['scalar', 'spectral']
spectral_or_scalar_calc_all = ['scalar']

datetime_str = strftime("%Y%m%d_%H%M%S")

matlab_data_path = ['../data/All_data_part_1.mat', '../data/All_data_part_2.mat']

optional_title_folders = '{0}sc_{1}sp_{2}_CPU'.format(
    train_datasize_fraction_scalar*100, train_datasize_fraction_spectral*100, n_cpus)
# optional_title_folders='DataSize_study_BigTest'

#feature_set = ['Geometry_TriangPrismIsosc', 'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire',
#               'Material_Au', 'Material_SiN', 'Material_SiO2',
#               'log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim']

feature_set = ['Geometry_TriangPrismIsosc', 'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire',
              'Material_Au', 'Material_SiN', 'Material_SiO2',
              'log Area/Vol', 'log ShortestDim', 'log MiddleDim', 'log LongDim']

# DON'T CHANGE THESE VARIABLES
my_x = np.logspace(13, 14.9, num=400)
scaling_factors = {'Area': 10.0**12, 'Volume': 10.0**18, 'Volume/Area': 10.0**9, 'Area/Volume': 10.0**-6, 'LongestDim': 10.0**6, 'MiddleDim': 10.0 **
                   6, 'ShortestDim': 10.0**6, 'PeakFrequency': 10.0**-14, 'PeakEmissivity': 10.0, 'LengthX': 10**6, 'Height': 10**6, 'Length': 10**6}


# %% Loading data ==============================================================
data_featurized_1, interpolated_ys, spectrum_parameters = load_spectrum_param_data_mat(matlab_data_path, my_x, scaling_factors)  # this data has more "sphere" results

# %% Data pre-processing =======================================================

# separating material and geometry features, for stratifying the data
feature_set_geom = [x for x in feature_set if 'Geometry' in x]
feature_set_mat = [x for x in feature_set if 'Material' in x]
feature_set_geom_mat = feature_set_geom + feature_set_mat
feature_set_dimensions = list(set(feature_set) - set(feature_set_geom_mat))

print("dataset size, original: {0}".format(len(data_featurized)))

# drop data with A/V outside the range
sclae_here = scaling_factors['Area/Volume']
idx_low = data_featurized['Area/Vol'].values < 0.9e6*sclae_here
idx_high = data_featurized['Area/Vol'].values > 1e8*sclae_here
idx_outrange = ~np.logical_or(idx_low, idx_high)
data_featurized = data_featurized[idx_outrange]
interpolated_ys = interpolated_ys[idx_outrange]
print(
    "after dropping A/V outside range, we have {0} datapoints".format(len(data_featurized)))

# drop analytical (except spheres) samples from training
idx_num = ~data_featurized['is_analytical'].astype(bool)
idx_anal = ~idx_num
print("we have {0} numerical simulations, and {1} analytical".format(
    sum(idx_num), sum(idx_anal)))

idx_sphere = data_featurized['Geometry_sphere'].astype(bool)
print("we have {0} spheres".format(sum(idx_sphere)))

# we will keep all numerical simulations or sphere simulations
idx = np.logical_or(idx_num.values, idx_sphere.values)
print("we have {0} spheres | numerical simulation".format(sum(idx)))
data_featurized = data_featurized[idx]
interpolated_ys = interpolated_ys[idx]


# to reduce the number of datapoints for each geometry
#idx_to_keep = []
idx_to_keep = np.array([])
for geom in feature_set_geom:
    n_data_desired_here = n_data_desired[geom]
    idx_geom_here = data_featurized[geom]
    for mat in feature_set_mat:
        idx_mat_here = data_featurized[mat]
        #idx_geom_mat_here = np.logical_and(idx_geom_here.values, idx_mat_here.values)
        idx_geom_mat_here = np.where((data_featurized[geom].values == 1) & (
            data_featurized[mat].values == 1))[0]
        if n_data_desired_here >= len(idx_geom_mat_here):
            # idx_to_keep.append(idx_geom_mat_here)
            idx_to_keep = np.append(idx_to_keep, idx_geom_mat_here)
        else:
            #idx_to_keep.append(np.random.choice(idx_geom_mat_here, size=n_data_desired_here, replace=False) )
            idx_to_keep = np.append(idx_to_keep, np.random.choice(
                idx_geom_mat_here, size=n_data_desired_here, replace=False))


data_featurized = data_featurized.iloc[idx_to_keep]
interpolated_ys = interpolated_ys[idx_to_keep.astype(int)]

assert len(interpolated_ys) == len(data_featurized)

data_featurized = data_featurized.drop('is_analytical', axis=1)

print("dataset size, after dropping: {0}".format(len(data_featurized)))


# %% 08/03/2020 ================================================================
# effect of number of estimators on RF performance
'''
n_estimators_list = [1,2,4,6,8,10,12,20]
n_estimators_list = [1,2]
num_fold_this_study = 4
RF_or_DT__to_be_studied = ['linear']

for spectral_or_scalar_calc in spectral_or_scalar_calc_all:
    save_folder = '../cache/r{0}_'.format(datetime_str) + 'study_n_estimators' + '/' + spectral_or_scalar_calc + '/'
    Erel_this_study={}
    nnnn = 0
    for n_estimators_here in n_estimators_list:
        
        # preparing features and labels
        X = data_featurized[feature_set].copy()    
        if spectral_or_scalar_calc == 'spectral':
            y = interpolated_ys
            train_data_size_fraction = train_datasize_fraction_spectral
        elif spectral_or_scalar_calc == 'scalar':
            y = data_featurized['Emissivity'].values
            train_data_size_fraction = train_datasize_fraction_scalar
            
        
            
        # here, we will reduce the "training data", not the "entire data"                
        test_size___ = 0.2
        save_folder_here = save_folder+'nestimator_'+str(n_estimators_here)
        
        All_errors = z_RF_DT_DTGEN_error_folds(X[feature_set], y, feature_set, feature_set_dimensions, feature_set_geom_mat, data_featurized, my_x, \
                num_folds=num_fold_this_study, test_size=test_size___, n_estimators=n_estimators_here, \
                n_cpus = n_cpus, keep_spheres = True, optional_title_folders=save_folder_here, \
                use_log_emissivity=use_log_emissivity, display_plots=False, display_txt_out = True, \
                RF_or_DT__ = RF_or_DT__to_be_studied, PlotTitle_extra = spectral_or_scalar_calc, \
                n_gen_to_data_ratio = n_gen_to_data_ratio)

        
        
        for key_here in All_errors.keys():
            if nnnn == 0:                
                Erel_this_study[key_here] = []

            try:
                Erel_this_study[key_here].append(np.average(All_errors[key_here]['Erel_all']))
            except:
                pass

        nnnn +=1

    for key_here in Erel_this_study.keys():
        try:
            plt.plot(n_estimators_list, Erel_this_study[key_here], label=spectral_or_scalar_calc+'_'+key_here)
        except:
            pass

    plt.legend()
'''


# %% Quantify random forest model uncertainty (i.e. the confidence interval)
spectral_or_scalar_calc = 'scalar'
test_size = 0.05
# preparing features and labels
X = data_featurized[feature_set].copy()
if spectral_or_scalar_calc == 'spectral':
    y = interpolated_ys
    train_data_size_fraction = train_datasize_fraction_spectral
elif spectral_or_scalar_calc == 'scalar':
    y = data_featurized['Emissivity'].values
    train_data_size_fraction = train_datasize_fraction_scalar

# splitting test and training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=X[feature_set_geom_mat])
if use_log_emissivity:
    y_train = np.log(y_train)
    # if we have 'ZEROS', its log will be -Inf. We are replacing anything zlose to zero with exp(-25)
    y_train[y_train < -25] = -25

# training RF
rf = RandomForestRegressor(n_estimators, n_jobs=n_cpus)
rf.fit(X_train[feature_set], y_train)

# testing RF
if use_log_emissivity:
    y_pred_rf = np.exp(rf.predict(X_test))
else:
    y_pred_rf = spectra_prediction_corrector(rf.predict(X_test))

#  calculate confidence interval
y_pred_standard_deviation = get_confidence_interval_for_random_forest(rf, X_test, y_test, use_log_emissivity=use_log_emissivity)


plt.errorbar(y_test, y_pred_rf, yerr=y_pred_standard_deviation, fmt='.')
plt.xlabel('test emissivity')
plt.ylabel('predicted emissivity')

print(np.average(y_pred_standard_deviation))


# %% effect of number of estimators on RF performance ---------- 2
#n_estimators_list = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 100, 200]
n_estimators_list = [1, 2, 4, 8, 10, 20, 50, 100]

#num_fold_this_study = 50
num_fold_this_study = 50

train_size_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
train_size_list = [0.1, 0.5, 0.9]

RF_or_DT__to_be_studied = ['RF']

all_figures = {}

spectral_or_scalar_for_n_estimator_study = ['scalar', 'spectral']

for spectral_or_scalar_calc in spectral_or_scalar_for_n_estimator_study:
    all_figures[spectral_or_scalar_calc] = plt.figure()
    save_folder = '../cache/r{0}_'.format(datetime_str) + \
        'study_n_estimators' + '/' + spectral_or_scalar_calc + '/'
    fig_here = plt.figure()
    for train_size_here in train_size_list:
        Erel_this_study = {}
        nnnn = 0
        test_size_here = 1 - train_size_here

        for n_estimators_here in n_estimators_list:

            # preparing features and labels
            X = data_featurized[feature_set].copy()
            if spectral_or_scalar_calc == 'spectral':
                y = interpolated_ys

            elif spectral_or_scalar_calc == 'scalar':
                y = data_featurized['Emissivity'].values

            # here, we will reduce the "training data", not the "entire data"
            save_folder_here = save_folder+'nestimator_'+str(n_estimators_here)

            All_errors = z_RF_DT_DTGEN_error_folds(X[feature_set], y, feature_set, feature_set_dimensions, feature_set_geom_mat, data_featurized, my_x,
                                                   num_folds=num_fold_this_study, test_size=test_size_here, n_estimators=n_estimators_here,
                                                   n_cpus=n_cpus, keep_spheres=True, optional_title_folders=save_folder_here,
                                                   use_log_emissivity=use_log_emissivity, display_plots=False, display_txt_out=False,
                                                   RF_or_DT__=RF_or_DT__to_be_studied, PlotTitle_extra=spectral_or_scalar_calc,
                                                   n_gen_to_data_ratio=n_gen_to_data_ratio)

            for key_here in All_errors.keys():
                if nnnn == 0:
                    Erel_this_study[key_here] = []

                try:
                    Erel_this_study[key_here].append(
                        np.average(All_errors[key_here]['Erel_all']))
                except:
                    pass

            nnnn += 1

        for key_here in Erel_this_study.keys():
            try:
                plt.plot(n_estimators_list, Erel_this_study[key_here], label=spectral_or_scalar_calc +
                         '_'+key_here+' - train size = '+str(train_size_here*100)+'%')

            except:
                pass

        plt.xlabel('number of estimators')
        plt.ylabel('relative error')
        plt.legend()

    # saving figure data to be plotted in Matlab
    plt2matlab(fig_here, save_path_figure_data=save_folder +
               'fig_data_multiple_trainsize.mat')


# %% exploring performance of linear model =============================================
'''
feature_set = ['Geometry_TriangPrismIsosc', 'Geometry_parallelepiped', 'Geometry_sphere', 'Geometry_wire',
               'Material_Au', 'Material_SiN', 'Material_SiO2',
               'log Area/Vol', 'ShortestDim', 'MiddleDim', 'LongDim']
'''

num_fold_this_study = 20
#num_fold_this_study = 2

train_size_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#train_size_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
#train_size_list = [0.4, 0.6, 0.8, 0.95]

train_size_list = np.array(train_size_list)

RF_or_DT__to_be_studied_ = ['linear', 'RF']
#RF_or_DT__to_be_studied_ = ['RF']

spectral_or_scalar_calc_all___ = ['spectral', 'scalar']
#spectral_or_scalar_calc_all___ = ['scalar']

for RF_or_DT__to_be_studied in RF_or_DT__to_be_studied_:
    for spectral_or_scalar_calc in spectral_or_scalar_calc_all___:
        fig_here_train_perc = plt.figure()
        fig_here_train_points = plt.figure()
        for geom_here in feature_set_geom:
            for mat_here in feature_set_mat:

                # setting the feature set
                if RF_or_DT__to_be_studied == 'linear':
                    if mat_here.endswith('Au'):
                        feature_set_here = ['LongDim']
                    elif mat_here.endswith('SiO2'):
                        feature_set_here = ['log Area/Vol']
                    elif mat_here.endswith('SiN'):
                        feature_set_here = ['MiddleDim']
                else:
                    feature_set_here = feature_set_dimensions
                    
                        

                idx_geom_mat_here = np.where((data_featurized[geom_here].values == 1) & (
                    data_featurized[mat_here].values == 1))[0]
                geom_mat_label = geom_here + '__' + mat_here

                save_folder = '../cache/r{0}_'.format(
                    datetime_str) + 'study_linear' + '/' + spectral_or_scalar_calc + '/'
                Erel_this_study = {}
                All_errors = {}
                nnnn = 0

                for train_size_here in train_size_list:
                    test_size_here = 1 - train_size_here

                    # preparing features and labels
                    X = data_featurized[feature_set].copy()
                    if spectral_or_scalar_calc == 'spectral':
                        y = interpolated_ys
                    elif spectral_or_scalar_calc == 'scalar':
                        y = data_featurized['Emissivity'].values

                    #X, y = X.iloc[idx_geom_mat_here, :], y[idx_geom_mat_here]
                    X, y = X.iloc[idx_geom_mat_here], y[idx_geom_mat_here]

                    # here, we will reduce the "training data", not the "entire data"
                    save_folder_here = save_folder + \
                        'trainsize_'+str(train_size_here)

                    All_errors = z_RF_DT_DTGEN_error_folds(X[feature_set], y, feature_set_here, feature_set_dimensions, feature_set_geom_mat, data_featurized, my_x,
                                                           num_folds=num_fold_this_study, test_size=test_size_here,
                                                           n_cpus=n_cpus, keep_spheres=True, optional_title_folders=save_folder_here,
                                                           use_log_emissivity=use_log_emissivity, display_plots=False, display_txt_out=False,
                                                           RF_or_DT__=[
                                                               RF_or_DT__to_be_studied], PlotTitle_extra=spectral_or_scalar_calc,
                                                           n_gen_to_data_ratio=n_gen_to_data_ratio)

                    for key_here in All_errors.keys():
                        if nnnn == 0:
                            Erel_this_study[key_here] = []

                        try:
                            Erel_this_study[key_here].append(
                                np.average(All_errors[key_here]['Erel_all']))
                        except:
                            pass

                    nnnn += 1

                for key_here in Erel_this_study.keys():
                    try:
                        plt.figure(fig_here_train_perc.number)
                        plt.plot(np.array(train_size_list)*100, np.array(
                            Erel_this_study[key_here])*100, label=spectral_or_scalar_calc+'_'+key_here+'_'+geom_mat_label, marker='.')

                        plt.figure(fig_here_train_points.number)
                        plt.plot(np.array(train_size_list)*len(X), np.array(
                            Erel_this_study[key_here])*100, label=spectral_or_scalar_calc+'_'+key_here+'_'+geom_mat_label, marker='.')
                        print(geom_here + '--' + mat_here)
                        print(Erel_this_study)
                    except:
                        pass
        plt.figure(fig_here_train_perc.number)
        plt.xlabel('training size [%]')
        plt.ylabel('relative error [%]')
        plt.legend()
        plt.title(spectral_or_scalar_calc + RF_or_DT__to_be_studied)

        plt.figure(fig_here_train_points.number)
        plt.xlabel('training size [# of training points]')
        plt.ylabel('relative error [%]')
        plt.legend()
        plt.title(spectral_or_scalar_calc + RF_or_DT__to_be_studied)

        # saving figure data to be plotted in Matlab
        plt2matlab(fig_here_train_perc, save_path_figure_data=save_folder +
                   'fig_data_'+RF_or_DT__to_be_studied+'_'+spectral_or_scalar_calc+'_xaxis_percent_training.mat')

        plt2matlab(fig_here_train_points, save_path_figure_data=save_folder +
                   'fig_data_'+RF_or_DT__to_be_studied+'_'+spectral_or_scalar_calc+'_xaxis_npoint_training.mat')


# %% testing the "get_iloc_for_test_on_edges_of_column"

test_frac = 0.7
column_ToTake_test_near_ItsEdges =  'vol' # 'Area/Vol'
X = data_featurized.copy()

idx_test_all = get_iloc_for_test_on_edges_of_column(X, column_ToTake_test_near_ItsEdges = column_ToTake_test_near_ItsEdges, test_frac=test_frac, plot_intermediate=False)

plt.figure()
plt.plot(np.log(data_featurized.vol), np.log(data_featurized['Area/Vol']), Marker='.', linestyle='None')
plt.plot(np.log(data_featurized.vol.iloc[idx_test_all]), np.log(data_featurized['Area/Vol'].iloc[idx_test_all]), Marker='.', linestyle='None')
plt.xlabel('log(vol)'); plt.ylabel('log(Area/Vol)')
plt.title('test data is {0}% the entire data'.format(test_frac*100))


# %% Explore ability of the model to extrapolate
# =================================================
# =================================================
'''
I will take the test data to be on the edges, and train the model with the rest.
'''


save_folder = '../cache/r{0}_'.format(datetime_str) + 'test_extrapolation'
os.makedirs(save_folder, exist_ok=True)

column_ToTake_test_near_ItsEdges_ = ['', 'vol', 'Area/Vol']
#column_ToTake_test_near_ItsEdges_ = []

num_fold_this_study = 20
#train_size_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
train_size_list = [0.1, 0.3, 0.5, 0.7, 0.9]
train_size_list = np.array(train_size_list)


#spectral_or_scalar_calc_all___ = ['spectral', 'scalar']
spectral_or_scalar_calc_all___ = ['scalar']

for column_ToTake_test_near_ItsEdges in column_ToTake_test_near_ItsEdges_:
    for spectral_or_scalar_calc in spectral_or_scalar_calc_all___:
        All_errors_trainsize = {}
        nn = 0
        for train_size_here in train_size_list:
            test_size_here = 1 - train_size_here

            # preparing features and labels
            X = data_featurized.copy()
            #X = data_featurized[feature_set].copy()
            if spectral_or_scalar_calc == 'spectral':
                y = interpolated_ys
            elif spectral_or_scalar_calc == 'scalar':
                y = data_featurized['Emissivity'].values

            All_errors = z_RF_DT_DTGEN_error_folds(X, y, feature_set, feature_set_dimensions, feature_set_geom_mat, data_featurized, my_x,
                                                   num_folds=num_fold_this_study, test_size=test_size_here, n_estimators=10, n_cpus=1, keep_spheres=True, optional_title_folders=save_folder,
                                                   use_log_emissivity=True, display_plots=False, display_txt_out=True, RF_or_DT__=['RF'], PlotTitle_extra='',
                                                   column_ToTake_test_near_ItsEdges=column_ToTake_test_near_ItsEdges)

            # store all errors
            dict_here = All_errors['RF']['Erel_matgeom']
            for key_here in dict_here.keys():
                if not (key_here in All_errors_trainsize.keys()):
                    All_errors_trainsize[key_here] = []

                All_errors_trainsize[key_here].append(
                    [train_size_here, np.average(dict_here[key_here])])
        # plot all errors
        fig_here = plt.figure()
        for key_here in All_errors_trainsize.keys():
            arr = np.array(All_errors_trainsize[key_here])
            plt.semilogy(arr[:, 0], arr[:, 1], label=key_here)
        plt.xlabel('train size [%]')
        plt.ylabel('relative error')
        plt.legend()

        # try saving the figure to be exported to Matlab
        try:
            file_name = save_folder+'/'+spectral_or_scalar_calc+'_ColSort_' + \
                column_ToTake_test_near_ItsEdges.replace(
                    '/', '_')+'_over_'+str(num_fold_this_study)+'_runs.mat'
            plt2matlab(fig_here, save_path_figure_data=file_name)

        except:
            pass






# %% sorted split, splitting test data by error
n_estimators_here = 200
spectral_or_scalar_calc = 'scalar'
test_size = 0.2
X = data_featurized[feature_set].copy()

error_rf_init_testdata = []
error_rf_init_alldata = []
error_rf_sortedsplit_testdata = []
for ii in range(20):
    print()
    # step #1: training RF -----------------------------------------------------------
    # -- preparing features and labels
    if spectral_or_scalar_calc == 'spectral':
        y = interpolated_ys
        train_data_size_fraction = train_datasize_fraction_spectral
    elif spectral_or_scalar_calc == 'scalar':
        y = data_featurized['Emissivity'].values
        train_data_size_fraction = train_datasize_fraction_scalar

    # -- splitting test and training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=X[feature_set_geom_mat])
    if use_log_emissivity:
        y_train = np.log(y_train)
        # if we have 'ZEROS', its log will be -Inf. We are replacing anything zlose to zero with exp(-25)
        y_train[y_train < -25] = -25

    # -- training the model
    rf = RandomForestRegressor(n_estimators_here, n_jobs=n_cpus)
    rf.fit(X_train[feature_set], y_train)

    # -- calculating the error using the test dataset
    if use_log_emissivity:
        y_pred_rf = np.exp(rf.predict(X_test[feature_set]))
    else:
        y_pred_rf = spectra_prediction_corrector(rf.predict(X_test[feature_set]))
    error_rel,error_rel_mean = error_integ_by_spectrum_integ(y_test, y_pred_rf)
    print('RF for calculating error in predicintg the entire dataset')
    print('The mean of the relative error, calculated using the test data = {0}'.format(error_rel_mean))
    error_rf_init_testdata.append(error_rel_mean)

    # step #2: calculate prediction error for the entire data set --------------------
    if use_log_emissivity:
        y_pred_rf = np.exp(rf.predict(X[feature_set]))
    else:
        y_pred_rf = spectra_prediction_corrector(rf.predict(X[feature_set]))
    error_rel,error_rel_mean = error_integ_by_spectrum_integ(y, y_pred_rf)
    print('RF for calculating error in predicintg the entire dataset')
    print('The mean of the relative error, calculated using the entire data = {0}'.format(error_rel_mean))
    error_rf_init_alldata.append(error_rel_mean)

    # adding the relative error to X
    X['error_relative'] = error_rel

    # step #3: use this error to sort the data, and show model performance. ----------
    # -- splitting test and training
    idx_test_all = get_iloc_for_test_on_edges_of_column(
                    X, column_ToTake_test_near_ItsEdges = 'error_relative', test_frac=test_size, plot_intermediate=False)
    idx_train_all = np.setdiff1d(range(len(X)), idx_test_all)
    X_train,X_test,y_train,y_test = X.iloc[idx_train_all], X.iloc[idx_test_all], y[idx_train_all], y[idx_test_all]
    if use_log_emissivity:
        y_train = np.log(y_train)
        # if we have 'ZEROS', its log will be -Inf. We are replacing anything zlose to zero with exp(-25)
        y_train[y_train < -25] = -25

    # -- training the model
    rf_sortedsplit = RandomForestRegressor(n_estimators_here, n_jobs=n_cpus)
    rf_sortedsplit.fit(X_train[feature_set], y_train)  

    # -- calculate the error
    if use_log_emissivity:
        y_pred_rf = np.exp(rf_sortedsplit.predict(X_test[feature_set]))
    else:
        y_pred_rf = spectra_prediction_corrector(rf_sortedsplit.predict(X_test[feature_set]))
    error_rel,error_rel_mean = error_integ_by_spectrum_integ(y_test, y_pred_rf)
    print('RF trained on sorted split data')
    print('the mean of the relative error, calculated using the sorted split = {0}'.format(error_rel_mean))
    error_rf_sortedsplit_testdata.append(error_rel_mean)





# %%
fig_here = plt.figure()
plt.plot(error_rf_sortedsplit_testdata, np.ones(np.shape(error_rf_sortedsplit_testdata)), linestyle='None', marker='.')
plt.plot(error_rf_init_testdata, np.ones(np.shape(error_rf_init_testdata)), linestyle='None', marker='.')
# saving figure data to be plotted in Matlab
plt2matlab(fig_here, save_path_figure_data='../cache/sorted_split_figure.mat')
