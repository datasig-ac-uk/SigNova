import os
import time
import warnings
import pickle
import iisignature
import itertools
import plot_lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.preprocessing
import sklearn.utils
import sklearn.model_selection
from distfit import distfit
from tqdm import tqdm
from joblib import Memory, Parallel, delayed
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pysegments import Interval, segment
from multiprocessing import Pool, cpu_count
from pynndescent import NNDescent
from omegaconf import DictConfig, ListConfig, OmegaConf
from matplotlib import cm
from matplotlib import colors

#================================================================================================================================
# Basic helper functions
#================================================================================================================================

def convert_np(df):
    array = np.vstack(df['Sig'].to_numpy())
    return array

def fill_binary_ans(arr, ans, i):
    if len(ans) == 0:
        arr[:, i] = 1
    else:
        for j in range(0,len(ans)):
            inf = int(ans[j][0])
            sup = int(ans[j][1])
            arr[inf:sup, i] = 0
    return arr 

def get_threshold_score(scores, distribution, threshold):
    # Initialize for common/popular distributions with bootstrapping.
    dfit = distfit(distr=distribution, n_boots=10, alpha=threshold, verbose='silent')
    # Estimate the best fit
    results = dfit.fit_transform(scores)
    return results['model']['CII_max_alpha']


#=================================================================================================================================
# Functions to vectorize the data using signatures
#=================================================================================================================================

def _path_transforms(stream, add_time=False, lead_lag=False, base_point=False):
    if add_time:
        stream = np.column_stack((stream, np.arange(stream.shape[0])))
    if lead_lag:
        stream = np.repeat(stream, 2, axis=0)
        stream = np.column_stack((stream[1:, :], stream[:-1, :]))
    if base_point:
        stream = np.vstack(((stream, stream[-1], np.zeros_like(stream[-1]))))
        stream = np.column_stack((stream, np.append(np.ones(stream.shape[0]-2), [0, 0])))
    return stream

def _vectorize(df, show_progress=False, trunc=2, expected_signature=True, chunk_size=1000):
   
    sigs = []
        
    for chunk in tqdm([df[i:i+chunk_size] for i in range(0,len(df), chunk_size)],\
                      disable=not show_progress, desc='Computing signatures'):

        largest_shape = chunk['Stream'].apply(lambda x: np.shape(x)).max()
        chunk['Stream'] = chunk['Stream'].apply(lambda x: np.concatenate((x, np.repeat(x[-1][np.newaxis, :], (largest_shape[0] - len(x)), axis=0))) if len(x) > 0 else np.zeros(largest_shape))
        check = np.stack(chunk['Stream'])

        sigs.append(iisignature.sig(np.stack(chunk['Stream']), trunc)) 

    sigs = np.concatenate(sigs)
    df['Sig'] = [sig for sig in sigs]

    if expected_signature:
        df_vec = df.groupby(['Ant1','FrCh'])['Sig'].apply(np.mean)

        N, ants = len(df.Ant1.unique()), df.Ant1.unique()

        #computing expected signatures for each antenna
        df_vec = pd.concat([\
                            df.query('Ant1 == {0}| Ant2 == {0}'.format(ants[i])).groupby(['FrCh'])\
                            ['Sig'].apply(np.mean).reset_index(False) for i in range(N)\
                           ])  
        ##For HERA Data (No Ant2)
#        df_vec = pd.concat([\
#                            df.query('Ant1 == {0}'.format(ants[i])).groupby(['FrCh'])\
#                            ['Sig'].apply(np.mean).reset_index(False) for i in range(N)\
#                           ]) ###NEW   
  
    else:
        return df
    
    return df_vec
     

def vectorize(df, show_progress=False, trunc=2, expected_signature=True, chunk_size=1000):
    
    df_vec = _vectorize(df, show_progress, trunc, expected_signature, chunk_size) 
    
    return df_vec

#================================================================================================================================
# Functions to flag
#================================================================================================================================
def _flag(args):
    ''' Flagger that operates at the level of a frequency channel. '''
    # this arg is not very pretty...
    
    df_to_process = args[0]
    A = args[1]
    threshold = args[2]
    inliers_scores = args[3]
    corpus = args[4]
    param_seg = args[5]
    param_vec = args[6]
    
    integration_times = df_to_process['Stream'].iloc[0].shape[0]
    base = Interval(0, integration_times)

    AD_on_data = lambda interval: AD(interval, df_to_process, inliers_scores, corpus, \
                                     A=A, threshold = threshold, \
                                     distfit = param_seg.distfit, param_vec=param_vec)

    # run pysegments
    with warnings.catch_warnings(record=True):
        ans = segment(base, AD_on_data, param_seg.tol,  param_seg.sig_tol)  # these should be hyperparameters to choose.

    if len(ans)==0:
        return []
    else:
        return [(interval.inf, interval.sup) for interval in ans]

def AD(interval, df_to_process, inliers_scores, corpus, A=None, threshold=0., distfit='none', param_vec=None):
    
        df_copy = df_to_process.copy(deep=True)

        # subset the data in time based on interval.inf and interval.sup
        low, high = int(interval.inf), int(interval.sup)
        length = high - low
        df_copy['Stream'] = df_copy['Stream'].apply(lambda x: x[low:high])


        # compute expected signatures
        vecs = convert_np(vectorize(df_copy, **param_vec))

        # compute score
        scores = variance_score(vecs, A, corpus)
        score = np.mean(scores)
#        score = np.median(scores)
        
        if distfit=='none':
            # threshold the score
            gr = len(inliers_scores[inliers_scores>score])
            per = gr/len(inliers_scores)
            no_anomaly =  per > threshold
        else:
            no_anomaly = score < threshold

        return no_anomaly

#=================================================================================================================================
# Functions to compute nearest neighbors in the Mahalanobis distance
#================================================================================================================================
def build_covariance_matrix(X):
    mu = np.mean(X, axis=0)
    X = X - mu
    U, s, V = np.linalg.svd(X, full_matrices = 0)
    C = np.dot(np.dot(V.T,np.diag(s**2)),V)
    #normalize 
    C = C / (X.shape[0])
#    return C
        # invert
    A_inv = np.linalg.pinv(C, hermitian=True)
    return A_inv

def variance_score(X, L_inv, X_corpus, approx=False):
    
    if not approx:
        N,M,D = X.shape[0], X_corpus.shape[0], X.shape[1]
        mat_diff = X.T[:,:,None]-X_corpus.T[:,None,:]   # (N, M, D)
        V = np.matmul(L_inv, mat_diff.reshape(D,N*M)).reshape(D,N,M)  # computes L^{-1}(x_i-x_j) for all i,j
        norm_square_V = (V**2).sum(axis=0) # (N,M)  norm_square_V[i,j]=mahalanobis(x_i,x_j)
        return norm_square_V.min(axis=1)
    else:
        return _variance_score_approx(X, L_inv, X_corpus)

def _variance_score_approx(X, L_inv, X_corpus):
    N,M,D = X.shape[0], X_corpus.shape[0], X.shape[1]
    
    # Transform the data
    X_corpus_transform = np.matmul(L_inv, X_corpus.transpose()).transpose() #(M,D)
    X_transform = np.matmul(L_inv, X.transpose()).transpose() #(N,D)
    
    nnd = NNDescent(X_corpus_transform)
    nn_ix, nn_d = nnd.query(X_transform)
    
    return nn_d

#=================================================================================================================================
# Main class for outlier detection
#=================================================================================================================================
    
class SigNova(object):
    
    def __init__(self, config_path='config.yaml'):
        self.config = OmegaConf.load(config_path)
        
        self.dataset = self.config.dataset
        self.path_transforms = self.config.path_transforms
        self.vectorization = self.config.vectorization
        self.pysegments = self.config.pysegments
        self.nn = self.config.nearest_neighbor
        
        # paths data
        self.corpus_paths_flag, self.inliers_paths_flag, self.test_paths_flag = False, False, False

    def load_calibrated_flagger(self, inliers_scores, A_inv, corpus_vec):
        self.inliers_scores = inliers_scores
        self.A_inv = A_inv
        self.corpus_vec = corpus_vec
        
        self.FrCh = self.corpus_vec.keys()
        
        if self.nn.per_freq:
            self.threshold = {FrCh: self.pysegments['threshold'] for FrCh in self.FrCh} 
        else:
            self.threshold = self.pysegments['threshold']  

    def update_hyperparam(self, grid, grid_name='path_transforms'):
        '''update the hyperparameters {'add_time':False, 'lead_lag':False, 'base_point':False, 'trunc':2}'''
        
        assert len(list(set(self.config[grid_name].keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in self.config[grid_name].keys()])
        
        self.config[grid_name].update(grid)
        
        # might be done in a more succint way...
        if grid_name == 'dataset':
            self.dataset.update(grid)
        elif grid_name == 'path_transforms':
            self.path_transforms.update(grid)
        elif grid_name == 'vectorization':
            self.vectorization.update(grid)
        elif grid_name == 'pysegments':
             self.pysegments.update(grid)
        elif grid_name == 'nearest_neighbor':
             self.nn.update(grid)
        else:
            raise NameError('unknown grid of parameters')

    def create_corpus(self, data_path=None):
        '''Currently just loads the paths to use as a reference corpus of normal/clean data'''
        if data_path is None:
            # load the default corpus
            dg = pickle.load(open(self.dataset.base_data_dir+self.dataset.corpus_pkl, 'rb'))
        else:
            dg = pickle.load(open(data_path), 'rb')
        self.corpus_paths = pd.DataFrame(dg)
        self.corpus_paths_flag = True
    
    def create_inliers(self, data_path=None):
        '''Currently just loads the paths to use as a inliers of normal/clean data to calibrate the flagger'''
        
        if data_path is None:
            # load the default inliers
            dg = pickle.load(open(self.dataset.base_data_dir+self.dataset.inliers_pkl, 'rb'))
        else:
            dg = pickle.load(open(data_path, 'rb'))
        self.inliers_paths = pd.DataFrame(dg)
        self.inliers_paths_flag = True
        
    def create_test(self, data_path=None):
        '''Loads the paths to process'''
        
        # load the default paths to process
        if data_path is None:
            dg = pickle.load(open(self.dataset.base_data_dir+self.dataset.test_pkl, 'rb'))
        else:
            dg = pickle.load(open(data_path, 'rb'))
        self.test_paths = pd.DataFrame(dg)
        self.test_paths_flag = True
                
    def fit(self, show_progress=False):
        '''Construct the relevant objects to compute the minimum Mahalanobis distance to a reference corpus'''
        
        # TODO: per_freq and mean could be set as hyperparameters.
        assert self.corpus_paths_flag, "please input a corpus"
        
        # apply path transforms
        self.corpus_paths['Stream'] = self.corpus_paths['Stream'].apply(self.path_transform) 
        
        # vectorize the streamed data.
        self.corpus_vec = vectorize(self.corpus_paths, show_progress, **self.vectorization)
        
        # Define the Mahalanobis distance: compute the inverse of the covariance matrix 
        if self.nn.per_freq:
            # One Mahalanobis distance per Frequency channel
            self.FrCh = self.corpus_vec.FrCh.unique()
           
            self.corpus_vec = {FrCh: convert_np(corpus_per_freq)\
                               for FrCh, corpus_per_freq in self.corpus_vec.groupby('FrCh')}
            self.A_inv = {FrCh: build_covariance_matrix(self.corpus_vec[FrCh]) for FrCh in self.FrCh}
        else:
            # A single Mahalanobis distance
            self.corpus_vec = convert_np(self.corpus_vec)
            self.A_inv = build_covariance_matrix(self.corpus_vec)
        
    def calibrate(self, show_progress=False):
        '''Compute the scores (minimum Mahalanobis distance) of inliers to use to calibrate the flagger'''
        
        assert self.corpus_paths_flag, "please input a corpus"
        assert self.inliers_paths_flag, "please input data to calibrate"
        
        # apply transforms
        self.inliers_paths['Stream'] = self.inliers_paths['Stream'].apply(self.path_transform)
        
        # vectorize the streamed data.
        self.inliers_vec = vectorize(self.inliers_paths, show_progress, **self.vectorization) 
        
        # compute scores (distance to the nearest neighbor in the Mahalanobis distance to the corpus)
        if self.nn.per_freq:
            self.FrCh = self.inliers_vec.FrCh.unique()
            # scores in the frequency channel-oriented Mahalanobis distance
            self.inliers_vec = {FrCh: convert_np(inliers) for FrCh, inliers in self.inliers_vec.groupby('FrCh')}
            self.inliers_scores = {FrCh: variance_score(self.inliers_vec[FrCh], self.A_inv[FrCh], self.corpus_vec[FrCh]) \
                                                        for FrCh in tqdm(self.FrCh, disable= not show_progress, \
                                                                         desc='Computing scores for each frequency channel')}
            
            if self.pysegments.distfit!='none':
                self.threshold = {FrCh: get_threshold_score(self.inliers_scores[FrCh], self.pysegments.distfit, self.pysegments['threshold'])\
                                  for FrCh in self.FrCh} 
            else:
                self.threshold = {FrCh: self.pysegments['threshold'] for FrCh in self.FrCh} 
        else:
            self.inliers_vec = convert_np(self.inliers_vec)
            self.inliers_scores = variance_score(self.inliers_vec, self.A_inv, self.corpus_vec)
            
            if self.pysegments.distfit!='none':
                self.threshold = get_threshold_score(self.inliers_scores, self.pysegments.distfit, self.pysegments['threshold']) 
            else:
                self.threshold = self.pysegments['threshold'] 

    def flag(self, all_inliers=False, output_name='arr', **parallel_kwarg):
        '''Process/Detect outliers in new data'''
        
        assert self.test_paths is not None, "please input a dataset to process"

        # apply paths transforms
        self.test_paths['Stream'] = self.test_paths['Stream'].apply(self.path_transform)

        # get interval length and frequency channels to process    
        self.channels_test = self.test_paths.FrCh.unique()
        n_chann = len(self.channels_test)
        
        pool = Pool(**parallel_kwarg)
        
        # set threshold
        if self.nn.per_freq:
            if all_inliers:
                assert self.pysegments.distfit=='none', "do not use distfit if aggregating all inliers"
                all_scores = np.concatenate([self.inliers_scores[FrCh] for FrCh in self.inliers_scores.keys()])
                inliers_scores = {FrCh: all_scores for FrCh in self.inliers_scores.keys()}
            else:
                inliers_scores = self.inliers_scores

                
                
        if self.nn.per_freq:
            to_process = [(self.test_paths[self.test_paths['FrCh']==FrCh],\
                           self.A_inv[FrCh],\
                           self.threshold[FrCh],
                           inliers_scores[FrCh],\
                           self.corpus_vec[FrCh],\
                           self.pysegments,\
                           self.vectorization)\
                           for i,FrCh in enumerate(self.channels_test)]
            
        else:
            to_process = [(self.test_paths[self.test_paths['FrCh']==FrCh],\
                           self.A_inv,\
                           self.threshold,\
                           self.inliers_scores,\
                           self.corpus_vec,\
                           self.pysegments,\
                           self.vectorization)\
                           for i,FrCh in enumerate(self.channels_test)]
        
        # process the channels with Pysegments in parallel    
        self.ans = list(tqdm(pool.imap(_flag, to_process)))
        pool.close()
        
        # process the results
        integration_times = self.test_paths['Stream'].iloc[0].shape[0]
        self.arr = np.zeros((integration_times,n_chann))
        self.arr.fill(1)
        
        for i in range(n_chann):
            self.arr = fill_binary_ans(self.arr, self.ans[i], i)
  
        # save the results
        with open(output_name+'.npy','wb') as f:
            np.save(f, self.arr)
   
    def path_transform(self, stream):
        return _path_transforms(stream, **self.path_transforms)

    def plot_result(self, array, outname='plots/fig', telescope=None):
        if telescope == 'MWA':
            y_ticks = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]
            y_ticklabels = [167, 170, 173, 177, 180, 183, 186, 189, 193, 196]
            x_ticks = np.arange(0, 50, 10)
            x_ticklabels = np.arange(0, 50, 10)
            xline = True
        if telescope == 'HERA':
            y = len(self.FrCh)
            y_ticks = np.arange(0,array.shape[1],20)
            y_ticklabels = np.arange(0,array.shape[1],20)
            x_ticks = np.arange(0, array.shape[0], 20)
            x_ticklabels = np.arange(0, array.shape[0], 20)
            xline = False
#        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6)) ##TRY for colored plots 
#        plot_lib.image_plot(fig, ax, array.T, title='Pysegments', xticks=x_ticks, xticklabels=x_ticklabels, cbar_label = "score/threshold", mask_color='navy', vmax=50, yticks = y_ticks, yticklabels = y_ticklabels,xlabel='Time (2s)', ylabel='Frequency (MHz)', xline=xline) ##TRY for colored plots
        cmap = colors.ListedColormap(['navy', 'yellow'])
        bounds=[0,0.5,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        plot_lib.image_plot(fig, ax, array.T,\
                            xticks=x_ticks, xticklabels=x_ticklabels, \
                            yticks = y_ticks, yticklabels = y_ticklabels,\
                            mask_color='yellow', \
                            xlabel='Time (2s)', ylabel='Frequency (MHz)', \
                            cbar=False, two_colors=True, cbar_ticks=[0, 0.5, 1], \
                            cmap=cmap, norm=norm, boundaries=bounds, vmin=0, vmax=1, \
                            cbar_ticks_labels=['Non-Flagged', '', 'Flagged'], cbar_label='', \
                            xline=xline, xlineticks=[50],font_size=16)
        ax.set_title('SigNova', fontsize=20)
        fig.savefig(outname+'.pdf')
        plt.show()
        
        
    def get_number_inliers(self):
        if self.nn.per_freq:
            return {FrCh: len(self.inliers_scores[FrCh]) for FrCh in self.inliers_scores.keys()}
        else:
            return len(self.inliers_scores)

    def get_inliers_scores(self,name): #NEW
        if self.nn.per_freq:
            a = {FrCh: self.inliers_scores[FrCh] for FrCh in self.inliers_scores.keys()}
            out = open('scores/{}.pkl'.format(name), 'wb')
            pickle.dump(a, out)
            out.close()

            return {FrCh: len(self.inliers_scores[FrCh]) for FrCh in self.inliers_scores.keys()}
        else:
            a = self.inliers_scores
            out = open('scores/{}.pkl'.format(name), 'wb')
            pickle.dump(a, out)
            out.close()
            return len(self.inliers_scores)
        
    def get_corpus_size(self):
        if self.nn.per_freq:
            return {FrCh: self.corpus_vec[FrCh].shape[0] for FrCh in self.corpus_vec.keys()}
        else:
            return self.corpus_vec.shape[0]
        
    def get_number_features(self):
        if self.nn.per_freq:
            return {FrCh: self.corpus_vec[FrCh].shape[1] for FrCh in self.corpus_vec.keys()}
        else:
            return self.corpus_vec.shape[1]
        
    def get_corpus_size(self):
        if self.nn.per_freq:
            return {FrCh: self.corpus_vec[FrCh].shape[0] for FrCh in self.corpus_vec.keys()}
        else:
            return self.corpus_vec.shape[0]
        
    def get_number_features(self):
        if self.nn.per_freq:
            return {FrCh: self.corpus_vec[FrCh].shape[1] for FrCh in self.corpus_vec.keys()}
        else:
            return self.corpus_vec.shape[1]
        
        
        
        
        
      
