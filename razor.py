import numpy as np
import pandas as pd
import scipy
import math
import os
from scipy.sparse import csr_matrix
from numpy import genfromtxt
from sys import getsizeof
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
from scipy.stats import spearmanr, pearsonr
from collections import Counter
import re
import pickle
import argparse

### Parse script parameters ###
parser = argparse.ArgumentParser(description='RaZoR : Estimation of cell type drug sensitivity from ex-vivo drug sensitivity assay and single-cell RNA-seq data.')
parser.add_argument('-o', '--out_dir', type=str, help='Output directory to write results.')
parser.add_argument('-b', '--bulk_matrix', type=str, help='Expression matrix for bulk-rnaseq to use.eFormat needs to be: ... ')
parser.add_argument('-p', '--prop_file', type=str, help='Cell type proportions matrix for N patients in bulk-rnaseq data')
parser.add_argument('-r', '--drug_file', type=str, help='Drug sensitivity assay raw data file. Contains the ratio of surviving cells. First column is patient name, second column is drug names. columns 2-9 are drug sensitivity raw data. ')
parser.add_argument('-s', '--sign_expr_file', type=str, help='Signature expression matrix derived from scRNA-seq reference dataset. ')
parser.add_argument('-d', '--drug', type=str, help='String indicating which drug in drug_file to consider.')
parser.add_argument('-nd', '--n_doses', type=int, default=7, help='Number of doses to consider (subset of the full data consisting of 7 doses)')
parser.add_argument('-i', '--n_iter', type=int, default=50, help='Maximum number of iteration for the EM algorithm')
parser.add_argument('-f', '--n_fold_cv', type=int, default=5, help='Number of folds to consider for the Cross-Validation.')
parser.add_argument('-c', '--c_param', type=float, default=1.0, help='Value for regularization parameter C used for the weighted logistic regression. c.g. scikit-learn, logistic regression with liblinear solver.')

args = parser.parse_args()

### Define constants ###
BULK_FILE = args.bulk_matrix
ALPHA_FILE = args.prop_file
DRUG_FILE = args.drug_file
SIGN_EXPR_FILE = args.sign_expr_file
DRUG = args.drug
N_DOSES = args.n_doses
N_ITER = args.n_iter
N_FOLD_CV = args.n_fold_cv
C = args.c_param
OUT_DIR = args.out_dir
SUBSET_SAMPLES = None

### Define general functions ###
# Return the name of the file without directory path nor extension
def basen_no_ext(my_str):
    return os.path.splitext(os.path.basename(my_str))[0]

# Function to print Class::Name - Comments from any classes

def sigmoid(x):
    return 1 / (1 + math.exp(-x) )

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def rmse(p, q):
    return np.sqrt((1/len(p))*np.sum((p - q)**2))

def logger(comment):
    """
    logger::Write class / function name with custom comment

    :param comment: custom comment as string
    """
    class_fun_str = inspect.stack()[1][0].f_locals["self"].__class__.__name__ + "::" + \
                    inspect.stack()[1].function + "()"
    print("{:<40} - {:<50}".format(class_fun_str, comment))

# Function to create directory
def create_dir(d):
    mkdir_cmd = "mkdir -p " + d
    sub.run(mkdir_cmd, check=True, shell=True)

# Detect gzip compressed file
def test_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'

def get_gene_symbol(g):
    if g in GENE_REF.index:
        g_symbol = GENE_REF.loc[g][6]
        if len(np.array([GENE_REF.loc[g][6], 1])) > 2:
            return g
        else :
            return g_symbol
    else:
        return g

def get_gene_symbols(g):
    gs_len = len(g)
    gsymbols = np.empty(gs_len, dtype=object)
    for i in range(gs_len):
        gsymbols[i] = get_gene_symbol(g[i])
    return gsymbols

def erase_if_exists(file):
    if os.path.exists(file):
        os.remove(file)

def prepare_data_Sjk(bulk_file, alpha_file, drug_file, sign_file, drug, n_dose, subset_samples=None):
    sign_mat = pd.read_csv(sign_file, delimiter=',', index_col=0)
    alphas = pd.read_csv(alpha_file, delimiter = ',', index_col = 0)
    bulk_expr = pd.read_csv(bulk_file, index_col = 1).transpose()
    bulk_expr.drop(labels='Gene', axis=0, inplace=True)
    bulk_expr = bulk_expr.transpose().iloc[np.array(bulk_expr.columns.isin(sign_mat.index))]
    bulk_expr.sort_index(axis=0, inplace=True)
    bulk_expr = bulk_expr.iloc[bulk_expr.index.isin(sign_mat.index)]
    sign_mat_sub = sign_mat.iloc[sign_mat.index.isin(bulk_expr.index)]
    assert(all(bulk_expr.index == sign_mat_sub.index))
    bulk_expr = bulk_expr.transpose().iloc[np.array(bulk_expr.columns.isin(alphas.index))].transpose()
    alphas = alphas.iloc[np.array(alphas.index.isin(bulk_expr.columns))]
    assert(all(alphas.index == bulk_expr.columns))
    drug_resp = pd.read_csv(drug_file, delimiter=',', index_col=0)
    drug_resp_sub = drug_resp.iloc[drug_resp.index == drug]
    drug_resp_sub = drug_resp_sub.iloc[np.array(~drug_resp_sub["dose=0.3704"].isna())]
    intersect = np.intersect1d(drug_resp_sub['sample_id'], alphas.index)
    if subset_samples is not None:
        intersect = intersect[0:subset_samples] # subset for testing
    print("there is " + str(len(intersect)) + " common sample_id between alpha table and drug response table")
    drug_resp_sub = drug_resp_sub.iloc[np.array(drug_resp_sub['sample_id'].isin(intersect))]
    alphas_sub = alphas.iloc[np.array(alphas.index.isin(intersect))]
    bulk_expr_sub = bulk_expr.T.iloc[np.array(bulk_expr.columns.isin(intersect))].T
    assert(all(bulk_expr_sub.columns == drug_resp_sub['sample_id']))
    # Normalize signature matrix by cell types
    smat_norm = (sign_mat_sub.transpose() / sign_mat_sub.sum(axis=1)).transpose()
    gnames = np.array(sign_mat_sub.index)
    smat_norm.sum(axis=1)[0:5]
    smat_norm_np = np.array(smat_norm)
    all_s_jk = list()
    for j in range(bulk_expr_sub.shape[1]):
        y_j = np.array(bulk_expr_sub.iloc[:,j]).astype(float)
        s_jk = (smat_norm_np.T * y_j)
        s_jk_scale = preprocessing.scale(s_jk)
        all_s_jk.append(s_jk_scale)
    G = sign_mat_sub.shape[0] # number of genes 
    K = sign_mat_sub.shape[1] # number of cell types 
    N = bulk_expr_sub.shape[1] # number of patients 
    ct_names = np.array(sign_mat_sub.columns)
    n_columns = G * K
    n_rows = N * K
    S_jk = np.zeros((n_rows, n_columns))
    # Now we fill the big Sjk matrix
    for j in range(N):
        s_jk = np.array(all_s_jk[j])
        for k in range(K):
            row_idx = j*K + k 
            col_idx_start = k * G
            col_idx_end = k * G + G
            S_jk[row_idx, col_idx_start:col_idx_end] = s_jk[k,:]
    np.nan_to_num(S_jk, copy=False)
    assert(np.isnan(S_jk).sum(axis=0).sum() == 0) #check there is no nan values left
    # Generate outputs
    y_j = np.array(drug_resp_sub.T.drop('sample_id').T)
    y_KN = np.repeat(y_j, K, axis=0)
    a_flat = np.array(alphas_sub).flatten()
    # Generate string value for each observation 
    ct_names = np.array(alphas.columns)
    subject_names = np.array(bulk_expr_sub.columns)
    ct_names_KN = np.tile(ct_names, (len(subject_names), 1)).flatten()
    subject_names_KN = np.repeat(subject_names, K)
    test_var_KN_lab = subject_names_KN + "_" + ct_names_KN
    # Return results 
    return S_jk, a_flat, y_KN, test_var_KN_lab, gnames, ct_names, K, N, G


### CLASSES DEFINITION ###

class DataContainer_Skj:
    """
    Allows to load all necessary matrices and vectors.

    :param C: single-cell expression matrix, can be in following format :
                            * a dense 2D numpy.ndarray
    :param a: array of normalized alphas (alpha_tilde). The sum of all alpha_tilde over all cells should be 1 for every patients.
    :param y: array of drug response as survival rate (ratio of cells that survive the given drug treatment).
    :param dC: doubled matrix C. Half of the matrix is associated with label = 1 and other half with label = 0.
    :param dlabs: doubled label array. Half of the array is equal to 1 and other half to 0.
    """
    def __init__(self, C, a, y, N, K, gene_names, ct_names, n_dose=7, obs_names_KN=None, dC=None, dlabs=None):
        self.check_input(C, a, y, n_dose) # check vector / matrix size
        self.C = C
        self.a = a
        self.y = y[:,0:n_dose]
        self.n_dose = n_dose
        self.N = N
        self.K = K
        self.G = len(gene_names)
        self.dL7 = self.init_7_labels() # 7*2 labels 
        self.dM7_D7 = self.init_7_mat() # 7*2 matrix for 7 drug doses
        self.obs_names_KN = obs_names_KN 
        self.gene_names = gene_names
        self.ct_names = ct_names
        
    def check_input(self, C, a, y, n_dose):
        check_n_cells = C.shape[1] == a.shape[0]
        check_n_patients = a.shape[0] == y.shape[0]
        if check_n_cells & check_n_patients != True:
            raise ValueError('Size of input matrices and vectors do not match')

    def init_7_mat(self):
        dC = self.init_double_mat() 
        DM7 = np.tile(dC, (self.n_dose, 1)).transpose()
        Dmat = self.init_7_hotencode(DM7, self.C)
        return np.concatenate([DM7, Dmat.transpose()]).transpose()
    
    def init_7_labels(self):
        #return np.tile(self.dlabs, (1, self.n_dose))[0]
        dlabs = self.init_double_labels()
        return np.tile( dlabs, (1, self.N * self.n_dose))[0]

    def init_7_hotencode(self, dM7, mat):
        Dmat = np.zeros((dM7.shape[1], self.n_dose))
        n_cells = mat.shape[1]
        for i in range(Dmat.shape[0]):
            for j in range(Dmat.shape[1]):
                start_1 = j*n_cells*2
                end_1 = start_1 + n_cells*2
                if i >= start_1 :
                    if i < end_1 :
                        Dmat[i,j] = 1
        return Dmat

    def init_double_mat(self):
        #return np.concatenate( (self.C, self.C), axis=1).transpose()
        this_c = np.hstack((self.C[:,0:21], self.C[:,0:21]))
        for x in range(1, self.N):
            start_idx = x*21
            end_idx = x*21 + 21
            c_piece = self.C[:,start_idx:end_idx]
            this_c = np.hstack((this_c, np.hstack((c_piece, c_piece))))
        return this_c.T

    def init_double_labels(self):
        #return np.concatenate( ( np.ones(self.C.shape[1]), np.zeros(self.C.shape[1]) ) )
        return np.concatenate( ( np.ones(self.K), np.zeros(self.K) ) )


class ExpectationMaximizationLauncher7_Sjk:
    """
    Launch the Expectation-Maximization algorithm by using arrays / matrices provided in the object DC (class DataContainer defined above).

    :param DC: DataContainer object containing input matrix, alphas and target survival rate y
    :param n_iter: integer defining the number of iteration to be used during the EM algorithm 
    :param b: beta coefficient associated with genes. If not provided, betas are initialized at betas_0 = (0, ..., 0)  
    :param w: weights associated with single-cells. If not provided, weights are initialized according to betas_0
    """
    def __init__(self, DC, n_iter, n_fold_cv, out_dir= '.', param_C=1, b=None, w=None, rand_groups=None, tol=1e-3):
        self.DC = DC # DC is a DataContainer object
        self.n_iter = n_iter # n_iter of EM 
        self.n_fold_cv = n_fold_cv # n fold for cross-validation
        self.out_dir = out_dir # output directory to write results
        self.param_C = param_C
        # Variables to be found during the run
        self.betas, self.y_hats, self.y_true = {}, {}, {} # records betas / y_hat at each iter
        self.pis, self.pis_mean_ct, self.beta_agg, self.ratio_sel_agg = {}, {}, {}, {}
        self.aucs_mean_ct, self.aucs = {}, {}
        self.b = b
        self.w = w
        self.N = self.DC.N
        self.K = self.DC.K
        self.init_beta0(b, w)
        self.rand_groups = rand_groups
        # Score metrics
        self.scor, self.pcor = {}, {} # correlation coefs
        #self.kld_q1, self.kld_q0 = {}, {} # KLD between q and alpha
        self.rmse = {} # root mean square error at each iter
        self.tol = tol
        self.last_step = None

    def run(self):
        
        prev_betas = self.b
        # Define group for CV folds
        test_alphas, train_alphas, test_y7_j, train_y7_j, rand_groups, rand_groups_KN, rand_groups_2KND = self.define_cv_folds()
        
        reached_tol_it = 0
        for x in range(self.n_iter):
            print("EM step " + str(x))
            # declare second level dict
            self.scor[x], self.pcor[x] = {}, {} # correlation coefs
            self.rmse[x] = {} # root mean square error at each iter
            self.betas[x], self.y_hats[x], self.y_true[x] = {}, {}, {}
            no_increase = 0
            for i in range(self.n_fold_cv):
                print("CV fold {}".format(i))
                # Define groups
                sel_train = ~(rand_groups == i)
                sel_train_KN = ~(rand_groups_KN == i)
                sel_train_2KND = ~(rand_groups_2KND == i)
                # Define betas 
                if x > 1:
                    prev_betas = self.betas[x-1][i]
                # E-step to re-calculate weights
                weights = self.e_step(self.DC.dM7_D7, train_alphas[i], prev_betas, train_y7_j[i], sel_train_KN)
                self.w = weights
                # M-step to update beta coefficient with weighted logistic regression
                new_betas = self.m_step(weights, sel_train_2KND)
                self.betas[x][i] = new_betas
                
                # Record results
                # declare third level dict
                self.scor[x][i], self.pcor[x][i] = {}, {} # correlation coefs
                self.y_hats[x][i], self.rmse[x][i], self.y_true[x][i]  = {}, {}, {}
                y_hats_train = self.get_y_hat_Sjk(train_alphas[i], new_betas, sel_train, sel_train_KN)
                y_true_train = train_y7_j[i][0::self.K]
                y_hats_test = self.get_y_hat_Sjk(test_alphas[i], new_betas, ~sel_train, ~sel_train_KN)
                y_true_test = test_y7_j[i][0::self.K]                
                self.record_results( y_hats_train, y_true_train, train_alphas[i], x, i, 'train')
                self.record_results( y_hats_test, y_true_test, test_alphas[i], x, i, 'test')
                
            # Record PIs by merging all folds
            mean_pis, mean_pis_ct = self.get_pis(x)
            self.pis[x] = mean_pis
            self.pis_mean_ct[x] = mean_pis_ct
            self.aucs[x] = self.compute_aucs(mean_pis)
            self.aucs_mean_ct[x] = self.compute_aucs(mean_pis_ct)
            # Record aggregated info about gene coefficeint for this step
            self.summarize_gene_coef(x)
            
            # Check if we reached tolerance for early stoping 
            if x > 1 :
                mean_rmse_test = np.mean(np.array([em_launcher.rmse[x][0]['train'][a] for a in range(self.DC.n_dose)]))
                prev_mean_rmse_test = np.mean(np.array([em_launcher.rmse[x-1][0]['train'][a] for a in range(self.DC.n_dose)]))
                print('Testing RMSE {}'.format(str(mean_rmse_test)))
                print('Prev testing RMSE {}'.format(str(prev_mean_rmse_test)))
                print('Diff : {}'.format(str(prev_mean_rmse_test - mean_rmse_test)))
                #print('difference in RMSE mean : {0:.10f}'.format(abs(mean_rmse_test - prev_mean_rmse_test)))
                if prev_mean_rmse_test - mean_rmse_test < self.tol:
                    reached_tol_it+=1
                    print('reach_tol_it var value : {}'.format(str(reached_tol_it)))
                    if reached_tol_it >= 2:
                        print('tol val reached : early stopping')
                        self.last_step = x # record last step
                        break
                else: 
                    reached_tol_it=0
        self.last_step = x # record last step

    def init_rand_groups(self):
        self.rand_groups = np.random.randint(0,self.n_fold_cv, size=(self.DC.a.shape[1]))            
           
    def define_cv_folds(self):
        the_min = 0
        while(the_min < 2):
            rand_groups = np.random.randint(0,em_launcher.n_fold_cv, size=N)
            the_min = np.min(np.unique(rand_groups, return_counts=True)[1])

        rand_groups_K = np.repeat(rand_groups, K)
        rand_groups_2K = np.repeat(rand_groups, 2*K)
        rand_groups_2KND = np.tile(rand_groups_2K, self.DC.n_dose)
        test_alphas, train_alphas = [], []
        test_y_j, train_y_j = [], []
        test_tvar, train_tvar = [], []
        for i in range(self.n_fold_cv):
            sel_group = rand_groups == i
            sel_groups_K = rand_groups_K == i
            sel_groups_2KND = rand_groups_2KND == i
            test_alphas.append(np.array(self.DC.a[sel_groups_K]))
            train_alphas.append(np.array(self.DC.a[~sel_groups_K]))
            test_y_j.append(np.array(self.DC.y[sel_groups_K,:]))
            train_y_j.append(np.array(self.DC.y[~sel_groups_K,:]))

        return test_alphas, train_alphas, test_y_j, train_y_j, rand_groups, rand_groups_K, rand_groups_2KND
    
    def init_beta0(self, b, w):
        if b == None:
            self.b = np.zeros(self.DC.C.shape[0] + self.DC.n_dose)

    def e_step(self, mat, a_train, b, y7, sel_train):
        final_w = np.empty(0)
        for i in range(y7.shape[1]):
            y_col_train = y7[:,i]
            mat_start = i*2*self.DC.C.shape[1]
            mat_end   = mat_start + self.DC.C.shape[1]
            mat_sub   = self.DC.dM7_D7[mat_start:mat_end,:].copy().transpose()
            pi = np.array([sigmoid(x) for x in np.matmul(b.transpose(), mat_sub)])
            pi_train = pi[sel_train]
            q_1_num = pi_train * a_train
            q_0_num = (1 - pi_train) * a_train
            this_w = np.empty(0)
            for x in range(len(y_col_train)):
                start_idx = x*21
                end_idx = x*21 + 21
                q1_n  = q_1_num[start_idx:end_idx]
                q1_den = q1_n.sum()
                q0_n  = q_0_num[start_idx:end_idx]
                q0_den = q0_n.sum()
                w_1 = (q1_n / q1_den) * y_col_train[start_idx:end_idx]
                w_0 = (q0_n / q0_den) * (1 - y_col_train[start_idx:end_idx])
                this_w = np.concatenate((this_w , w_1))
                this_w = np.concatenate((this_w , w_0))
            final_w = np.concatenate( (final_w, this_w), axis=None)
        return np.nan_to_num(final_w, copy=False)

    def m_step(self, w, sel_train):
        X_in = self.DC.dM7_D7[sel_train,:]
        y_in = self.DC.dL7.astype('str')[sel_train]
        clf = LogisticRegression(max_iter=1e3, solver='liblinear', fit_intercept = False, 
                                 penalty='l1', C=self.param_C)
        clf = clf.fit(X_in, y_in, sample_weight = w)
        print("Number of genes selected by model : {}".format(np.sum(clf.coef_[0] != 0)))
        return clf.coef_[0]

    def get_pis(self, em_step):
        list_pis = []
        for f in range(self.n_fold_cv): # self.DC.n_dose represents the number of distinct drug doses
            all_pis = []
            for x in range(self.DC.n_dose): # self.DC.n_dose represents the number of distinct drug doses
                ohe_7 = np.zeros((self.DC.C.shape[1], self.DC.n_dose))
                ohe_7[:,x] = np.ones(self.DC.C.shape[1])
                mat7 = np.concatenate([self.DC.C, ohe_7.transpose()])
                betas = self.betas[em_step][f]
                pi = np.array([sigmoid(i) for i in np.matmul(betas.transpose(), mat7)])
                all_pis.append(pi)
            list_pis.append(pd.DataFrame(all_pis, columns = self.DC.obs_names_KN)) #columns = sign_mat.columns))
        ### Combine PIs found in each folds
        mean_pis = list_pis[0]
        for x in range(self.n_fold_cv-1):
            mean_pis = mean_pis + list_pis[x+1]
        mean_pis = mean_pis / self.n_fold_cv
        ### Compute average pis per cell type
        ct_summary = np.zeros((self.DC.K, n_doses))
        for index, ct in enumerate(self.DC.ct_names):
            ct_summary[index, :] = mean_pis.iloc[:, index::self.DC.K].mean(axis=1)
        mean_pis_ct = pd.DataFrame(ct_summary, index = self.DC.ct_names).transpose()
        return mean_pis, mean_pis_ct
    
    def compute_aucs(self, pis_df):
        all_aucs = []
        for ct in pis_df.columns:
            all_aucs.append(auc(np.array([x for x in range(n_doses)])/(n_doses-1), pis_df[ct]))
        return pd.DataFrame(all_aucs,index = pis_df.columns)

    def summarize_gene_coef(self, em_step):
        n_genes = self.DC.G ; n_fold = self.n_fold_cv ; ct_df = [] ; sel_ct_names = []
        for index, ct in enumerate(self.DC.ct_names):
            idx_srt = index * n_genes
            idx_end = (index+1) * n_genes
            sel_gene_names, beta_coefs = np.empty(0), np.empty(0)
            for fold in range(n_fold):
                sel_non_zero = em_launcher.betas[em_step][fold][idx_srt:idx_end] != 0
                sel_gene_names = np.concatenate((sel_gene_names, self.DC.gene_names[sel_non_zero]))
                beta_coefs = np.concatenate((beta_coefs, em_launcher.betas[em_step][fold][idx_srt:idx_end][sel_non_zero]))
            df_summary = pd.DataFrame([sel_gene_names, beta_coefs, np.ones(len(beta_coefs))], index = ['gene_names', 'beta', 'freq']).transpose()
            if df_summary.shape[0] > 0:
                df_summary['beta'] = pd.to_numeric(df_summary['beta'])
                df_summary['freq'] = pd.to_numeric(df_summary['freq'])
                df_agg = df_summary.groupby('gene_names').sum()
                df_agg['beta'] = df_agg['beta'] / n_fold
                df_agg['freq'] = df_agg['freq'] / n_fold        
                ct_df.append(df_agg)
                sel_ct_names.append(ct)
        concatdict = {}
        for i in range(len(sel_ct_names)):
            concatdict[sel_ct_names[i]] = ct_df[i]
        df_concat_betas = pd.concat(concatdict)['beta'].unstack(level=0).fillna(0)
        df_concat_ratio = pd.concat(concatdict)['freq'].unstack(level=0).fillna(0)
        self.beta_agg[em_step] = df_concat_betas
        self.ratio_sel_agg[em_step] = df_concat_ratio
    
    def get_y_hat_Sjk(self, a, b, sel_train, sel_train_KN):
        y_hats = np.zeros((np.sum(sel_train), self.DC.y.shape[1]))
        for x in range(self.DC.n_dose): # self.DC.n_dose represents the number of distinct drug doses
            ohe_7 = np.zeros((self.DC.C.shape[1], self.DC.n_dose))
            ohe_7[:,x] = np.ones(self.DC.C.shape[1])
            mat7 = np.concatenate([self.DC.C, ohe_7.transpose()])
            pi = np.array([sigmoid(i) for i in np.matmul(b.transpose(), mat7)])
            ct_contrib = pi[sel_train_KN] * a
            y_hats_list = []
            for y in range(np.sum(sel_train)):
                start_range = y*self.K
                end_range = y*self.K + self.K
                y_hats_list.append(np.sum(ct_contrib[start_range : end_range]))
            y_hats[:,x] = np.array(y_hats_list)
        return(y_hats)

    def get_results7(y7_j, yhat7_j):
        res = np.zeros((self.DC.n_dose, 3))
        for x in range(self.DC.n_dose):
            res[x,0] = spearmanr(y7_j[:,x], yhat7_j[:,x])[0]
            res[x,1] = pearsonr(y7_j[:,x], yhat7_j[:,x])[0]
            res[x,2] = rmse(y7_j[:,x], yhat7_j[:,x])
        return res

    def write_np_betas(self, fold, tr_or_te):
        np_betas = np.zeros([len(self.betas[fold][0][tr_or_te]), self.n_iter])
        for i in range(self.n_iter):
            np_betas[:, i] = em_launcher.betas[fold][i][tr_or_te]
        np.savetxt(self.out_dir + "/betas_" + tr_or_te + "_fold" + str(fold) + ".csv", np_betas, delimiter=',')
    
    def write_np_y_hats(self, y_j, fold, tr_or_te):
        np_y_hat = np.zeros([len(self.y_hats[fold][0][tr_or_te]), self.n_iter])
        for i in range(self.n_iter):
            np_y_hat[:, i] = em_launcher.y_hats[fold][i][tr_or_te]
        #np_y_hat[:, self.n_iter+1] = y_j
        np.savetxt(self.out_dir + "/y_hats_" + tr_or_te + "_fold" + str(fold) + ".csv", np_y_hat, delimiter=',')
    
    def record_results(self, y_hat, y_true, a, x, i, tr_or_te):
        self.scor[x][i][tr_or_te], self.pcor[x][i][tr_or_te] = {}, {}
        self.rmse[x][i][tr_or_te] = {}
        self.y_hats[x][i][tr_or_te], self.y_true[x][i][tr_or_te] = {}, {}
        
        # add results
        for d in range(self.DC.n_dose):
            self.scor[x][i][tr_or_te][d] = spearmanr(y_true[:,d], y_hat[:,d])[0]
            self.pcor[x][i][tr_or_te][d] = pearsonr(y_true[:,d], y_hat[:,d])[0]
            self.rmse[x][i][tr_or_te][d] = rmse(y_true[:,d], y_hat[:,d])
            self.y_true[x][i][tr_or_te][d] = y_true[:,d]
            self.y_hats[x][i][tr_or_te][d] = y_hat[:,d]
        
        print('{0} : {1}'.format(tr_or_te, np.mean(list(em_launcher.scor[x][i][tr_or_te].values()))))
        
    def get_train_test_matrix(self, dict_res):
        train_mat = np.zeros([self.n_iter, self.n_fold_cv])
        test_mat = np.zeros([self.n_iter, self.n_fold_cv])
        for i in range(self.n_fold_cv):
            for j in range(self.n_iter):
                train_mat[j][i] = dict_res[i][j]['train']
                test_mat[j][i] = dict_res[i][j]['test']
        return train_mat, test_mat



if __name__ == "__main__":
   
    if DRUG == 'all':
        all_drugs = pd.read_csv(DRUG_FILE)['drug_name'].unique()
        n_fold = 5; max_em_steps = 5; n_doses = 6
        param_C = 1.0

        for drug_ in all_drugs:
            print(drug_)

            # Prepare input data
            S_jk, a_flat, y_KN, obs_names_KN, gene_names, ct_names, K, N, G = prepare_data_Sjk(BULK_FILE, ALPHA_FILE, DRUG_FILE,
                                                                                               SIGN_EXPR_FILE, drug_, n_dose=N_DOSES,subset_samples=None)

            # Create data container object
            dc = DataContainer_Skj(S_jk.T, a_flat, y_KN, N=N, K=K, n_dose=n_doses, obs_names_KN = obs_names_KN,
                                   gene_names = gene_names, ct_names = ct_names)

            # Build EM launcher and define n_iter
            em_launcher = ExpectationMaximizationLauncher7_Sjk(dc, n_iter = N_ITER, n_fold_cv = N_FOLD, param_C = C)

            # Launch with error checking
            # Error checking - if model could be launched properly, continue
            str_error = True
            for x in range(0, 4):  # try 4 times
                try:
                    em_launcher.run()
                    str_error = False
                except:
                    pass
                if not str_error:
                    break
            if x >= 3:
                print("em_launcher failed for drug {}".format(drug_))
                continue

            # Remove large matrix from result table
            em_launcher.DC.dM7_D7, em_launcher.DC.C, em_launcher.DC.dL7 = None, None, None

            # Save results in pickle
            out_file = OUT_DIR + "/" + drug_ + "_C" + str(C) + "_emLauncherObj.pkl"
            with open(out_file, "wb") as out:
                pickle.dump(em_launcher, out)
    else:

        ## 1. Load data 
        S_jk, a_flat, y_KN, obs_names_KN, gene_names, ct_names, K, N, G = prepare_data_Sjk(BULK_FILE, ALPHA_FILE, DRUG_FILE, SIGN_EXPR_FILE, 
                                                                                           DRUG, n_dose=N_DOSES, subset_samples = SUBSET_SAMPLES)

        ## 2. Build DataContainers
        dc = DataContainer_Skj(S_jk.T, a_flat, y_KN, N=N, K=K, n_dose=N_DOSES, test_var = obs_names_KN)

        ## 3. Create EMlaucher object 
        em_launcher = ExpectationMaximizationLauncher7_Sjk(dc, n_iter = N_ITER, n_fold_cv = N_FOLD_CV, param_C = C, N_subjects = N)
        # Launch EM algorithm 
        em_launcher.run()

        ## 4. Save object with pickle
        # First we set the big matrix to None to avoid writing it 
        em_launcher.DC.dM7_D7 = None
        # We write results
        out_file = OUT_DIR + "/" + DRUG + "_C" + str(C) + "_emLauncherObj.pkl"
        with open(out_file, "wb") as out:
            pickle.dump(em_launcher, out)

