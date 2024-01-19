# cython: embedsignature=True
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# distutils: language = c++
#
# Cython version of the Navarro & Fuss, 2009 DDM PDF. Based on the following code by Navarro & Fuss:
# http://www.psychocmath.logy.adelaide.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
#
# This implementation is about 170 times faster than the matlab
# reference version.
#
# Copyleft Thomas Wiecki (thomas_wiecki[at]brown.edu) & Imri Sofer, 2011
# GPLv3

import hddm

import scipy.integrate as integrate
from copy import copy
import numpy as np
# from math import comb
from scipy.special import comb
# cimport math
import itertools
import scipy.stats as st
cimport numpy as np
cimport cython

from cython.parallel import *
# cimport openmp

# include "pdf.pxi"
include 'integrate.pxi'

# Added for Bayesian Q in 2022-08-04:
# import pymc as pm
# import pymc3 as pm
from collections import deque
import random


# Added for uncertainty modulation in 2022-10-01:

from scipy.stats import entropy as scientropy
# from scipy.stats import beta

np.warnings.filterwarnings('ignore', '(overflow|invalid)')

np.random.seed(seed=1234)

cdef extern from "math.h" nogil:
    double sin(double)
    double cos(double)
    double log(double)
    double exp(double)
    double sqrt(double)
#    double fmax(double, double)
    double pow(double, double)
    double ceil(double)
    double floor(double)
    double fabs(double)
    double M_PI

# JY added in 2022-10-01 for uncertainty modulation

def alphaf(int k):
    return k+1

def betaf(int n, int k):
    return n-k+1

def var_beta(double a,double b):
    return (a*b) / (((a+b)**2)*(a+b+1))

def exp_beta(double a, double b):
    return a / (a+b)

def mode_beta(double a, double b): # when a, b > 1
    return (a-1) / (a + b -2)

def softmax_stable(np.ndarray[double, ndim=1] x):
    return(exp(x - np.max(x)) / exp(x - np.max(x)).sum())



# def entr(double x, out=None):
#     cdef double x_
#     if x>0:
#         x_ = -x*np.log(x)
#     elif x==0:
#         x_ = 0
#     else:
#         x_ = -np.inf
#     return x_

def entropyf(np.ndarray[double, ndim=1] pk, double base=2, int axis=0):
    if np.sum(pk, axis=axis) != 1.0:
        pk = softmax_stable(pk)
    return -np.sum(pk * np.log(pk)) / np.log(base)
    # cdef double vec
    # cdef double S
    # pk = np.asarray(pk)
    # pk = 1.0*pk / np.sum(pk, axis=axis, keepdims=True)
    # vec = entr(pk)
    # S = np.sum(vec, axis=axis)
    # if base is not None:
    #     S /= np.log(base)
    # return S
def search(np.ndarray[double, ndim=1] arr, double x):
    cdef int i
    for i in range(len(arr)):

        if arr[i] == x:
            return i

    return -1
# class BayesianQ_Agent:
#
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         # self.memory_mf = deque(maxlen=20000)
#         # self.memory_mb = deque(maxlen=20000)
#         self.model = None
#         # self.MIN_VAL = 99.1
#         # self.MAX_VAL = 100.9
#         # self.D = int(((self.MAX_VAL - self.MIN_VAL) * 10) * 3) + 1  # times 3 for the 3 possible current_position values
#
#         self.Qmb_mus_estimates_mu = np.zeros((self.action_size, self.state_size)) #np.zeros((3, self.D))
#         self.Qmb_mus_estimates_sd = np.ones((self.action_size, self.state_size)) #np.ones((3, self.D))
#
#         self.Qmb_sds_estimates_mu = np.ones((self.action_size, self.state_size)) * -2. #np.ones((3, self.D)) * -2.
#         self.Qmb_sds_estimates_sd = np.ones((self.action_size, self.state_size)) * 10. #np.ones((3, self.D)) * 10.
#
#         self.Qmf_mus_estimates_mu = np.zeros((self.action_size, comb(self.state_size,2,exact=True))) #np.zeros((3, self.D))
#         self.Qmf_mus_estimates_sd = np.ones((self.action_size, comb(self.state_size,2,exact=True))) #np.ones((3, self.D))
#
#         self.Qmf_sds_estimates_mu = np.ones((self.action_size, comb(self.state_size,2,exact=True))) * -2. #np.ones((3, self.D)) * -2.
#         self.Qmf_sds_estimates_sd = np.ones((self.action_size, comb(self.state_size,2,exact=True))) * 10. #np.ones((3, self.D)) * 10.
#
#     # def reset_memory(self):
#     #     self.memory = deque(maxlen=20000)
#
#     def act(self, stage, state, Tm = np.array([[0.7, 0.3], [0.3, 0.7]]), use_explo=True):
#
#         if self.model is None:
#             return random.randrange(self.action_size)
#
#         def calculate_VPI(mus, sds):
#
#             def gain(i, i2, x):
#                 gains = []
#                 for j in range(len(mus)):
#                     if j == i:
#                         # special case: this is the best action
#                         g = mus[i2] - np.minimum(x, mus[i2])
#                     else:
#                         g = np.maximum(x, mus[i]) - mus[i]
#
#                     gains.append(g)
#
#                 gains = np.reshape(np.array(gains), [-1, len(x)]).transpose()
#                 return gains
#
#             SAMPLE_SIZE = 1000
#             Q_LOW = -1.
#             Q_HIGH = 1.
#             x = np.random.uniform(Q_LOW, Q_HIGH, SAMPLE_SIZE)
#             x = np.reshape(x, [-1, 1])
#
#             dist = st.norm(mus, np.exp(sds))
#
#             probs = dist.pdf(x)
#
#             best_action_idx = np.argmax(mus)
#
#             tmp_mus = np.copy(mus)
#             tmp_mus[best_action_idx] = -9999.
#             second_best_action_idx = np.argmax(tmp_mus)
#
#             gains = gain(best_action_idx, second_best_action_idx, x)
#
#             return np.mean(gains * probs, axis=0)
#
#         if stage==1: # 1st stage -> get TM & 2nd-stage Q values,
#             # important: two states (planets) as input, unlike stage 2!
#             # output is Qmb (next code is dtq = Qmb[1] - Qmb[0])
#             planets = state
#             # Qmb = np.dot(Tm, [np.max(qs_mb[planets[0], :]), np.max(qs_mb[planets[1], :])])
#             planet1_mus = self.Qmb_mus_estimates[:,planets[0]]
#             planet2_mus = self.Qmb_mus_estimates[:,planets[1]]
#
#             planet1_sds = self.Qmb_sds_estimates[:,planets[0]]
#             planet2_sds = self.Qmb_sds_estimates[:,planets[1]]
#             if use_explo:
#                 action_scores_1 =  calculate_VPI(planet1_mus, planet1_sds) + planet1_mus
#                 action_scores_2 =  calculate_VPI(planet2_mus, planet2_sds) + planet2_mus
#             else:
#                 action_scores_1 = planet1_mus
#                 action_scores_2 = planet2_mus
#
#             return np.dot(Tm, [np.max(action_scores_1), np.max(action_scores_2)]) # Qmb
#         elif stage==2: # 2nd stage -> directly pick between 2nd-stage Q values
#             # important: two states (planets) as input, unlike stage 2!
#             # output is action score (qs; next code is dtq = qs[0] - qs[1])
#             state_idx = state
#             state_mus = self.Qmb_mus_estimates[:, state_idx]
#             state_sds = self.Qmb_sds_estimates[:, state_idx]
#             if use_explo:
#                 VPI_per_action = calculate_VPI(state_mus, state_sds)
#                 action_scores = VPI_per_action + state_mus
#                 # idx_selected_action = np.argmax(action_scores)
#
#                 return action_scores
#             else:
#                 return np.argmax(state_mus)
#
#             # if use_explo:
#             # VPI_per_action = calculate_VPI(state_mus, state_sds)
#             # action_scores = VPI_per_action + state_mus
#             # idx_selected_action = np.argmax(action_scores)
#
#             # return action_scores
#             # else:
#             #     return np.argmax(state_mus)
#
#
#     def update(self,state,action,reward):
#         # mem_mb = np.array(self.memory_mb)
#         # mem_mf = np.array(self.memory_mf)
#
#         # states = mem_mb[:, :self.state_size]
#         # actions = np.reshape(mem_mb[:, self.state_size], [-1, 1])
#         # rewards = mem_mb[:, -1]
#         #
#         # full_tensor = []
#         #
#         # qvalues = [N x 3]
#         # 1 - action index
#         # 2 - state index
#         # 3 - reward
#         # qvalues = np.array(full_tensor)
#
#
#
#         #COMMENT THIS OUT FOR NOW TO RUN OTHER MODELS THAT DON'T USE BAYESIAN-Q
#         with pm.Model() as self.model:
#
#             # state_mus = self.Qmus_estimates[:, state_idx]
#             Qmus = pm.Normal('Qmus', mu=self.Qmb_mus_estimates_mu, sd=self.Qmb_mus_estimates_sd, shape=[self.action_size, self.state_size])
#             Qsds = pm.Normal('Qsds', mu=self.Qmb_sds_estimates_mu, sd=self.Qmb_sds_estimates_sd, shape=[self.action_size, self.state_size])
#
#             # idx0 = qvalues[:, 0].astype(int) action
#             # idx1 = qvalues[:, 1].astype(int) state
#             idx0 = action
#             idx1 = state
#
#             # pm.Normal('likelihood', mu=Qmus[idx0, idx1], sd=np.exp(Qsds[idx0, idx1]), observed=qvalues[:, 2])
#             pm.Normal('likelihood', mu=Qmus[idx0, idx1], sd=np.exp(Qsds[idx0, idx1]), observed=reward)
#
#             mean_field = pm.fit(n=15000, method='advi', obj_optimizer=pm.adam(learning_rate=0.1))
#             self.trace = mean_field.sample(5000)
#
#         # with VAR = EXPR:
#         #     try:
#         #         BLOCK
#         #     finally:
#         #         VAR.__exit__()
#
#
#         self.Qmb_mus_estimates = np.mean(self.trace['Qmus'], axis=0)
#         self.Qmb_sds_estimates = np.mean(self.trace['Qsds'], axis=0)
#
#         self.Qmb_mus_estimates_mu = self.Qmb_mus_estimates
#         self.Qmb_mus_estimates_sd = np.std(self.trace['Qmus'], axis=0)
#
#         self.Qmb_sds_estimates_mu = self.Qmb_sds_estimates
#         self.Qmb_sds_estimates_sd = np.std(self.trace['Qsds'], axis=0)
#
#         # self.reset_memory()
#         # return
#     def forget(self, gamma, state, action):
#
#         # qs_mb[s_, a_] *= (1 - gamma_)
#         self.Qmb_mus_estimates_mu[action, state] *= (1-gamma)
#     def uncertainty(self, state):
#         # think of other ways to change this
#         return np.mean(self.Qmb_sds_estimates_mu[:,state])


def pdf_array(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz,
              double t, double st, double err=1e-4, bint logp=0, int n_st=2, int n_sz=2, bint use_adaptive=1,
              double simps_err=1e-3, double p_outlier=0, double w_outlier=0):

    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim = 1] y = np.empty(size, dtype=np.double)

    for i in prange(size, nogil=True):
        y[i] = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                        n_st, n_sz, use_adaptive, simps_err)

    y = y * (1 - p_outlier) + (w_outlier * p_outlier)
    if logp == 1:
        return np.log(y)
    else:
        return y

cdef inline bint p_outlier_in_range(double p_outlier):
    return (p_outlier >= 0) & (p_outlier <= 1)



# def gain(double best_i, double secondbest_i, double x, double mu):
#     gains = []
#     for j in range(len(mu)):
#         if j == best_i:
#             # special case: this is the best action
#             g = mu[secondbest_i] - np.minimum(x,mu[secondbest_i])
#         else:
#             g = np.maximum(x, mu[best_i]) - mu[best_i]
#         gains.append(g)
#     gains = np.reshape(np.array(gains), [-1, len(x)]).transpose()
#     return gains
#
#
#
# def calculate_VPI(double mu, double std):
#     sample_size = 1000
#     q_low = 0.
#     q_high = 1.
#     x = np.random.uniform(q_low, q_high, sample_size)
#     x = np.reshape(x,[-1,1])
#     dist = st.norm(mu, np.exp(std))
#     probs = dist.pdf(x)
#     best_action_idx = np.argmax(mu)
#     tmp_mu = np.compy(mu)
#     tmp_mu[best_action_idx] = -9999.
#     second_best_action_idx = np.argmax(tmp_mu)
#
#     gains = gain(best_action_idx, second_best_action_idx, x, mu)
#
#     return np.mean(gains * probs, axis=0)




def wiener_like(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz, double t,
                double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                double p_outlier=0, double w_outlier=0.1):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    for i in range(size):
        p = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                     n_st, n_sz, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        p = p * (1 - p_outlier) + wp_outlier
        if p == 0:
            return -np.inf

        sum_logp += log(p)

    return sum_logp

def wiener_like_rlddm(np.ndarray[double, ndim=1] x,
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, double alpha, double pos_alpha, double v, 
                      double sv, double a, double z, double sz, double t,
                      double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] xs
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha

    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        xs = x[split_by == s]
        s_size = xs.shape[0]
        qs[0] = q
        qs[1] = q

        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses[0]]:
            alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        else:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # received on current trial.
        qs[responses[0]] = qs[responses[0]] + \
            alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(1, s_size):
            p = full_pdf(xs[i], ((qs[1] - qs[0]) * v), sv, a, z,
                         sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
            # If one probability = 0, the log sum will be -Inf
            p = p * (1 - p_outlier) + wp_outlier
            if p == 0:
                return -np.inf
            sum_logp += log(p)

            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            if feedbacks[i] > qs[responses[i]]:
                alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            else:
                alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            qs[responses[i]] = qs[responses[i]] + \
                alfa * (feedbacks[i] - qs[responses[i]])
    return sum_logp

# JY added on 2022-01-03 for simultaneous regression on two-step tasks
def wiener_like_rlddm_2step(np.ndarray[double, ndim=1] x1, # 1st-stage RT
# def wiener_like_rlddm_2step_reg_sliding_window(np.ndarray[double, ndim=1] x1, # 1st-stage RT
                      np.ndarray[double, ndim=1] x2, # 2nd-stage RT
                      # np.ndarray[long, ndim=1] isleft1, # whether left response 1st-stage,
                      # np.ndarray[long, ndim=1] isleft2, # whether left response 2nd-stage
                      np.ndarray[long,ndim=1] s1, # 1st-stage state
                      np.ndarray[long,ndim=1] s2, # 2nd-stage state
                      np.ndarray[long, ndim=1] response1,
                      np.ndarray[long, ndim=1] response2,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, double alpha, double pos_alpha,

                      # double w,
                      double gamma, double gamma2,
                      double lambda_,

                      double v0, double v1, double v2,
                      double v, # don't use second stage
                      double sv,
                      double a,
                      double z0, double z1, double z2,
                      double z,
                      double sz,
                      double t,
                      int nstates,
                      # double v_qval, double z_qval,
                      double v_interaction, double z_interaction,
                      double two_stage,

                      double a_2,
                      double z_2,
                      double t_2,
                      double v_2,
                      double sz2,
                      double st2,
                      double sv2,
                      double alpha2,
                      double w, double w2, double z_scaler,
                      double z_sigma, double z_sigma2,
                      double window_start, double window_size,
                      double beta_ndt, double beta_ndt2, double beta_ndt3,


                      double st,


                      double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0,
                      ):


    # cdef double a = 1
    # cdef double sz = 0
    # cdef double st = 0
    # cdef double sv = 0


    cdef Py_ssize_t size = x1.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef double alfa2

    cdef double gamma_
    cdef double gamma__
    cdef double lambda__

    # cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=2] qs_mf = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
    cdef np.ndarray[double, ndim=2] qs_mb = np.ones((nstates, 2))*q # second-stage Q-values

    cdef np.ndarray[double, ndim=2] ndt_counter_set = np.ones((comb(nstates,2,exact=True),1)) # first-stage MF Q-values
    cdef np.ndarray[double, ndim=2] ndt_counter_ind = np.ones((nstates, 1)) # first-stage MF Q-values


    cdef double dtQ1
    cdef double dtQ2

    cdef double dtq_mb
    cdef double dtq_mf

    cdef long s_
    cdef long a_
    cdef double v_
    cdef double z_
    cdef double t_
    cdef double sig
    cdef double v_2_
    cdef double z_2_
    cdef double a_2_
    cdef double t_2_


    cdef np.ndarray[double, ndim=1] x1s
    cdef np.ndarray[double, ndim=1] x2s
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses1
    cdef np.ndarray[long, ndim=1] responses2
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    cdef np.ndarray[long, ndim=1] s1s
    cdef np.ndarray[long, ndim=1] s2s

    # Added by Jungsun Yoo on 2021-11-27 for two-step tasks
    # parameters added for two-step


    cdef np.ndarray[long, ndim=1] planets
    cdef np.ndarray[double, ndim=1] counter = np.zeros(comb(nstates,2,exact=True))
    cdef np.ndarray[double, ndim=1] Qmb
    cdef double dtq
    cdef double rt
    cdef np.ndarray[double, ndim=2] Tm = np.array([[0.7, 0.3], [0.3, 0.7]]) # transition matrix
    cdef np.ndarray[long, ndim=2] state_combinations = np.array(list(itertools.combinations(np.arange(nstates),2)))

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha

    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses1 = response1[split_by == s]
        responses2 = response2[split_by == s]
        x1s = x1[split_by == s]
        x2s = x2[split_by == s]
        s1s = s1[split_by == s]
        s2s = s2[split_by == s]

        s_size = x1s.shape[0]
        qs_mf[:,0] = q
        qs_mf[:,1] = q

        qs_mb[:,0] = q
        qs_mb[:,1] = q


        alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)
        gamma_ = (2.718281828459**gamma) / (1 + 2.718281828459**gamma)

        if gamma2 != 100.00:
            gamma__ = (2.718281828459**gamma2) / (1 + 2.718281828459**gamma2)
        else:
            gamma__ = gamma_

        if alpha2 != 100.00:
            alfa2 = (2.718281828459**alpha2) / (1 + 2.718281828459**alpha2)
        else:
            alfa2 = alfa
        if lambda_ != 100.00:
            lambda__ = (2.718281828459**lambda_) / (1 + 2.718281828459**lambda_)
        if w != 100.00:
            w = (2.718281828459**w) / (1 + 2.718281828459**w)
        if w2 != 100.00:
            w2 = (2.718281828459**w2) / (1 + 2.718281828459**w2)

        # if beta_ndt != 100.00:
        #     beta_ndt = (2.718281828459**beta_ndt) / (1 + 2.718281828459**beta_ndt)
        #
        # if beta_ndt2 != 100.00:
        #     beta_ndt2 = (2.718281828459**beta_ndt2) / (1 + 2.718281828459**beta_ndt2)


        # don't calculate pdf for first trial but still update q
        # if feedbacks[0] > qs[responses[0]]:
            # alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        # else:
            # alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # # received on current trial.
        # qs[responses[0]] = qs[responses[0]] + \
        #     alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        # print(window_size, window_start)
        for i in range(0, s_size):
            if window_start <= i < window_start + window_size:  # and (window_start <= i < window_start+window_size):
                if counter[s1s[i]] > 0 and x1s[i]>0.15:
                # proceed with pdf only if 1) the current 1st-stage state have been updated and 2) "plausible" RT (150 ms)

                    # 1st stage
                    planets = state_combinations[s1s[i]]
                    Qmb = np.dot(Tm, [np.max(qs_mb[planets[0],:]), np.max(qs_mb[planets[1],:])])
                    # qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial

                    # dtq = qs[1] - qs[0]
                    dtq_mb = Qmb[1] - Qmb[0] # 1 is upper, 0 is lower
                    dtq_mf = qs_mf[s1s[i],1] - qs_mf[s1s[i],0]
                    if v == 100.00: # if v_reg
                        v_ = v0 + (dtq_mb * v1) + (dtq_mf * v2) + (v_interaction * dtq_mb * dtq_mf)

                    else: # if don't use v_reg:
                        # if v_qval == 0: # use both qmb and qmf
                        qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial
                        dtq = qs[1] - qs[0]
                        v_ = dtq * v

                    # if z == 0.5: # if use z_reg:
                    if w2 == 100.00: # if use z_reg
                        # Transform regression parameters so that >0
                        z_ = z0 + (dtq_mb * z1) + (dtq_mf * z2) + (z_interaction * dtq_mb * dtq_mf)
                        sig = 1/(1+np.exp(-z_))
                    else: # if don't use z_reg:
                        qs = w2 * Qmb + (1-w2) * qs_mf[s1s[i],:] # Update for 1st trial
                        dtq = qs[1] - qs[0]
                        z_ = dtq * z_scaler
                        sig = 1 / (1 + np.exp(-z_))

                    rt = x1s[i]

                    # Modeling ndt
                    t_ = ((np.log(ndt_counter_ind[planets[0],0]) + np.log(ndt_counter_ind[planets[1],0]))/2)*beta_ndt + \
                         np.log(ndt_counter_set[s1s[i],0])*beta_ndt2 + \
                         t

                    p = full_pdf(rt, v_, sv, a, sig * a,
                                 sz, t_, st, err, n_st, n_sz, use_adaptive, simps_err)
                    # If one probability = 0, the log sum will be -Inf
                    p = p * (1 - p_outlier) + wp_outlier
                    if p == 0:
                        return -np.inf
                    sum_logp += log(p)


                    # # # 2nd stage
                    if two_stage == 1.00:

                        v_2_ = v if v_2==100.00 else v_2
                        a_2_ = a if a_2 == 100.00 else a_2

                        # CONFIGURE Z_2 USING Z_SIGMA AND V HERE!!!
                        if z_sigma ==100.00: # if don't use 1st-stage dependent drift rate
                            # z_2_ = z if z_2 == 0.5 else z_2
                            z_2_ = z_2
                        else: # if use 1st-stage dependent dr

                            if z_sigma2 == 100.00: # don't use baseline
                                z_2_ = 1 / (1 + np.exp(-v_*z_sigma))
                            else: # use baseline
                                z_2_ = 1 / (1 + np.exp(-(v_ * z_sigma + z_sigma2)))

                        t_2_ = t if t_2 == 100.00 else t_2

                        qs = qs_mb[s2s[i],:]
                        dtq = qs[1] - qs[0]
                        rt = x2s[i]

                        p = full_pdf(rt, (dtq * v_2_), sv2, a_2_, z_2_, sz2, t_2_, st2, err, n_st, n_sz, use_adaptive, simps_err)
                        # If one probability = 0, the log sum will be -Inf
                        p = p * (1 - p_outlier) + wp_outlier
                        if p == 0:
                            return -np.inf
                        sum_logp += log(p)


            # update Q values, regardless of pdf

            ndt_counter_set[s1s[i],0] += 1
            ndt_counter_ind[s2s[i],0] += 1

            # just update 1st-stage MF values if estimating
            dtQ1 = qs_mb[s2s[i],responses2[i]] - qs_mf[s1s[i], responses1[i]] # delta stage 1
            qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + alfa * dtQ1 # delta update for qmf

            dtQ2 = feedbacks[i] - qs_mb[s2s[i],responses2[i]] # delta stage 2
            qs_mb[s2s[i], responses2[i]] = qs_mb[s2s[i],responses2[i]] + alfa2 * dtQ2 # delta update for qmb
            if lambda_ != 100.00: # if using eligibility trace
                qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + lambda__ * dtQ2 # eligibility trace


            # memory decay for unexperienced options in this trial

            for s_ in range(nstates):
                for a_ in range(2):
                    if (s_ is not s2s[i]) or (a_ is not responses2[i]):
                        # qs_mb[s_, a_] = qs_mb[s_, a_] * (1-gamma)
                        qs_mb[s_,a_] *= (1-gamma__)

            for s_ in range(comb(nstates,2,exact=True)):
                for a_ in range(2):
                    if (s_ is not s1s[i]) or (a_ is not responses1[i]):
                        qs_mf[s_,a_] *= (1-gamma_)

            counter[s1s[i]] += 1

    return sum_logp


# JY added on 2022-06-22 for choice model
def wiener_like_rl_2step(np.ndarray[double, ndim=1] x1, # 1st-stage RT
                      np.ndarray[double, ndim=1] x2, # 2nd-stage RT
                      np.ndarray[long,ndim=1] s1, # 1st-stage state
                      np.ndarray[long,ndim=1] s2, # 2nd-stage state
                      np.ndarray[long, ndim=1] response1,
                      np.ndarray[long, ndim=1] response2,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, double alpha, double pos_alpha,

                      double gamma, double gamma2,
                      double lambda_,

                      double v, # don't use second stage
                      double z,

                      int nstates,
                      double two_stage,

                      double z_2,
                      double v_2,
                      double alpha2,
                      double w, double window_start, double window_size,
                      double sv, 
                      double sz, 
                      double st, 
                      double sv2, 
                      double sz2, 
                      double st2, 


                      double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0,
                      ):



    cdef Py_ssize_t size = x1.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double p
    cdef double drift

    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef double alfa2

    cdef double gamma_
    cdef double gamma__
    cdef double lambda__

    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=2] qs_mf = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
    cdef np.ndarray[double, ndim=2] qs_mb = np.ones((nstates, 2))*q # second-stage Q-values

    cdef double dtQ1
    cdef double dtQ2

    cdef double dtq_mb
    cdef double dtq_mf

    cdef long s_
    cdef long a_

    cdef np.ndarray[double, ndim=1] x1s
    cdef np.ndarray[double, ndim=1] x2s
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses1
    cdef np.ndarray[long, ndim=1] responses2
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    cdef np.ndarray[long, ndim=1] s1s
    cdef np.ndarray[long, ndim=1] s2s
    # cdef np.ndarray[long, ndim=1] isleft1s
    # cdef np.ndarray[long, ndim=1] isleft2s

    # Added by Jungsun Yoo on 2021-11-27 for two-step tasks
    # parameters added for two-step


    cdef np.ndarray[long, ndim=1] planets
    cdef np.ndarray[double, ndim=1] counter = np.zeros(comb(nstates,2,exact=True))
    cdef np.ndarray[double, ndim=1] Qmb
    cdef double dtq
    cdef double rt
    cdef np.ndarray[double, ndim=2] Tm = np.array([[0.7, 0.3], [0.3, 0.7]]) # transition matrix
    cdef np.ndarray[long, ndim=2] state_combinations = np.array(list(itertools.combinations(np.arange(nstates),2)))

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha

    # if alpha2==100.00: # if either only 1st stage or don't share lr:


    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses1 = response1[split_by == s]
        responses2 = response2[split_by == s]
        x1s = x1[split_by == s]
        x2s = x2[split_by == s]
        s1s = s1[split_by == s]
        s2s = s2[split_by == s]

        # isleft1s = isleft1[split_by == s]
        # isleft2s = isleft2[split_by == s]

        s_size = x1s.shape[0]
        qs_mf[:,0] = q
        qs_mf[:,1] = q

        qs_mb[:,0] = q
        qs_mb[:,1] = q


        alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)
        gamma_ = (2.718281828459**gamma) / (1 + 2.718281828459**gamma)
        if gamma2 != 100.00:
            gamma__ = (2.718281828459**gamma2) / (1 + 2.718281828459**gamma2)
        else:
            gamma__ = gamma_
        if alpha2 != 100.00:
            alfa2 = (2.718281828459**alpha2) / (1 + 2.718281828459**alpha2)
        else:
            alfa2 = alfa
        if lambda_ != 100.00:
            lambda__ = (2.718281828459**lambda_) / (1 + 2.718281828459**lambda_)
        if w != 100.00:
            w_ = (2.718281828459**w) / (1 + 2.718281828459**w)

        # loop through all trials in current condition
        for i in range(0, s_size):
            if window_start <= i < window_start + window_size:  # and (window_start <= i < window_start+window_size):
                if counter[s1s[i]] > 0 and x1s[i]>0.15:

                    # proceed with pdf only if 1) the current 1st-stage state have been updated and 2) "plausible" RT (150 ms)
                    # 1st stage
                    planets = state_combinations[s1s[i]]
                    Qmb = np.dot(Tm, [np.max(qs_mb[planets[0],:]), np.max(qs_mb[planets[1],:])])

                    qs = w_ * Qmb + (1 - w_) * qs_mf[s1s[i], :]  # Update for 1st trial
                    drift = (qs[1] - qs[0]) * v

                    if drift == 0:
                        p = 0.5
                    else:
                        if responses1[i] == 1:
                            p = (2.718281828459**(-2 * z * drift) - 1) / \
                                (2.718281828459**(-2 * drift) - 1)
                        else:
                            p = 1 - (2.718281828459**(-2 * z * drift) - 1) / \
                                (2.718281828459**(-2 * drift) - 1)

                    # If one probability = 0, the log sum will be -Inf
                    p = p * (1 - p_outlier) + wp_outlier
                    if p == 0:
                        return -np.inf
                    sum_logp += log(p)

                    # # # # 2nd stage
                    if two_stage == 1.00:

                        # v_2_ =  v if v_2 = 100.00 else v_2
                        if v_2 == 100.00:
                            v_2 = v


                        qs = qs_mb[s2s[i],:]
                        drift = (qs[1] - qs[0]) * v_2
                        if drift == 0:
                            p = 0.5
                        else:
                            if responses2[i] == 1:
                                p = (2.718281828459 ** (-2 * z_2 * drift) - 1) / \
                                    (2.718281828459 ** (-2 * drift) - 1)
                            else:
                                p = 1 - (2.718281828459 ** (-2 * z_2 * drift) - 1) / \
                                    (2.718281828459 ** (-2 * drift) - 1)

                        # If one probability = 0, the log sum will be -Inf
                        p = p * (1 - p_outlier) + wp_outlier
                        if p == 0:
                            return -np.inf
                        sum_logp += log(p)

                # update Q values, regardless of pdf
                dtQ1 = qs_mb[s2s[i],responses2[i]] - qs_mf[s1s[i], responses1[i]] # delta stage 1
                qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + alfa * dtQ1 # delta update for qmf

                dtQ2 = feedbacks[i] - qs_mb[s2s[i],responses2[i]] # delta stage 2
                qs_mb[s2s[i], responses2[i]] = qs_mb[s2s[i],responses2[i]] + alfa2 * dtQ2 # delta update for qmb
                if lambda_ != 100.00: # if using eligibility trace
                    qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + lambda__ * dtQ2 


                for s_ in range(nstates):
                    for a_ in range(2):
                        if (s_ is not s2s[i]) or (a_ is not responses2[i]):
                            # qs_mb[s_, a_] = qs_mb[s_, a_] * (1-gamma)
                            qs_mb[s_,a_] *= (1-gamma__)

                for s_ in range(comb(nstates,2,exact=True)):
                    for a_ in range(2):
                        if (s_ is not s1s[i]) or (a_ is not responses1[i]):
                            qs_mf[s_,a_] *= (1-gamma_)

                counter[s1s[i]] += 1
    return sum_logp



# JY added on 2022-10-01 for modeling ndt as a function of uncertainty & entropy
def wiener_like_rlddm_uncertainty(np.ndarray[double, ndim=1] x1, # 1st-stage RT
                      np.ndarray[double, ndim=1] x2, # 2nd-stage RT
                      np.ndarray[long,ndim=1] s1, # 1st-stage state
                      np.ndarray[long,ndim=1] s2, # 2nd-stage state
                      np.ndarray[long, ndim=1] response1,
                      np.ndarray[long, ndim=1] response2,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      np.ndarray[int, ndim=1] bandit_type,
                      double q, 
                      double alpha, double pos_alpha,
                      double alpha_pos, double alpha_neu, double alpha_neg, # YC added for aversive outcomes, 01-19-24

                      # double w,
                      double gamma, double gamma2,
                      double lambda_,

                      double v0, double v1, double v2,
                      double v, # don't use second stage
                      double sv,
                      double a,
                      double z0, double z1, double z2,
                      double z,
                      double sz,
                      double t,
                      int nstates,
                      # double v_qval, double z_qval,
                      double v_interaction, double z_interaction,
                      double two_stage,

                      double a_2,
                      double z_2,
                      double t_2,
                      double v_2,
                      double sz2,
                      double st2,
                      double sv2,
                      double alpha2,
                      double w, double w2, double z_scaler, double z_scaler_2,
                      double z_sigma, double z_sigma2,
                      double window_start, double window_size,
                      double beta_ndt, double beta_ndt2, double beta_ndt3, double beta_ndt4,
                      double model_unc_rep, double mem_unc_rep, double unc_hybrid, double w_unc,


                      double st,

                      double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0,
                      ):


    # cdef double a = 1
    # cdef double sz = 0
    # cdef double st = 0
    # cdef double sv = 0


    cdef Py_ssize_t size = x1.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef double alfa2

    cdef double gamma_
    cdef double gamma__
    cdef double lambda__

    cdef double w_
    cdef double w2_
    cdef double w_unc_



    cdef double dtQ1
    cdef double dtQ2

    cdef double dtq_mb
    cdef double dtq_mf

    cdef long s_
    cdef long a_
    cdef double v_
    cdef double z_
    cdef double t_
    cdef double sig
    cdef double v_2_
    cdef double z_2_
    cdef double a_2_
    cdef double t_2_


    cdef double var1
    cdef double var2
    cdef double var_tr
    cdef double var_tr_
    cdef double var_tr__
    cdef double var_val
    cdef double entropy1
    cdef double entropy2
    cdef double memory_weight_tr
    cdef double total_memory_weight_val
    cdef double alpha_b
    cdef double beta_b

    cdef double mode1_1
    cdef double mode1_2
    cdef double mode2_1
    cdef double mode2_2
    cdef double mode1
    cdef double mode2

    cdef int chosen_state


    cdef np.ndarray[double, ndim=1] x1s
    cdef np.ndarray[double, ndim=1] x2s
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses1
    cdef np.ndarray[long, ndim=1] responses2
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    cdef np.ndarray[long, ndim=1] s1s
    cdef np.ndarray[long, ndim=1] s2s

    # Added by Jungsun Yoo on 2021-11-27 for two-step tasks
    # parameters added for two-step




        # cdef np.ndarray[double, ndim=1] qs = np.array([q, q])


    cdef np.ndarray[double, ndim=2] qs_mf = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
    cdef np.ndarray[double, ndim=2] qs_mb = np.ones((nstates, 2))*q # second-stage Q-values


    # cdef np.ndarray[double, ndim=2] qs_mf_success = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
    # cdef np.ndarray[double, ndim=2] qs_mf_n = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values

    cdef np.ndarray[double, ndim=2] qs_mb_n = np.ones((nstates, 2)) * 2 # second-stage Q-values
    cdef np.ndarray[double, ndim=2] qs_mb_success = np.ones((nstates, 2))  # second-stage Q-values

    cdef np.ndarray[double, ndim=2] ndt_counter_set = np.ones((comb(nstates,2,exact=True),1))
    cdef np.ndarray[double, ndim=2] ndt_counter_ind = np.ones((nstates, 1))

    cdef np.ndarray[double, ndim=2] memory_weight_tr_set = np.ones((comb(nstates,2,exact=True),1))
    cdef np.ndarray[double, ndim=2] memory_weight_tr_ind = np.ones((nstates, 1))

    cdef np.ndarray[double, ndim=2] memory_weight_val = np.ones((nstates, 2))

    # JY added for uncertainty modeling
    cdef np.ndarray[double, ndim=2] beta_n_set = np.ones((comb(nstates,2,exact=True),1)) * 2
    cdef np.ndarray[double, ndim=2] beta_success_set = np.ones((comb(nstates,2,exact=True),1))

    cdef np.ndarray[double, ndim=2] beta_n_ind = np.ones((nstates, 1)) * 2
    cdef np.ndarray[double, ndim=2] beta_success_ind = np.ones((nstates,1))

    cdef np.ndarray[double, ndim=1] counter = np.zeros(comb(nstates,2,exact=True))


    cdef np.ndarray[long, ndim=1] planets

    cdef np.ndarray[double, ndim=1] Qmb
    cdef double dtq
    cdef double rt
    # cdef np.ndarray[double, ndim=2] Tm = np.array([[0.7, 0.3], [0.3, 0.7]]) # transition matrix
    cdef np.ndarray[double, ndim=2] Tm
    cdef np.ndarray[long, ndim=2] state_combinations = np.array(list(itertools.combinations(np.arange(nstates),2)))

    # cdef double transition_priors = np.ones((comb(nstates,2,exact=True),1)) * 0.5



    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha

    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]



        ####### 2023-08-03 INITIALIZING PARAMETERS FOR NEXT CONDITION ##### 
        # cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
        if j>0:  # if in the second condition

            qs_mf = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
            qs_mb = np.ones((nstates, 2))*q # second-stage Q-values


            # cdef np.ndarray[double, ndim=2] qs_mf_success = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
            # cdef np.ndarray[double, ndim=2] qs_mf_n = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values

            qs_mb_n = np.ones((nstates, 2)) * 2 # second-stage Q-values
            qs_mb_success = np.ones((nstates, 2))  # second-stage Q-values

            ndt_counter_set = np.ones((comb(nstates,2,exact=True),1))
            ndt_counter_ind = np.ones((nstates, 1))

            memory_weight_tr_set = np.ones((comb(nstates,2,exact=True),1))
            memory_weight_tr_ind = np.ones((nstates, 1))

            memory_weight_val = np.ones((nstates, 2))

            # JY added for uncertainty modeling
            beta_n_set = np.ones((comb(nstates,2,exact=True),1)) * 2    # set, configural
            beta_success_set = np.ones((comb(nstates,2,exact=True),1))

            beta_n_ind = np.ones((nstates, 1)) * 2  # each planet as independent, piecemeal
            beta_success_ind = np.ones((nstates,1))

            counter = np.zeros(comb(nstates,2,exact=True))



        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses1 = response1[split_by == s]
        responses2 = response2[split_by == s]
        x1s = x1[split_by == s]
        x2s = x2[split_by == s]
        s1s = s1[split_by == s]
        s2s = s2[split_by == s]

        # YC added 01-19-24
        bandit_types = bandit_type[split_by==s]

        s_size = x1s.shape[0]
        qs_mf[:,0] = q
        qs_mf[:,1] = q

        qs_mb[:,0] = q
        qs_mb[:,1] = q

        if alpha != 100.00:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # YC added for aversive outcomes, 01-19-24
        if alpha_pos != 100.00:
            alfa_pos = (2.718281828459**alpha_pos) / (1 + 2.718281828459**alpha_pos)
        if alpha_neu != 100.00:
            alfa_neu = (2.718281828459**alpha_neu) / (1 + 2.718281828459**alpha_neu)
        if alpha_neg != 100.00:
            alfa_neg = (2.718281828459**alfa_neg) / (1 + 2.718281828459**alfa_neg)

        if gamma != 100.00:
            gamma_ = (2.718281828459**gamma) / (1 + 2.718281828459**gamma)
        if gamma2 != 100.00:
            gamma__ = (2.718281828459**gamma2) / (1 + 2.718281828459**gamma2)
        else:
            gamma__ = gamma_
        if alpha2 != 100.00:
            alfa2 = (2.718281828459**alpha2) / (1 + 2.718281828459**alpha2)
        else:
            alfa2 = alfa
        if lambda_ != 100.00:
            lambda__ = (2.718281828459**lambda_) / (1 + 2.718281828459**lambda_)
        if w != 100.00:
            w_ = (2.718281828459**w) / (1 + 2.718281828459**w)
        if w2 != 100.00:
            w2_ = (2.718281828459**w2) / (1 + 2.718281828459**w2)
        if w_unc != 0.00:
            w_unc_ = (2.718281828459 ** w_unc) / (1 + 2.718281828459 ** w_unc)








        # if beta_ndt != 100.00:
        #     beta_ndt = (2.718281828459**beta_ndt) / (1 + 2.718281828459**beta_ndt)
        #
        # if beta_ndt2 != 100.00:
        #     beta_ndt2 = (2.718281828459**beta_ndt2) / (1 + 2.718281828459**beta_ndt2)


        # don't calculate pdf for first trial but still update q
        # if feedbacks[0] > qs[responses[0]]:
            # alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        # else:
            # alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # # received on current trial.
        # qs[responses[0]] = qs[responses[0]] + \
        #     alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        # print(window_size, window_start)
        for i in range(0, s_size):
            if window_start <= i < window_start + window_size:  # and (window_start <= i < window_start+window_size):
                if counter[s1s[i]] > 0 and x1s[i]>0.15:
                # proceed with pdf only if 1) the current 1st-stage state have been updated and 2) "plausible" RT (150 ms)

                    # 1st stage
                    planets = state_combinations[s1s[i]]
                    curr_bandit_type = bandit_types[i]  # YC added, 01-19-24

                    # Transition matrix
                    # Tm = np.array([[transition_priors[s1s[i]], 1-transition_priors[s1s[i]]], [1 - transition_priors[s1s[i]], transition_priors[s1s[i]]]])
                    if unc_hybrid == 0.00: # if don't use hybrid
                        if model_unc_rep == 1: # if ind
                            #// Tm = np.array([[mode_beta(alphaf(beta_success_ind[planets[0]]), betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]])),
                            #//                 mode_beta(alphaf(beta_success_ind[planets[1]]), betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]]))],
                            #//                [mode_beta(alphaf(beta_success_ind[planets[1]]), betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]])),
                            #//                 mode_beta(alphaf(beta_success_ind[planets[0]]), betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]]))]])

                            Tm = np.array([[mode_beta(alphaf(beta_success_ind[planets[0]]), betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]])),
                                            1 - mode_beta(alphaf(beta_success_ind[planets[0]]), betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]]))],
                                           [1 - mode_beta(alphaf(beta_success_ind[planets[1]]), betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]])),
                                            mode_beta(alphaf(beta_success_ind[planets[1]]), betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]]))]])


                        elif model_unc_rep == -1: # if set
                            Tm = np.array([[mode_beta(alphaf(beta_success_set[s1s[i]]), betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])),
                                            1-mode_beta(alphaf(beta_success_set[s1s[i]]), betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]]))],
                                           [1-mode_beta(alphaf(beta_success_set[s1s[i]]), betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])),
                                            mode_beta(alphaf(beta_success_set[s1s[i]]), betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]]))]])
                        else: # if don't model transition matrix
                            Tm = np.array([[0.7, 0.3], [0.3, 0.7]])  # transition matrix
                    elif unc_hybrid == 1.00: # bellman ind, uncertainty set
                        Tm = np.array([[mode_beta(alphaf(beta_success_ind[planets[0]]),
                                                  betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]])),
                                        1 - mode_beta(alphaf(beta_success_ind[planets[0]]),
                                                  betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]]))],
                                       [1 - mode_beta(alphaf(beta_success_ind[planets[1]]),
                                                  betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]])),
                                        mode_beta(alphaf(beta_success_ind[planets[1]]),
                                                  betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]]))]])
                    elif unc_hybrid == 2.00: # bellman set, uncertainty ind:
                        Tm = np.array([[mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                  betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])),
                                        1 - mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                      betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]]))],
                                       [1 - mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                      betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])),
                                        mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                  betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]]))]])
                    elif unc_hybrid == 3.00: # both simulaneously
                        # ind
                        Tm = np.array([[mode_beta(alphaf(beta_success_ind[planets[0]]),
                                                  betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]])),
                                        1 - mode_beta(alphaf(beta_success_ind[planets[0]]),
                                                  betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]]))],
                                       [1 - mode_beta(alphaf(beta_success_ind[planets[1]]),
                                                  betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]])),
                                        mode_beta(alphaf(beta_success_ind[planets[1]]),
                                                  betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]]))]])
                        # set
                        Tm += np.array([[mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                  betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])),
                                        1 - mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                      betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]]))],
                                       [1 - mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                      betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])),
                                        mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                  betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]]))]])
                        # average
                        Tm /= 2
                    elif unc_hybrid == 4.00:
                        Tm = (1-w_unc_) * np.array([[mode_beta(alphaf(beta_success_ind[planets[0]]),
                                                  betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]])),
                                        1 - mode_beta(alphaf(beta_success_ind[planets[0]]),
                                                  betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]]))],
                                       [1 - mode_beta(alphaf(beta_success_ind[planets[1]]),
                                                  betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]])),
                                        mode_beta(alphaf(beta_success_ind[planets[1]]),
                                                  betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]]))]]) + \
                             w_unc_ * np.array([[mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                  betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])),
                                        1 - mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                      betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]]))],
                                       [1 - mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                      betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])),
                                        mode_beta(alphaf(beta_success_set[s1s[i]]),
                                                  betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]]))]])


                    if beta_ndt2 != 0.00: # if we estimate value with uncertainty
                        mode1_1 = mode_beta(alphaf(qs_mb_success[planets[0],0]),
                                            betaf(qs_mb_n[planets[0],0], qs_mb_success[planets[0],0]))
                        mode1_2 = mode_beta(alphaf(qs_mb_success[planets[0],1]),
                                            betaf(qs_mb_n[planets[0],1], qs_mb_success[planets[0],1]))
                        mode2_1 = mode_beta(alphaf(qs_mb_success[planets[1], 0]),
                                            betaf(qs_mb_n[planets[1], 0], qs_mb_success[planets[1], 0]))
                        mode2_2 = mode_beta(alphaf(qs_mb_success[planets[1], 1]),
                                            betaf(qs_mb_n[planets[1], 1], qs_mb_success[planets[1], 1]))

                        Qmb = np.dot(Tm, [np.max([mode1_1, mode1_2]), np.max([mode2_1, mode2_2])])

                    else:
                        Qmb = np.dot(Tm, [np.max(qs_mb[planets[0], :]), np.max(qs_mb[planets[1], :])])

                    # dtq_mb = Qmb[1] - Qmb[0]  # 1 is upper, 0 is lower
                    # dtq_mf = qs_mf[s1s[i], 1] - qs_mf[s1s[i], 0]





                    # cdef np.ndarray[double, ndim=2] Tm = np.array([[0.7, 0.3], [0.3, 0.7]])  # transition matrix

                    # planet 1 mode


                    # Qmb = np.dot(Tm, [np.max(qs_mb[planets[0],:]), np.max(qs_mb[planets[1],:])])
                    # qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial

                    # dtq = qs[1] - qs[0]
                    dtq_mb = Qmb[1] - Qmb[0] # 1 is upper, 0 is lower
                    dtq_mf = qs_mf[s1s[i],1] - qs_mf[s1s[i],0]
                    if v == 100.00: # if v_reg
                        v_ = v0 + (dtq_mb * v1) + (dtq_mf * v2) + (v_interaction * dtq_mb * dtq_mf)

                    else: # if don't use v_reg:
                        # if v_qval == 0: # use both qmb and qmf
                        if w != 100.00: # if using w parameter:
                            qs = w_ * Qmb + (1-w_) * qs_mf[s1s[i],:] # Update for 1st trial
                            dtq = qs[1] - qs[0]
                            v_ = dtq * v
                        else:

                            v_ = dtq_mb * v # for uncertainty modeling: use just mb for now

                    # if z == 0.5: # if use z_reg:
                    # if w2 == 100.00: # if use z_reg
                    if (z0 != 0.00) or (z1 != 0.00) or (z2 != 0.00) or (z_interaction != 0.00): # if use regression
                        # Transform regression parameters so that >0
                        z_ = z0 + (dtq_mb * z1) + (dtq_mf * z2) + (z_interaction * dtq_mb * dtq_mf)
                        sig = 1/(1+np.exp(-z_))
                    else: # if don't use z_reg:
                        # if w2 != 100.00:
                        if w2 != 100.00:   # use w parameter for starting point bias
                            qs = w2_ * Qmb + (1-w2_) * qs_mf[s1s[i],:] # Update for 1st trial
                            dtq = qs[1] - qs[0]
                            z_ = dtq * z_scaler
                            sig = 1 / (1 + np.exp(-z_))
                        else:
                            # 2022-12-28: This is the part that's being used for most recent models (model 57, etc)
                            # Changes to be made as of 12-28: remove sigmoid and multiplication with a
                            if z_scaler != 100.00: # if using z_scaler but don't use w2 nor z0,z1,z2,z_int
                                z_ = dtq_mb * z_scaler + z_sigma
                                # sig = 1 / (1 + exp(-z_)) 
                                # sig *= a
                                sig = z_
                            else:
                                sig = z # 0.5



                    rt = x1s[i]

                    # Modeling ndt
                    # 1. Model uncertainty of the transition matrix
                    # 1) Transition uncertainty (common/rare)
                    # variance for beta distribution of lower boundary
                    # _, var1, _, _ = beta.stats(alphaf(beta_success[planets[0]]), betaf(beta_n[planets[0]], beta_success[planets[0]]), moments='mvsk')
                    # # variance for beta distribution of upper boundary
                    # _, var2, _, _ = beta.stats(alphaf(beta_success[planets[1]]),
                    #                            betaf(beta_n[planets[1]], beta_success[planets[1]]),
                    #                            moments='mvsk')
                    if beta_ndt != 0.00:
                        if unc_hybrid == 0.00:
                            if model_unc_rep == 1: # ind

                                # 1. Get variance of beta (transition)
                                alpha_b = alphaf(beta_success_ind[planets[0]])
                                beta_b = betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]])
                                var_tr = var_beta(alpha_b, beta_b)

                                alpha_b = alphaf(beta_success_ind[planets[1]])
                                beta_b = betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]])
                                var_tr += var_beta(alpha_b, beta_b)
                                var_tr /= 2

                                # 2. Get variance of beta (value)
                                # alpha_b = alphaf()

                            elif model_unc_rep == -1: # set

                                # 1. Get variance of beta (transition)
                                alpha_b = alphaf(beta_success_set[s1s[i]])
                                beta_b = betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])
                                var_tr = var_beta(alpha_b, beta_b)
                        elif unc_hybrid == 1.00: # bellman ind, uncertainty set
                            alpha_b = alphaf(beta_success_set[s1s[i]])
                            beta_b = betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])
                            var_tr = var_beta(alpha_b, beta_b)
                        elif unc_hybrid == 2.00: # bellman set, uncertainty ind
                            # 1. Get variance of beta (transition)
                            alpha_b = alphaf(beta_success_ind[planets[0]])
                            beta_b = betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]])
                            var_tr = var_beta(alpha_b, beta_b)

                            alpha_b = alphaf(beta_success_ind[planets[1]])
                            beta_b = betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]])
                            var_tr += var_beta(alpha_b, beta_b)
                            var_tr /= 2

                            # 2. Get variance of beta (value)
                            # alpha_b = alphaf()
                        elif unc_hybrid == 3.00: # average of set and ind
                            # first, set
                            alpha_b = alphaf(beta_success_set[s1s[i]])
                            beta_b = betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])
                            var_tr = var_beta(alpha_b, beta_b)

                            # then, add ind and then take the average
                            alpha_b = alphaf(beta_success_ind[planets[0]])
                            beta_b = betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]])
                            var_tr_ = var_beta(alpha_b, beta_b)

                            alpha_b = alphaf(beta_success_ind[planets[1]])
                            beta_b = betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]])
                            var_tr_ += var_beta(alpha_b, beta_b)
                            var_tr_ /= 2

                            var_tr += var_tr_
                            var_tr /= 2
                        elif unc_hybrid == 4.00: # regressing both (additional parameter)
                            alpha_b = alphaf(beta_success_set[s1s[i]])
                            beta_b = betaf(beta_n_set[s1s[i]], beta_success_set[s1s[i]])
                            var_tr_ = var_beta(alpha_b, beta_b)

                            # then, add ind and then take the average
                            alpha_b = alphaf(beta_success_ind[planets[0]])
                            beta_b = betaf(beta_n_ind[planets[0]], beta_success_ind[planets[0]])
                            var_tr__ = var_beta(alpha_b, beta_b)

                            alpha_b = alphaf(beta_success_ind[planets[1]])
                            beta_b = betaf(beta_n_ind[planets[1]], beta_success_ind[planets[1]])
                            var_tr__ += var_beta(alpha_b, beta_b)
                            var_tr__ /= 2

                            var_tr = w_unc_ * var_tr_ + (1-w_unc_) * var_tr__




                    else:
                        var_tr = 0

                    # 2. Subjective entropy
                    # softmax -> entropy of 2nd stage of planet 1
                    # or, actually, SIGMOID because two probabilities are independent from each other?
                    # entropy of two options in lower_boundary_planet stage 2:
                    # entropy1 = scientropy(qs_mb[planets[0], 0], qs_mb[planets[0],1])
                    # entropy of two options in upper_boundary_planet stage 2:
                    # entropy2 = scientropy(qs_mb[planets[1], 0], qs_mb[planets[1],1])
                    #
                    # entropy1 = entropyf(np.array([qs_mb[planets[0], 0], qs_mb[planets[0],1]]))
                    # entropy2 = entropyf(np.array([qs_mb[planets[1], 0], qs_mb[planets[1],1]]))

                    # 2023-01-28: good idea, but 2nd-stage value estimation should be done by Kalman filter
                    # if beta_ndt2 != 0.00:
                        # 2. Model uncertainty of the 2nd-stage value
                    #    var_val = var_beta(alphaf(qs_mb_success[planets[0],0]),
                    #                        betaf(qs_mb_n[planets[0],0], qs_mb_success[planets[0],0]))
                    #    var_val += var_beta(alphaf(qs_mb_success[planets[0],1]),
                    #                        betaf(qs_mb_n[planets[0],1], qs_mb_success[planets[0],1]))
                    #    var_val += var_beta(alphaf(qs_mb_success[planets[1], 0]),
                    #                        betaf(qs_mb_n[planets[1], 0], qs_mb_success[planets[1], 0]))
                    #    var_val += var_beta(alphaf(qs_mb_success[planets[1], 1]),
                    #                        betaf(qs_mb_n[planets[1], 1], qs_mb_success[planets[1], 1]))
                    #    var_val /= 4
                    #else: # beta_ndt2 = 0
                    #    var_val = 0 # exception handling

                    # 2023-01-28: regress with trialnum
                    if beta_ndt2 != 0.00: 
                        var_val = i # trial number; to regress out the linear effect
                    else: 
                        var_val = 0 # exception handling

                    # 
                    # commented out in 2023-01-28
                    # 3. Memory (representational) uncertainty of the transition: beta_ndt3
                    #if beta_ndt3 != 0.00:
                    #    if mem_unc_rep == 1: # ind
                    #        memory_weight_tr = (memory_weight_tr_ind[planets[0], 0] + memory_weight_tr_ind[planets[1], 0])/2
                    #    elif mem_unc_rep == -1: # set
                    #        memory_weight_tr = memory_weight_tr_set[s1s[i], 0]
                    #else: # beta_ndt3 = 0
                    #    memory_weight_tr = 0

                    # 2023-01-28: regress with configural use
                    # just keep the variable names for now
                    if beta_ndt3 != 0.00: 
                        memory_weight_tr = w_unc
                    else: 
                        memory_weight_tr = 0

                    # commented out in 2023-01-28    
                    # 4. Memory (representational) uncertainty of value: beta_ndt4
                    # if beta_ndt4 != 0.00:
                        # memory_weight_val =
                        # total_memory_weight_val = memory_weight_val[planets[0],0] + memory_weight_val[planets[0],1] + \
                        #                           memory_weight_val[planets[1],0] + memory_weight_val[planets[1],1]
                    #    total_memory_weight_val = memory_weight_val[planets[0],0]
                    #    total_memory_weight_val += memory_weight_val[planets[0],1]
                    #    total_memory_weight_val += memory_weight_val[planets[1],0]
                    #    total_memory_weight_val += memory_weight_val[planets[1],1]
                    #    total_memory_weight_val /= 4
                    # else:
                    #    total_memory_weight_val = 0

                    # 2023-01-28: regress with piecemeal use
                    if beta_ndt4 != 0.00: 
                        total_memory_weight_val = 1-w_unc
                    else: 
                        total_memory_weight_val = 0



                    # sum together
                    t_ = t + \
                         (beta_ndt * var_tr) + \
                         (beta_ndt2 * var_val) + \
                         (beta_ndt3 * memory_weight_tr) + \
                         (beta_ndt4 * total_memory_weight_val)
                        # 1. Model uncertainty: transition
                        # 2. Model uncertainty: value
                        # 3. Memory uncertainty: transition
                        # 4. Memory uncertainty: value

                        # JUST USE 1 AND 3 FOR NOW (Just transition matrix for now)



                #beta_ndt3*np.log(ndt_counter_set[s1s[i],0]) + t

                    # t_ = ((np.log(ndt_counter_ind[planets[0],0]) + np.log(ndt_counter_ind[planets[1],0]))/2)*beta_ndt + \
                    #      np.log(ndt_counter_set[s1s[i],0])*beta_ndt2 + \
                    #      t

                    p = full_pdf(rt, v_, sv, a, sig,
                                 sz, t_, st, err, n_st, n_sz, use_adaptive, simps_err)
                    # If one probability = 0, the log sum will be -Inf
                    p = p * (1 - p_outlier) + wp_outlier
                    if p == 0:
                        return -np.inf
                    sum_logp += log(p)


                    # # # 2nd stage
                    if two_stage == 1.00:


                        v_2_ = v if v_2==100.00 else v_2
                        a_2_ = a if a_2 == 100.00 else a_2

                        # CONFIGURE Z_2 USING Z_SIGMA AND V HERE!!!
                        # if z_sigma ==100.00: # if don't use 1st-stage dependent drift rate
                        #     # z_2_ = z if z_2 == 0.5 else z_2
                        #     z_2_ = z_2
                        # else: # if use 1st-stage dependent dr
                        #
                        #     if z_sigma2 == 100.00: # don't use baseline
                        #         z_2_ = 1 / (1 + np.exp(-v_*z_sigma))
                        #     else: # use baseline
                        #         z_2_ = 1 / (1 + np.exp(-(v_ * z_sigma + z_sigma2)))

                        t_2_ = t if t_2 == 100.00 else t_2
                        if beta_ndt2 != 0.00: # if we estimate value with uncertainty
                            mode1 = mode_beta(alphaf(qs_mb_success[s2s[i], 0]),
                                      betaf(qs_mb_n[s2s[i], 0], qs_mb_success[s2s[i], 0]))
                            mode2 = mode_beta(alphaf(qs_mb_success[s2s[i], 1]),
                                      betaf(qs_mb_n[s2s[i], 1], qs_mb_success[s2s[i], 1]))
                            qs = [mode1, mode2]
                        else:
                            qs = qs_mb[s2s[i], :]
                            # dtq = qs[1] - qs[0]



                        dtq = qs[1] - qs[0]
                        rt = x2s[i]
                        # z_ = dtq * z_scaler
                        if z_scaler_2 == 100.00:
                            # sig = 0.5
                            sig = z_2
                        else:
                            # 2022-12-28: This is the part that's being used for most recent models (model 57, etc)
                            # Changes to be made as of 12-28: remove sigmoid and multiplication with a

                            # z_ = dtq_mb * z_scaler + z_sigma
                            # # sig = 1 / (1 + exp(-z_))
                            # # sig *= a
                            # sig = z_

                            z_2_ = dtq * z_scaler_2 + z_sigma2
                            # sig = 1 / (1 + exp(-z_2_))
                            # sig *= a_2_
                            sig = z_2_
                        # p = full_pdf(rt, v_, sv, a, sig * a,
                        #              sz, t_, st, err, n_st, n_sz, use_adaptive, simps_err)
                        p = full_pdf(rt, (dtq * v_2_), sv2, a_2_, sig, sz2, t_2_, st2, err, n_st, n_sz, use_adaptive,
                                     simps_err)
                        # p = full_pdf(rt, (dtq * v_2_), sv, a_2_, z_2_, sz, t_2_, st, err, n_st, n_sz, use_adaptive, simps_err)
                        # If one probability = 0, the log sum will be -Inf
                        p = p * (1 - p_outlier) + wp_outlier
                        if p == 0:
                            return -np.inf
                        sum_logp += log(p)
            # update Q values, regardless of pdf

            # ndt_counter_set[s1s[i], 0] += 1
            # ndt_counter_ind[s2s[i], 0] += 1

            # just update 1st-stage MF values if estimating
            # 2023-02-22: Revive the QMF value update
            # Just assume single learning rate for two stages for now?

            # YC added, 01-19-24
            if alpha_pos!=100.00:
                if curr_bandit_type==0:
                    alfa = alfa_pos
                    alfa2 = alfa_pos
                elif curr_bandit_type==1:
                    alfa = alfa_neu
                    alfa2 = alfa_neu
                elif curr_bandit_type==2:
                    alfa = alfa_neg
                    alfa2 = alfa_neg

            if w != 100.00: # if so, we need to update both Qmb and Qmf
                if alpha != 100.00 or alpha_pos!=100.00: # there should be at least one learning rate to do this (alpha), whether using same or separate lr
            
                    dtQ1 = qs_mb[s2s[i], responses2[i]] - qs_mf[s1s[i], responses1[i]]  # delta stage 1
                    qs_mf[s1s[i], responses1[i]] = qs_mf[
                                                    s1s[i], responses1[i]] + alfa * dtQ1  # delta update for qmf
                # if alpha2 != 100.00:
                    dtQ2 = feedbacks[i] - qs_mb[s2s[i], responses2[i]]  # delta stage 2
                    qs_mb[s2s[i], responses2[i]] = qs_mb[
                                                    s2s[i], responses2[i]] + alfa2 * dtQ2  # delta update for qmb

            else: # if not using both Qmb and Qmf, just update Qmb
                if alpha != 100.00 or alpha_pos!=100.00:
                    dtQ2 = feedbacks[i] - qs_mb[s2s[i], responses2[i]]  # delta stage 2
                    qs_mb[s2s[i], responses2[i]] = qs_mb[
                                                    s2s[i], responses2[i]] + alfa * dtQ2  # delta update for qmb

            if lambda_ != 100.00:  # if using eligibility trace
                qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[
                    i]] + lambda__ * dtQ2  # eligibility trace

            # Updating ndt-related variables, regardless of pdf
            # Updating encountraces
            # ndt_counter_ind[s2s[i], 0] += 1
            # ndt_counter_set[s1s[i], 0] += 1

            # Updating memory for transition matrix

            # memory_weight_tr_ind *= (1 - gamma__) # forgetting for all
            # memory_weight_tr_set *= (1 - gamma__) # forgetting for all
            # memory_weight_tr_ind[s2s[i], 0] += 1
            # memory_weight_tr_set[s1s[i], 0] += 1
            if beta_ndt3 != 0.00:
                if mem_unc_rep == 1: # ind
                    for s_ in range(nstates):
                        if s_ is not s2s[i]:
                            memory_weight_tr_ind[s_,0] *= (1-gamma__)
                if mem_unc_rep == -1: # set
                    for s_ in range(comb(nstates,2,exact=True)):
                        if s_ is not s1s[i]:
                            memory_weight_tr_set[s_,0] *= (1-gamma__)


            # if mem_unc_rep == 1:  # ind
            #     memory_weight_tr = (memory_weight_tr_ind[planets[0], 0] + memory_weight_tr_ind[planets[1], 0]) / 2
            # elif mem_unc_rep == -1:  # set
            #     memory_weight_tr = memory_weight_tr_set[s1s[i], 0]



            # Updating memory for values
            if beta_ndt4 != 0.00:
                for s_ in range(nstates):
                    for a_ in range(2):
                        if (s_ == s2s[i]) and (a_ == responses2[i]):
                            pass
                        else:
                            memory_weight_val[s_, a_] *= (1-gamma_)
            else: # just discount pointwise values
                # memory decay for unexperienced options in this trial
                if w != 100.00: # should update both Qmf and Qmb
                    for s_ in range(nstates):
                        for a_ in range(2):
                            if (s_ is not s2s[i]) or (a_ is not responses2[i]):
                                # qs_mb[s_, a_] = qs_mb[s_, a_] * (1-gamma)
                                qs_mb[s_, a_] *= (1 - gamma__)
                                
                # 2023-02-22: Revive the QMF decay
                    for s_ in range(comb(nstates, 2, exact=True)):
                        for a_ in range(2):
                            if (s_ is not s1s[i]) or (a_ is not responses1[i]):
                                qs_mf[s_, a_] *= (1 - gamma_)
                else: # don't have to update Qmf
                    for s_ in range(nstates):
                        for a_ in range(2):
                            if (s_ is not s2s[i]) or (a_ is not responses2[i]):
                                # qs_mb[s_, a_] = qs_mb[s_, a_] * (1-gamma)
                                qs_mb[s_, a_] *= (1 - gamma_)                    

            # memory_weight_val *= (1 - gamma_) # forgetting for all
            # memory_weight_val[s2s[i], responses2[i]] += 1

            # Updating common/rare transition uncertainty (beta parameters)
            planets = state_combinations[s1s[i]]
            chosen_state = planets[responses1[i]]
            beta_n_ind[chosen_state] += 1
            beta_n_set[s1s[i]] += 1
            if chosen_state == s2s[i]:
                beta_success_ind[chosen_state] += 1
                beta_success_set[s1s[i]] += 1
            # Update mode of beta



            # update Q values, regardless of pdf
            # just update 1st-stage MF values if estimating
            # dtQ1 = qs_mb[s2s[i],responses2[i]] - qs_mf[s1s[i], responses1[i]] # delta stage 1
            # qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + alfa * dtQ1 # delta update for qmf

            # dtQ2 = feedbacks[i] - qs_mb[s2s[i],responses2[i]] # delta stage 2
            # qs_mb[s2s[i], responses2[i]] = qs_mb[s2s[i],responses2[i]] + alfa2 * dtQ2 # delta update for qmb
            # if lambda_ != 100.00: # if using eligibility trace
            #     qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + lambda__ * dtQ2 # eligibility trace

            # Replace TD updating rule with beta (bayesian) updating
            qs_mb_n[s2s[i], responses2[i]] += 1
            if feedbacks[i] == 1:
                qs_mb_success[s2s[i], responses2[i]] += 1


            # memory decay for unexperienced options in this trial

            # for s_ in range(nstates):
            #     for a_ in range(2):
            #         if (s_ is not s2s[i]) or (a_ is not responses2[i]):
            #             # qs_mb[s_, a_] = qs_mb[s_, a_] * (1-gamma)
            #             # qs_mb[s_,a_] *= (1-gamma__)
            #             qs_mb[s_, a_] *= (1 - gamma_)
            #
            # for s_ in range(comb(nstates,2,exact=True)):
            #     for a_ in range(2):
            #         if (s_ is not s1s[i]) or (a_ is not responses1[i]):
            #             qs_mf[s_,a_] *= (1-gamma_)

            counter[s1s[i]] += 1

    return sum_logp

# # JY added on 2022-01-03 for simultaneous regression on two-step tasks
# def wiener_like_rlddm_2step_thinkact(np.ndarray[double, ndim=1] x1, # 1st-stage RT                      
#                       np.ndarray[double, ndim=1] x2, # 2nd-stage RT                     
#                       np.ndarray[long, ndim=1] isleft1, # whether left response 1st-stage, 
#                       np.ndarray[long, ndim=1] isleft2, # whether left response 2nd-stage  
#                       np.ndarray[long,ndim=1] s1, # 1st-stage state
#                       np.ndarray[long,ndim=1] s2, # 2nd-stage state
#                       np.ndarray[long, ndim=1] response1,
#                       np.ndarray[long, ndim=1] response2,
#                       np.ndarray[double, ndim=1] feedback,
#                       np.ndarray[long, ndim=1] split_by,
#                       double q, double alpha, double pos_alpha, 
#                       double gamma, 
#                       double lambda_, 

#                       double v0, double v1, double v2, 
#                       double v, # don't use second stage
#                       # double sv, 
#                       double a, 
#                       double z0, double z1, double z2,
#                       double z, 
#                       # double sz, 
#                       double t,
#                       int nstates,
#                       double qval,
#                       double two_stage,

#                       double a_2,
#                       double z_2, 
#                       double t_2,
#                       double v_2,
#                       double alpha2,
#                       double w,
#                       # double st, 

#                       double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
#                       double p_outlier=0, double w_outlier=0,
#                       ):

#     if a==100.00: # if fixed threshold
#         a = 1

#     cdef double sz = 0
#     cdef double st = 0
#     cdef double sv = 0

#     cdef Py_ssize_t size = x1.shape[0]
#     cdef Py_ssize_t i, j
#     cdef Py_ssize_t s_size
#     cdef int s
#     cdef double p
#     cdef double sum_logp = 0
#     cdef double wp_outlier = w_outlier * p_outlier
#     cdef double alfa
#     cdef double pos_alfa
#     cdef double alfa2

#     cdef double gamma_
#     cdef double lambda__

#     # cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
#     cdef np.ndarray[double, ndim=2] qs_mf = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
#     cdef np.ndarray[double, ndim=2] qs_mb = np.ones((nstates, 2))*q # second-stage Q-values

#     cdef double dtQ1
#     cdef double dtQ2

#     cdef double dtq_mb
#     cdef double dtq_mf

#     cdef long s_
#     cdef long a_ 
#     cdef double v_
#     cdef double z_
#     cdef double sig

#     cdef np.ndarray[double, ndim=1] x1s
#     cdef np.ndarray[double, ndim=1] x2s
#     cdef np.ndarray[double, ndim=1] feedbacks
#     cdef np.ndarray[long, ndim=1] responses1
#     cdef np.ndarray[long, ndim=1] responses2
#     cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)    

#     cdef np.ndarray[long, ndim=1] s1s
#     cdef np.ndarray[long, ndim=1] s2s   
#     cdef np.ndarray[long, ndim=1] isleft1s
#     cdef np.ndarray[long, ndim=1] isleft2s        

#     # Added by Jungsun Yoo on 2021-11-27 for two-step tasks
#     # parameters added for two-step

#     cdef np.ndarray[long, ndim=1] planets
#     cdef np.ndarray[double, ndim=1] counter = np.zeros(comb(nstates,2,exact=True))
#     cdef np.ndarray[double, ndim=1] Qmb
#     cdef double dtq
#     cdef double rt
#     cdef np.ndarray[double, ndim=2] Tm = np.array([[0.7, 0.3], [0.3, 0.7]]) # transition matrix
#     cdef np.ndarray[long, ndim=2] state_combinations = np.array(list(itertools.combinations(np.arange(nstates),2)))

#     float delta_t = 0.001, # timesteps fraction of seconds
#     # cdef float thinking_t
#     cdef float t_particle
#     cdef float max_t    

#     t_particle = 0.0 # reset time
#     while t_particle <= max_t:




#         t_particle += delta_t


#     if not p_outlier_in_range(p_outlier):
#         return -np.inf

#     if pos_alpha==100.00:
#         pos_alfa = alpha
#     else:
#         pos_alfa = pos_alpha

#     # unique represent # of conditions
#     for j in range(unique.shape[0]):
#         s = unique[j]
#         # select trials for current condition, identified by the split_by-array
#         feedbacks = feedback[split_by == s]
#         responses1 = response1[split_by == s]
#         responses2 = response2[split_by == s]
#         x1s = x1[split_by == s]
#         x2s = x2[split_by == s]
#         s1s = s1[split_by == s]
#         s2s = s2[split_by == s]

#         isleft1s = isleft1[split_by == s]
#         isleft2s = isleft2[split_by == s]

#         s_size = x1s.shape[0]
#         qs_mf[:,0] = q
#         qs_mf[:,1] = q

#         qs_mb[:,0] = q
#         qs_mb[:,1] = q

#         alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)
#         gamma_ = (2.718281828459**gamma) / (1 + 2.718281828459**gamma)
#         if alpha2 != 100.00:
#             alfa2 = (2.718281828459**alpha2) / (1 + 2.718281828459**alpha2)
#         else:
#             alfa2 = alfa            
#         if lambda_ != 100.00:
#             lambda__ = (2.718281828459**lambda_) / (1 + 2.718281828459**lambda_)
#         if w != 100.00:
#             w = (2.718281828459**w) / (1 + 2.718281828459**w)

#         # loop through all trials in current condition
#         for i in range(0, s_size):

#             if counter[s1s[i]] > 0 and x1s[i]>0.15: 
#             # proceed with pdf only if 1) the current 1st-stage state have been updated and 2) "plausible" RT (150 ms)

#                 for 


#                 # 1st stage
#                 planets = state_combinations[s1s[i]]
#                 Qmb = np.dot(Tm, [np.max(qs_mb[planets[0],:]), np.max(qs_mb[planets[1],:])])
#                 # qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial 

#                 # dtq = qs[1] - qs[0]
#                 dtq_mb = Qmb[0] - Qmb[1]
#                 dtq_mf = qs_mf[s1s[i],0] - qs_mf[s1s[i],1]
#                 if v == 100.00: # if v_reg
#                     if qval == 0:
#                         v_ = v0 + (dtq_mb * v1) + (dtq_mf * v2) # use both Qvals
#                     elif qval == 1: # just mb
#                         v_ = v0 + (dtq_mb * v1)
#                     elif qval == 2: 
#                         v_ = v0 + (dtq_mf * v2) # just qmf
#                 else: # if don't use v_reg:
#                     if qval == 0: # use both qmb and qmf
#                         qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial 
#                         dtq = qs[1] - qs[0]
#                         v_ = dtq * v
#                     elif qval == 1:
#                         v_ = dtq_mb * v
#                     elif qval==2:
#                         v_ = dtq_mf * v 

#                 if z0 != 100.00: # if use z_reg:
#                     if qval == 0:
#                         z_ = z0 + (dtq_mb * z1) + (dtq_mf * z2) # use both Qvals
#                     elif qval == 1: # just mb
#                         z_ = z0 + (dtq_mb * z1)
#                     elif qval == 2: 
#                         z_ = z0 + (dtq_mf * z2) # just qmf
#                     sig = 1/(1+np.exp(-z_))
#                 else: # if don't use z_reg:
#                     sig = z


#                 # z_ = z0 + (dtq_mb * z1) + (dtq_mf * z2)
#                 # sig =  np.where(z_<0, np.exp(z_)/(1+np.exp(z_)), 1/(1+np.exp(-z_))) # perform sigmoid on z to bound it [0,1]
#                 # sig = 1/(1+np.exp(-z_))
                
#                 rt = x1s[i]
#                 # if qs[0] > qs[1]:
#                 #     dtq = -dtq
#                 #     rt = -rt

#                 if isleft1s[i] == 0: # if chosen right
#                     rt = -rt
#                     v_ = -v_

#                 # p = full_pdf(rt, (dtq * v), sv, a, z,
#                 #              sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
#                 p = full_pdf(rt, v_, sv, a, sig,
#                              sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)                
#                 # If one probability = 0, the log sum will be -Inf
#                 p = p * (1 - p_outlier) + wp_outlier
#                 if p == 0:
#                     return -np.inf
#                 sum_logp += log(p)


#                 # # # 2nd stage
#                 if two_stage == 1.00:

#                     v_2_ = v if v_2==100.00 else v_2
#                     a_2_ = a if a_2 == 100.00 else a_2 
#                     z_2_ = z if z_2 == 0.5 else z_2 
#                     t_2_ = t if t_2 == 100.00 else t_2                                        

#                     qs = qs_mb[s2s[i],:]
#                     dtq = qs[1] - qs[0]
#                     rt = x2s[i]
#                     if isleft2s[i] == 0:
#                     # if qs[0] > qs[1]:
#                         dtq = -dtq
#                         rt = -rt           
#                     p = full_pdf(rt, (dtq * v_2_), sv, a_2_, z_2_, sz, t_2_, st, err, n_st, n_sz, use_adaptive, simps_err)
#                     # If one probability = 0, the log sum will be -Inf
#                     p = p * (1 - p_outlier) + wp_outlier
#                     if p == 0:
#                         return -np.inf
#                     sum_logp += log(p)


#             # update Q values, regardless of pdf    


#             # get learning rate for current trial. if pos_alpha is not in
#             # include it will be same as alpha so can still use this
#             # calculation:
#             # if feedbacks[i] > qs[responses[i]]:
#             #     alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
#             # else:


#             # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
#             # received on current trial.
#             # qs[responses[i]] = qs[responses[i]] + \
#             #     alfa * (feedbacks[i] - qs[responses[i]])


#             dtQ1 = qs_mb[s2s[i],responses2[i]] - qs_mf[s1s[i], responses1[i]] # delta stage 1
#             qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + alfa * dtQ1 # delta update for qmf

#             dtQ2 = feedbacks[i] - qs_mb[s2s[i],responses2[i]] # delta stage 2 
#             qs_mb[s2s[i], responses2[i]] = qs_mb[s2s[i],responses2[i]] + alfa2 * dtQ2 # delta update for qmb
#             if lambda_ != 100.00: # if using eligibility trace
#                 qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + lambda__ * dtQ2 # eligibility trace        


#             # memory decay for unexperienced options in this trial

#             for s_ in range(nstates):
#                 for a_ in range(2):
#                     if (s_ is not s2s[i]) or (a_ is not responses2[i]):
#                         # qs_mb[s_, a_] = qs_mb[s_, a_] * (1-gamma)
#                         qs_mb[s_,a_] *= (1-gamma_)

#             for s_ in range(comb(nstates,2,exact=True)):
#                 for a_ in range(2):
#                     if (s_ is not s1s[i]) or (a_ is not responses1[i]):
#                         qs_mf[s_,a_] *= (1-gamma_)
           
#             counter[s1s[i]] += 1



#     return sum_logp



def wiener_like_rl(np.ndarray[long, ndim=1] response,
                   np.ndarray[double, ndim=1] feedback,
                   np.ndarray[long, ndim=1] split_by,
                   double q, double alpha, double pos_alpha, double v, double z,
                   double err=1e-4, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                   double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = response.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double drift
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha
        
    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        s_size = responses.shape[0]
        qs[0] = q
        qs[1] = q

        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses[0]]:
            alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        else:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # received on current trial.
        qs[responses[0]] = qs[responses[0]] + \
            alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(1, s_size):

            drift = (qs[1] - qs[0]) * v

            if drift == 0:
                p = 0.5
            else:
                if responses[i] == 1:
                    p = (2.718281828459**(-2 * z * drift) - 1) / \
                        (2.718281828459**(-2 * drift) - 1)
                else:
                    p = 1 - (2.718281828459**(-2 * z * drift) - 1) / \
                        (2.718281828459**(-2 * drift) - 1)

            # If one probability = 0, the log sum will be -Inf
            p = p * (1 - p_outlier) + wp_outlier
            if p == 0:
                return -np.inf

            sum_logp += log(p)

            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            if feedbacks[i] > qs[responses[i]]:
                alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            else:
                alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            qs[responses[i]] = qs[responses[i]] + \
                alfa * (feedbacks[i] - qs[responses[i]])
    return sum_logp


def wiener_like_multi(np.ndarray[double, ndim=1] x, v, sv, a, z, sz, t, st, double err, multi=None,
                      int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if multi is None:
        return full_pdf(x, v, sv, a, z, sz, t, st, err)
    else:
        params = {'v': v, 'z': z, 't': t, 'a': a, 'sv': sv, 'sz': sz, 'st': st}
        params_iter = copy(params)
        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]
            if abs(x[i]) != 999.:
                p = full_pdf(x[i], params_iter['v'],
                             params_iter['sv'], params_iter['a'], params_iter['z'],
                             params_iter['sz'], params_iter['t'], params_iter['st'],
                             err, n_st, n_sz, use_adaptive, simps_err)
                p = p * (1 - p_outlier) + wp_outlier
            elif x[i] == 999.:
                p = prob_ub(params_iter['v'], params_iter['a'], params_iter['z'])
            else: # x[i] == -999.
                p = 1 - prob_ub(params_iter['v'], params_iter['a'], params_iter['z'])

            sum_logp += log(p)

        return sum_logp


def wiener_like_multi_rlddm(np.ndarray[double, ndim=1] x, 
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, v, sv, a, z, sz, t, st, alpha, double err, multi=None,
                      int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t ij
    cdef Py_ssize_t s_size
    cdef double p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef int s
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])

    if multi is None:
        return full_pdf(x, v, sv, a, z, sz, t, st, err)
    else:
        params = {'v': v, 'z': z, 't': t, 'a': a, 'sv': sv, 'sz': sz, 'st': st, 'alpha':alpha}
        params_iter = copy(params)
        qs[0] = q
        qs[1] = q
        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]

            if (i != 0):
                if (split_by[i] != split_by[i-1]):
                    qs[0] = q
                    qs[1] = q

            p = full_pdf(x[i], params_iter['v'] * (qs[1] - qs[0]),
                         params_iter['sv'], params_iter['a'], params_iter['z'],
                         params_iter['sz'], params_iter[
                             't'], params_iter['st'],
                         err, n_st, n_sz, use_adaptive, simps_err)
            p = p * (1 - p_outlier) + wp_outlier
            sum_logp += log(p)

            alfa = (2.718281828459**params_iter['alpha']) / (1 + 2.718281828459**params_iter['alpha'])   
            qs[response[i]] = qs[response[i]] + alfa * (feedback[i] - qs[response[i]])

        return sum_logp


def gen_rts_from_cdf(double v, double sv, double a, double z, double sz, double t,
                     double st, int samples=1000, double cdf_lb=-6, double cdf_ub=6, double dt=1e-2):

    cdef np.ndarray[double, ndim = 1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim = 1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef int idx

    l_cdf[0] = 0
    for i from 1 <= i < size:
        pdf = full_pdf(x[i], v, sv, a, z, sz, 0, 0, 1e-4)
        l_cdf[i] = l_cdf[i - 1] + pdf

    l_cdf /= l_cdf[x.shape[0] - 1]

    cdef np.ndarray[double, ndim = 1] rts = np.empty(samples, dtype=np.double)
    cdef np.ndarray[double, ndim = 1] f = np.random.rand(samples)
    cdef np.ndarray[double, ndim = 1] delay

    if st != 0:
        delay = (np.random.rand(samples) * st + (t - st / 2.))
    for i from 0 <= i < samples:
        idx = np.searchsorted(l_cdf, f[i])
        rt = x[idx]
        if st == 0:
            rt = rt + np.sign(rt) * t
        else:
            rt = rt + np.sign(rt) * delay[i]
        rts[i] = rt
    return rts


# JY added for simulation with factorial design

        # rts = hddm.wfpt.gen_rts_from_cdf_factorial(            
        #     params["v"],
        #     params["sv"],
        #     params["a"],
        #     params["z"],
        #     params["sz"],
        #     params["t"],
        #     params["st"],
        #     # JY added on 2022-02-17 for factorial
        #     params["a2"], 
        #     params["t2"], 
        #     params["z0"], 
        #     params["z1"], 
        #     params["z2"], 
        #     params["v0"], 
        #     params["v1"], 
        #     params["v2"],
        #     # ===========
        #     size,
        #     range_[0],
        #     range_[1],
        #     dt,
        #     )

def gen_rts_from_cdf_factorial(double v, double sv, double a, double z, double sz, double t,
                     double st, 
                     double a2, double t2,                      
                     double z0, double z1, double z2, 
                     double v0, double v1, double v2, 
                     int samples=1000, double cdf_lb=-6, double cdf_ub=6, double dt=1e-2):

    cdef np.ndarray[double, ndim = 1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim = 1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef int idx

    l_cdf[0] = 0
    for i from 1 <= i < size:
        pdf = full_pdf(x[i], v, sv, a, z, sz, 0, 0, 1e-4)
        l_cdf[i] = l_cdf[i - 1] + pdf

    l_cdf /= l_cdf[x.shape[0] - 1]

    cdef np.ndarray[double, ndim = 1] rts = np.empty(samples, dtype=np.double)
    cdef np.ndarray[double, ndim = 1] f = np.random.rand(samples)
    cdef np.ndarray[double, ndim = 1] delay

    if st != 0:
        delay = (np.random.rand(samples) * st + (t - st / 2.))
    for i from 0 <= i < samples:
        idx = np.searchsorted(l_cdf, f[i])
        rt = x[idx]
        if st == 0:
            rt = rt + np.sign(rt) * t
        else:
            rt = rt + np.sign(rt) * delay[i]
        rts[i] = rt
    return rts


def wiener_like_contaminant(np.ndarray[double, ndim=1] x, np.ndarray[int, ndim=1] cont_x, double v,
                            double sv, double a, double z, double sz, double t, double st, double t_min,
                            double t_max, double err, int n_st=10, int n_sz=10, bint use_adaptive=1,
                            double simps_err=1e-8):
    """Wiener likelihood function where RTs could come from a
    separate, uniform contaminant distribution.

    Reference: Lee, Vandekerckhove, Navarro, & Tuernlinckx (2007)
    """
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef int n_cont = np.sum(cont_x)
    cdef int pos_cont = 0

    for i in prange(size, nogil=True):
        if cont_x[i] == 0:
            p = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                         n_st, n_sz, use_adaptive, simps_err)
            if p == 0:
                with gil:
                    return -np.inf
            sum_logp += log(p)
        # If one probability = 0, the log sum will be -Inf

    # add the log likelihood of the contaminations
    sum_logp += n_cont * log(0.5 * 1. / (t_max - t_min))

    return sum_logp

def gen_cdf_using_pdf(double v, double sv, double a, double z, double sz, double t, double st, double err,
                      int N=500, double time=5., int n_st=2, int n_sz=2, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    """
    generate cdf vector using the pdf
    """
    if (sv < 0) or (a <= 0 ) or (z < 0) or (z > 1) or (sz < 0) or (sz > 1) or (z + sz / 2. > 1) or \
            (z - sz / 2. < 0) or (t - st / 2. < 0) or (t < 0) or (st < 0) or not p_outlier_in_range(p_outlier):
        raise ValueError(
            "at least one of the parameters is out of the support")

    cdef np.ndarray[double, ndim = 1] x = np.linspace(-time, time, 2 * N + 1)
    cdef np.ndarray[double, ndim = 1] cdf_array = np.empty(x.shape[0], dtype=np.double)
    cdef int idx

    # compute pdf on the real line
    cdf_array = pdf_array(x, v, sv, a, z, sz, t, st, err, 0,
                          n_st, n_sz, use_adaptive, simps_err, p_outlier, w_outlier)

    # integrate
    cdf_array[1:] = integrate.cumtrapz(cdf_array)

    # normalize
    cdf_array /= cdf_array[x.shape[0] - 1]

    return x, cdf_array


def split_cdf(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] data):

    # get length of data
    cdef int N = (len(data) - 1) / 2

    # lower bound is reversed
    cdef np.ndarray[double, ndim = 1] x_lb = -x[:N][::-1]
    cdef np.ndarray[double, ndim = 1] lb = data[:N][::-1]
    # lower bound is cumulative in the wrong direction
    lb = np.cumsum(np.concatenate([np.array([0]), -np.diff(lb)]))

    cdef np.ndarray[double, ndim = 1] x_ub = x[N + 1:]
    cdef np.ndarray[double, ndim = 1] ub = data[N + 1:]
    # ub does not start at 0
    ub -= ub[0]

    return (x_lb, lb, x_ub, ub)


# NEW WITH NN-EXTENSION

#############
# Basic MLP Likelihoods
def wiener_like_nn_mlp(np.ndarray[float, ndim = 1] rt,
                       np.ndarray[float, ndim = 1] response,
                       np.ndarray[float, ndim = 1] params,
                       double p_outlier = 0,
                       double w_outlier = 0,
                       network = None):

    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t n_params = params.shape[0]
    cdef float log_p = 0
    cdef float ll_min = -16.11809

    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile(params, (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([rt, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return log_p

def wiener_like_nn_mlp_pdf(np.ndarray[float, ndim = 1] rt,
                            np.ndarray[float, ndim = 1] response,
                            np.ndarray[float, ndim = 1] params,
                            double p_outlier = 0, 
                            double w_outlier = 0,
                            bint logp = 0,
                            network = None):
    
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t n_params = params.shape[0]

    cdef np.ndarray[float, ndim = 1] log_p = np.zeros(size, dtype = np.float32)
    cdef float ll_min = -16.11809

    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile(params, (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([rt, response], axis = 1)

    # Call to network:
    if p_outlier == 0: # ddm_model
        log_p = np.squeeze(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else: # ddm_model
        log_p = np.squeeze(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    if logp == 0:
        log_p = np.exp(log_p) # shouldn't be called log_p anymore but no need for an extra array here
    return log_p


################
# Regression style likelihoods: (Can prob simplify and make all mlp likelihoods of this form)

def wiener_like_multi_nn_mlp(np.ndarray[float, ndim = 2] data,
                             double p_outlier = 0, 
                             double w_outlier = 0,
                             network = None):
                             #**kwargs):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    # Call to network:
    if p_outlier == 0: # previous ddm_model
        log_p = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    return log_p 

def wiener_like_multi_nn_mlp_pdf(np.ndarray[float, ndim = 2] data,
                                 double p_outlier = 0, 
                                 double w_outlier = 0,
                                 network = None):
                                 #**kwargs):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    # Call to network:
    if p_outlier == 0: # previous ddm_model
        log_p = np.squeeze(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.squeeze(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    return log_p

###########
# Basic CNN likelihoods

#def wiener_like_cnn_2(np.ndarray[long, ndim = 1] x, 
#                      np.ndarray[long, ndim = 1] response, 
#                      np.ndarray[float, ndim = 1] parameters,
#                      double p_outlier = 0, 
#                      double w_outlier = 0,
#                      **kwargs):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i 
#    cdef float log_p = 0
#    cdef np.ndarray[float, ndim = 2] pred = kwargs['network'](parameters)
#    #log_p = 0
#    
#    for i in range(size):
#        if response[i] == 0:
#            log_p += np.log(pred[0, 2 * x[i]] * (1 - p_outlier) + w_outlier * p_outlier)
#        else: 
#            log_p += np.log(pred[0, 2 * x[i] + 1] * (1 - p_outlier) + w_outlier * p_outlier)
#
#    # Call to network:
#    return log_p
#
#def wiener_pdf_cnn_2(np.ndarray[long, ndim = 1] x, 
#                     np.ndarray[long, ndim = 1] response, 
#                     np.ndarray[float, ndim = 1] parameters,
#                     double p_outlier = 0, 
#                     double w_outlier = 0,
#                     bint logp = 0,
#                     **kwargs):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i
#    cdef np.ndarray[float, ndim = 1] log_p = np.zeros(size, dtype = np.float32)
#    cdef np.ndarray[float, ndim = 2] pred = kwargs['network'](parameters)
#    #print(pred.shape)
#    #log_p = 0
#    for i in range(size):
#        if response[i] == 0:
#            log_p[i] += np.log(pred[0, 2 * x[i]] * (1 - p_outlier) + w_outlier * p_outlier)
#        else: 
#            log_p[i] += np.log(pred[0, 2 * x[i] + 1] * (1 - p_outlier) + w_outlier * p_outlier)
#    
#    if logp == 0:
#        log_p = np.exp(log_p)
#
#    # Call to network:
#    return log_p
#
#def wiener_like_reg_cnn_2(np.ndarray[long, ndim = 1] x, 
#                          np.ndarray[long, ndim = 1] response, 
#                          np.ndarray[float, ndim = 2] parameters,
#                          double p_outlier = 0, 
#                          double w_outlier = 0,
#                          bint logp = 0,
#                          **kwargs):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i
#    cdef float log_p = 0
#    cdef np.ndarray[float, ndim = 2] pred = kwargs['network'](parameters)
#    #log_p = 0
#    #print(pred.shape)
#    #print(pred)
#    for i in range(size):
#        if response[i] == 0:
#            log_p += np.log(pred[i, 2 * x[i]] * (1 - p_outlier) + w_outlier * p_outlier)
#        else: 
#            log_p += np.log(pred[i, 2 * x[i] + 1] * (1 - p_outlier) + w_outlier * p_outlier)
#    
#    # Call to network:
#    return log_p
#
#