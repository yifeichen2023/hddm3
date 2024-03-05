"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_rl, wiener_like_rl_2step #, wiener_like_rl_2step_sliding_window
from collections import OrderedDict


class Hrl(HDDM):
    """RL model that can be used to analyze data from two-armed bandit tasks."""

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop("non_centered", False)
        self.dual = kwargs.pop("dual", False)
        self.alpha = kwargs.pop("alpha", True)
        self.gamma = kwargs.pop("gamma", True)
        self.z = kwargs.pop("z", False)
        self.rl_class = RL_2step
        self.two_stage = kwargs.pop("two_stage", False) # whether to RLDDM just 1st stage or both stages
        self.sep_alpha = kwargs.pop("sep_alpha", False)  # use different learning rates for second stage
        self.sep_gamma = kwargs.pop("sep_gamma", False)
        self.lambda_ = kwargs.pop("lambda_", False)  # added for two-step task

        self.w = kwargs.pop("w", False)

        self.choice_model = kwargs.pop("choice_model", True)

        # below: unneecessary for choice model
        self.v_reg = kwargs.pop("v_reg", False)  # added for regression in two-step task
        self.z_reg = kwargs.pop("z_reg", False)
        self.a_fix = kwargs.pop("a_fix", False)

        # self.two_stage = kwargs.pop("two_stage", False)  # whether to RLDDM just 1st stage or both stages
        # self.sep_alpha = kwargs.pop("sep_alpha", False)  # use different learning rates for second stage
        #
        self.v_sep_q = kwargs.pop("v_sep_q",
                                  False)  # In 1st stage, whether to use Qmf/Qmb separately for v (drift rate) regression
        self.v_qmb = kwargs.pop("v_qmb", False)  # Given sep_q, True = qmb, False = Qmf
        self.v_interaction = kwargs.pop("v_interaction", False)  # whether to include interaction term for v

        self.z_sep_q = kwargs.pop("z_sep_q",
                                  False)  # In 1st stage, whether to use Qmf/Qmb separately for z (starting point) regression
        self.z_qmb = kwargs.pop("z_qmb", False)  # Given sep_q, True = qmb, False = Qmf
        self.z_interaction = kwargs.pop("z_interaction", False)  # whether to include interaction term for z

        self.a_share = kwargs.pop("a_share", False)  # whether to share a btw 1st & 2nd stage (if a!=1)
        self.v_share = kwargs.pop("v_share", False)  # whether to share v btw 1st & 2nd stage (if v!=reg)
        self.z_share = kwargs.pop("z_share", False)  # whether to share z btw 1st & 2nd stage (if z!=reg)
        self.t_share = kwargs.pop("t_share", False)  # whether to share t btw 1st & 2nd stage

        # JY added on 2022-03-15 for configuring starting point bias
        # if second-stage starting point depends on 1st-stage parameters

        self.z_2_depend = kwargs.pop("z_2_depend", False)  # whether z_2 depends on previous stage
        self.z_sigma2 = kwargs.pop("z_sigma2", False)  # for model 21

        self.free_z_2 = kwargs.pop("free_z_2", False)  # free parameter for z_2

        self.window_start = kwargs.pop("window_start", False)
        self.window_size = kwargs.pop("window_size", False)





        super(Hrl, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        # params = ["v"]
        # if "p_outlier" in self.include:
        #     params.append("p_outlier")
        # if "z" in self.include:
        #     params.append("z")
        # if self.two_stage:
        #     params.append("v_2")
        #     params.append("z_2")
        # include = set(params)

        knodes = super(Hrl, self)._create_stochastic_knodes(include)
        if self.non_centered:
            print("setting learning rate parameter(s) to be non-centered")
            if self.alpha:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "alpha",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.gamma:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "gamma",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.w: 
                    knodes.update(
                    self._create_family_normal_non_centered(
                        "w",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            # if self.two_stage:
            if self.sep_alpha:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "alpha2",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.sep_gamma:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "gamma2",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            # knodes.update(
            #     self._create_family_normal_non_centered(
            #         "w",
            #         value=0,
            #         g_mu=0.2,
            #         g_tau=3 ** -2,
            #         std_lower=1e-10,
            #         std_upper=10,
            #         std_value=0.1,
            #     )
            # )
            if self.lambda_:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "lambda_",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )

            if self.dual:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "pos_alpha",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
        else:
            if self.alpha:
                knodes.update(
                    self._create_family_normal(
                        "alpha",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.gamma:
                knodes.update(
                    self._create_family_normal(
                        "gamma",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.w: 
                knodes.update(
                    self._create_family_normal(
                        "w",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            # if self.two_stage:
            if self.sep_alpha:
                knodes.update(
                    self._create_family_normal(
                        "alpha2",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
                
            if self.sep_gamma:
                knodes.update(
                    self._create_family_normal(
                        "gamma2",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )

            if self.lambda_:
                knodes.update(
                    self._create_family_normal(
                        "lambda_",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.dual:
                knodes.update(
                    self._create_family_normal(
                        "pos_alpha",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )

        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents = super(Hrl, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents["v"] = knodes["v_bottom"]
        wfpt_parents["v_2"] = knodes["v_2_bottom"] if self.two_stage else 100.00
        wfpt_parents["alpha"] = knodes["alpha_bottom"] if self.alpha else 100.00
        wfpt_parents["alpha2"] = knodes["alpha2_bottom"] if self.sep_alpha else 100.00
        wfpt_parents["gamma2"] = knodes["gamma2_bottom"] if self.sep_gamma else 100.00
        wfpt_parents["pos_alpha"] = knodes["pos_alpha_bottom"] if self.dual else 100.00
        wfpt_parents["z"] = knodes["z_bottom"] if "z" in self.include else 0.5
        wfpt_parents["z_2"] = knodes["z_2_bottom"] if "z_2" in self.include else 0.5

        wfpt_parents["gamma"] = knodes["gamma_bottom"] if self.gamma else 100.00
        wfpt_parents["w"] = knodes["w_bottom"] if self.w else 100.00
        wfpt_parents["lambda_"] = knodes["lambda__bottom"] if self.lambda_ else 100.00

        if self.window_size is False:
            wfpt_parents['window_start'] = -1.00
            wfpt_parents['window_size'] = 999.00
        else:
            wfpt_parents['window_start'] = self.window_start
            wfpt_parents['window_size'] = self.window_size

        if self.two_stage: # two stage RLDDM
            wfpt_parents['two_stage'] = 1.00
        else:
            wfpt_parents['two_stage'] = 0.00

        return wfpt_parents

    # def _create_wfpt_knode(self, knodes):
    #     wfpt_parents = self._create_wfpt_parents_dict(knodes)
    #     return Knode(
    #         self.rl_class,
    #         "wfpt",
    #         observed=True,
    #         col_name=["split_by", "feedback", "response", "q_init"],
    #         **wfpt_parents
    #     )
    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(
            self.rl_class,
            "wfpt",
            observed=True,
            col_name=["split_by", "feedback", "response1", "response2", "rt1", "rt2",  "q_init", "state1", "state2", ],
            # col_name=["split_by", "feedback", "response1", "response2", "rt1", "rt2",  "q_init", "state1", "state2", "isleft1", "isleft2"],
            **wfpt_parents
        )

def RL_like(x, v, alpha, pos_alpha, z=0.5, p_outlier=0):

    wiener_params = {
        "err": 1e-4,
        "n_st": 2,
        "n_sz": 2,
        "use_adaptive": 1,
        "simps_err": 1e-3,
        "w_outlier": 0.1,
    }
    sum_logp = 0
    wp = wiener_params
    response = x["response"].values.astype(int)
    q = x["q_init"].iloc[0]
    feedback = x["feedback"].values.astype(float)
    split_by = x["split_by"].values.astype(int)
    return wiener_like_rl(
        response,
        feedback,
        split_by,
        q,
        alpha,
        pos_alpha,
        v,
        z,
        p_outlier=p_outlier,
        **wp
    )

# def RL_like_2step(x, v, v_2, alpha, alpha2, two_stage, pos_alpha, gamma, lambda_, w, z=0.5, z_2 = 0.5, p_outlier=0):
#
#     # wiener_params = {
#     #     "err": 1e-4,
#     #     "n_st": 2,
#     #     "n_sz": 2,
#     #     "use_adaptive": 1,
#     #     "simps_err": 1e-3,
#     #     "w_outlier": 0.1,
#     # }
#     # sum_logp = 0
#     # wp = wiener_params
#     # response = x["response"].values.astype(int)
#     # q = x["q_init"].iloc[0]
#     # feedback = x["feedback"].values.astype(float)
#     # split_by = x["split_by"].values.astype(int)
#     # return wiener_like_rl_2step(
#     #     response,
#     #     feedback,
#     #     split_by,
#     #     q,
#     #     alpha,
#     #     pos_alpha,
#     #     v,
#     #     z,
#     #     p_outlier=p_outlier,
#     #     **wp
#     # )
#
#     wiener_params = {
#         "err": 1e-4,
#         "n_st": 2,
#         "n_sz": 2,
#         "use_adaptive": 1,
#         "simps_err": 1e-3,
#         "w_outlier": 0.1,
#     }
#     wp = wiener_params
#     response1 = x["response1"].values.astype(int)
#     response2 = x["response2"].values.astype(int)
#     state1 = x["state1"].values.astype(int)
#     state2 = x["state2"].values.astype(int)
#
#     q = x["q_init"].iloc[0]
#     feedback = x["feedback"].values.astype(float)
#     split_by = x["split_by"].values.astype(int)
#
#
#     # JY added for two-step tasks on 2021-12-05
#     # nstates = x["nstates"].values.astype(int)
#     nstates = max(x["state2"].values.astype(int)) + 1
#
#
#     return wiener_like_rl_2step(
#         x["rt1"].values,
#         x["rt2"].values,
#         state1,
#         state2,
#         response1,
#         response2,
#         feedback,
#         split_by,
#         q,
#         alpha,
#         pos_alpha,
#         gamma, # added for two-step task
#         lambda_, # added for two-step task
#         v, # don't use second stage for now
#         z,
#         nstates,
#         two_stage,
#         z_2,
#         v_2,
#         alpha2,
#         w,
#         p_outlier=p_outlier,
#         **wp
#     )


def RL_like_2step(x, v, v_2, alpha, alpha2, two_stage, pos_alpha, gamma, gamma2, lambda_, w, window_start, window_size, sv, sz, st, sv2, sz2, st2, z=0.5, z_2 = 0.5, p_outlier=0):

    # wiener_params = {
    #     "err": 1e-4,
    #     "n_st": 2,
    #     "n_sz": 2,
    #     "use_adaptive": 1,
    #     "simps_err": 1e-3,
    #     "w_outlier": 0.1,
    # }
    # sum_logp = 0
    # wp = wiener_params
    # response = x["response"].values.astype(int)
    # q = x["q_init"].iloc[0]
    # feedback = x["feedback"].values.astype(float)
    # split_by = x["split_by"].values.astype(int)
    # return wiener_like_rl_2step(
    #     response,
    #     feedback,
    #     split_by,
    #     q,
    #     alpha,
    #     pos_alpha,
    #     v,
    #     z,
    #     p_outlier=p_outlier,
    #     **wp
    # )

    wiener_params = {
        "err": 1e-4,
        "n_st": 2,
        "n_sz": 2,
        "use_adaptive": 1,
        "simps_err": 1e-3,
        "w_outlier": 0.1,
    }
    wp = wiener_params
    response1 = x["response1"].values.astype(int)
    response2 = x["response2"].values.astype(int)
    state1 = x["state1"].values.astype(int)
    state2 = x["state2"].values.astype(int)

    q = x["q_init"].iloc[0]
    feedback = x["feedback"].values.astype(float)
    split_by = x["split_by"].values.astype(int)


    # JY added for two-step tasks on 2021-12-05
    # nstates = x["nstates"].values.astype(int)
    nstates = max(x["state2"].values.astype(int)) + 1


    return wiener_like_rl_2step(
        x["rt1"].values,
        x["rt2"].values,
        state1,
        state2,
        response1,
        response2,
        feedback,
        split_by,
        q,
        alpha,
        pos_alpha,
        gamma, # added for two-step task
        gamma2,
        lambda_, # added for two-step task
        v, # don't use second stage for now
        z,
        nstates,
        two_stage,
        z_2,
        v_2,
        alpha2,
        w,
        window_start,
        window_size,
        sv, sz, st, sv2, sz2, st2, 
        p_outlier=p_outlier,
        
        **wp
    )
# RL = stochastic_from_dist("RL", RL_like)
RL_2step = stochastic_from_dist("RL_2step", RL_like_2step)
# RL_2step_sliding_window = stochastic_from_dist("RL_2step_sliding_window", RL_like_2step_sliding_window)
#