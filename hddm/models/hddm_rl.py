"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_rlddm, wiener_like_rlddm_2step , wiener_like_rlddm_uncertainty  #wiener_like_rlddm_2step_reg, wiener_like_rlddm_2step_reg_sliding_window # wiener_like_rlddm_2step,
from collections import OrderedDict


class HDDMrl(HDDM):
    """HDDM model that can be used for two-armed bandit tasks."""

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop("non_centered", False)
        self.dual = kwargs.pop("dual", False)
        self.alpha = kwargs.pop("alpha", False)
        self.gamma = kwargs.pop("gamma", False) # added for two-step task

        self.lambda_ = kwargs.pop("lambda_", False) # added for two-step task
        self.v_reg = kwargs.pop("v_reg", False) # added for regression in two-step task
        self.z_reg = kwargs.pop("z_reg", False)
        self.a_fix = kwargs.pop("a_fix", False)

        self.w = kwargs.pop("w", False)
        self.w2 = kwargs.pop("w2", False)

        self.z_scaler = kwargs.pop("z_scaler", False)
        self.z_scaler_2 = kwargs.pop("z_scaler_2", False)

        self.two_stage = kwargs.pop("two_stage", False) # whether to RLDDM just 1st stage or both stages
        self.sep_alpha = kwargs.pop("sep_alpha", False) # use different learning rates for second stage
        # JY added on 2022-10-01 for separate gamma
        self.sep_gamma = kwargs.pop("sep_gamma", False)  # use different learning rates for second stage

        # self.v_sep_q = kwargs.pop("v_sep_q", False) # In 1st stage, whether to use Qmf/Qmb separately for v (drift rate) regression
        # self.v_qmb = kwargs.pop("v_qmb", False) # Given sep_q, True = qmb, False = Qmf

        self.v0 = kwargs.pop("v0", False) # whether to use v0
        self.v1 = kwargs.pop("v1", False)  # whether to use v0
        self.v2 = kwargs.pop("v2", False)  # whether to use v0
        self.v_interaction = kwargs.pop("v_interaction", False)  # whether to include interaction term for v
        

        # self.z_sep_q = kwargs.pop("z_sep_q", False) # In 1st stage, whether to use Qmf/Qmb separately for z (starting point) regression
        # self.z_qmb = kwargs.pop("z_qmb", False) # Given sep_q, True = qmb, False = Qmf
        # self.z_interaction = kwargs.pop("z_interaction", False) # whether to include interaction term for z
        self.z0 = kwargs.pop("z0", False) # whether to use z0
        self.z1 = kwargs.pop("z1", False)  # whether to use z0
        self.z2 = kwargs.pop("z2", False)  # whether to use z0
        self.z_interaction = kwargs.pop("z_interaction", False)  # whether to use z0


        self.a_share = kwargs.pop("a_share", False) # whether to share a btw 1st & 2nd stage (if a!=1)
        self.v_share = kwargs.pop("v_share", False) # whether to share v btw 1st & 2nd stage (if v!=reg)
        self.z_share = kwargs.pop("z_share", False) # whether to share z btw 1st & 2nd stage (if z!=reg)
        self.t_share = kwargs.pop("t_share", False) # whether to share t btw 1st & 2nd stage

        # JY added on 2022-03-15 for configuring starting point bias
        # if second-stage starting point depends on 1st-stage parameters

        # self.z_2_depend = kwargs.pop("z_2_depend", False)  # whether z_2 depends on previous stage
        self.z_sigma = kwargs.pop("z_sigma", False)  # for model 21
        self.z_sigma2 = kwargs.pop("z_sigma2", False)  # for model 21

        self.free_z_2 = kwargs.pop("free_z_2",False) # free parameter for z_2

        self.window_start = kwargs.pop("window_start", False)
        self.window_size = kwargs.pop("window_size", False)
        # print(self.window_start, self.window_size)
        self.regress_ndt = kwargs.pop("regress_ndt", False)
        self.regress_ndt2 = kwargs.pop("regress_ndt2", False)
        self.regress_ndt3 = kwargs.pop("regress_ndt3", False)
        self.regress_ndt4 = kwargs.pop("regress_ndt4", False)

        self.model_unc_rep = kwargs.pop("model_unc_rep", False) # uncertainty of model : set or ind?
        self.mem_unc_rep = kwargs.pop("mem_unc_rep", False) # uncertainty of memory: set or ind?

        self.unc_hybrid = kwargs.pop("unc_hybrid", False) # whether to use hybrid

        # YC added for new TST with aversive outcomues, 01-19-24
        self.aversive = kwargs.pop("aversive", False)

        self.choice_model = False # just a placeholder for compatibility

        self.wfpt_rl_class = WienerRL

        super(HDDMrl, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        # params = ["t"]
        # if "p_outlier" in self.include:
        #     params.append("p_outlier")
        # # if "z" in self.include:
        # if not self.v_reg:
        #     params.append("v")
        # if not self.z_reg:
        #     params.append("z")
        # if not self.a_fix:
        #     params.append("a")

        # include = set(params)

        knodes = super(HDDMrl, self)._create_stochastic_knodes(include)
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

            # if (not self.v_reg) and (not self.v_sep_q):
            # if not self.v_reg:
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
            if self.unc_hybrid == 'fourth': # regressing both will need an additional parameter
                knodes.update(
                    self._create_family_normal_non_centered(
                        "w_unc",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.regress_ndt:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "beta_ndt",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.regress_ndt2:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "beta_ndt2",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.regress_ndt3:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "beta_ndt3",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.regress_ndt4:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "beta_ndt4",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            # if (not self.z_reg) and (not self.z_sep_q):
            if not self.z_reg:
                if self.w2:
                    knodes.update(
                        self._create_family_normal_non_centered(
                            "w2",
                            value=0,
                            g_mu=0.2,
                            g_tau=3 ** -2,
                            std_lower=1e-10,
                            std_upper=10,
                            std_value=1,
                        )
                    )
                if self.z_scaler:
                    knodes.update(
                        self._create_family_normal_non_centered(
                            "z_scaler",
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
            
            # YC added for new TST with aversive outcomes, 01-19-24
            if self.aversive:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "alpha_pos",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
                knodes.update(
                    self._create_family_normal_non_centered(
                        "alpha_neu",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
                knodes.update(
                    self._create_family_normal_non_centered(
                        "alpha_neg",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )

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

            if self.v_reg:
                if self.v0:
                    knodes.update(
                        self._create_family_normal_non_centered(
                            "v0",
                            value=0,
                            g_mu=0.2,
                            g_tau=3 ** -2,
                            std_lower=1e-10,
                            std_upper=10,
                            std_value=1,
                        )
                    )
                # if self.v_sep_q:
                #     if self.v_qmb: # == 'mb': # just use MB Qvalues
                if self.v1:
                    knodes.update(
                        self._create_family_normal_non_centered(
                            "v1",
                            value=0,
                            g_mu=0.2,
                            g_tau=3 ** -2,
                            std_lower=1e-10,
                            std_upper=10,
                            std_value=1,
                        )
                    )
                    # else:
                if self.v2:
                    knodes.update(
                        self._create_family_normal_non_centered(
                            "v2",
                            value=0,
                            g_mu=0.2,
                            g_tau=3 ** -2,
                            std_lower=1e-10,
                            std_upper=10,
                            std_value=1,
                        )
                    )

                if self.v_interaction: # if include interaction term for v1 and v2
                    knodes.update(
                        self._create_family_normal_non_centered(
                            "v_interaction",
                            value=0,
                            g_mu=0.2,
                            g_tau=3 ** -2,
                            std_lower=1e-10,
                            std_upper=10,
                            std_value=1,
                        )
                    )
            if self.z_reg:
                if self.z0:
                    knodes.update(
                        self._create_family_invlogit(
                            "z0", value=0.5, g_tau=0.5 ** -2, std_std=0.05)
                    )
                if self.z1:
                # if self.z_sep_q:
                #     if self.z_qmb: # == 'mb': # just use MB Qvalues
                    knodes.update(
                        self._create_family_invlogit(
                            "z1", value=0.5, g_tau=0.5 ** -2, std_std=0.05)
                    )
                    # else:
                if self.z2:
                    knodes.update(
                        self._create_family_invlogit(
                            "z2", value=0.5, g_tau=0.5 ** -2, std_std=0.05)
                    )
                # else: # if both
                if self.z_interaction:
                    knodes.update(
                        self._create_family_invlogit(
                            "z_interaction", value=0.5, g_tau=0.5 ** -2, std_std=0.05)
                    )

            # if self.z_2_depend: # if second-stage starting point depends on first stage v -> only z std is needed, use z0 as std?
            if self.z_sigma:  # if second-stage starting point depends on first stage v -> only z std is needed, use z0 as std?
                knodes.update(
                    self._create_family_invlogit(
                        "z_sigma", value=0.5, g_tau=0.5 ** -2, std_std=0.05)

                )
            if self.z_sigma2:
                knodes.update(
                    self._create_family_invlogit(
                        "z_sigma2", value=0.5, g_tau=0.5 ** -2, std_std=0.05)

                )
            if self.two_stage and self.z_scaler_2:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "z_scaler_2",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )



        else: # if not non-centered (default)
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
            # if (not self.v_reg) and (not self.v_sep_q):
            # if not self.v_reg:
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
            if self.unc_hybrid == 'fourth': # regressing both will need an additional parameter
                knodes.update(
                    self._create_family_normal(
                        "w_unc",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            # if (not self.z_reg) and (not self.z_sep_q):
            if not self.z_reg:
                if self.w2:
                    knodes.update(
                        self._create_family_normal(
                            "w2",
                            value=0,
                            g_mu=0.2,
                            g_tau=3 ** -2,
                            std_lower=1e-10,
                            std_upper=10,
                            std_value=1,
                        )
                    )
                if self.z_scaler:
                    knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "z_scaler", value=2, g_mu=2, g_tau=3 ** -2, std_std=2
                        )
                    )
            if self.regress_ndt:
                knodes.update(
                    self._create_family_normal(
                        "beta_ndt",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.regress_ndt2:
                knodes.update(
                    self._create_family_normal(
                        "beta_ndt2",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.regress_ndt3:
                knodes.update(
                    self._create_family_normal(
                        "beta_ndt3",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )
            if self.regress_ndt4:
                knodes.update(
                    self._create_family_normal(
                        "beta_ndt4",
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

            # YC added for new TST with aversive outcomes, 01-19-24
            if self.aversive:
                knodes.update(
                    self._create_family_normal(
                        "alpha_pos",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                )   
                knodes.update(
                    self._create_family_normal(
                        "alpha_neu",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=1,
                    )
                ) 
                knodes.update(
                    self._create_family_normal(
                        "alpha_neg",
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

            if self.v_reg:
                if self.v0:
                    knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "v0", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                        )
                    )
                if self.v_interaction:
                    knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "v_interaction", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                        )
                    )
                if self.v1:
                    knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "v1", value=2, g_mu=2, g_tau=3 ** -2, std_std=2 # informative prior
                        )
                    )
                if self.v2:
                    knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "v2", value=2, g_mu=2, g_tau=3 ** -2, std_std=2 # informative prior
                        )
                    )

            if self.z_reg:
                if self.z0:
                    knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "z0", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                        )
                    )
                if self.z_interaction:
                    knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "z_interaction", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                        )
                    )
                if self.z1:
                    knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "z1", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                        )
                    )
                if self.z2:
                    knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "z2", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                        )
                    )
            # if self.z_2_depend: # if second-stage starting point depends on first stage v -> only z std is needed, use z0 as std?
            if self.z_sigma:
                knodes.update(
                    self._create_family_normal_normal_hnormal(
                        "z_sigma", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                    )
                )
            if self.z_sigma2:
                knodes.update(
                    self._create_family_normal_normal_hnormal(
                        "z_sigma2", value=0, g_tau=50 ** -2, std_std=10  # uninformative prior
                    )
                )
            if self.two_stage and self.z_scaler_2:
                knodes.update(
                    self._create_family_normal_normal_hnormal(
                        "z_scaler_2", value=2, g_mu=2, g_tau=3 ** -2, std_std=2
                    )
                )

        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents = super(HDDMrl, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents["alpha"] = knodes["alpha_bottom"] if self.alpha else 100.00
        wfpt_parents["pos_alpha"] = knodes["pos_alpha_bottom"] if self.dual else 100.00

        # YC added, 01-19-24
        wfpt_parents["alpha_pos"] = knodes["alpha_pos_bottom"] if self.aversive else 100.00
        wfpt_parents["alpha_neu"] = knodes["alpha_neu_bottom"] if self.aversive else 100.00
        wfpt_parents["alpha_neg"] = knodes["alpha_neg_bottom"] if self.aversive else 100.00

        wfpt_parents["alpha2"] = knodes["alpha2_bottom"] if self.sep_alpha else 100.00
        wfpt_parents["gamma2"] = knodes["gamma2_bottom"] if self.sep_gamma else 100.00
        # if not self.v_reg) and (not self.v_sep_q):
        if not self.v_reg:
            if self.w:
                wfpt_parents["w"] = knodes["w_bottom"]
            else:
                wfpt_parents["w"] = 100.00
        else:
            wfpt_parents["w"] = 100.00
        # if (not self.z_reg) and (not self.z_sep_q):
        if not self.z_reg:
            if self.w2:
                wfpt_parents["w2"] = knodes["w2_bottom"]
            else:
                wfpt_parents["w2"] = 100.00
            if self.z_scaler:
                wfpt_parents["z_scaler"] = knodes["z_scaler_bottom"]
            else:
                wfpt_parents["z_scaler"] = 100.00
        else:
            wfpt_parents["w2"] = 100.00
            wfpt_parents["z_scaler"] = 100.00
        wfpt_parents["gamma"] = knodes["gamma_bottom"] if self.gamma else 100.00
        wfpt_parents["lambda_"] = knodes["lambda__bottom"] if self.lambda_ else 100.00


        wfpt_parents["beta_ndt"] = knodes["beta_ndt_bottom"] if self.regress_ndt else 0.00
        wfpt_parents["beta_ndt2"] = knodes["beta_ndt2_bottom"] if self.regress_ndt2 else 0.00
        wfpt_parents["beta_ndt3"] = knodes["beta_ndt3_bottom"] if self.regress_ndt3 else 0.00
        wfpt_parents["beta_ndt4"] = knodes["beta_ndt4_bottom"] if self.regress_ndt4 else 0.00

        if self.v_reg: # if using v_regression 
            wfpt_parents['v'] = 100.00

            wfpt_parents["v0"] = knodes["v0_bottom"] if self.v0 else 0.00
            wfpt_parents["v1"] = knodes["v1_bottom"] if self.v1 else 0.00
            wfpt_parents["v2"] = knodes["v2_bottom"] if self.v2 else 0.00
            wfpt_parents["v_interaction"] = knodes["v_interaction_bottom"] if self.v_interaction else 0.00

        else: # if not using v_regression: just multiplying v * Q
            wfpt_parents["v0"] = 0.00
            wfpt_parents["v1"] = 0.00
            wfpt_parents["v2"] = 0.00
            wfpt_parents["v_interaction"] = 0.00

        if self.z_reg:

            wfpt_parents["z0"] = knodes["z0_bottom"] if self.z0 else 0.00
            wfpt_parents["z1"] = knodes["z1_bottom"] if self.z1 else 0.00
            wfpt_parents["z2"] = knodes["z2_bottom"] if self.z2 else 0.00
            wfpt_parents["z_interaction"] = knodes["z_interaction_bottom"] if self.z_interaction else 0.00

        else: # if not using z_regression
            wfpt_parents["z0"] = 0.00
            wfpt_parents["z1"] = 0.00
            wfpt_parents["z2"] = 0.00
            wfpt_parents['z_interaction'] = 0.00

        # if self.z_2_depend:
        wfpt_parents['z_sigma'] = knodes['z_sigma_bottom'] if self.z_sigma else 0.00
        wfpt_parents['z_sigma2'] = knodes['z_sigma2_bottom'] if self.z_sigma2 else 0.00
        # wfpt_parents['z_sigma'] = 100.00
        #
        # if self.z_reg:
        #     wfpt_parents['z'] = 100.00
        if self.a_fix:
            wfpt_parents['a'] = 1.00   # threshold set to 1

        if self.two_stage: # two stage RLDDM
            wfpt_parents['two_stage'] = 1.00
            if self.v_share:
                wfpt_parents['v_2'] = 100.00
            if self.a_share:
                wfpt_parents['a_2'] = 100.00
            # elif not self.a_share and self.a_fix:
            #     wfpt_parents['a_2'] = 100.00
            # if self.z_share:
            #     wfpt_parents['z_2'] = 100.00
            if self.t_share:
                wfpt_parents['t_2'] = 100.00

        else:
            wfpt_parents['two_stage'] = 0.00
            # since only first-stage is modeled, none of v_2,a_2,t_2,z_2 is used
            wfpt_parents['v_2'] = 100.00
            wfpt_parents['a_2'] = 100.00
            # wfpt_parents['z_2'] = 100.00
            wfpt_parents['t_2'] = 100.00

        if self.window_size is False:
            wfpt_parents['window_start'] = -1.00
            wfpt_parents['window_size'] = 999.00
        else:
            wfpt_parents['window_start'] = self.window_start
            wfpt_parents['window_size'] = self.window_size

        # form of representation - set or ind
        if self.mem_unc_rep == 'ind':
            wfpt_parents['mem_unc_rep'] = 1.00
        elif self.mem_unc_rep == 'set':
            wfpt_parents['mem_unc_rep'] = -1.00
        else:
            wfpt_parents['mem_unc_rep'] = 0.00

        if self.model_unc_rep == 'ind':
            wfpt_parents['model_unc_rep'] = 1.00
        elif self.model_unc_rep == 'set':
            wfpt_parents['model_unc_rep'] = -1.00
        else:
            wfpt_parents['model_unc_rep'] = 0.00

        # wfpt_parents["z"] = knodes["z_bottom"] if "z" in self.include else 0.5
        if self.two_stage:
            if self.z_scaler_2:
                wfpt_parents['z_scaler_2'] = knodes['z_scaler_2_bottom']
            else:
                wfpt_parents['z_scaler_2'] = 100.00
        else:
            wfpt_parents['z_scaler_2'] = 100.00

        if self.unc_hybrid:
            if self.unc_hybrid == 'first': # bellman ind, uncertainty set
                wfpt_parents['unc_hybrid'] = 1.00
                wfpt_parents['w_unc'] = 0.00
            elif self.unc_hybrid == 'second': # bellman set, uncertainty ind
                wfpt_parents['unc_hybrid'] = 2.00
                wfpt_parents['w_unc'] = 0.00
            elif self.unc_hybrid == 'third': # regress the average of ind and set
                wfpt_parents['unc_hybrid'] = 3.00
                wfpt_parents['w_unc'] = 0.00
            elif self.unc_hybrid == 'fourth': # regress both within ndt1
                wfpt_parents['unc_hybrid'] = 4.00
                wfpt_parents['w_unc'] = knodes['w_unc_bottom']
        else:
            wfpt_parents['unc_hybrid'] = 0.00
            wfpt_parents['w_unc'] = 0.00

        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(
            self.wfpt_rl_class,
            "wfpt",
            observed=True,
            col_name=["split_by", "feedback", "response1", "response2", "rt1", "rt2",  "q_init", "state1", "state2", "bandit_type"],
            # col_name=["split_by", "feedback", "response1", "response2", "rt1", "rt2",  "q_init", "state1", "state2", "isleft1", "isleft2"],
            **wfpt_parents
        )


def wienerRL_like(x, v, alpha, pos_alpha, sv, a, z, sz, t, st, p_outlier=0):

    wiener_params = {
        "err": 1e-4,
        "n_st": 2,
        "n_sz": 2,
        "use_adaptive": 1,
        "simps_err": 1e-3,
        "w_outlier": 0.1,
    }
    wp = wiener_params
    response = x["response"].values.astype(int)
    q = x["q_init"].iloc[0]
    feedback = x["feedback"].values.astype(float)
    split_by = x["split_by"].values.astype(int)
    return wiener_like_rlddm(
        x["rt"].values,
        response,
        feedback,
        split_by,
        q,
        alpha,
        pos_alpha,
        v,
        sv,
        a,
        z,
        sz,
        t,
        st,
        p_outlier=p_outlier,
        **wp
    )


def wienerRL_like_2step(x, v0, v1, v2, v_interaction, z0, z1, z2, z_interaction, lambda_, alpha, pos_alpha, gamma,gamma2, a,z,sz,t,st,v,sv, a_2, z_2, t_2,v_2,alpha2,
                                           two_stage, w, w2,z_scaler,z_sigma,z_sigma2,window_start,window_size, beta_ndt, beta_ndt2, beta_ndt3,
                        st2, sv2, sz2, p_outlier=0): # regression ver2: bounded, a fixed to 1

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

    conditions = x['conditions'].values.astype(int)

    # isleft1 = x["isleft1"].values.astype(int)
    # isleft2 = x["isleft2"].values.astype(int)


    q = x["q_init"].iloc[0]
    feedback = x["feedback"].values.astype(float)
    split_by = x["split_by"].values.astype(int)


    # JY added for two-step tasks on 2021-12-05
    # nstates = x["nstates"].values.astype(int)
    nstates = max(x["state2"].values.astype(int)) + 1

    # # JY added for which q-value to use (if sep, qmb or qmf)
    # qval = 0 # default: simultaneous

    # if

    return wiener_like_rlddm_2step(
    # return wiener_like_rlddm_2step_reg_sliding_window(
        x["rt1"].values,
        x["rt2"].values,

        # isleft1,
        # isleft2,

        state1,
        state2,
        response1,
        response2,
        feedback,
        split_by,
        q,
        alpha,
        pos_alpha,
        # w, # added for two-step task
        gamma, # added for two-step task
        gamma2,
        lambda_, # added for two-step task
        v0, # intercept for first stage rt regression
        v1, # slope for mb
        v2, # slobe for mf
        v, # don't use second stage for now
        sv,
        a,
        z0, # bias: added for intercept regression 1st stage
        z1, # bias: added for slope regression mb 1st stage
        z2, # bias: added for slope regression mf 1st stage
        z,
        sz,
        t,
        nstates,
        # v_qval,
        # z_qval,
        v_interaction,
        z_interaction,
        two_stage,

        a_2,
        z_2,
        t_2,
        v_2,
        sz2,
        st2,
        sv2,
        alpha2,
        w,
        w2,
        z_scaler,
        z_sigma,
        z_sigma2,

        window_start,
        window_size,
        beta_ndt,
        beta_ndt2,
        beta_ndt3,
        st,
        p_outlier=p_outlier,
        **wp
    )
# def wienerRL_like_bayesianQ(x, v0, v1, v2, v_interaction, z0, z1, z2, z_interaction, lambda_, alpha, pos_alpha, gamma, gamma2,a,z,t,v, a_2, z_2, t_2,v_2,alpha2,
#                                            two_stage, w, w2,z_scaler,z_sigma,z_sigma2,window_start,window_size, beta_ndt, beta_ndt2, beta_ndt3, p_outlier=0): # regression ver2: bounded, a fixed to 1
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
#     # isleft1 = x["isleft1"].values.astype(int)
#     # isleft2 = x["isleft2"].values.astype(int)
#
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
#     # # JY added for which q-value to use (if sep, qmb or qmf)
#     # qval = 0 # default: simultaneous
#
#     # if
#
#     return wiener_like_rlddm_bayesianQ(
#     # return wiener_like_rlddm_2step_reg_sliding_window(
#         x["rt1"].values,
#         x["rt2"].values,
#
#         # isleft1,
#         # isleft2,
#
#         state1,
#         state2,
#         response1,
#         response2,
#         feedback,
#         split_by,
#         q,
#         alpha,
#         pos_alpha,
#         # w, # added for two-step task
#         gamma, # added for two-step task
#         gamma2,
#         lambda_, # added for two-step task
#         v0, # intercept for first stage rt regression
#         v1, # slope for mb
#         v2, # slobe for mf
#         v, # don't use second stage for now
#         # sv,
#         a,
#         z0, # bias: added for intercept regression 1st stage
#         z1, # bias: added for slope regression mb 1st stage
#         z2, # bias: added for slope regression mf 1st stage
#         z,
#         # sz,
#         t,
#         nstates,
#         # v_qval,
#         # z_qval,
#         v_interaction,
#         z_interaction,
#         two_stage,
#
#         a_2,
#         z_2,
#         t_2,
#         v_2,
#         alpha2,
#         w,
#         w2,
#         z_scaler,
#         z_sigma,
#         z_sigma2,
#         # st,
#         window_start,
#         window_size,
#         beta_ndt,
#         beta_ndt2,
#         beta_ndt3,
#         p_outlier=p_outlier,
#         **wp
#     )
def wienerRL_like_uncertainty(x, v0, v1, v2, v_interaction, z0, z1, z2, z_interaction, lambda_, alpha, pos_alpha, alpha_pos, alpha_neu, alpha_neg, gamma, gamma2, a,z,sz,t,st,v,sv, a_2, z_2, t_2,v_2,alpha2,
                                           two_stage, w, w2,z_scaler, z_scaler_2, z_sigma,z_sigma2,window_start,window_size, beta_ndt, beta_ndt2, beta_ndt3, beta_ndt4,
                              model_unc_rep, mem_unc_rep, unc_hybrid, w_unc, st2, sv2, sz2, p_outlier=0): # regression ver2: bounded, a fixed to 1

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

    # isleft1 = x["isleft1"].values.astype(int)
    # isleft2 = x["isleft2"].values.astype(int)


    q = x["q_init"].iloc[0]
    feedback = x["feedback"].values.astype(float)
    split_by = x["split_by"].values.astype(int)

    # YC added, 01-19-24
    bandit_type = x['bandit_type'].values.astype(int)

    # JY added for two-step tasks on 2021-12-05
    # nstates = x["nstates"].values.astype(int)
    nstates = max(x["state2"].values.astype(int)) + 1

    # # JY added for which q-value to use (if sep, qmb or qmf)
    # qval = 0 # default: simultaneous

    # if
    print(type(alpha_pos),type(alpha_neu),type(alpha_neg))
    print(type(bandit_type))
    return wiener_like_rlddm_uncertainty(
    # return wiener_like_rlddm_2step_reg_sliding_window(
        x["rt1"].values,
        x["rt2"].values,

        # isleft1,
        # isleft2,

        state1,
        state2,
        response1,
        response2,
        feedback,
        split_by,
        bandit_type,    # YC added 01-19-23
        q,
        alpha,
        pos_alpha,
        alpha_pos,
        alpha_neu,
        alpha_neg,
        # w, # added for two-step task
        gamma, # added for two-step task
        gamma2,
        lambda_, # added for two-step task
        v0, # intercept for first stage rt regression
        v1, # slope for mb
        v2, # slobe for mf
        v, # don't use second stage for now
        sv,
        a,
        z0, # bias: added for intercept regression 1st stage
        z1, # bias: added for slope regression mb 1st stage
        z2, # bias: added for slope regression mf 1st stage
        z,
        sz,
        t,
        nstates,
        # v_qval,
        # z_qval,
        v_interaction,
        z_interaction,
        two_stage,

        a_2,
        z_2,
        t_2,
        v_2,
        sz2,
        st2,
        sv2,
        alpha2,
        w,
        w2,
        z_scaler,
        z_scaler_2,
        z_sigma,
        z_sigma2,

        window_start,
        window_size,
        beta_ndt,
        beta_ndt2,
        beta_ndt3,
        beta_ndt4,

        model_unc_rep,
        mem_unc_rep,
        unc_hybrid,
        w_unc,
        st,
        p_outlier=p_outlier,
        **wp
    )
# WienerRL = stochastic_from_dist("wienerRL_2step", wienerRL_like_2step)
# WienerRL = stochastic_from_dist("wienerRL_bayesianQ", wienerRL_like_bayesianQ)
WienerRL = stochastic_from_dist("wienerRL_uncertainty", wienerRL_like_uncertainty)