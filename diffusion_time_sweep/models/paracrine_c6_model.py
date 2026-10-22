#!/usr/bin/env python3
"""
Paracrine (c-6) timelapse model with finite bidirectional signal-transfer time.

Two communicating cells (Lux/Las cross-talk) are separated by a configurable
distance. Cell A carries pCon-LuxI, pCon-LasR, and pLas-GFP; Cell B carries
pCon-LuxR, pLux-GFP, and pLux-LasI. Autoinducers AHL (from cell A) and S (Las
signal from cell B) diffuse between two extracellular micro-compartments before
reaching the partner cell. Diffusion is approximated by first-order exchange with
rate k_diff. By default the mean travel time for each signal is τ = 5 s
(k_diff = 1/τ). You can override this by supplying diffusion coefficients, in
which case k_diff = 2·D / L² for distance L.

Defaults for direct model use: N=500, T_end=6000, distance=1 cm, τ_AHL=τ_S=5 s.
Outputs include raw reporter traces for both compartments, a compact summary table, and an optional timelapse plot.
"""
import math
import os
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _exchange_rate(diff_coef: float, distance_cm: float) -> float:
    if distance_cm <= 0:
        raise ValueError("distance_cm must be positive")
    return 2.0 * diff_coef / (distance_cm ** 2)


@dataclass
class Params6:
    RNAP_total: int = 50
    Ribo_total: int = 200
    n_pCon_luxI_A: int = 10
    n_pCon_lasR_A: int = 10
    n_pLas_gfp_A: int = 10
    n_pCon_luxR_B: int = 10
    n_pLux_gfp_B: int = 10
    n_pLux_lasI_B: int = 10
    kon_act: float = 1e-3
    koff_act: float = 1e-3
    kon_RL: float = 1e-3
    koff_RL: float = 1e-2
    kon_RS: float = 1e-3
    koff_RS: float = 1e-2
    kon_RNAP: float = 5e-4
    kcat_tx_active: float = 0.08
    kcat_tx_basal: float = 0.008
    kon_ribo: float = 1e-3
    kcat_tl: float = 0.5
    gamma_m: float = 0.01
    gamma_p: float = 5e-4
    kmat: float = 0.01
    k_I: float = 0.02
    k_S: float = 0.02
    gamma_AHL: float = 0.001
    gamma_S: float = 0.001
    gamma_G: float = 0.01
    T_end: float = 6000.0
    distance_cm: float = 1.0
    transfer_time_ahl: Optional[float] = 5.0
    transfer_time_s: Optional[float] = 5.0
    diff_coef_ahl: Optional[float] = None
    diff_coef_s: Optional[float] = None

    @property
    def k_diff_ahl(self) -> float:
        if self.diff_coef_ahl is not None:
            return _exchange_rate(self.diff_coef_ahl, self.distance_cm)
        if self.transfer_time_ahl is None or self.transfer_time_ahl <= 0:
            raise ValueError("transfer_time_ahl must be >0 or diff_coef_ahl provided")
        return 1.0 / self.transfer_time_ahl

    @property
    def k_diff_s(self) -> float:
        if self.diff_coef_s is not None:
            return _exchange_rate(self.diff_coef_s, self.distance_cm)
        if self.transfer_time_s is None or self.transfer_time_s <= 0:
            raise ValueError("transfer_time_s must be >0 or diff_coef_s provided")
        return 1.0 / self.transfer_time_s

    @property
    def effective_diff_coef_ahl(self) -> float:
        if self.diff_coef_ahl is not None:
            return self.diff_coef_ahl
        return self.distance_cm ** 2 / (2.0 * self.transfer_time_ahl)

    @property
    def effective_diff_coef_s(self) -> float:
        if self.diff_coef_s is not None:
            return self.diff_coef_s
        return self.distance_cm ** 2 / (2.0 * self.transfer_time_s)


def gillespie_cond6(P: Params6, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    AHL_A = 0
    AHL_B = 0
    S_A = 0
    S_B = 0
    A = {
        "RNAP_free": P.RNAP_total,
        "Ribo_free": P.Ribo_total,
        "LasR": 0,
        "RS": 0,
        "pCon_luxI_free": P.n_pCon_luxI_A,
        "pCon_luxI_RNAP": 0,
        "pCon_lasR_free": P.n_pCon_lasR_A,
        "pCon_lasR_RNAP": 0,
        "pLas_gfp_free": P.n_pLas_gfp_A,
        "pLas_gfp_act": 0,
        "pLas_gfp_RNAP_inact": 0,
        "pLas_gfp_RNAP_act": 0,
        "m_luxI": 0,
        "m_lasR": 0,
        "ribo_luxI": 0,
        "ribo_lasR": 0,
        "LuxI": 0,
        "m_gfp_A": 0,
        "ribo_gfp_A": 0,
        "Gi_A": 0,
        "G_A": 0,
    }
    B = {
        "RNAP_free": P.RNAP_total,
        "Ribo_free": P.Ribo_total,
        "LuxR": 0,
        "RL": 0,
        "pCon_luxR_free": P.n_pCon_luxR_B,
        "pCon_luxR_RNAP": 0,
        "pLux_gfp_free": P.n_pLux_gfp_B,
        "pLux_gfp_act": 0,
        "pLux_gfp_RNAP_inact": 0,
        "pLux_gfp_RNAP_act": 0,
        "pLux_lasI_free": P.n_pLux_lasI_B,
        "pLux_lasI_act": 0,
        "pLux_lasI_RNAP_inact": 0,
        "pLux_lasI_RNAP_act": 0,
        "m_luxR": 0,
        "m_gfp_B": 0,
        "ribo_luxR": 0,
        "ribo_gfp_B": 0,
        "Gi_B": 0,
        "G_B": 0,
        "m_lasI_B": 0,
        "ribo_lasI_B": 0,
        "LasI_B": 0,
    }
    t = 0.0
    it = 0
    max_it = int(3e6)
    times = [0.0]
    GA = [0]
    GB = [0]
    while t < P.T_end and it < max_it:
        it += 1
        props = []
        rxns = []
        # Cell A: constitutive LasR expression, sensing S_A, and producing LuxI / GFP
        if A["LasR"] > 0 and S_A > 0:
            props.append(P.kon_RS * A["LasR"] * S_A)
            rxns.append(("A", "bind_RS"))
        if A["RS"] > 0:
            props.append(P.koff_RS * A["RS"])
            rxns.append(("A", "unbind_RS"))
        if A["RS"] > 0 and A["pLas_gfp_free"] > 0:
            props.append(P.kon_act * A["RS"] * A["pLas_gfp_free"])
            rxns.append(("A", "act_pLas_gfp"))
        if A["pLas_gfp_act"] > 0:
            props.append(P.koff_act * A["pLas_gfp_act"])
            rxns.append(("A", "deact_pLas_gfp"))
        if A["RNAP_free"] > 0:
            if A["pCon_luxI_free"] > 0:
                props.append(P.kon_RNAP * A["RNAP_free"] * A["pCon_luxI_free"])
                rxns.append(("A", "bind_pCon_luxI"))
            if A["pCon_lasR_free"] > 0:
                props.append(P.kon_RNAP * A["RNAP_free"] * A["pCon_lasR_free"])
                rxns.append(("A", "bind_pCon_lasR"))
            if A["pLas_gfp_free"] > 0:
                props.append(P.kon_RNAP * A["RNAP_free"] * A["pLas_gfp_free"])
                rxns.append(("A", "bind_pLas_gfp_inact"))
            if A["pLas_gfp_act"] > 0:
                props.append(P.kon_RNAP * A["RNAP_free"] * A["pLas_gfp_act"])
                rxns.append(("A", "bind_pLas_gfp_act"))
        if A["pCon_luxI_RNAP"] > 0:
            props.append(P.kcat_tx_active * A["pCon_luxI_RNAP"])
            rxns.append(("A", "tx_pCon_luxI"))
        if A["pCon_lasR_RNAP"] > 0:
            props.append(P.kcat_tx_active * A["pCon_lasR_RNAP"])
            rxns.append(("A", "tx_pCon_lasR"))
        if A["pLas_gfp_RNAP_inact"] > 0:
            props.append(P.kcat_tx_basal * A["pLas_gfp_RNAP_inact"])
            rxns.append(("A", "tx_pLas_gfp_inact"))
        if A["pLas_gfp_RNAP_act"] > 0:
            props.append(P.kcat_tx_active * A["pLas_gfp_RNAP_act"])
            rxns.append(("A", "tx_pLas_gfp_act"))
        if A["Ribo_free"] > 0 and A["m_luxI"] > 0:
            props.append(P.kon_ribo * A["Ribo_free"] * A["m_luxI"])
            rxns.append(("A", "bind_ribo_luxI"))
        if A["Ribo_free"] > 0 and A["m_lasR"] > 0:
            props.append(P.kon_ribo * A["Ribo_free"] * A["m_lasR"])
            rxns.append(("A", "bind_ribo_lasR"))
        if A["Ribo_free"] > 0 and A["m_gfp_A"] > 0:
            props.append(P.kon_ribo * A["Ribo_free"] * A["m_gfp_A"])
            rxns.append(("A", "bind_ribo_gfp_A"))
        if A["ribo_luxI"] > 0:
            props.append(P.kcat_tl * A["ribo_luxI"])
            rxns.append(("A", "tl_luxI"))
        if A["ribo_lasR"] > 0:
            props.append(P.kcat_tl * A["ribo_lasR"])
            rxns.append(("A", "tl_lasR"))
        if A["ribo_gfp_A"] > 0:
            props.append(P.kcat_tl * A["ribo_gfp_A"])
            rxns.append(("A", "tl_gfp_A"))
        if A["m_luxI"] > 0:
            props.append(P.gamma_m * A["m_luxI"])
            rxns.append(("A", "deg_m_luxI"))
        if A["m_lasR"] > 0:
            props.append(P.gamma_m * A["m_lasR"])
            rxns.append(("A", "deg_m_lasR"))
        if A["m_gfp_A"] > 0:
            props.append(P.gamma_m * A["m_gfp_A"])
            rxns.append(("A", "deg_m_gfp_A"))
        if A["LuxI"] > 0:
            props.append(P.gamma_p * A["LuxI"])
            rxns.append(("A", "deg_LuxI"))
            props.append(P.k_I * A["LuxI"])
            rxns.append(("E", "syn_AHL_A"))
        if A["LasR"] > 0:
            props.append(P.gamma_p * A["LasR"])
            rxns.append(("A", "deg_LasR"))
        if A["Gi_A"] > 0:
            props.append(P.kmat * A["Gi_A"])
            rxns.append(("A", "mat_G_A"))
        if A["G_A"] > 0:
            props.append(P.gamma_G * A["G_A"])
            rxns.append(("A", "deg_G_A"))
        # Cell B: constitutive LuxR expression, sensing AHL_B, and producing LasI / GFP
        if B["LuxR"] > 0 and AHL_B > 0:
            props.append(P.kon_RL * B["LuxR"] * AHL_B)
            rxns.append(("B", "bind_RL"))
        if B["RL"] > 0:
            props.append(P.koff_RL * B["RL"])
            rxns.append(("B", "unbind_RL"))
        if B["RL"] > 0 and B["pLux_gfp_free"] > 0:
            props.append(P.kon_act * B["RL"] * B["pLux_gfp_free"])
            rxns.append(("B", "act_pLux_gfp"))
        if B["RL"] > 0 and B["pLux_lasI_free"] > 0:
            props.append(P.kon_act * B["RL"] * B["pLux_lasI_free"])
            rxns.append(("B", "act_pLux_lasI"))
        if B["pLux_gfp_act"] > 0:
            props.append(P.koff_act * B["pLux_gfp_act"])
            rxns.append(("B", "deact_pLux_gfp"))
        if B["pLux_lasI_act"] > 0:
            props.append(P.koff_act * B["pLux_lasI_act"])
            rxns.append(("B", "deact_pLux_lasI"))
        if B["RNAP_free"] > 0:
            if B["pCon_luxR_free"] > 0:
                props.append(P.kon_RNAP * B["RNAP_free"] * B["pCon_luxR_free"])
                rxns.append(("B", "bind_pCon_luxR"))
            if B["pLux_gfp_free"] > 0:
                props.append(P.kon_RNAP * B["RNAP_free"] * B["pLux_gfp_free"])
                rxns.append(("B", "bind_pLux_gfp_inact"))
            if B["pLux_gfp_act"] > 0:
                props.append(P.kon_RNAP * B["RNAP_free"] * B["pLux_gfp_act"])
                rxns.append(("B", "bind_pLux_gfp_act"))
            if B["pLux_lasI_free"] > 0:
                props.append(P.kon_RNAP * B["RNAP_free"] * B["pLux_lasI_free"])
                rxns.append(("B", "bind_pLux_lasI_inact"))
            if B["pLux_lasI_act"] > 0:
                props.append(P.kon_RNAP * B["RNAP_free"] * B["pLux_lasI_act"])
                rxns.append(("B", "bind_pLux_lasI_act"))
        if B["pCon_luxR_RNAP"] > 0:
            props.append(P.kcat_tx_active * B["pCon_luxR_RNAP"])
            rxns.append(("B", "tx_pCon_luxR"))
        if B["pLux_gfp_RNAP_inact"] > 0:
            props.append(P.kcat_tx_basal * B["pLux_gfp_RNAP_inact"])
            rxns.append(("B", "tx_pLux_gfp_inact"))
        if B["pLux_gfp_RNAP_act"] > 0:
            props.append(P.kcat_tx_active * B["pLux_gfp_RNAP_act"])
            rxns.append(("B", "tx_pLux_gfp_act"))
        if B["pLux_lasI_RNAP_inact"] > 0:
            props.append(P.kcat_tx_basal * B["pLux_lasI_RNAP_inact"])
            rxns.append(("B", "tx_pLux_lasI_inact"))
        if B["pLux_lasI_RNAP_act"] > 0:
            props.append(P.kcat_tx_active * B["pLux_lasI_RNAP_act"])
            rxns.append(("B", "tx_pLux_lasI_act"))
        if B["Ribo_free"] > 0 and B["m_luxR"] > 0:
            props.append(P.kon_ribo * B["Ribo_free"] * B["m_luxR"])
            rxns.append(("B", "bind_ribo_luxR"))
        if B["Ribo_free"] > 0 and B["m_gfp_B"] > 0:
            props.append(P.kon_ribo * B["Ribo_free"] * B["m_gfp_B"])
            rxns.append(("B", "bind_ribo_gfp_B"))
        if B["Ribo_free"] > 0 and B["m_lasI_B"] > 0:
            props.append(P.kon_ribo * B["Ribo_free"] * B["m_lasI_B"])
            rxns.append(("B", "bind_ribo_lasI_B"))
        if B["ribo_luxR"] > 0:
            props.append(P.kcat_tl * B["ribo_luxR"])
            rxns.append(("B", "tl_luxR"))
        if B["ribo_gfp_B"] > 0:
            props.append(P.kcat_tl * B["ribo_gfp_B"])
            rxns.append(("B", "tl_gfp_B"))
        if B["ribo_lasI_B"] > 0:
            props.append(P.kcat_tl * B["ribo_lasI_B"])
            rxns.append(("B", "tl_lasI_B"))
        if B["m_luxR"] > 0:
            props.append(P.gamma_m * B["m_luxR"])
            rxns.append(("B", "deg_m_luxR"))
        if B["m_gfp_B"] > 0:
            props.append(P.gamma_m * B["m_gfp_B"])
            rxns.append(("B", "deg_m_gfp_B"))
        if B["m_lasI_B"] > 0:
            props.append(P.gamma_m * B["m_lasI_B"])
            rxns.append(("B", "deg_m_lasI_B"))
        if B["Gi_B"] > 0:
            props.append(P.kmat * B["Gi_B"])
            rxns.append(("B", "mat_G_B"))
        if B["G_B"] > 0:
            props.append(P.gamma_G * B["G_B"])
            rxns.append(("B", "deg_G_B"))
        if B["LuxR"] > 0:
            props.append(P.gamma_p * B["LuxR"])
            rxns.append(("B", "deg_LuxR"))
        if B["LasI_B"] > 0:
            props.append(P.gamma_p * B["LasI_B"])
            rxns.append(("B", "deg_LasI_B"))
            props.append(P.k_S * B["LasI_B"])
            rxns.append(("E", "syn_S_B"))
        # Extracellular dynamics: decay + diffusion
        if AHL_A > 0:
            props.append(P.gamma_AHL * AHL_A)
            rxns.append(("E", "deg_AHL_A"))
            props.append(P.k_diff_ahl * AHL_A)
            rxns.append(("E", "diff_AHL_A_to_B"))
        if AHL_B > 0:
            props.append(P.gamma_AHL * AHL_B)
            rxns.append(("E", "deg_AHL_B"))
            props.append(P.k_diff_ahl * AHL_B)
            rxns.append(("E", "diff_AHL_B_to_A"))
        if S_A > 0:
            props.append(P.gamma_S * S_A)
            rxns.append(("E", "deg_S_A"))
            props.append(P.k_diff_s * S_A)
            rxns.append(("E", "diff_S_A_to_B"))
        if S_B > 0:
            props.append(P.gamma_S * S_B)
            rxns.append(("E", "deg_S_B"))
            props.append(P.k_diff_s * S_B)
            rxns.append(("E", "diff_S_B_to_A"))

        a0 = sum(props)
        if a0 <= 0:
            break
        tau = -math.log(rng.random()) / a0
        t += tau
        r2 = rng.random() * a0
        cum = 0.0
        idx = -1
        for i, p in enumerate(props):
            cum += p
            if r2 <= cum:
                idx = i
                break
        who, rx = rxns[idx]
        if who == "A":
            if rx == "bind_RS":
                A["LasR"] -= 1
                S_A -= 1
                A["RS"] += 1
            elif rx == "unbind_RS":
                A["RS"] -= 1
                A["LasR"] += 1
                S_A += 1
            elif rx == "act_pLas_gfp":
                A["RS"] -= 1
                A["pLas_gfp_free"] -= 1
                A["pLas_gfp_act"] += 1
            elif rx == "deact_pLas_gfp":
                A["pLas_gfp_act"] -= 1
                A["pLas_gfp_free"] += 1
                A["RS"] += 1
            elif rx == "bind_pCon_luxI":
                A["RNAP_free"] -= 1
                A["pCon_luxI_free"] -= 1
                A["pCon_luxI_RNAP"] += 1
            elif rx == "bind_pCon_lasR":
                A["RNAP_free"] -= 1
                A["pCon_lasR_free"] -= 1
                A["pCon_lasR_RNAP"] += 1
            elif rx == "bind_pLas_gfp_inact":
                A["RNAP_free"] -= 1
                A["pLas_gfp_free"] -= 1
                A["pLas_gfp_RNAP_inact"] += 1
            elif rx == "bind_pLas_gfp_act":
                A["RNAP_free"] -= 1
                A["pLas_gfp_act"] -= 1
                A["pLas_gfp_RNAP_act"] += 1
            elif rx == "tx_pCon_luxI":
                A["pCon_luxI_RNAP"] -= 1
                A["RNAP_free"] += 1
                A["pCon_luxI_free"] += 1
                A["m_luxI"] += 1
            elif rx == "tx_pCon_lasR":
                A["pCon_lasR_RNAP"] -= 1
                A["RNAP_free"] += 1
                A["pCon_lasR_free"] += 1
                A["m_lasR"] += 1
            elif rx == "tx_pLas_gfp_inact":
                A["pLas_gfp_RNAP_inact"] -= 1
                A["RNAP_free"] += 1
                A["pLas_gfp_free"] += 1
                A["m_gfp_A"] += 1
            elif rx == "tx_pLas_gfp_act":
                A["pLas_gfp_RNAP_act"] -= 1
                A["RNAP_free"] += 1
                A["pLas_gfp_act"] += 1
                A["m_gfp_A"] += 1
            elif rx == "bind_ribo_luxI":
                A["Ribo_free"] -= 1
                A["m_luxI"] -= 1
                A["ribo_luxI"] += 1
            elif rx == "bind_ribo_lasR":
                A["Ribo_free"] -= 1
                A["m_lasR"] -= 1
                A["ribo_lasR"] += 1
            elif rx == "bind_ribo_gfp_A":
                A["Ribo_free"] -= 1
                A["m_gfp_A"] -= 1
                A["ribo_gfp_A"] += 1
            elif rx == "tl_luxI":
                A["ribo_luxI"] -= 1
                A["Ribo_free"] += 1
                A["LuxI"] += 1
            elif rx == "tl_lasR":
                A["ribo_lasR"] -= 1
                A["Ribo_free"] += 1
                A["LasR"] += 1
            elif rx == "tl_gfp_A":
                A["ribo_gfp_A"] -= 1
                A["Ribo_free"] += 1
                A["Gi_A"] += 1
            elif rx == "deg_m_luxI":
                A["m_luxI"] -= 1
            elif rx == "deg_m_lasR":
                A["m_lasR"] -= 1
            elif rx == "deg_m_gfp_A":
                A["m_gfp_A"] -= 1
            elif rx == "deg_LuxI":
                A["LuxI"] -= 1
            elif rx == "deg_LasR":
                A["LasR"] -= 1
            elif rx == "mat_G_A":
                A["Gi_A"] -= 1
                A["G_A"] += 1
            elif rx == "deg_G_A":
                A["G_A"] -= 1
        elif who == "B":
            if rx == "bind_pCon_luxR":
                B["RNAP_free"] -= 1
                B["pCon_luxR_free"] -= 1
                B["pCon_luxR_RNAP"] += 1
            elif rx == "tx_pCon_luxR":
                B["pCon_luxR_RNAP"] -= 1
                B["RNAP_free"] += 1
                B["pCon_luxR_free"] += 1
                B["m_luxR"] += 1
            elif rx == "bind_ribo_luxR":
                B["Ribo_free"] -= 1
                B["m_luxR"] -= 1
                B["ribo_luxR"] += 1
            elif rx == "tl_luxR":
                B["ribo_luxR"] -= 1
                B["Ribo_free"] += 1
                B["LuxR"] += 1
            elif rx == "deg_m_luxR":
                B["m_luxR"] -= 1
            elif rx == "deg_LuxR":
                B["LuxR"] -= 1
            elif rx == "bind_RL":
                B["LuxR"] -= 1
                AHL_B -= 1
                B["RL"] += 1
            elif rx == "unbind_RL":
                B["RL"] -= 1
                B["LuxR"] += 1
                AHL_B += 1
            elif rx == "act_pLux_gfp":
                B["RL"] -= 1
                B["pLux_gfp_free"] -= 1
                B["pLux_gfp_act"] += 1
            elif rx == "act_pLux_lasI":
                B["RL"] -= 1
                B["pLux_lasI_free"] -= 1
                B["pLux_lasI_act"] += 1
            elif rx == "deact_pLux_gfp":
                B["pLux_gfp_act"] -= 1
                B["pLux_gfp_free"] += 1
                B["RL"] += 1
            elif rx == "deact_pLux_lasI":
                B["pLux_lasI_act"] -= 1
                B["pLux_lasI_free"] += 1
                B["RL"] += 1
            elif rx == "bind_pLux_gfp_inact":
                B["RNAP_free"] -= 1
                B["pLux_gfp_free"] -= 1
                B["pLux_gfp_RNAP_inact"] += 1
            elif rx == "bind_pLux_gfp_act":
                B["RNAP_free"] -= 1
                B["pLux_gfp_act"] -= 1
                B["pLux_gfp_RNAP_act"] += 1
            elif rx == "bind_pLux_lasI_inact":
                B["RNAP_free"] -= 1
                B["pLux_lasI_free"] -= 1
                B["pLux_lasI_RNAP_inact"] += 1
            elif rx == "bind_pLux_lasI_act":
                B["RNAP_free"] -= 1
                B["pLux_lasI_act"] -= 1
                B["pLux_lasI_RNAP_act"] += 1
            elif rx == "tx_pLux_gfp_inact":
                B["pLux_gfp_RNAP_inact"] -= 1
                B["RNAP_free"] += 1
                B["pLux_gfp_free"] += 1
                B["m_gfp_B"] += 1
            elif rx == "tx_pLux_gfp_act":
                B["pLux_gfp_RNAP_act"] -= 1
                B["RNAP_free"] += 1
                B["pLux_gfp_act"] += 1
                B["m_gfp_B"] += 1
            elif rx == "tx_pLux_lasI_inact":
                B["pLux_lasI_RNAP_inact"] -= 1
                B["RNAP_free"] += 1
                B["pLux_lasI_free"] += 1
                B["m_lasI_B"] += 1
            elif rx == "tx_pLux_lasI_act":
                B["pLux_lasI_RNAP_act"] -= 1
                B["RNAP_free"] += 1
                B["pLux_lasI_act"] += 1
                B["m_lasI_B"] += 1
            elif rx == "bind_ribo_gfp_B":
                B["Ribo_free"] -= 1
                B["m_gfp_B"] -= 1
                B["ribo_gfp_B"] += 1
            elif rx == "bind_ribo_lasI_B":
                B["Ribo_free"] -= 1
                B["m_lasI_B"] -= 1
                B["ribo_lasI_B"] += 1
            elif rx == "tl_gfp_B":
                B["ribo_gfp_B"] -= 1
                B["Ribo_free"] += 1
                B["Gi_B"] += 1
            elif rx == "tl_lasI_B":
                B["ribo_lasI_B"] -= 1
                B["Ribo_free"] += 1
                B["LasI_B"] += 1
            elif rx == "deg_m_gfp_B":
                B["m_gfp_B"] -= 1
            elif rx == "deg_m_lasI_B":
                B["m_lasI_B"] -= 1
            elif rx == "mat_G_B":
                B["Gi_B"] -= 1
                B["G_B"] += 1
            elif rx == "deg_G_B":
                B["G_B"] -= 1
            elif rx == "deg_LasI_B":
                B["LasI_B"] -= 1
        else:
            if rx == "syn_AHL_A":
                AHL_A += 1
            elif rx == "syn_S_B":
                S_B += 1
            elif rx == "deg_AHL_A":
                AHL_A -= 1
            elif rx == "deg_AHL_B":
                AHL_B -= 1
            elif rx == "deg_S_A":
                S_A -= 1
            elif rx == "deg_S_B":
                S_B -= 1
            elif rx == "diff_AHL_A_to_B":
                AHL_A -= 1
                AHL_B += 1
            elif rx == "diff_AHL_B_to_A":
                AHL_B -= 1
                AHL_A += 1
            elif rx == "diff_S_A_to_B":
                S_A -= 1
                S_B += 1
            elif rx == "diff_S_B_to_A":
                S_B -= 1
                S_A += 1
        times.append(t)
        GA.append(A["G_A"])
        GB.append(B["G_B"])
    return np.array(times), np.array(GA), np.array(GB)


def run(N=500, T_end=6000, window=(2500, 3000), outdir="paracrine_c6_timelapse_out", seed=633,
        distance_cm=1.0, transfer_time_ahl=5.0, transfer_time_s=5.0,
        diff_coef_ahl=None, diff_coef_s=None, copies=10,
        RNAP_total=50, Ribo_total=200, make_plot=True):
    os.makedirs(outdir, exist_ok=True)
    P = Params6(RNAP_total=RNAP_total, Ribo_total=Ribo_total,
                T_end=T_end, distance_cm=distance_cm,
                transfer_time_ahl=transfer_time_ahl,
                transfer_time_s=transfer_time_s,
                diff_coef_ahl=diff_coef_ahl,
                diff_coef_s=diff_coef_s)
    P.n_pCon_luxI_A = copies
    P.n_pCon_lasR_A = copies
    P.n_pLas_gfp_A = copies
    P.n_pCon_luxR_B = copies
    P.n_pLux_gfp_B = copies
    P.n_pLux_lasI_B = copies
    t_grid = np.arange(int(T_end) + 1)
    GA_all = np.zeros((N, len(t_grid)))
    GB_all = np.zeros((N, len(t_grid)))
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2 ** 31 - 1, size=N, dtype=np.int64)
    for i in range(N):
        ti, GA, GB = gillespie_cond6(P, seed=int(seeds[i]))
        idx = np.searchsorted(ti, t_grid, side="right") - 1
        idx[idx < 0] = 0
        GA_all[i, :] = GA[idx]
        GB_all[i, :] = GB[idx]
    cols = [f"GA_t{t}" for t in t_grid] + [f"GB_t{t}" for t in t_grid]
    raw = np.hstack([GA_all, GB_all])
    pd.DataFrame(raw, columns=cols).assign(replicate_id=lambda d: np.arange(N)) \
      .to_csv(os.path.join(outdir, "paracrine_c6_timelapse_raw.csv"), index=False)
    w0, w1 = window
    A_means = GA_all[:, w0:w1 + 1].mean(axis=1)
    B_means = GB_all[:, w0:w1 + 1].mean(axis=1)
    TOT = A_means + B_means
    mean = float(TOT.mean())
    sd = float(TOT.std(ddof=0))
    cv = float(sd / mean) if mean > 0 else float("nan")
    cv2 = float(cv * cv) if mean > 0 else float("nan")
    pd.DataFrame([{
        "condition": "Paracrine (c-6)",
        "RNAP": RNAP_total,
        "Ribo": Ribo_total,
        "load": 0,
        "mean": mean,
        "sd": sd,
        "CV": cv,
        "CV2": cv2,
        "N": N,
        "T_end": T_end,
        "window_start": w0,
        "window_end": w1,
        "distance_cm": distance_cm,
        "transfer_time_ahl": transfer_time_ahl if diff_coef_ahl is None else None,
        "transfer_time_s": transfer_time_s if diff_coef_s is None else None,
        "diff_coef_ahl": P.effective_diff_coef_ahl,
        "diff_coef_s": P.effective_diff_coef_s,
        "k_diff_ahl": P.k_diff_ahl,
        "k_diff_s": P.k_diff_s,
        "copies": copies,
        "receptor_model": "pCon_lasR_and_pCon_luxR_expression",
    }]).to_csv(os.path.join(outdir, "paracrine_c6_summary.csv"), index=False)
    if not make_plot:
        return
    meanA = GA_all.mean(axis=0)
    meanB = GB_all.mean(axis=0)
    p10A = np.percentile(GA_all, 10, axis=0)
    p90A = np.percentile(GA_all, 90, axis=0)
    p10B = np.percentile(GB_all, 10, axis=0)
    p90B = np.percentile(GB_all, 90, axis=0)
    plt.figure(figsize=(10, 5))
    for i in range(min(N, 60)):
        plt.plot(t_grid, GA_all[i, :], alpha=0.05, linewidth=0.4, color="tab:green")
        plt.plot(t_grid, GB_all[i, :], alpha=0.05, linewidth=0.4, color="tab:blue")
    plt.plot(t_grid, meanA, linewidth=2.0, label="Mean A", color="tab:green")
    plt.plot(t_grid, meanB, linewidth=2.0, label="Mean B", color="tab:blue")
    plt.fill_between(t_grid, p10A, p90A, alpha=0.25, color="tab:green", label="10–90% A")
    plt.fill_between(t_grid, p10B, p90B, alpha=0.25, color="tab:blue", label="10–90% B")
    plt.axvspan(w0, w1, alpha=0.1, label="Steady window")
    plt.xlabel("Time")
    plt.ylabel("GFP (A & B)")
    plt.title(f"Paracrine (c-6) timelapse (distance = {distance_cm} cm, copies = {copies})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "paracrine_c6_timelapse_plot.png"), dpi=150)
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=500)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--w0", type=int, default=2500)
    ap.add_argument("--w1", type=int, default=3000)
    ap.add_argument("--distance", type=float, default=1.0,
                    help="Cell separation in cm controlling diffusion delay")
    ap.add_argument("--tau_ahl", type=float, default=5.0,
                    help="Mean travel time (s) for AHL from cell A to B; ignored if --diff_ahl is set")
    ap.add_argument("--tau_s", type=float, default=5.0,
                    help="Mean travel time (s) for S from cell B to A; ignored if --diff_s is set")
    ap.add_argument("--diff_ahl", type=float, default=None,
                    help="Diffusion coefficient for AHL (cm^2/s); overrides --tau_ahl if provided")
    ap.add_argument("--diff_s", type=float, default=None,
                    help="Diffusion coefficient for S (cm^2/s); overrides --tau_s if provided")
    ap.add_argument("--out", type=str, default="paracrine_c6_timelapse_out")
    ap.add_argument("--seed", type=int, default=633)
    ap.add_argument("--copies", type=int, default=10,
                    help="Uniform plasmid copy number for all plasmids in both cells")
    ap.add_argument("--RNAP", type=int, default=50, help="RNAP_total per cell")
    ap.add_argument("--Ribo", type=int, default=200, help="Ribo_total per cell")
    ap.add_argument("--no_plot", action="store_true", help="Skip PNG timelapse plotting")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(N=args.N, T_end=args.T, window=(args.w0, args.w1),
        outdir=args.out, seed=args.seed,
        distance_cm=args.distance,
        transfer_time_ahl=args.tau_ahl if args.diff_ahl is None else None,
        transfer_time_s=args.tau_s if args.diff_s is None else None,
        diff_coef_ahl=args.diff_ahl,
        diff_coef_s=args.diff_s,
        copies=args.copies,
        RNAP_total=args.RNAP,
        Ribo_total=args.Ribo,
        make_plot=not args.no_plot)
