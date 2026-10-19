#!/usr/bin/env python3
import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


EVENTS = [
    "bind_RL",
    "unbind_RL",
    "act_pLux_gfp",
    "deact_pLux_gfp",
    "bind_pLux_gfp_inact",
    "bind_pLux_gfp_act",
    "bind_pCon_luxI",
    "bind_pCon_luxR",
    "tx_pLux_gfp_inact",
    "tx_pLux_gfp_act",
    "tx_pCon_luxI",
    "tx_pCon_luxR",
    "bind_ribo_gfp",
    "bind_ribo_luxI",
    "bind_ribo_luxR",
    "tl_gfp",
    "tl_luxI",
    "tl_luxR",
    "deg_m_gfp",
    "deg_m_luxI",
    "deg_m_luxR",
    "deg_LuxI",
    "deg_LuxR",
    "mat_G",
    "deg_G",
    "syn_AHL",
    "deg_AHL",
    "deg_AHL_A",
    "deg_AHL_B",
    "diff_AHL_A_to_B",
    "diff_AHL_B_to_A",
]
EVENT_INDEX = {name: idx for idx, name in enumerate(EVENTS)}


@dataclass
class ParamsAuto:
    RNAP_total: int = 50
    Ribo_total: int = 200
    n_pLux_gfp: int = 10
    n_pCon_luxI: int = 10
    n_pCon_luxR: int = 10
    kon_act: float = 1e-3
    koff_act: float = 1e-3
    kon_RL: float = 1e-3
    koff_RL: float = 1e-2
    kon_RNAP: float = 5e-4
    kcat_tx_active: float = 0.08
    kcat_tx_basal: float = 0.008
    kon_ribo: float = 1e-3
    kcat_tl: float = 0.5
    gamma_m: float = 0.01
    gamma_p: float = 5e-4
    kmat: float = 0.01
    k_I: float = 0.02
    gamma_AHL: float = 0.001
    gamma_G: float = 0.01
    T_end: float = 3500.0


@dataclass
class ParamsPara:
    RNAP_total: int = 50
    Ribo_total: int = 200
    n_pCon_luxI_A: int = 10
    n_pCon_luxR_B: int = 10
    n_pLux_gfp_B: int = 10
    kon_act: float = 1e-3
    koff_act: float = 1e-3
    kon_RL: float = 1e-3
    koff_RL: float = 1e-2
    kon_RNAP: float = 5e-4
    kcat_tx_active: float = 0.08
    kcat_tx_basal: float = 0.008
    kon_ribo: float = 1e-3
    kcat_tl: float = 0.5
    gamma_m: float = 0.01
    gamma_p: float = 5e-4
    kmat: float = 0.01
    k_I: float = 0.02
    gamma_AHL: float = 0.001
    gamma_G: float = 0.01
    T_end: float = 3500.0
    distance_cm: float = 1.0
    transfer_time: float = 240.0

    @property
    def k_diff_ahl(self) -> float:
        if self.transfer_time <= 0:
            raise ValueError("transfer_time must be positive")
        return 1.0 / self.transfer_time

    @property
    def effective_diff_coef(self) -> float:
        return self.distance_cm ** 2 / (2.0 * self.transfer_time)


def n_bins(T_end: float, bin_size: float) -> int:
    return int(math.ceil(T_end / bin_size))


def record_event(counts: np.ndarray, t: float, T_end: float, bin_size: float, event: str) -> None:
    if t < 0 or t > T_end:
        return
    bin_idx = min(int(t // bin_size), counts.shape[0] - 1)
    counts[bin_idx, EVENT_INDEX[event]] += 1


def choose_reaction(rng: np.random.Generator, props: List[float]) -> Tuple[float, int]:
    a0 = sum(props)
    if a0 <= 0:
        return 0.0, -1
    tau = -math.log(rng.random()) / a0
    r2 = rng.random() * a0
    cum = 0.0
    for i, p in enumerate(props):
        cum += p
        if r2 <= cum:
            return tau, i
    return tau, len(props) - 1


def gillespie_autocrine(
    P: ParamsAuto,
    seed: int,
    bin_size: float,
    max_events: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    AHL = 0
    c = {
        "RNAP_free": P.RNAP_total,
        "Ribo_free": P.Ribo_total,
        "LuxR": 0,
        "RL": 0,
        "LuxI": 0,
        "Gi": 0,
        "G": 0,
        "pLux_gfp_free": P.n_pLux_gfp,
        "pLux_gfp_act": 0,
        "pLux_gfp_RNAP_inact": 0,
        "pLux_gfp_RNAP_act": 0,
        "pCon_luxI_free": P.n_pCon_luxI,
        "pCon_luxI_RNAP": 0,
        "pCon_luxR_free": P.n_pCon_luxR,
        "pCon_luxR_RNAP": 0,
        "m_gfp": 0,
        "m_luxI": 0,
        "m_luxR": 0,
        "ribo_gfp": 0,
        "ribo_luxI": 0,
        "ribo_luxR": 0,
    }
    t = 0.0
    event_count = 0
    times = [0.0]
    G_trace = [0]
    counts = np.zeros((n_bins(P.T_end, bin_size), len(EVENTS)), dtype=np.int64)

    while t < P.T_end and event_count < max_events:
        props: List[float] = []
        rxns: List[str] = []

        if c["LuxR"] > 0 and AHL > 0:
            props.append(P.kon_RL * c["LuxR"] * AHL)
            rxns.append("bind_RL")
        if c["RL"] > 0:
            props.append(P.koff_RL * c["RL"])
            rxns.append("unbind_RL")
        if c["RL"] > 0 and c["pLux_gfp_free"] > 0:
            props.append(P.kon_act * c["RL"] * c["pLux_gfp_free"])
            rxns.append("act_pLux_gfp")
        if c["pLux_gfp_act"] > 0:
            props.append(P.koff_act * c["pLux_gfp_act"])
            rxns.append("deact_pLux_gfp")
        if c["RNAP_free"] > 0:
            if c["pLux_gfp_free"] > 0:
                props.append(P.kon_RNAP * c["RNAP_free"] * c["pLux_gfp_free"])
                rxns.append("bind_pLux_gfp_inact")
            if c["pLux_gfp_act"] > 0:
                props.append(P.kon_RNAP * c["RNAP_free"] * c["pLux_gfp_act"])
                rxns.append("bind_pLux_gfp_act")
            if c["pCon_luxI_free"] > 0:
                props.append(P.kon_RNAP * c["RNAP_free"] * c["pCon_luxI_free"])
                rxns.append("bind_pCon_luxI")
            if c["pCon_luxR_free"] > 0:
                props.append(P.kon_RNAP * c["RNAP_free"] * c["pCon_luxR_free"])
                rxns.append("bind_pCon_luxR")
        if c["pLux_gfp_RNAP_inact"] > 0:
            props.append(P.kcat_tx_basal * c["pLux_gfp_RNAP_inact"])
            rxns.append("tx_pLux_gfp_inact")
        if c["pLux_gfp_RNAP_act"] > 0:
            props.append(P.kcat_tx_active * c["pLux_gfp_RNAP_act"])
            rxns.append("tx_pLux_gfp_act")
        if c["pCon_luxI_RNAP"] > 0:
            props.append(P.kcat_tx_active * c["pCon_luxI_RNAP"])
            rxns.append("tx_pCon_luxI")
        if c["pCon_luxR_RNAP"] > 0:
            props.append(P.kcat_tx_active * c["pCon_luxR_RNAP"])
            rxns.append("tx_pCon_luxR")
        if c["Ribo_free"] > 0 and c["m_gfp"] > 0:
            props.append(P.kon_ribo * c["Ribo_free"] * c["m_gfp"])
            rxns.append("bind_ribo_gfp")
        if c["Ribo_free"] > 0 and c["m_luxI"] > 0:
            props.append(P.kon_ribo * c["Ribo_free"] * c["m_luxI"])
            rxns.append("bind_ribo_luxI")
        if c["Ribo_free"] > 0 and c["m_luxR"] > 0:
            props.append(P.kon_ribo * c["Ribo_free"] * c["m_luxR"])
            rxns.append("bind_ribo_luxR")
        if c["ribo_gfp"] > 0:
            props.append(P.kcat_tl * c["ribo_gfp"])
            rxns.append("tl_gfp")
        if c["ribo_luxI"] > 0:
            props.append(P.kcat_tl * c["ribo_luxI"])
            rxns.append("tl_luxI")
        if c["ribo_luxR"] > 0:
            props.append(P.kcat_tl * c["ribo_luxR"])
            rxns.append("tl_luxR")
        if c["m_gfp"] > 0:
            props.append(P.gamma_m * c["m_gfp"])
            rxns.append("deg_m_gfp")
        if c["m_luxI"] > 0:
            props.append(P.gamma_m * c["m_luxI"])
            rxns.append("deg_m_luxI")
        if c["m_luxR"] > 0:
            props.append(P.gamma_m * c["m_luxR"])
            rxns.append("deg_m_luxR")
        if c["LuxI"] > 0:
            props.append(P.gamma_p * c["LuxI"])
            rxns.append("deg_LuxI")
        if c["LuxR"] > 0:
            props.append(P.gamma_p * c["LuxR"])
            rxns.append("deg_LuxR")
        if c["Gi"] > 0:
            props.append(P.kmat * c["Gi"])
            rxns.append("mat_G")
        if c["G"] > 0:
            props.append(P.gamma_G * c["G"])
            rxns.append("deg_G")
        if c["LuxI"] > 0:
            props.append(P.k_I * c["LuxI"])
            rxns.append("syn_AHL")
        if AHL > 0:
            props.append(P.gamma_AHL * AHL)
            rxns.append("deg_AHL")

        tau, idx = choose_reaction(rng, props)
        if idx < 0:
            break
        t += tau
        rx = rxns[idx]
        record_event(counts, t, P.T_end, bin_size, rx)
        event_count += 1

        if rx == "bind_RL":
            c["LuxR"] -= 1; AHL -= 1; c["RL"] += 1
        elif rx == "unbind_RL":
            c["RL"] -= 1; c["LuxR"] += 1; AHL += 1
        elif rx == "act_pLux_gfp":
            c["RL"] -= 1; c["pLux_gfp_free"] -= 1; c["pLux_gfp_act"] += 1
        elif rx == "deact_pLux_gfp":
            c["pLux_gfp_act"] -= 1; c["pLux_gfp_free"] += 1; c["RL"] += 1
        elif rx == "bind_pLux_gfp_inact":
            c["RNAP_free"] -= 1; c["pLux_gfp_free"] -= 1; c["pLux_gfp_RNAP_inact"] += 1
        elif rx == "bind_pLux_gfp_act":
            c["RNAP_free"] -= 1; c["pLux_gfp_act"] -= 1; c["pLux_gfp_RNAP_act"] += 1
        elif rx == "bind_pCon_luxI":
            c["RNAP_free"] -= 1; c["pCon_luxI_free"] -= 1; c["pCon_luxI_RNAP"] += 1
        elif rx == "bind_pCon_luxR":
            c["RNAP_free"] -= 1; c["pCon_luxR_free"] -= 1; c["pCon_luxR_RNAP"] += 1
        elif rx == "tx_pLux_gfp_inact":
            c["pLux_gfp_RNAP_inact"] -= 1; c["RNAP_free"] += 1; c["pLux_gfp_free"] += 1; c["m_gfp"] += 1
        elif rx == "tx_pLux_gfp_act":
            c["pLux_gfp_RNAP_act"] -= 1; c["RNAP_free"] += 1; c["pLux_gfp_act"] += 1; c["m_gfp"] += 1
        elif rx == "tx_pCon_luxI":
            c["pCon_luxI_RNAP"] -= 1; c["RNAP_free"] += 1; c["pCon_luxI_free"] += 1; c["m_luxI"] += 1
        elif rx == "tx_pCon_luxR":
            c["pCon_luxR_RNAP"] -= 1; c["RNAP_free"] += 1; c["pCon_luxR_free"] += 1; c["m_luxR"] += 1
        elif rx == "bind_ribo_gfp":
            c["Ribo_free"] -= 1; c["m_gfp"] -= 1; c["ribo_gfp"] += 1
        elif rx == "bind_ribo_luxI":
            c["Ribo_free"] -= 1; c["m_luxI"] -= 1; c["ribo_luxI"] += 1
        elif rx == "bind_ribo_luxR":
            c["Ribo_free"] -= 1; c["m_luxR"] -= 1; c["ribo_luxR"] += 1
        elif rx == "tl_gfp":
            c["ribo_gfp"] -= 1; c["Ribo_free"] += 1; c["Gi"] += 1
        elif rx == "tl_luxI":
            c["ribo_luxI"] -= 1; c["Ribo_free"] += 1; c["LuxI"] += 1
        elif rx == "tl_luxR":
            c["ribo_luxR"] -= 1; c["Ribo_free"] += 1; c["LuxR"] += 1
        elif rx == "deg_m_gfp":
            c["m_gfp"] -= 1
        elif rx == "deg_m_luxI":
            c["m_luxI"] -= 1
        elif rx == "deg_m_luxR":
            c["m_luxR"] -= 1
        elif rx == "deg_LuxI":
            c["LuxI"] -= 1
        elif rx == "deg_LuxR":
            c["LuxR"] -= 1
        elif rx == "mat_G":
            c["Gi"] -= 1; c["G"] += 1
        elif rx == "deg_G":
            c["G"] -= 1
        elif rx == "syn_AHL":
            AHL += 1
        elif rx == "deg_AHL":
            AHL -= 1

        times.append(t)
        G_trace.append(c["G"])

    qc = {"last_t": t, "event_count": event_count, "hit_event_cap": float(event_count >= max_events)}
    return np.array(times), np.array(G_trace), counts, qc


def gillespie_paracrine(
    P: ParamsPara,
    seed: int,
    bin_size: float,
    max_events: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    AHL_A = 0
    AHL_B = 0
    A = {
        "RNAP_free": P.RNAP_total,
        "Ribo_free": P.Ribo_total,
        "pCon_luxI_free": P.n_pCon_luxI_A,
        "pCon_luxI_RNAP": 0,
        "m_luxI": 0,
        "ribo_luxI": 0,
        "LuxI": 0,
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
        "m_luxR": 0,
        "m_gfp": 0,
        "ribo_luxR": 0,
        "ribo_gfp": 0,
        "Gi": 0,
        "G": 0,
    }
    t = 0.0
    event_count = 0
    times = [0.0]
    G_trace = [0]
    counts = np.zeros((n_bins(P.T_end, bin_size), len(EVENTS)), dtype=np.int64)

    while t < P.T_end and event_count < max_events:
        props: List[float] = []
        rxns: List[Tuple[str, str]] = []

        if A["RNAP_free"] > 0 and A["pCon_luxI_free"] > 0:
            props.append(P.kon_RNAP * A["RNAP_free"] * A["pCon_luxI_free"])
            rxns.append(("A", "bind_pCon_luxI"))
        if A["pCon_luxI_RNAP"] > 0:
            props.append(P.kcat_tx_active * A["pCon_luxI_RNAP"])
            rxns.append(("A", "tx_pCon_luxI"))
        if A["Ribo_free"] > 0 and A["m_luxI"] > 0:
            props.append(P.kon_ribo * A["Ribo_free"] * A["m_luxI"])
            rxns.append(("A", "bind_ribo_luxI"))
        if A["ribo_luxI"] > 0:
            props.append(P.kcat_tl * A["ribo_luxI"])
            rxns.append(("A", "tl_luxI"))
        if A["m_luxI"] > 0:
            props.append(P.gamma_m * A["m_luxI"])
            rxns.append(("A", "deg_m_luxI"))
        if A["LuxI"] > 0:
            props.append(P.gamma_p * A["LuxI"])
            rxns.append(("A", "deg_LuxI"))
            props.append(P.k_I * A["LuxI"])
            rxns.append(("E", "syn_AHL"))

        if B["RNAP_free"] > 0 and B["pCon_luxR_free"] > 0:
            props.append(P.kon_RNAP * B["RNAP_free"] * B["pCon_luxR_free"])
            rxns.append(("B", "bind_pCon_luxR"))
        if B["pCon_luxR_RNAP"] > 0:
            props.append(P.kcat_tx_active * B["pCon_luxR_RNAP"])
            rxns.append(("B", "tx_pCon_luxR"))
        if B["Ribo_free"] > 0 and B["m_luxR"] > 0:
            props.append(P.kon_ribo * B["Ribo_free"] * B["m_luxR"])
            rxns.append(("B", "bind_ribo_luxR"))
        if B["ribo_luxR"] > 0:
            props.append(P.kcat_tl * B["ribo_luxR"])
            rxns.append(("B", "tl_luxR"))
        if B["m_luxR"] > 0:
            props.append(P.gamma_m * B["m_luxR"])
            rxns.append(("B", "deg_m_luxR"))
        if B["LuxR"] > 0:
            props.append(P.gamma_p * B["LuxR"])
            rxns.append(("B", "deg_LuxR"))
        if B["LuxR"] > 0 and AHL_B > 0:
            props.append(P.kon_RL * B["LuxR"] * AHL_B)
            rxns.append(("B", "bind_RL"))
        if B["RL"] > 0:
            props.append(P.koff_RL * B["RL"])
            rxns.append(("B", "unbind_RL"))
        if B["RL"] > 0 and B["pLux_gfp_free"] > 0:
            props.append(P.kon_act * B["RL"] * B["pLux_gfp_free"])
            rxns.append(("B", "act_pLux_gfp"))
        if B["pLux_gfp_act"] > 0:
            props.append(P.koff_act * B["pLux_gfp_act"])
            rxns.append(("B", "deact_pLux_gfp"))
        if B["RNAP_free"] > 0:
            if B["pLux_gfp_free"] > 0:
                props.append(P.kon_RNAP * B["RNAP_free"] * B["pLux_gfp_free"])
                rxns.append(("B", "bind_pLux_gfp_inact"))
            if B["pLux_gfp_act"] > 0:
                props.append(P.kon_RNAP * B["RNAP_free"] * B["pLux_gfp_act"])
                rxns.append(("B", "bind_pLux_gfp_act"))
        if B["pLux_gfp_RNAP_inact"] > 0:
            props.append(P.kcat_tx_basal * B["pLux_gfp_RNAP_inact"])
            rxns.append(("B", "tx_pLux_gfp_inact"))
        if B["pLux_gfp_RNAP_act"] > 0:
            props.append(P.kcat_tx_active * B["pLux_gfp_RNAP_act"])
            rxns.append(("B", "tx_pLux_gfp_act"))
        if B["Ribo_free"] > 0 and B["m_gfp"] > 0:
            props.append(P.kon_ribo * B["Ribo_free"] * B["m_gfp"])
            rxns.append(("B", "bind_ribo_gfp"))
        if B["ribo_gfp"] > 0:
            props.append(P.kcat_tl * B["ribo_gfp"])
            rxns.append(("B", "tl_gfp"))
        if B["m_gfp"] > 0:
            props.append(P.gamma_m * B["m_gfp"])
            rxns.append(("B", "deg_m_gfp"))
        if B["Gi"] > 0:
            props.append(P.kmat * B["Gi"])
            rxns.append(("B", "mat_G"))
        if B["G"] > 0:
            props.append(P.gamma_G * B["G"])
            rxns.append(("B", "deg_G"))

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

        tau, idx = choose_reaction(rng, props)
        if idx < 0:
            break
        t += tau
        who, rx = rxns[idx]
        record_event(counts, t, P.T_end, bin_size, rx)
        event_count += 1

        if who == "A":
            if rx == "bind_pCon_luxI":
                A["RNAP_free"] -= 1; A["pCon_luxI_free"] -= 1; A["pCon_luxI_RNAP"] += 1
            elif rx == "tx_pCon_luxI":
                A["pCon_luxI_RNAP"] -= 1; A["RNAP_free"] += 1; A["pCon_luxI_free"] += 1; A["m_luxI"] += 1
            elif rx == "bind_ribo_luxI":
                A["Ribo_free"] -= 1; A["m_luxI"] -= 1; A["ribo_luxI"] += 1
            elif rx == "tl_luxI":
                A["ribo_luxI"] -= 1; A["Ribo_free"] += 1; A["LuxI"] += 1
            elif rx == "deg_m_luxI":
                A["m_luxI"] -= 1
            elif rx == "deg_LuxI":
                A["LuxI"] -= 1
        elif who == "B":
            if rx == "bind_pCon_luxR":
                B["RNAP_free"] -= 1; B["pCon_luxR_free"] -= 1; B["pCon_luxR_RNAP"] += 1
            elif rx == "tx_pCon_luxR":
                B["pCon_luxR_RNAP"] -= 1; B["RNAP_free"] += 1; B["pCon_luxR_free"] += 1; B["m_luxR"] += 1
            elif rx == "bind_ribo_luxR":
                B["Ribo_free"] -= 1; B["m_luxR"] -= 1; B["ribo_luxR"] += 1
            elif rx == "tl_luxR":
                B["ribo_luxR"] -= 1; B["Ribo_free"] += 1; B["LuxR"] += 1
            elif rx == "deg_m_luxR":
                B["m_luxR"] -= 1
            elif rx == "deg_LuxR":
                B["LuxR"] -= 1
            elif rx == "bind_RL":
                B["LuxR"] -= 1; AHL_B -= 1; B["RL"] += 1
            elif rx == "unbind_RL":
                B["RL"] -= 1; B["LuxR"] += 1; AHL_B += 1
            elif rx == "act_pLux_gfp":
                B["RL"] -= 1; B["pLux_gfp_free"] -= 1; B["pLux_gfp_act"] += 1
            elif rx == "deact_pLux_gfp":
                B["pLux_gfp_act"] -= 1; B["pLux_gfp_free"] += 1; B["RL"] += 1
            elif rx == "bind_pLux_gfp_inact":
                B["RNAP_free"] -= 1; B["pLux_gfp_free"] -= 1; B["pLux_gfp_RNAP_inact"] += 1
            elif rx == "bind_pLux_gfp_act":
                B["RNAP_free"] -= 1; B["pLux_gfp_act"] -= 1; B["pLux_gfp_RNAP_act"] += 1
            elif rx == "tx_pLux_gfp_inact":
                B["pLux_gfp_RNAP_inact"] -= 1; B["RNAP_free"] += 1; B["pLux_gfp_free"] += 1; B["m_gfp"] += 1
            elif rx == "tx_pLux_gfp_act":
                B["pLux_gfp_RNAP_act"] -= 1; B["RNAP_free"] += 1; B["pLux_gfp_act"] += 1; B["m_gfp"] += 1
            elif rx == "bind_ribo_gfp":
                B["Ribo_free"] -= 1; B["m_gfp"] -= 1; B["ribo_gfp"] += 1
            elif rx == "tl_gfp":
                B["ribo_gfp"] -= 1; B["Ribo_free"] += 1; B["Gi"] += 1
            elif rx == "deg_m_gfp":
                B["m_gfp"] -= 1
            elif rx == "mat_G":
                B["Gi"] -= 1; B["G"] += 1
            elif rx == "deg_G":
                B["G"] -= 1
        else:
            if rx == "syn_AHL":
                AHL_A += 1
            elif rx == "deg_AHL_A":
                AHL_A -= 1
            elif rx == "deg_AHL_B":
                AHL_B -= 1
            elif rx == "diff_AHL_A_to_B":
                AHL_A -= 1; AHL_B += 1
            elif rx == "diff_AHL_B_to_A":
                AHL_B -= 1; AHL_A += 1

        times.append(t)
        G_trace.append(B["G"])

    qc = {"last_t": t, "event_count": event_count, "hit_event_cap": float(event_count >= max_events)}
    return np.array(times), np.array(G_trace), counts, qc


def sample_trace(times: np.ndarray, values: np.ndarray, T_end: int) -> np.ndarray:
    t_grid = np.arange(T_end + 1)
    idx = np.searchsorted(times, t_grid, side="right") - 1
    idx[idx < 0] = 0
    return values[idx]


def cv_summary(block: np.ndarray) -> Dict[str, float]:
    mean = float(block.mean())
    sd = float(block.std(ddof=0))
    cv = float(sd / mean) if mean > 0 else float("nan")
    return {"mean": mean, "sd": sd, "CV": cv, "CV2": float(cv * cv) if mean > 0 else float("nan")}


def write_reporter_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_event_bins(path: Path, condition_rows: List[Tuple[str, int, int, np.ndarray]], bin_size: int) -> None:
    fieldnames = ["condition", "copies", "replicate_id", "bin_start_s", "bin_end_s"] + EVENTS
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, copies, rep_id, counts in condition_rows:
            for bin_idx in range(counts.shape[0]):
                row = {
                    "condition": condition,
                    "copies": copies,
                    "replicate_id": rep_id,
                    "bin_start_s": bin_idx * bin_size,
                    "bin_end_s": (bin_idx + 1) * bin_size,
                }
                row.update({event: int(counts[bin_idx, EVENT_INDEX[event]]) for event in EVENTS})
                writer.writerow(row)


def write_event_summaries(outdir: Path, condition_rows: List[Tuple[str, int, int, np.ndarray]], bin_size: int, window: Tuple[int, int]) -> None:
    by_condition: Dict[Tuple[str, int], List[np.ndarray]] = {}
    for condition, copies, _rep_id, counts in condition_rows:
        by_condition.setdefault((condition, copies), []).append(counts)

    with (outdir / "event_bins_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "copies", "bin_start_s", "bin_end_s", "event", "mean_count", "sd_count", "total_count"],
        )
        writer.writeheader()
        for (condition, copies), matrices in sorted(by_condition.items()):
            stack = np.stack(matrices, axis=0)
            for bin_idx in range(stack.shape[1]):
                for event in EVENTS:
                    vals = stack[:, bin_idx, EVENT_INDEX[event]]
                    writer.writerow({
                        "condition": condition,
                        "copies": copies,
                        "bin_start_s": bin_idx * bin_size,
                        "bin_end_s": (bin_idx + 1) * bin_size,
                        "event": event,
                        "mean_count": float(vals.mean()),
                        "sd_count": float(vals.std(ddof=0)),
                        "total_count": int(vals.sum()),
                    })

    w0, w1 = window
    first_bin = int(w0 // bin_size)
    last_bin = int(math.ceil(w1 / bin_size)) - 1
    with (outdir / "event_totals_by_replicate.csv").open("w", newline="") as handle:
        fieldnames = ["condition", "copies", "replicate_id", "period"] + EVENTS
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, copies, rep_id, counts in condition_rows:
            for period, block in [
                ("full", counts),
                ("steady_window", counts[first_bin : last_bin + 1, :]),
            ]:
                totals = block.sum(axis=0)
                row = {"condition": condition, "copies": copies, "replicate_id": rep_id, "period": period}
                row.update({event: int(totals[EVENT_INDEX[event]]) for event in EVENTS})
                writer.writerow(row)


def run_condition(
    condition: str,
    copies: int,
    N: int,
    T_end: int,
    window: Tuple[int, int],
    bin_size: int,
    RNAP_total: int,
    Ribo_total: int,
    tau: float,
    seed: int,
    max_events: int,
) -> Tuple[List[Dict[str, object]], List[Tuple[str, int, int, np.ndarray]], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**31 - 1, size=N, dtype=np.int64)
    curves = np.zeros((N, T_end + 1))
    event_rows = []
    qc_rows = []

    for rep_id, rep_seed in enumerate(seeds):
        if condition == "autocrine_nc3":
            params = ParamsAuto(
                RNAP_total=RNAP_total,
                Ribo_total=Ribo_total,
                n_pLux_gfp=copies,
                n_pCon_luxI=copies,
                n_pCon_luxR=copies,
                T_end=T_end,
            )
            times, trace, counts, qc = gillespie_autocrine(params, int(rep_seed), bin_size, max_events)
        elif condition == "paracrine_c3":
            params = ParamsPara(
                RNAP_total=RNAP_total,
                Ribo_total=Ribo_total,
                n_pCon_luxI_A=copies,
                n_pCon_luxR_B=copies,
                n_pLux_gfp_B=copies,
                T_end=T_end,
                transfer_time=tau,
            )
            times, trace, counts, qc = gillespie_paracrine(params, int(rep_seed), bin_size, max_events)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        curves[rep_id, :] = sample_trace(times, trace, T_end)
        event_rows.append((condition, copies, rep_id, counts))
        qc_rows.append({
            "condition": condition,
            "copies": copies,
            "replicate_id": rep_id,
            "last_t": qc["last_t"],
            "event_count": int(qc["event_count"]),
            "hit_event_cap": int(qc["hit_event_cap"]),
        })

    w0, w1 = window
    block = curves[:, w0 : w1 + 1].mean(axis=1)
    summary = cv_summary(block)
    summary_row = {
        "condition": condition,
        "copies": copies,
        "mean": summary["mean"],
        "sd": summary["sd"],
        "CV": summary["CV"],
        "CV2": summary["CV2"],
        "N": N,
        "T_end": T_end,
        "window_start": w0,
        "window_end": w1,
        "bin_size_s": bin_size,
        "RNAP_total": RNAP_total,
        "Ribo_total": Ribo_total,
        "transfer_time_s": tau if condition == "paracrine_c3" else "",
        "max_events": max_events,
    }
    return [summary_row], event_rows, qc_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run binned SSA event counters for Autocrine (nc-3) vs Paracrine (c-3)."
    )
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--T", type=int, default=3500)
    parser.add_argument("--w0", type=int, default=3000)
    parser.add_argument("--w1", type=int, default=3500)
    parser.add_argument("--bin-size", type=int, default=10)
    parser.add_argument("--copies", nargs="+", type=int, default=[5, 40])
    parser.add_argument("--RNAP", type=int, default=50)
    parser.add_argument("--Ribo", type=int, default=200)
    parser.add_argument("--tau", type=float, default=240.0)
    parser.add_argument("--max-events", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reporter_rows: List[Dict[str, object]] = []
    all_event_rows: List[Tuple[str, int, int, np.ndarray]] = []
    qc_rows: List[Dict[str, object]] = []

    for copy_index, copies in enumerate(args.copies):
        for condition_index, condition in enumerate(["autocrine_nc3", "paracrine_c3"]):
            condition_seed = args.seed + copy_index * 1000 + condition_index * 100
            print(f"Running {condition}, copies={copies}, N={args.N}")
            summaries, event_rows, qc = run_condition(
                condition=condition,
                copies=copies,
                N=args.N,
                T_end=args.T,
                window=(args.w0, args.w1),
                bin_size=args.bin_size,
                RNAP_total=args.RNAP,
                Ribo_total=args.Ribo,
                tau=args.tau,
                seed=condition_seed,
                max_events=args.max_events,
            )
            reporter_rows.extend(summaries)
            all_event_rows.extend(event_rows)
            qc_rows.extend(qc)

    write_reporter_summary(args.out / "reporter_summary.csv", reporter_rows)
    write_event_bins(args.out / "event_bins_by_replicate.csv", all_event_rows, args.bin_size)
    write_event_summaries(args.out, all_event_rows, args.bin_size, (args.w0, args.w1))
    with (args.out / "run_qc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "copies", "replicate_id", "last_t", "event_count", "hit_event_cap"])
        writer.writeheader()
        writer.writerows(qc_rows)

    print(f"Done. Outputs written to {args.out}")


if __name__ == "__main__":
    main()
