#!/usr/bin/env python3
"""
Paracrine (c-3) timelapse model with finite signal-transfer time.

Cell A (LuxI producer) and Cell B (pCon-LuxR/pLux-GFP responder) are separated by
a specified distance, so AHL must diffuse between two extracellular
sub-compartments before activating Cell B. Diffusion is approximated as a
first-order exchange with rate k_diff. By default we set k_diff = 1 / tau with
tau = 5 s (mean travel time from Cell A to Cell B). If a diffusion coefficient D
(cm²/s) is supplied instead, we use k_diff = 2·D / L² for distance L (cm).

Defaults for direct model use: N=500, T_end=6000, distance=1 cm, tau=5 s.
Outputs include raw reporter traces, a compact summary table, and an optional timelapse plot.
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
    """Return first-order exchange rate constant (s⁻¹) for two compartments."""
    if distance_cm <= 0:
        raise ValueError("distance_cm must be positive")
    return 2.0 * diff_coef / (distance_cm ** 2)


@dataclass
class Params5:
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
    T_end: float = 6000.0
    distance_cm: float = 1.0
    transfer_time: Optional[float] = 5.0  # seconds; mean hop time A -> B
    diff_coef_ahl: Optional[float] = None  # cm^2/s, overrides transfer_time if set

    @property
    def k_diff_ahl(self) -> float:
        if self.diff_coef_ahl is not None:
            return _exchange_rate(self.diff_coef_ahl, self.distance_cm)
        if self.transfer_time is None or self.transfer_time <= 0:
            raise ValueError("Either transfer_time (>0) or diff_coef_ahl must be provided")
        return 1.0 / self.transfer_time

    @property
    def effective_diff_coef(self) -> float:
        if self.diff_coef_ahl is not None:
            return self.diff_coef_ahl
        return self.distance_cm ** 2 / (2.0 * self.transfer_time)


def gillespie_cond5(P: Params5, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    AHL_A = 0  # extracellular near cell A
    AHL_B = 0  # extracellular near cell B
    A = {
        "RNAP_free": P.RNAP_total,
        "Ribo_free": P.Ribo_total,
        "pCon_luxI_free": P.n_pCon_luxI_A,
        "pCon_luxI_RNAP": 0,
        "m_luxI": 0,
        "ribo_luxI": 0,
        "LuxI": 0,
        "Gi": 0,
        "G": 0,
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
    it = 0
    max_it = int(2e6)
    times = [0.0]
    GA = [0]
    GB = [0]
    while t < P.T_end and it < max_it:
        it += 1
        props = []
        rxns = []
        # Cell A: LuxI production and secretion to local pool
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
            rxns.append(("E", "syn_AHL_A"))
        # Cell B: constitutive LuxR expression + pLux-GFP sensing local pool B
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
        # Extracellular processes
        if AHL_A > 0:
            props.append(P.gamma_AHL * AHL_A)
            rxns.append(("E", "deg_AHL_A"))
            props.append(P.k_diff_ahl * AHL_A)
            rxns.append(("E", "diff_A_to_B"))
        if AHL_B > 0:
            props.append(P.gamma_AHL * AHL_B)
            rxns.append(("E", "deg_AHL_B"))
            props.append(P.k_diff_ahl * AHL_B)
            rxns.append(("E", "diff_B_to_A"))

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
            if rx == "bind_pCon_luxI":
                A["RNAP_free"] -= 1
                A["pCon_luxI_free"] -= 1
                A["pCon_luxI_RNAP"] += 1
            elif rx == "tx_pCon_luxI":
                A["pCon_luxI_RNAP"] -= 1
                A["RNAP_free"] += 1
                A["pCon_luxI_free"] += 1
                A["m_luxI"] += 1
            elif rx == "bind_ribo_luxI":
                A["Ribo_free"] -= 1
                A["m_luxI"] -= 1
                A["ribo_luxI"] += 1
            elif rx == "tl_luxI":
                A["ribo_luxI"] -= 1
                A["Ribo_free"] += 1
                A["LuxI"] += 1
            elif rx == "deg_m_luxI":
                A["m_luxI"] -= 1
            elif rx == "deg_LuxI":
                A["LuxI"] -= 1
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
            elif rx == "deact_pLux_gfp":
                B["pLux_gfp_act"] -= 1
                B["pLux_gfp_free"] += 1
                B["RL"] += 1
            elif rx == "bind_pLux_gfp_inact":
                B["RNAP_free"] -= 1
                B["pLux_gfp_free"] -= 1
                B["pLux_gfp_RNAP_inact"] += 1
            elif rx == "bind_pLux_gfp_act":
                B["RNAP_free"] -= 1
                B["pLux_gfp_act"] -= 1
                B["pLux_gfp_RNAP_act"] += 1
            elif rx == "tx_pLux_gfp_inact":
                B["pLux_gfp_RNAP_inact"] -= 1
                B["RNAP_free"] += 1
                B["pLux_gfp_free"] += 1
                B["m_gfp"] += 1
            elif rx == "tx_pLux_gfp_act":
                B["pLux_gfp_RNAP_act"] -= 1
                B["RNAP_free"] += 1
                B["pLux_gfp_act"] += 1
                B["m_gfp"] += 1
            elif rx == "bind_ribo_gfp":
                B["Ribo_free"] -= 1
                B["m_gfp"] -= 1
                B["ribo_gfp"] += 1
            elif rx == "tl_gfp":
                B["ribo_gfp"] -= 1
                B["Ribo_free"] += 1
                B["Gi"] += 1
            elif rx == "deg_m_gfp":
                B["m_gfp"] -= 1
            elif rx == "mat_G":
                B["Gi"] -= 1
                B["G"] += 1
            elif rx == "deg_G":
                B["G"] -= 1
        else:
            if rx == "syn_AHL_A":
                AHL_A += 1
            elif rx == "deg_AHL_A":
                AHL_A -= 1
            elif rx == "deg_AHL_B":
                AHL_B -= 1
            elif rx == "diff_A_to_B":
                AHL_A -= 1
                AHL_B += 1
            elif rx == "diff_B_to_A":
                AHL_B -= 1
                AHL_A += 1
        times.append(t)
        GA.append(A["G"])
        GB.append(B["G"])
    return np.array(times), np.array(GA), np.array(GB)


def run(N=500, T_end=6000, window=(2500, 3000), outdir="paracrine_c3_timelapse_out", seed=522,
        distance_cm=1.0, transfer_time=5.0, diff_coef_ahl=None, copies=10,
        RNAP_total=50, Ribo_total=200, make_plot=True):
    os.makedirs(outdir, exist_ok=True)
    P = Params5(RNAP_total=RNAP_total, Ribo_total=Ribo_total,
                T_end=T_end, distance_cm=distance_cm,
                transfer_time=transfer_time, diff_coef_ahl=diff_coef_ahl)
    P.n_pCon_luxI_A = copies
    P.n_pCon_luxR_B = copies
    P.n_pLux_gfp_B = copies
    t_grid = np.arange(int(T_end) + 1)
    GA_all = np.zeros((N, len(t_grid)))
    GB_all = np.zeros((N, len(t_grid)))
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2 ** 31 - 1, size=N, dtype=np.int64)
    for i in range(N):
        ti, GA, GB = gillespie_cond5(P, seed=int(seeds[i]))
        idx = np.searchsorted(ti, t_grid, side="right") - 1
        idx[idx < 0] = 0
        GA_all[i, :] = GA[idx]
        GB_all[i, :] = GB[idx]
    # save raw
    cols = [f"GA_t{t}" for t in t_grid] + [f"GB_t{t}" for t in t_grid]
    raw = np.hstack([GA_all, GB_all])
    pd.DataFrame(raw, columns=cols).assign(replicate_id=lambda d: np.arange(N)) \
      .to_csv(os.path.join(outdir, "paracrine_c3_timelapse_raw.csv"), index=False)
    # summary on GFP_total
    w0, w1 = window
    A_means = GA_all[:, w0:w1 + 1].mean(axis=1)
    B_means = GB_all[:, w0:w1 + 1].mean(axis=1)
    TOT = A_means + B_means
    mean = float(TOT.mean())
    sd = float(TOT.std(ddof=0))
    cv = float(sd / mean) if mean > 0 else float("nan")
    cv2 = float(cv * cv) if mean > 0 else float("nan")
    pd.DataFrame([{
        "condition": "Paracrine (c-3)",
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
        "transfer_time": transfer_time if diff_coef_ahl is None else None,
        "diff_coef_ahl": P.effective_diff_coef,
        "k_diff_ahl": P.k_diff_ahl,
        "copies": copies,
        "receptor_model": "pCon_luxR_expression",
    }]).to_csv(os.path.join(outdir, "paracrine_c3_summary.csv"), index=False)
    if not make_plot:
        return
    # plot GB
    meanB = GB_all.mean(axis=0)
    p10 = np.percentile(GB_all, 10, axis=0)
    p90 = np.percentile(GB_all, 90, axis=0)
    plt.figure(figsize=(10, 5))
    for i in range(min(N, 100)):
        plt.plot(t_grid, GB_all[i, :], alpha=0.08, linewidth=0.5)
    plt.plot(t_grid, meanB, linewidth=2.0, label="Mean (B)")
    plt.fill_between(t_grid, p10, p90, alpha=0.25, label="10–90% (B)")
    plt.axvspan(w0, w1, alpha=0.1, label="Steady window")
    plt.xlabel("Time")
    plt.ylabel("GFP (cell B)")
    plt.title(f"Paracrine (c-3) timelapse (distance = {distance_cm} cm, copies = {copies})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "paracrine_c3_timelapse_plot.png"), dpi=150)
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=500)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--w0", type=int, default=2500)
    ap.add_argument("--w1", type=int, default=3000)
    ap.add_argument("--distance", type=float, default=1.0,
                    help="Cell separation in cm (affects diffusion time)")
    ap.add_argument("--tau", type=float, default=5.0,
                    help="Mean travel time (s) for AHL from cell A to cell B; ignored if --diff is set")
    ap.add_argument("--diff", type=float, default=None,
                    help="Diffusion coefficient for AHL in cm^2/s (overrides --tau if provided)")
    ap.add_argument("--out", type=str, default="paracrine_c3_timelapse_out")
    ap.add_argument("--seed", type=int, default=522)
    ap.add_argument("--copies", type=int, default=10,
                    help="Uniform plasmid copy number for pCon-luxI (A), pCon-luxR (B), and pLux-GFP (B)")
    ap.add_argument("--RNAP", type=int, default=50, help="RNAP_total per cell")
    ap.add_argument("--Ribo", type=int, default=200, help="Ribo_total per cell")
    ap.add_argument("--no_plot", action="store_true", help="Skip PNG timelapse plotting")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(N=args.N, T_end=args.T, window=(args.w0, args.w1),
        outdir=args.out, seed=args.seed,
        distance_cm=args.distance,
        transfer_time=args.tau if args.diff is None else None,
        diff_coef_ahl=args.diff,
        copies=args.copies,
        RNAP_total=args.RNAP,
        Ribo_total=args.Ribo,
        make_plot=not args.no_plot)
