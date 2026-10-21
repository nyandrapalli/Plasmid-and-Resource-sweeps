#!/usr/bin/env python3
"""
Autocrine (nc-6) timelapse (single cell containing all plasmids from Paracrine (c-6) without finite-distance transfer).
Plasmids: pCon–LuxI, pCon–LasR, pCon–LuxR, pLas–GFP, pLux–GFP, pLux–LasI.
Exports pLas–GFP and pLux–GFP trajectories separately and their sum.
"""
import argparse
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Params4:
    RNAP_total: int = 50
    Ribo_total: int = 200
    n_pCon_luxI: int = 10
    n_pCon_lasR: int = 10
    n_pCon_luxR: int = 10
    n_pLas_gfp: int = 10
    n_pLux_gfp: int = 10
    n_pLux_lasI: int = 10
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


def gillespie_cond4(P: Params4, seed: int | None = None):
    rng = np.random.default_rng(seed)
    AHL = 0
    S = 0
    cell = {
        'RNAP_free': P.RNAP_total,
        'Ribo_free': P.Ribo_total,
        'LuxI': 0,
        'LuxR': 0,
        'LasR': 0,
        'RL': 0,
        'RS': 0,
        'pCon_luxI_free': P.n_pCon_luxI,
        'pCon_luxI_RNAP': 0,
        'pCon_luxR_free': P.n_pCon_luxR,
        'pCon_luxR_RNAP': 0,
        'pCon_lasR_free': P.n_pCon_lasR,
        'pCon_lasR_RNAP': 0,
        'pLas_gfp_free': P.n_pLas_gfp,
        'pLas_gfp_act': 0,
        'pLas_gfp_RNAP_inact': 0,
        'pLas_gfp_RNAP_act': 0,
        'pLux_gfp_free': P.n_pLux_gfp,
        'pLux_gfp_act': 0,
        'pLux_gfp_RNAP_inact': 0,
        'pLux_gfp_RNAP_act': 0,
        'pLux_lasI_free': P.n_pLux_lasI,
        'pLux_lasI_act': 0,
        'pLux_lasI_RNAP_inact': 0,
        'pLux_lasI_RNAP_act': 0,
        'm_luxI': 0,
        'm_luxR': 0,
        'm_lasR': 0,
        'm_gfp_las': 0,
        'm_gfp_lux': 0,
        'm_lasI': 0,
        'ribo_luxI': 0,
        'ribo_luxR': 0,
        'ribo_lasR': 0,
        'ribo_gfp_las': 0,
        'ribo_gfp_lux': 0,
        'ribo_lasI': 0,
        'Gi_las': 0,
        'G_las': 0,
        'Gi_lux': 0,
        'G_lux': 0,
        'LasI': 0,
    }

    t = 0.0
    times = [0.0]
    G_las_trace = [0]
    G_lux_trace = [0]
    it = 0
    max_it = int(3e6)

    while t < P.T_end and it < max_it:
        it += 1
        props: list[float] = []
        rxns: list[tuple[str, str]] = []

        # Constitutive expression (LuxI, LuxR, LasR)
        if cell['RNAP_free'] > 0 and cell['pCon_luxI_free'] > 0:
            props.append(P.kon_RNAP * cell['RNAP_free'] * cell['pCon_luxI_free'])
            rxns.append(('C', 'bind_pCon_luxI'))
        if cell['pCon_luxI_RNAP'] > 0:
            props.append(P.kcat_tx_active * cell['pCon_luxI_RNAP'])
            rxns.append(('C', 'tx_pCon_luxI'))
        if cell['RNAP_free'] > 0 and cell['pCon_luxR_free'] > 0:
            props.append(P.kon_RNAP * cell['RNAP_free'] * cell['pCon_luxR_free'])
            rxns.append(('C', 'bind_pCon_luxR'))
        if cell['pCon_luxR_RNAP'] > 0:
            props.append(P.kcat_tx_active * cell['pCon_luxR_RNAP'])
            rxns.append(('C', 'tx_pCon_luxR'))
        if cell['RNAP_free'] > 0 and cell['pCon_lasR_free'] > 0:
            props.append(P.kon_RNAP * cell['RNAP_free'] * cell['pCon_lasR_free'])
            rxns.append(('C', 'bind_pCon_lasR'))
        if cell['pCon_lasR_RNAP'] > 0:
            props.append(P.kcat_tx_active * cell['pCon_lasR_RNAP'])
            rxns.append(('C', 'tx_pCon_lasR'))

        # Translation for constitutive genes
        if cell['Ribo_free'] > 0 and cell['m_luxI'] > 0:
            props.append(P.kon_ribo * cell['Ribo_free'] * cell['m_luxI'])
            rxns.append(('C', 'bind_ribo_luxI'))
        if cell['Ribo_free'] > 0 and cell['m_luxR'] > 0:
            props.append(P.kon_ribo * cell['Ribo_free'] * cell['m_luxR'])
            rxns.append(('C', 'bind_ribo_luxR'))
        if cell['Ribo_free'] > 0 and cell['m_lasR'] > 0:
            props.append(P.kon_ribo * cell['Ribo_free'] * cell['m_lasR'])
            rxns.append(('C', 'bind_ribo_lasR'))
        if cell['ribo_luxI'] > 0:
            props.append(P.kcat_tl * cell['ribo_luxI'])
            rxns.append(('C', 'tl_luxI'))
        if cell['ribo_luxR'] > 0:
            props.append(P.kcat_tl * cell['ribo_luxR'])
            rxns.append(('C', 'tl_luxR'))
        if cell['ribo_lasR'] > 0:
            props.append(P.kcat_tl * cell['ribo_lasR'])
            rxns.append(('C', 'tl_lasR'))

        # Degradation of constitutive mRNA/protein
        if cell['m_luxI'] > 0:
            props.append(P.gamma_m * cell['m_luxI'])
            rxns.append(('C', 'deg_m_luxI'))
        if cell['m_luxR'] > 0:
            props.append(P.gamma_m * cell['m_luxR'])
            rxns.append(('C', 'deg_m_luxR'))
        if cell['m_lasR'] > 0:
            props.append(P.gamma_m * cell['m_lasR'])
            rxns.append(('C', 'deg_m_lasR'))
        if cell['LuxI'] > 0:
            props.append(P.gamma_p * cell['LuxI'])
            rxns.append(('C', 'deg_LuxI'))
        if cell['LuxR'] > 0:
            props.append(P.gamma_p * cell['LuxR'])
            rxns.append(('C', 'deg_LuxR'))
        if cell['LasR'] > 0:
            props.append(P.gamma_p * cell['LasR'])
            rxns.append(('C', 'deg_LasR'))

        # Signal production from LuxI / LasI
        if cell['LuxI'] > 0:
            props.append(P.k_I * cell['LuxI'])
            rxns.append(('Sg', 'syn_AHL'))
        if cell['LasI'] > 0:
            props.append(P.k_S * cell['LasI'])
            rxns.append(('Sg', 'syn_S'))

        # LasR•S activation of pLas–GFP
        if cell['LasR'] > 0 and S > 0:
            props.append(P.kon_RS * cell['LasR'] * S)
            rxns.append(('C', 'bind_RS'))
        if cell['RS'] > 0:
            props.append(P.koff_RS * cell['RS'])
            rxns.append(('C', 'unbind_RS'))
        if cell['RS'] > 0 and cell['pLas_gfp_free'] > 0:
            props.append(P.kon_act * cell['RS'] * cell['pLas_gfp_free'])
            rxns.append(('C', 'act_pLas_gfp'))
        if cell['pLas_gfp_act'] > 0:
            props.append(P.koff_act * cell['pLas_gfp_act'])
            rxns.append(('C', 'deact_pLas_gfp'))

        # LuxR•AHL activation of pLux–GFP and pLux–LasI
        if cell['LuxR'] > 0 and AHL > 0:
            props.append(P.kon_RL * cell['LuxR'] * AHL)
            rxns.append(('C', 'bind_RL'))
        if cell['RL'] > 0:
            props.append(P.koff_RL * cell['RL'])
            rxns.append(('C', 'unbind_RL'))
        if cell['RL'] > 0 and cell['pLux_gfp_free'] > 0:
            props.append(P.kon_act * cell['RL'] * cell['pLux_gfp_free'])
            rxns.append(('C', 'act_pLux_gfp'))
        if cell['RL'] > 0 and cell['pLux_lasI_free'] > 0:
            props.append(P.kon_act * cell['RL'] * cell['pLux_lasI_free'])
            rxns.append(('C', 'act_pLux_lasI'))
        if cell['pLux_gfp_act'] > 0:
            props.append(P.koff_act * cell['pLux_gfp_act'])
            rxns.append(('C', 'deact_pLux_gfp'))
        if cell['pLux_lasI_act'] > 0:
            props.append(P.koff_act * cell['pLux_lasI_act'])
            rxns.append(('C', 'deact_pLux_lasI'))

        # RNAP binding to regulated promoters
        if cell['RNAP_free'] > 0:
            if cell['pLas_gfp_free'] > 0:
                props.append(P.kon_RNAP * cell['RNAP_free'] * cell['pLas_gfp_free'])
                rxns.append(('C', 'bind_pLas_gfp_free'))
            if cell['pLas_gfp_act'] > 0:
                props.append(P.kon_RNAP * cell['RNAP_free'] * cell['pLas_gfp_act'])
                rxns.append(('C', 'bind_pLas_gfp_act'))
            if cell['pLux_gfp_free'] > 0:
                props.append(P.kon_RNAP * cell['RNAP_free'] * cell['pLux_gfp_free'])
                rxns.append(('C', 'bind_pLux_gfp_free'))
            if cell['pLux_gfp_act'] > 0:
                props.append(P.kon_RNAP * cell['RNAP_free'] * cell['pLux_gfp_act'])
                rxns.append(('C', 'bind_pLux_gfp_act'))
            if cell['pLux_lasI_free'] > 0:
                props.append(P.kon_RNAP * cell['RNAP_free'] * cell['pLux_lasI_free'])
                rxns.append(('C', 'bind_pLux_lasI_free'))
            if cell['pLux_lasI_act'] > 0:
                props.append(P.kon_RNAP * cell['RNAP_free'] * cell['pLux_lasI_act'])
                rxns.append(('C', 'bind_pLux_lasI_act'))

        # Transcription from regulated promoters
        if cell['pLas_gfp_RNAP_inact'] > 0:
            props.append(P.kcat_tx_basal * cell['pLas_gfp_RNAP_inact'])
            rxns.append(('C', 'tx_pLas_gfp_inact'))
        if cell['pLas_gfp_RNAP_act'] > 0:
            props.append(P.kcat_tx_active * cell['pLas_gfp_RNAP_act'])
            rxns.append(('C', 'tx_pLas_gfp_act'))
        if cell['pLux_gfp_RNAP_inact'] > 0:
            props.append(P.kcat_tx_basal * cell['pLux_gfp_RNAP_inact'])
            rxns.append(('C', 'tx_pLux_gfp_inact'))
        if cell['pLux_gfp_RNAP_act'] > 0:
            props.append(P.kcat_tx_active * cell['pLux_gfp_RNAP_act'])
            rxns.append(('C', 'tx_pLux_gfp_act'))
        if cell['pLux_lasI_RNAP_inact'] > 0:
            props.append(P.kcat_tx_basal * cell['pLux_lasI_RNAP_inact'])
            rxns.append(('C', 'tx_pLux_lasI_inact'))
        if cell['pLux_lasI_RNAP_act'] > 0:
            props.append(P.kcat_tx_active * cell['pLux_lasI_RNAP_act'])
            rxns.append(('C', 'tx_pLux_lasI_act'))

        # Translation from regulated transcripts
        if cell['Ribo_free'] > 0 and cell['m_gfp_las'] > 0:
            props.append(P.kon_ribo * cell['Ribo_free'] * cell['m_gfp_las'])
            rxns.append(('C', 'bind_ribo_gfp_las'))
        if cell['Ribo_free'] > 0 and cell['m_gfp_lux'] > 0:
            props.append(P.kon_ribo * cell['Ribo_free'] * cell['m_gfp_lux'])
            rxns.append(('C', 'bind_ribo_gfp_lux'))
        if cell['Ribo_free'] > 0 and cell['m_lasI'] > 0:
            props.append(P.kon_ribo * cell['Ribo_free'] * cell['m_lasI'])
            rxns.append(('C', 'bind_ribo_lasI'))
        if cell['ribo_gfp_las'] > 0:
            props.append(P.kcat_tl * cell['ribo_gfp_las'])
            rxns.append(('C', 'tl_gfp_las'))
        if cell['ribo_gfp_lux'] > 0:
            props.append(P.kcat_tl * cell['ribo_gfp_lux'])
            rxns.append(('C', 'tl_gfp_lux'))
        if cell['ribo_lasI'] > 0:
            props.append(P.kcat_tl * cell['ribo_lasI'])
            rxns.append(('C', 'tl_lasI'))

        # Degradation / maturation for regulated products
        if cell['m_gfp_las'] > 0:
            props.append(P.gamma_m * cell['m_gfp_las'])
            rxns.append(('C', 'deg_m_gfp_las'))
        if cell['m_gfp_lux'] > 0:
            props.append(P.gamma_m * cell['m_gfp_lux'])
            rxns.append(('C', 'deg_m_gfp_lux'))
        if cell['m_lasI'] > 0:
            props.append(P.gamma_m * cell['m_lasI'])
            rxns.append(('C', 'deg_m_lasI'))
        if cell['Gi_las'] > 0:
            props.append(P.kmat * cell['Gi_las'])
            rxns.append(('C', 'mat_G_las'))
        if cell['Gi_lux'] > 0:
            props.append(P.kmat * cell['Gi_lux'])
            rxns.append(('C', 'mat_G_lux'))
        if cell['G_las'] > 0:
            props.append(P.gamma_G * cell['G_las'])
            rxns.append(('C', 'deg_G_las'))
        if cell['G_lux'] > 0:
            props.append(P.gamma_G * cell['G_lux'])
            rxns.append(('C', 'deg_G_lux'))
        if cell['LasI'] > 0:
            props.append(P.gamma_p * cell['LasI'])
            rxns.append(('C', 'deg_LasI'))

        # Signal decay
        if AHL > 0:
            props.append(P.gamma_AHL * AHL)
            rxns.append(('Sg', 'deg_AHL'))
        if S > 0:
            props.append(P.gamma_S * S)
            rxns.append(('Sg', 'deg_S'))

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
        if who == 'C':
            if rx == 'bind_pCon_luxI':
                cell['RNAP_free'] -= 1
                cell['pCon_luxI_free'] -= 1
                cell['pCon_luxI_RNAP'] += 1
            elif rx == 'tx_pCon_luxI':
                cell['pCon_luxI_RNAP'] -= 1
                cell['RNAP_free'] += 1
                cell['pCon_luxI_free'] += 1
                cell['m_luxI'] += 1
            elif rx == 'bind_pCon_luxR':
                cell['RNAP_free'] -= 1
                cell['pCon_luxR_free'] -= 1
                cell['pCon_luxR_RNAP'] += 1
            elif rx == 'tx_pCon_luxR':
                cell['pCon_luxR_RNAP'] -= 1
                cell['RNAP_free'] += 1
                cell['pCon_luxR_free'] += 1
                cell['m_luxR'] += 1
            elif rx == 'bind_pCon_lasR':
                cell['RNAP_free'] -= 1
                cell['pCon_lasR_free'] -= 1
                cell['pCon_lasR_RNAP'] += 1
            elif rx == 'tx_pCon_lasR':
                cell['pCon_lasR_RNAP'] -= 1
                cell['RNAP_free'] += 1
                cell['pCon_lasR_free'] += 1
                cell['m_lasR'] += 1
            elif rx == 'bind_ribo_luxI':
                cell['Ribo_free'] -= 1
                cell['m_luxI'] -= 1
                cell['ribo_luxI'] += 1
            elif rx == 'bind_ribo_luxR':
                cell['Ribo_free'] -= 1
                cell['m_luxR'] -= 1
                cell['ribo_luxR'] += 1
            elif rx == 'bind_ribo_lasR':
                cell['Ribo_free'] -= 1
                cell['m_lasR'] -= 1
                cell['ribo_lasR'] += 1
            elif rx == 'tl_luxI':
                cell['ribo_luxI'] -= 1
                cell['Ribo_free'] += 1
                cell['LuxI'] += 1
            elif rx == 'tl_luxR':
                cell['ribo_luxR'] -= 1
                cell['Ribo_free'] += 1
                cell['LuxR'] += 1
            elif rx == 'tl_lasR':
                cell['ribo_lasR'] -= 1
                cell['Ribo_free'] += 1
                cell['LasR'] += 1
            elif rx == 'deg_m_luxI':
                cell['m_luxI'] -= 1
            elif rx == 'deg_m_luxR':
                cell['m_luxR'] -= 1
            elif rx == 'deg_m_lasR':
                cell['m_lasR'] -= 1
            elif rx == 'deg_LuxI':
                cell['LuxI'] -= 1
            elif rx == 'deg_LuxR':
                cell['LuxR'] -= 1
            elif rx == 'deg_LasR':
                cell['LasR'] -= 1
            elif rx == 'bind_RS':
                cell['LasR'] -= 1
                S -= 1
                cell['RS'] += 1
            elif rx == 'unbind_RS':
                cell['RS'] -= 1
                cell['LasR'] += 1
                S += 1
            elif rx == 'bind_RL':
                cell['LuxR'] -= 1
                AHL -= 1
                cell['RL'] += 1
            elif rx == 'unbind_RL':
                cell['RL'] -= 1
                cell['LuxR'] += 1
                AHL += 1
            elif rx == 'act_pLas_gfp':
                cell['RS'] -= 1
                cell['pLas_gfp_free'] -= 1
                cell['pLas_gfp_act'] += 1
            elif rx == 'deact_pLas_gfp':
                cell['pLas_gfp_act'] -= 1
                cell['pLas_gfp_free'] += 1
                cell['RS'] += 1
            elif rx == 'act_pLux_gfp':
                cell['RL'] -= 1
                cell['pLux_gfp_free'] -= 1
                cell['pLux_gfp_act'] += 1
            elif rx == 'act_pLux_lasI':
                cell['RL'] -= 1
                cell['pLux_lasI_free'] -= 1
                cell['pLux_lasI_act'] += 1
            elif rx == 'deact_pLux_gfp':
                cell['pLux_gfp_act'] -= 1
                cell['pLux_gfp_free'] += 1
                cell['RL'] += 1
            elif rx == 'deact_pLux_lasI':
                cell['pLux_lasI_act'] -= 1
                cell['pLux_lasI_free'] += 1
                cell['RL'] += 1
            elif rx == 'bind_pLas_gfp_free':
                cell['RNAP_free'] -= 1
                cell['pLas_gfp_free'] -= 1
                cell['pLas_gfp_RNAP_inact'] += 1
            elif rx == 'bind_pLas_gfp_act':
                cell['RNAP_free'] -= 1
                cell['pLas_gfp_act'] -= 1
                cell['pLas_gfp_RNAP_act'] += 1
            elif rx == 'bind_pLux_gfp_free':
                cell['RNAP_free'] -= 1
                cell['pLux_gfp_free'] -= 1
                cell['pLux_gfp_RNAP_inact'] += 1
            elif rx == 'bind_pLux_gfp_act':
                cell['RNAP_free'] -= 1
                cell['pLux_gfp_act'] -= 1
                cell['pLux_gfp_RNAP_act'] += 1
            elif rx == 'bind_pLux_lasI_free':
                cell['RNAP_free'] -= 1
                cell['pLux_lasI_free'] -= 1
                cell['pLux_lasI_RNAP_inact'] += 1
            elif rx == 'bind_pLux_lasI_act':
                cell['RNAP_free'] -= 1
                cell['pLux_lasI_act'] -= 1
                cell['pLux_lasI_RNAP_act'] += 1
            elif rx == 'tx_pLas_gfp_inact':
                cell['pLas_gfp_RNAP_inact'] -= 1
                cell['RNAP_free'] += 1
                cell['pLas_gfp_free'] += 1
                cell['m_gfp_las'] += 1
            elif rx == 'tx_pLas_gfp_act':
                cell['pLas_gfp_RNAP_act'] -= 1
                cell['RNAP_free'] += 1
                cell['pLas_gfp_act'] += 1
                cell['m_gfp_las'] += 1
            elif rx == 'tx_pLux_gfp_inact':
                cell['pLux_gfp_RNAP_inact'] -= 1
                cell['RNAP_free'] += 1
                cell['pLux_gfp_free'] += 1
                cell['m_gfp_lux'] += 1
            elif rx == 'tx_pLux_gfp_act':
                cell['pLux_gfp_RNAP_act'] -= 1
                cell['RNAP_free'] += 1
                cell['pLux_gfp_act'] += 1
                cell['m_gfp_lux'] += 1
            elif rx == 'tx_pLux_lasI_inact':
                cell['pLux_lasI_RNAP_inact'] -= 1
                cell['RNAP_free'] += 1
                cell['pLux_lasI_free'] += 1
                cell['m_lasI'] += 1
            elif rx == 'tx_pLux_lasI_act':
                cell['pLux_lasI_RNAP_act'] -= 1
                cell['RNAP_free'] += 1
                cell['pLux_lasI_act'] += 1
                cell['m_lasI'] += 1
            elif rx == 'bind_ribo_gfp_las':
                cell['Ribo_free'] -= 1
                cell['m_gfp_las'] -= 1
                cell['ribo_gfp_las'] += 1
            elif rx == 'bind_ribo_gfp_lux':
                cell['Ribo_free'] -= 1
                cell['m_gfp_lux'] -= 1
                cell['ribo_gfp_lux'] += 1
            elif rx == 'bind_ribo_lasI':
                cell['Ribo_free'] -= 1
                cell['m_lasI'] -= 1
                cell['ribo_lasI'] += 1
            elif rx == 'tl_gfp_las':
                cell['ribo_gfp_las'] -= 1
                cell['Ribo_free'] += 1
                cell['Gi_las'] += 1
            elif rx == 'tl_gfp_lux':
                cell['ribo_gfp_lux'] -= 1
                cell['Ribo_free'] += 1
                cell['Gi_lux'] += 1
            elif rx == 'tl_lasI':
                cell['ribo_lasI'] -= 1
                cell['Ribo_free'] += 1
                cell['LasI'] += 1
            elif rx == 'deg_m_gfp_las':
                cell['m_gfp_las'] -= 1
            elif rx == 'deg_m_gfp_lux':
                cell['m_gfp_lux'] -= 1
            elif rx == 'deg_m_lasI':
                cell['m_lasI'] -= 1
            elif rx == 'mat_G_las':
                cell['Gi_las'] -= 1
                cell['G_las'] += 1
            elif rx == 'mat_G_lux':
                cell['Gi_lux'] -= 1
                cell['G_lux'] += 1
            elif rx == 'deg_G_las':
                cell['G_las'] -= 1
            elif rx == 'deg_G_lux':
                cell['G_lux'] -= 1
            elif rx == 'deg_LasI':
                cell['LasI'] -= 1
        else:
            if rx == 'syn_AHL':
                AHL += 1
            elif rx == 'syn_S':
                S += 1
            elif rx == 'deg_AHL':
                AHL -= 1
            elif rx == 'deg_S':
                S -= 1

        times.append(t)
        G_las_trace.append(cell['G_las'])
        G_lux_trace.append(cell['G_lux'])

    return np.array(times), np.array(G_las_trace), np.array(G_lux_trace)


def run(
    N: int = 500,
    T_end: int = 6000,
    window: tuple[int, int] = (2500, 3000),
    outdir: str = "autocrine_nc6_timelapse_out",
    seed: int = 444,
    copies: int = 10,
    make_plot: bool = True,
):
    os.makedirs(outdir, exist_ok=True)
    P = Params4(
        T_end=T_end,
        n_pCon_luxI=copies,
        n_pCon_lasR=copies,
        n_pCon_luxR=copies,
        n_pLas_gfp=copies,
        n_pLux_gfp=copies,
        n_pLux_lasI=copies,
    )
    t_grid = np.arange(int(T_end) + 1)
    G_las_curves = np.zeros((N, len(t_grid)))
    G_lux_curves = np.zeros((N, len(t_grid)))
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**31 - 1, size=N, dtype=np.int64)

    for i in range(N):
        ti, G_las, G_lux = gillespie_cond4(P, seed=int(seeds[i]))
        idx = np.searchsorted(ti, t_grid, side='right') - 1
        idx[idx < 0] = 0
        G_las_curves[i, :] = G_las[idx]
        G_lux_curves[i, :] = G_lux[idx]

    total_curves = G_las_curves + G_lux_curves
    df = pd.DataFrame(G_las_curves, columns=[f'GLas_t{t}' for t in t_grid])
    df = df.join(pd.DataFrame(G_lux_curves, columns=[f'GLux_t{t}' for t in t_grid]))
    df = df.assign(replicate_id=lambda d: np.arange(N))
    df.to_csv(os.path.join(outdir, 'autocrine_nc6_timelapse_raw.csv'), index=False)

    w0, w1 = window
    block_las = G_las_curves[:, w0 : w1 + 1].mean(axis=1)
    block_lux = G_lux_curves[:, w0 : w1 + 1].mean(axis=1)
    block_total = total_curves[:, w0 : w1 + 1].mean(axis=1)

    def stats(arr):
        mean = float(arr.mean())
        sd = float(arr.std(ddof=0))
        cv = float(sd / mean) if mean > 0 else float('nan')
        cv2 = float(cv * cv) if mean > 0 else float('nan')
        return mean, sd, cv, cv2

    mean_tot, sd_tot, cv_tot, cv2_tot = stats(block_total)
    mean_las, sd_las, cv_las, cv2_las = stats(block_las)
    mean_lux, sd_lux, cv_lux, cv2_lux = stats(block_lux)
    pd.DataFrame([{'mean_total': mean_tot, 'sd_total': sd_tot, 'CV_total': cv_tot, 'CV2_total': cv2_tot,
                  'mean_pLasGFP': mean_las, 'sd_pLasGFP': sd_las, 'CV_pLasGFP': cv_las, 'CV2_pLasGFP': cv2_las,
                  'mean_pLuxGFP': mean_lux, 'sd_pLuxGFP': sd_lux, 'CV_pLuxGFP': cv_lux, 'CV2_pLuxGFP': cv2_lux,
                  'N': N, 'T_end': T_end, 'window_start': w0, 'window_end': w1, 'copies': copies}]).to_csv(
        os.path.join(outdir, 'autocrine_nc6_summary.csv'), index=False
    )

    if make_plot:
        mean_las_curve = G_las_curves.mean(axis=0)
        mean_lux_curve = G_lux_curves.mean(axis=0)
        total_mean_curve = total_curves.mean(axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(t_grid, mean_las_curve, label='Mean pLas-GFP', linewidth=2.0)
        plt.plot(t_grid, mean_lux_curve, label='Mean pLux-GFP', linewidth=2.0)
        plt.plot(t_grid, total_mean_curve, label='Mean total GFP', linewidth=2.5, color='black', alpha=0.7)
        plt.axvspan(w0, w1, alpha=0.1, label='Steady window')
        plt.xlabel('Time')
        plt.ylabel('GFP counts')
        plt.title(f'Autocrine (nc-6) timelapse (copies={copies})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'autocrine_nc6_timelapse_plot.png'), dpi=150)
        plt.close()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=500)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--w0", type=int, default=2500)
    ap.add_argument("--w1", type=int, default=3000)
    ap.add_argument("--copies", type=int, default=10, help="Plasmid copy number per plasmid type")
    ap.add_argument("--out", type=str, default="autocrine_nc6_timelapse_out")
    ap.add_argument("--no-plot", action="store_true", help="Skip timelapse plot generation")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(N=args.N, T_end=args.T, window=(args.w0, args.w1), outdir=args.out, copies=args.copies, make_plot=not args.no_plot)



