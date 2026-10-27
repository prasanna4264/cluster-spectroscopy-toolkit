"""Diagnostic plots for one cluster (takes the dict returned by ``analyse_cluster``)."""
import numpy as np
import matplotlib.pyplot as plt


def phase_space(res, ax=None, zoom=4.0, title=None):
    """Projected radius vs line-of-sight velocity; members highlighted, +/-3 sigma_v lines."""
    ax = ax or plt.subplots(figsize=(5.5, 4.5))[1]
    m = res['members']
    ax.scatter(res['R'][~m], res['v_pec'][~m], s=8, c='0.6', alpha=0.5, label='field')
    ax.scatter(res['R'][m], res['v_pec'][m], s=18, c='C0', ec='k', lw=0.3, label='members')
    s = res['sigma_v']
    for k in (-3, 3):
        ax.axhline(k * s, ls='--', c='C3', lw=0.8)
    if np.isfinite(res.get('R200', np.nan)):
        ax.axvline(res['R200'], ls=':', c='k', lw=0.8, label='$R_{200}$')
    ax.set_ylim(-zoom * s, zoom * s)
    ax.set_xlabel('projected radius (Mpc)')
    ax.set_ylabel('$v_{\\rm pec}$ (km s$^{-1}$)')
    ax.legend(fontsize=8)
    ax.set_title(title or res.get('cluster', ''))
    return ax


def ds_histogram(res, ax=None):
    """Monte Carlo distribution of Delta/N with the observed value."""
    ax = ax or plt.subplots(figsize=(5.5, 4))[1]
    t = res['ds']
    ax.hist(t['delta_mc'], bins=30, histtype='step', color='C0', label='shuffled')
    ax.axvline(t['delta_obs'], c='C3', ls='--', label=f"observed, p = {t['p_value']:.3f}")
    ax.set_xlabel(r'$\Delta / N$')
    ax.set_ylabel('count')
    ax.legend(fontsize=8)
    return ax


def ds_bubbles(res, ra, dec, ax=None, scale=6.0):
    """Sky positions of members, circle size ~ exp(delta_i), colour = v_pec."""
    ax = ax or plt.subplots(figsize=(5.5, 5))[1]
    m = res['members']
    ra, dec = np.asarray(ra)[m], np.asarray(dec)[m]
    sc = ax.scatter(ra, dec, s=scale * np.exp(res['ds']['delta_i']), c=res['v_pec'][m],
                    cmap='coolwarm', ec='k', lw=0.3, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='$v_{\\rm pec}$ (km s$^{-1}$)')
    ax.invert_xaxis()
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    return ax
