#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical calculations for the paper

    U. Thattarampilly and Y. Zheng,
    "Momentum-space entanglement entropy of Kaluza-Klein reduction".

How to get each result of the paper
-----------------------------------
The script is run from a terminal, with the names of the calculations you want after
the file name.  Add the word  figure  to also draw the corresponding panel of figure 1.

    What you want                                              What to type
    ---------------------------------------------------------  -----------------------
    Cubic theory: J_n^(3) -> A_3/(nR), eq. (6.3), fig. 1(a)    cubic figure
    Cubic theory: zero-mode coefficient D_3, eq. (6.7)         d3
    Quartic (2,2) channel: J_n -> A/(nR^3), eq. (4.8)          j22
    Quartic (2,2)+mixed: K_n and a, b, eq. (5.10), fig. 1(b)   kn beta figure
    Quartic: only the constants a', beta, a, b, eq. (5.9)      beta
    Quartic (1,3) channel: c and c', eq. (5.11), fig. 1(c)     j13 figure
    Closed-form numbers of sections 4, 6 and 8                 misc
    Everything, and the whole of figure 1                      (nothing)

For example, to get the (1,3) channel of the quartic theory with its plot:

    python kk_entanglement_numerics.py j13 figure

In a Jupyter notebook, put the same words after  %run :

    %run kk_entanglement_numerics.py j13 figure

(or paste the whole script into a cell and then type  main(["j13", "figure"]) ).

By default the script uses smaller samples, which take minutes and reproduce the paper
within the statistical errors.  Add  --full  to use the sample sizes of the paper,
for example  j13 figure --full .  The whole run with --full takes a few hours on a
laptop, most of it for K_n at Lambda R = 640.

What you get:
  * the numbers, printed on the screen with the equation they belong to;
  * the file results.json, where every result is stored.  Later runs add to it, so a
    slow calculation only has to be done once, and  figure  on its own redraws the plot
    from the stored results;
  * the plot: fig_mc.pdf when all three panels of figure 1 are available, and otherwise
    a file with only the panels you have computed, for example fig_1c.pdf for the
    (1,3) channel.  In Jupyter the plot is also shown below the cell.

The script needs Python 3.9 or later, with numpy, scipy and matplotlib.
Equation and section numbers refer to the submitted version of the paper.

Conventions
-----------
We work in units R = 1, so that the mass of KK level n is simply m_n = n, and take
mu R = 1 unless we say otherwise.  The resolution scale mu bounds the momenta of the
soft zero modes, |k| <= mu.

Energies are always free energies: omega = |k| for a zero-mode quantum,
Omega = sqrt(q^2 + n^2) for a quantum at level n, and E for the total energy of the
intermediate state.

All the kernels in the paper have the same structure.  Each one integrates

        1 / [ 2^(number of quanta) x (product of their energies) x E^2 ]

over the independent momenta of the intermediate state, with one momentum fixed by
three-momentum conservation, and divides by (2 pi)^3 for every independent momentum.
Eqs. (3.15)-(3.17) are examples.

In the quartic theory we use the cutoff of section 5.1.  Every tower quantum must
satisfy q^2 + m_n^2 <= Lambda^2, and the hard zero mode must satisfy |k_0| <= Lambda.

How the integrals are done
--------------------------
Most kernels are evaluated by Monte Carlo integration with importance sampling, as
described in appendix C.  Soft momenta are drawn from the ball |k| <= mu.  When the
integrand contains 1/omega, we draw |k| with a density proportional to k, which
cancels this factor exactly.  Unbounded hard momenta are drawn in a random direction,
with a magnitude q distributed as p(q) = (2/pi) s / (s^2 + q^2), where s is the typical
hard scale of the problem.  At large q this density falls like the integrand itself,
so the sample weights stay bounded and the statistical error decreases like
1/sqrt(number of samples).

Every Monte Carlo result is the average of several independent batches.  The error we
quote is the standard error of that average.

Where an integral can be done by deterministic quadrature, we do that instead, or as
a cross-check.

"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate

PI = math.pi
RESULTS_FILE = "results.json"

# Coefficients that the paper gives in closed form (for mu R = 1).
A_COEFF = PI**4 / (32 * (2 * PI) ** 9)          # eq. (4.8): J_n of the (2,2) channel tends to A/(n R^3)
A3_COEFF = 1.0 / (1024 * PI**3)                 # eq. (6.3): J_n^(3) of the cubic channel tends to A_3/(n R)
D3_COEFF = (math.log(2) - 0.5) / (128 * PI**4)  # eq. (6.7): zero-mode term D_3 mu in the cubic entropy
QUARTIC_NORM = 2 * PI / (16 * (2 * PI) ** 9)    # section 5.3: K_n = QUARTIC_NORM x mu^2 x F(m_n)


# ======================================================================================
# Tools for drawing random momenta
# ======================================================================================
def make_rng(seed):
    """A random-number generator with a fixed seed, so that every run gives the same
    numbers."""
    return np.random.default_rng(seed)


def isotropic_unit(rng, n):
    """n directions chosen uniformly at random on the sphere."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def uniform_ball(rng, n, radius):
    """n points spread uniformly through a ball of the given radius.

    Each point represents a share (volume of the ball)/n of the integral, so we also
    return the volume of the ball."""
    r = radius * rng.uniform(size=n) ** (1.0 / 3.0)
    return isotropic_unit(rng, n) * r[:, None], 4.0 / 3.0 * PI * radius**3


def soft_one_over_omega(rng, n, mu):
    """Soft momenta |k| <= mu, for integrals that contain the factor 1/omega = 1/|k|.

    We draw |k| with probability density 2k/mu^2.  The volume element 4 pi k^2 dk
    divided by omega = k is then proportional to this density, so 1/omega drops out and
    each sample carries the same weight, 2 pi mu^2.  We return the momenta, their
    magnitudes and this weight."""
    k = mu * np.sqrt(rng.uniform(size=n))
    return isotropic_unit(rng, n) * k[:, None], k, 2 * PI * mu**2


def hard_lorentzian(rng, n, s):
    """Hard momenta of unbounded magnitude, in random directions.

    The magnitude q is drawn from p(q) = (2/pi) s/(s^2 + q^2), which is largest for
    q of order s and falls like 1/q^2, the same way as the integrands at large q.  The
    weight of a sample in an integral over d^3q is 4 pi q^2 / p(q)."""
    q = s * np.tan(0.5 * PI * rng.uniform(size=n))
    p = (2.0 / PI) * s / (s**2 + q**2)
    return isotropic_unit(rng, n) * q[:, None], q, 4 * PI * q**2 / p


def batched_mean(estimator, chunks, seed):
    """Run a Monte Carlo estimate on several independent batches and combine them.

    Returns the mean over the batches and its standard error, which is the error bar
    we quote."""
    vals = np.array([estimator(make_rng(seed + i)) for i in range(chunks)])
    return vals.mean(axis=0), vals.std(axis=0, ddof=1) / np.sqrt(chunks)


# ======================================================================================
# 1. The cubic theory (section 6)
#
#    A soft zero-mode quantum (momentum k, |k| <= mu) is produced together with a pair
#    of quanta at levels +n and -n (momenta q and -k-q).  The contribution of level n to
#    the kernel is
#
#        J_n^(3) = (2 pi)^-6  int d^3k d^3q  1 / (8 omega Omega_1 Omega_2 E^2).
#
#    The integral converges without a cutoff.  For large n it tends to A_3/(nR),
#    eq. (6.3), because the q-integral is then the one of eq. (4.6).
# ======================================================================================
def cubic_quadrature(m, mu=1.0, nk=48, nc=48, nq=200):
    """J_n^(3) at level mass m by direct (Gauss-Legendre) quadrature.

    The three integration variables are the soft momentum |k| in [0, mu], the cosine
    of the angle between k and q, and the hard momentum, written as |q| = m tan(t) so
    that its infinite range becomes t in [0, pi/2)."""
    xk, wk = leggauss(nk)
    k = 0.5 * mu * (xk + 1)
    wk = 0.5 * mu * wk
    c, wc = leggauss(nc)
    xt, wt = leggauss(nq)
    t = PI / 4 * (xt + 1)
    wt = PI / 4 * wt
    q = m * np.tan(t)
    wq = wt * m / np.cos(t) ** 2
    K, C, Q = np.meshgrid(k, c, q, indexing="ij")
    W = wk[:, None, None] * wc[None, :, None] * wq[None, None, :]
    Om1 = np.sqrt(Q**2 + m * m)
    Om2 = np.sqrt(Q**2 + K**2 + 2 * Q * K * C + m * m)
    E = K + Om1 + Om2
    # In these variables d^3k d^3q = (4 pi k^2 dk) (2 pi q^2 dq dcos).
    integrand = 4 * PI * K**2 * 2 * PI * Q**2 / (8 * K * Om1 * Om2 * E**2)
    return (W * integrand).sum() / (2 * PI) ** 6


def cubic_mc(n, mu=1.0, samples=1_000_000, chunks=10, seed=1):
    """The same kernel J_n^(3) by Monte Carlo, as an independent check of the
    quadrature."""
    def est(rng):
        kvec, k, wk = soft_one_over_omega(rng, samples, mu)
        qvec, q, wq = hard_lorentzian(rng, samples, s=float(n))
        Om1 = np.sqrt(q**2 + n * n)
        Om2 = np.sqrt(((kvec + qvec) ** 2).sum(1) + n * n)
        E = k + Om1 + Om2
        return np.mean(wk * wq / (8 * Om1 * Om2 * E**2)) / (2 * PI) ** 6
    return batched_mean(est, chunks, seed)


def run_cubic(cfg):
    """Prints n R J_n^(3) / A_3 for several levels.  The paper states that this ratio
    approaches 1 from below, is about 25% low at n = 1, and is within 1% of 1 for
    n >= 30 (section 6.3)."""
    print("\n[cubic]  n R J_n^(3) / A_3  should approach 1 from below (eq. (6.3))")
    levels = [1, 2, 4, 8, 16, 30, 32, 64]
    out = {"levels": levels, "quad": [], "mc": [], "mc_err": []}
    for n in levels:
        jq = cubic_quadrature(n)
        jm, em = cubic_mc(n, samples=cfg["cubic_samples"], chunks=cfg["chunks"])
        out["quad"].append(jq)
        out["mc"].append(jm)
        out["mc_err"].append(em)
        print(f"  n={n:3d}   quadrature {n*jq/A3_COEFF:.4f}   Monte Carlo {n*jm/A3_COEFF:.4f} "
              f"+- {n*em/A3_COEFF:.4f}   MC/quad - 1 = {jm/jq-1:+.4f}")
    # A smooth curve for figure 1(a).
    grid = np.geomspace(0.8, 100, 60)
    out["curve_n"] = grid.tolist()
    out["curve"] = [m * cubic_quadrature(m) / A3_COEFF for m in grid]
    print(f"  so: n=1 is {100*(1-out['quad'][0]/A3_COEFF):.0f}% below A_3, and"
          f" n=30 is {100*(1-30*out['quad'][5]/A3_COEFF):.2f}% below A_3")
    return out


# ======================================================================================
# 2. The (2,2) channel of the quartic theory (section 4.1)
#
#    Two soft zero-mode quanta (k1, k2, both with |k| <= mu) are produced together with
#    a pair at levels +n and -n (momenta q1 and -k1-k2-q1).  From eq. (3.15),
#
#        J_n = (1/2) (2 pi)^-9  int d^3k1 d^3k2 d^3q1  1 / (16 omega_1 omega_2 Omega_1 Omega_2 E^2).
#
#    The factor 1/2 is there because the two soft quanta are identical.  The integral
#    converges, and for large n it tends to A/(n R^3), eq. (4.8).
# ======================================================================================
def j22_mc(n, mu=1.0, samples=1_000_000, chunks=10, seed=2):
    """Monte Carlo estimate of J_n.  Both soft momenta are drawn uniformly from the
    ball |k| <= mu, and q1 from the hard distribution."""
    def est(rng):
        k1, Vb = uniform_ball(rng, samples, mu)
        k2, _ = uniform_ball(rng, samples, mu)
        q1, q, wq = hard_lorentzian(rng, samples, s=float(n))
        om1 = np.linalg.norm(k1, axis=1)
        om2 = np.linalg.norm(k2, axis=1)
        Om1 = np.sqrt(q**2 + n * n)
        Om2 = np.sqrt(((k1 + k2 + q1) ** 2).sum(1) + n * n)
        E = om1 + om2 + Om1 + Om2
        f = Vb * Vb * wq / (16 * om1 * om2 * Om1 * Om2 * E**2)
        return 0.5 * np.mean(f) / (2 * PI) ** 9   # the 1/2 for identical soft quanta
    return batched_mean(est, chunks, seed)


def run_j22(cfg):
    """Checks the three statements of section 4.1.2: J_n approaches A/(n R^3) from
    below, a power-law fit to levels 13-20 gives an exponent close to -1, and the
    coefficient scales as (mu R)^4."""
    print("\n[j22]  (2,2) channel: n J_n / A  should approach 1 from below (eq. (4.8))")
    levels = [2, 4, 8, 13, 14, 15, 16, 17, 18, 19, 20]
    vals, errs = [], []
    for n in levels:
        j, e = j22_mc(n, samples=cfg["j22_samples"], chunks=cfg["chunks"])
        vals.append(j)
        errs.append(e)
        print(f"  n={n:3d}   n J_n / A = {n*j/A_COEFF:.4f} +- {n*e/A_COEFF:.4f}")
    # Power-law fit to levels 13-20, with each point weighted by its error bar.
    sel = [i for i, n in enumerate(levels) if 13 <= n <= 20]
    x = np.log([levels[i] for i in sel])
    y = np.log([vals[i] for i in sel])
    w = np.array([vals[i] / errs[i] for i in sel])        # inverse error of log J
    (slope, icpt), cov = np.polyfit(x, y, 1, w=w, cov="unscaled")
    print(f"  power-law fit to levels 13-20:  J_n ~ n^({slope:.3f} +- {math.sqrt(cov[0,0]):.3f})")
    # The (mu R)^4 scaling, tested at level 16 with mu R = 0.5 and 2.
    scal = {}
    for mu in (0.5, 2.0):
        j, e = j22_mc(16, mu=mu, samples=cfg["j22_samples"], chunks=cfg["chunks"], seed=5)
        scal[mu] = j / (vals[levels.index(16)] * mu**4)
        print(f"  mu R = {mu}:  J_16(mu) / [mu^4 J_16(1)] = {scal[mu]:.3f}"
              "  (this would be exactly 1 for pure mu^4 scaling)")
    return {"levels": levels, "J": vals, "err": errs, "slope": slope,
            "slope_err": math.sqrt(cov[0, 0]), "mu_scaling": scal}


# ======================================================================================
# 3. The (2,2) and mixed channels together (section 5.3)
#
#    In the mutual information these two channels combine into a single integral,
#    eq. (5.6), in which the second zero-mode quantum k0 may be soft or hard:
#
#        K_n = (2 pi)^-9  int d^3k d^3k0 d^3q1  1 / (16 omega omega_0 Omega_1 Omega_2 E^2),
#
#    with |k| <= mu, |k0| <= Lambda, the tower pair at q1 and -k-k0-q1, and the cutoff
#    q^2 + n^2 <= Lambda^2 on both tower quanta.  The part with k0 soft (|k0| <= mu) is
#    2 J_n^(2,2), and the part with k0 hard is the mixed channel J_n^mix.
# ======================================================================================
def _sin2_angles(rng, n):
    """Angles theta between 0 and pi/2, drawn with probability proportional to
    sin^2(theta).  Used by kn_mc below."""
    out = np.empty(0)
    while out.size < n:
        th = rng.uniform(0, PI / 2, 2 * n)
        keep = rng.uniform(0, 1, 2 * n) < np.sin(th) ** 2
        out = np.concatenate([out, th[keep]])
    return out[:n]


def kn_mc(n, Lam, mu=1.0, samples=2_000_000, chunks=10, seed=3):
    """Monte Carlo estimate of K_n at cutoff Lambda.

    Returns the soft part 2 J_n^(2,2), the mixed part J_n^mix and their sum K_n,
    each with its error bar.

    How the momenta are drawn:
      k  (soft)  : |k| with density 2k/mu^2, which cancels 1/omega (weight 2 pi mu^2).
      k0 (second zero mode, soft or hard) : |k0| <= Lambda with density 2 k0/Lambda^2,
                   which cancels 1/omega_0 (weight 2 pi Lambda^2).
      q1 (tower) : the two tower quanta carry momenta that add up to -(k + k0), so we
                   write q1 = -(k + k0)/2 + P and draw the relative momentum P from a
                   distribution that follows the product 1/(Omega_1 Omega_2) of their
                   energies.  In three dimensions this density is s/(pi^2 (s^2+P^2)^2),
                   with s = sqrt(n^2 + (k0/2)^2)."""
    def est(rng):
        kvec, k, wk = soft_one_over_omega(rng, samples, mu)
        r0 = Lam * np.sqrt(rng.uniform(size=samples))
        k0 = isotropic_unit(rng, samples) * r0[:, None]
        wk0 = 2 * PI * Lam**2
        s = np.sqrt(n**2 + (r0 / 2) ** 2)
        p = s * np.tan(_sin2_angles(rng, samples))
        P = isotropic_unit(rng, samples) * p[:, None]
        fP = s / (PI**2 * (s**2 + p**2) ** 2)
        q1 = -(kvec + k0) / 2 + P
        q2 = -(kvec + k0 + q1)
        Q1 = (q1**2).sum(1)
        Q2 = (q2**2).sum(1)
        cut = (Q1 + n * n <= Lam**2) & (Q2 + n * n <= Lam**2)
        Om1 = np.sqrt(Q1 + n * n)
        Om2 = np.sqrt(Q2 + n * n)
        E = k + r0 + Om1 + Om2
        val = wk * wk0 / (Om1 * Om2 * E**2) / fP * cut / (16 * (2 * PI) ** 9)
        soft = r0 <= mu
        # We return the sum as well as the two parts, so that the error bar on K_n is
        # computed directly and accounts for the correlation between the parts.
        return np.array([np.mean(val * soft), np.mean(val * ~soft), np.mean(val)])
    return batched_mean(est, chunks, seed)


def run_kn(cfg):
    """Computes K_n for levels 2, 4, 8 and 16 at three cutoffs.  Plotted as
    K_n/(Lambda R) against n/(Lambda R), the results for different cutoffs should fall
    on one curve, which approaches the straight line a - b n/(Lambda R) of eq. (5.10)
    for small n/(Lambda R).  This is figure 1(b)."""
    print("\n[kn]  K_n = 2 J_n^(2,2) + J_n^mix at finite cutoff (eq. (5.6));"
          " K_n/(Lambda R) against n/(Lambda R) should lie on one curve")
    levels = [2, 4, 8, 16]
    out = {"levels": levels, "Lambda": cfg["kn_lambdas"], "K": {}, "K_err": {}}
    for Lam in cfg["kn_lambdas"]:
        Ks, Es = [], []
        for n in levels:
            (soft, mix, kn), (es, em, ek) = kn_mc(n, Lam, samples=cfg["kn_samples"], chunks=cfg["chunks"])
            Ks.append(kn)
            Es.append(ek)
            print(f"  Lambda R={Lam:4d}  n={n:3d}:  2J22={soft:.4e}  Jmix={mix:.4e}  "
                  f"K_n={kn:.4e} +- {ek:.1e}   K_n/(Lambda R)={1e7*kn/Lam:.3f}e-7")
        out["K"][str(Lam)] = Ks
        out["K_err"][str(Lam)] = Es
        # A straight-line fit in n.  It approaches a Lambda - b n only when n is much
        # smaller than Lambda R, so it is meaningful mainly at the largest cutoff.  The
        # values of a and b quoted in the paper come from the 'beta' section.
        fit = np.polyfit(levels[:3], Ks[:3], 1)
        print(f"    straight-line fit to n = 2,4,8:  intercept/Lambda = {fit[1]/Lam:.3e}  (compare a),"
              f"  minus slope = {-fit[0]:.3e}  (compare b)")
    # The (mu R)^2 scaling of section 5.3, tested at n = 4 with mu R = 0.5 and 1.
    Lam = cfg["kn_lambdas"][0]
    (_, _, k1), _ = kn_mc(4, Lam, mu=1.0, samples=cfg["kn_samples"], chunks=cfg["chunks"], seed=7)
    (_, _, k5), _ = kn_mc(4, Lam, mu=0.5, samples=cfg["kn_samples"], chunks=cfg["chunks"], seed=7)
    ratio = k5 / (k1 * 0.25)
    print(f"  mu R = 0.5 against 1 at n=4, Lambda R={Lam}:  K(0.5)/[0.25 K(1)] = {ratio:.3f}"
          "  (exactly 1 for pure mu^2 scaling; the small (2,2) part scales as mu^4)")
    out["mu_scaling"] = ratio
    return out


# ======================================================================================
# 4. The function F(m) and the constants a', beta, a and b (section 5.3)
#
#    For levels much heavier than mu, K_n reduces to QUARTIC_NORM x mu^2 x F(m_n), with
#
#        F(m) = int d^3k0 d^3q1  1 / [omega_0 Omega_1 Omega_2 (omega_0 + Omega_1 + Omega_2)^2],
#
#    eq. (5.7), where Omega_1 = sqrt(q1^2 + m^2) and Omega_2 = sqrt((k0 + q1)^2 + m^2).
#    Measuring momenta in units of m gives F(m) = m Phi(Lambda/m), and for large x
#    Phi(x) = a' x - beta, eqs. (5.8)-(5.9).
#
#    The slope a' depends on exactly how the cutoff is imposed.  The constant beta does
#    not: it comes from F(m) - F(0), an integral that converges, so the region near the
#    cutoff stops contributing as the cutoff is removed.  We check this with three
#    different cutoffs.
# ======================================================================================
def _beta_integrand(k0, q, q2, m):
    """The integrand of F(m), written in terms of |k0|, |q1| and |q2| = |k0 + q1|."""
    Om1 = np.sqrt(q * q + m * m)
    Om2 = np.sqrt(q2 * q2 + m * m)
    return 1.0 / (k0 * Om1 * Om2 * (k0 + Om1 + Om2) ** 2)


def beta_and_slope(Lam=None, mode=None, nx=800, nt=48):
    """Computes beta and a' at cutoff Lambda by deterministic quadrature.

    We integrate over the magnitudes |k0|, |q1| and |q2|.  In these variables
    d^3k0 d^3q1 = 8 pi^2 k0 q1 q2 dk0 dq1 dq2, on the triangle |k0-q1| <= q2 <= k0+q1.
    Because the integrand varies over many orders of magnitude, |k0| and |q1| are
    sampled evenly in their logarithms.

    Returns
        beta = -[F(1) - F(0)], the finite part, and
        a'   = F(0)/Lambda, the coefficient of the linear growth.

    The three versions of the cutoff (all with |k0| <= Lambda) are
        'i'   : no cutoff on the tower momenta,
        'ii'  : |q1| <= Lambda and |q2| <= Lambda, which is the cutoff of section 5.1
                for a massless level, and the one used in the paper,
        'iii' : k0^2 + q1^2 <= Lambda^2."""
    lo = -16.0
    hi = 16.0 if Lam is None else math.log(Lam) + (8.0 if mode == "i" else 0.5)
    khi = hi if Lam is None else math.log(Lam)
    xg, wx = leggauss(nx)
    tg, wt = leggauss(nt)
    xs = 0.5 * (khi - lo) * xg + 0.5 * (khi + lo)
    wk = 0.5 * (khi - lo) * wx
    ys = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
    wq = 0.5 * (hi - lo) * wx
    res = np.zeros(2)
    for i in range(0, nx, 100):                      # work through k0 in blocks to save memory
        K0 = np.exp(xs[i:i + 100])[:, None, None]
        Q = np.exp(ys)[None, :, None]
        a, b = np.abs(K0 - Q), K0 + Q
        T = tg[None, None, :]
        Q2 = 0.5 * (b - a) * T + 0.5 * (b + a)
        WT = 0.5 * (b - a) * wt[None, None, :]
        W = (wk[i:i + 100, None, None] * K0) * (wq[None, :, None] * Q) * WT * 8 * PI**2 * K0 * Q * Q2
        M = np.ones_like(Q2, bool)
        if mode == "ii":
            M = (Q <= Lam) & (Q2 <= Lam)
        if mode == "iii":
            M = (K0**2 + Q**2 <= Lam**2)
        diff = _beta_integrand(K0, Q, Q2, 1.0) - _beta_integrand(K0, Q, Q2, 0.0)
        res += np.array([-(W * diff * M).sum(),
                         (W * _beta_integrand(K0, Q, Q2, 0.0) * M).sum() / (Lam if Lam else 1.0)])
    return res


def richardson(lams, vals):
    """Estimates the value at infinite cutoff from three values at cutoffs Lambda,
    4 Lambda and 16 Lambda.  We assume the approach is v(Lambda) = v_inf - C/Lambda^p
    and determine the power p from the data themselves."""
    d1, d2 = vals[-2] - vals[-3], vals[-1] - vals[-2]
    r = d1 / d2                                   # equals 4^p
    return vals[-1] + d2 / (r - 1), math.log(r, 4)


def run_beta(cfg):
    """Computes beta and a' for the three cutoffs, extrapolates beta to infinite cutoff,
    and converts the results into the constants a and b of eq. (5.10).

    (We take beta from finite cutoffs and extrapolate, because evaluating F(1) - F(0)
    directly without a cutoff converges poorly in the logarithmic variables.)"""
    print("\n[beta]  Phi(x) = a' x - beta (eq. (5.9));  a = QUARTIC_NORM x a',  b = QUARTIC_NORM x beta")
    lams = cfg["beta_lambdas"]
    table = {}
    for Lam in lams:
        row = {mode: tuple(beta_and_slope(Lam, mode)) for mode in ("i", "ii", "iii")}
        table[str(Lam)] = row
        print(f"  Lambda={Lam:5d}:  " + " | ".join(
            f"({m}) beta={row[m][0]:6.2f} a'={row[m][1]:6.2f}" for m in ("i", "ii", "iii")))
    # Repeat the largest cutoff on a finer grid, to check the quadrature.
    bfine, afine = beta_and_slope(lams[-1], "ii", nx=1200, nt=64)
    print(f"  finer grid at Lambda={lams[-1]} (cutoff ii, 1200x64):  beta={bfine:.2f}  a'={afine:.2f}")
    ext = {}
    for mode in ("i", "ii", "iii"):
        v, pexp = richardson(lams, [table[str(L)][mode][0] for L in lams])
        ext[mode] = v
        print(f"  beta extrapolated to infinite cutoff, cutoff ({mode}): {v:.2f}"
              f"   (approached like Lambda^-{pexp:.2f})")
    beta = ext["ii"]
    a_prime = table[str(lams[-1])]["ii"][1]
    out = {"table": table, "beta_extrapolated": ext, "beta": beta, "a_prime": a_prime,
           "a": QUARTIC_NORM * a_prime, "b": QUARTIC_NORM * beta}
    print(f"  so beta = {beta:.1f} for every cutoff, while the slopes are a' = "
          + ", ".join(f"{table[str(lams[-1])][m][1]:.1f}" for m in ("i", "ii", "iii"))
          + " for cutoffs i, ii and iii (ii is the cutoff used in the paper)")
    spread = max(ext.values()) - min(ext.values())
    grid_err = abs(bfine - table[str(lams[-1])]["ii"][0])
    print(f"  accuracy of beta: the three cutoffs differ by {spread:.2f} and the finer grid changes it by"
          f" {grid_err:.2f}; the extrapolation adds an uncertainty of about 0.1")
    print(f"  a = {out['a']:.3e},  b = {out['b']:.3e}   (paper: a = 5.76e-7, b = 2.1e-6)")
    return out


# ======================================================================================
# 5. The (1,3) channel (sections 4.2 and 5.3)
#
#    A soft zero-mode quantum is produced together with three tower quanta, at levels
#    n1 + n2 + n3 = 0.  For the mutual information with level n we need J_n^(1,3), the
#    contribution of all such states that have one quantum at level +n or -n.
#
#    We fix the first tower quantum at level +n and sum over the level n2 of the
#    second; the third then sits at n3 = -n - n2.  The factor 1/3! of eq. (3.16) is
#    cancelled by the three ways of choosing which quantum sits at level n and the two
#    signs of n, which leaves
#
#        J_n^(1,3) = (2 pi)^-9  sum over n2  int d^3k d^3q1 d^3q2
#                    1 / (16 omega Omega_1 Omega_2 Omega_3 E^2),
#
#    with the third momentum q3 = -(k + q1 + q2) and the cutoff on all three tower
#    quanta.
#
#    One special case needs care.  If n2 = n (so that n3 = -2n), or n3 = n, the state
#    contains two identical quanta at level +n.  Either of them could be the one we
#    fixed, so these states would be counted twice as often as the others.  We give them
#    weight 1/2, so that every state is counted once.  (Two identical quanta at the other
#    levels, n2 = n3 = -n/2, need no correction, because the quantum at level n is then
#    unique.)
# ======================================================================================
def j13_mc(n, Lam, mu=1.0, samples=2_000_000, chunks=8, seed=4):
    """Monte Carlo estimate of J_n^(1,3) at cutoff Lambda.  The level n2 is drawn
    uniformly from the allowed values, and both free tower momenta uniformly from the
    ball |q| <= Lambda."""
    N = int(math.floor(Lam))
    allowed = np.array([m for m in range(-N, N + 1) if m != 0 and m != -n and abs(n + m) <= N])

    def est(rng):
        kvec, k, wk = soft_one_over_omega(rng, samples, mu)
        n2 = rng.choice(allowed, samples)             # level of the second tower quantum
        n3 = -n - n2
        q1, Vb = uniform_ball(rng, samples, Lam)       # both free tower momenta
        q2, _ = uniform_ball(rng, samples, Lam)
        q3 = -(kvec + q1 + q2)
        Q1, Q2, Q3 = (q1**2).sum(1), (q2**2).sum(1), (q3**2).sum(1)
        cut = (Q1 + n * n <= Lam**2) & (Q2 + n2**2 <= Lam**2) & (Q3 + n3**2 <= Lam**2)
        Om1, Om2, Om3 = np.sqrt(Q1 + n * n), np.sqrt(Q2 + n2**2), np.sqrt(Q3 + n3**2)
        E = k + Om1 + Om2 + Om3
        mult = np.where((n2 == n) | (n3 == n), 0.5, 1.0)   # weight 1/2 for the states (n, n, -2n)
        val = wk * Vb * Vb * len(allowed) * mult / (Om1 * Om2 * Om3 * E**2) * cut / (16 * (2 * PI) ** 9)
        return np.mean(val)
    return batched_mean(est, chunks, seed)


def run_j13(cfg):
    """Computes J_n^(1,3) for levels 2, 4, 8 and 16 at three cutoffs.  Plotted as
    J_n^(1,3)/(Lambda R)^2 against n/(Lambda R), the results should follow the line
    c - c' n/(Lambda R) of eq. (5.11).  This is figure 1(c)."""
    print("\n[j13]  (1,3) channel: J_n^(1,3)/(Lambda R)^2 against n/(Lambda R) (eq. (5.11))")
    levels = [2, 4, 8, 16]
    out = {"levels": levels, "Lambda": cfg["j13_lambdas"], "J": {}, "err": {}, "c": {}, "cp": {}}
    for Lam in cfg["j13_lambdas"]:
        Js, Es = [], []
        for n in levels:
            j, e = j13_mc(n, Lam, samples=cfg["j13_samples"], chunks=cfg["chunks"])
            Js.append(j)
            Es.append(e)
        # Straight-line fit y = c - c' x to all four levels, weighted by the error bars.
        x = np.array(levels) / Lam
        y = np.array(Js) / Lam**2
        sig = np.array(Es) / Lam**2
        (slope, icpt), cov = np.polyfit(x, y, 1, w=1 / sig, cov="unscaled")
        out["J"][str(Lam)], out["err"][str(Lam)] = Js, Es
        out["c"][str(Lam)], out["cp"][str(Lam)] = icpt, -slope
        # As a check on the fit, repeat it with only the three lowest levels.
        (s3, i3), _ = np.polyfit(x[:3], y[:3], 1, w=1 / sig[:3], cov="unscaled")
        out.setdefault("c_n248", {})[str(Lam)], out.setdefault("cp_n248", {})[str(Lam)] = i3, -s3
        print(f"  Lambda R={Lam:4d}:  " + "  ".join(f"n={n}: {j/Lam**2*1e7:.3f}({e/Lam**2*1e7:.3f})"
                                                   for n, j, e in zip(levels, Js, Es))
              + f"  [units 1e-7]  ->  c = {icpt*1e7:.2f}({math.sqrt(cov[1,1])*1e7:.2f})e-7,"
                f"  c' = {-slope*1e7:.2f}({math.sqrt(cov[0,0])*1e7:.2f})e-7"
                f"   [with n = 2,4,8 only: c = {i3*1e7:.2f}e-7, c' = {-s3*1e7:.2f}e-7]")
    return out


# ======================================================================================
# 6. The zero-mode coefficient D_3 in the cubic theory (section 6.4, eq. (6.7))
#
#    The entropy of the accessible sector in the cubic theory has, besides the tower
#    term, a term D_3 mu from states made only of zero modes, with at least one soft and
#    at least one hard quantum:
#
#        D_3 mu = (1/3!) (8 (2 pi)^6)^-1  int d^3k1 d^3k2  1 / [omega_1 omega_2 omega_3 (omega_1+omega_2+omega_3)^2].
#
#    The factor 1/3! is there because the three quanta are identical.  Using the three
#    energies themselves as variables, the measure becomes 8 pi^2 d omega_1 d omega_2
#    d omega_3 on the triangle |omega_1 - omega_2| <= omega_3 <= omega_1 + omega_2, and
#    the integrand is just 1/(omega_1 + omega_2 + omega_3)^2.  This can be integrated by
#    hand:
#        one soft quantum  : 8 pi^2 (5 ln 2 - 3 ln 3) mu, for each of the 3 choices,
#        two soft quanta   : 8 pi^2 (3 ln 3 - 4 ln 2 - 1/2) mu, for each of the 3 choices,
#    which add up to 24 pi^2 (ln 2 - 1/2) mu.  This is the massless limit of the
#    integral I_3(mu) of BMV.  We check the closed form numerically in two ways.
# ======================================================================================
def run_d3(cfg):
    """Evaluates I_3 in closed form, by quadrature and by Monte Carlo, and compares it
    with the value quoted by BMV (footnote to eq. (6.7))."""
    print("\n[d3]  zero-mode coefficient D_3 of the cubic theory")
    L2, L3 = math.log(2), math.log(3)
    one_soft = 5 * L2 - 3 * L3
    two_soft = 3 * L3 - 4 * L2 - 0.5
    I3 = 3 * 8 * PI**2 * (one_soft + two_soft)          # = 24 pi^2 (ln 2 - 1/2)
    print(f"  closed form:  I_3/mu = 24 pi^2 (ln2-1/2) = {I3:.4f},  so D_3 = {D3_COEFF:.4e}")

    # Check 1: direct quadrature in the original variables |k1|, |k2| and the cosine c
    # of the angle between them.  In both cases below the third quantum must be hard,
    # |k1 + k2| > mu, which happens when c is larger than c_min = (mu^2 - k^2 - q^2)/(2kq).
    # We therefore integrate c from c_min to 1 (with c_min kept between -1 and 1).  This
    # keeps the soft/hard boundary at the edge of the integration region, so the
    # integrand is smooth inside it and the quadrature converges without warnings.
    def f(c, q, k):
        w3 = math.sqrt(max(k * k + q * q + 2 * k * q * c, 0.0))
        E = k + q + w3
        return 4 * PI * k * k * 2 * PI * q * q / (k * q * w3 * E * E)

    def c_min(k, q):
        return min(max((1.0 - k * k - q * q) / (2 * k * q), -1.0), 1.0)

    one = sum(integrate.tplquad(f, 0, 1, lo, hi, c_min, 1, epsrel=1e-6)[0]
              for lo, hi in [(1, 3), (3, 30), (30, 3000)])
    two = integrate.tplquad(f, 0, 1, 0, 1, c_min, 1, epsrel=1e-6)[0]
    # 'one' has the first quantum soft and the other two hard; 'two' has the first two
    # soft.  Each case occurs in 3 ways, so the total is 3 (one + two).  The integral of
    # the one-soft case beyond |q| = 3000 is added by hand: there the integrand falls
    # like 4 pi^2 k/q^2, which gives 2 pi^2/3000 for each of the 3 ways.
    tail = 3 * 2 * PI**2 / 3000.0
    print(f"  quadrature :  I_3/mu = {3*(one+two)+tail:.4f}")

    # Check 2: Monte Carlo in the original six variables, with both momenta drawn
    # evenly in the logarithm of their magnitude.
    rng = make_rng(7)
    rmin, rmax = 1e-4, 1e4
    LN = math.log(rmax / rmin)
    vals = []
    for _ in range(cfg["d3_chunks"]):
        M = cfg["d3_samples"]
        r1 = rmin * np.exp(rng.uniform(size=M) * LN)
        r2 = rmin * np.exp(rng.uniform(size=M) * LN)
        k1 = isotropic_unit(rng, M) * r1[:, None]
        k2 = isotropic_unit(rng, M) * r2[:, None]
        p1 = 1 / (4 * PI * r1**3 * LN)                  # the corresponding density in 3D
        p2 = 1 / (4 * PI * r2**3 * LN)
        w3 = np.linalg.norm(k1 + k2, axis=1)
        ns = (r1 <= 1).astype(int) + (r2 <= 1) + (w3 <= 1)
        f6 = ((ns >= 1) & (ns <= 2)) / (r1 * r2 * w3 * (r1 + r2 + w3) ** 2)
        vals.append(np.mean(f6 / (p1 * p2)))
    vals = np.array(vals)
    print(f"  Monte Carlo:  I_3/mu = {vals.mean():.2f} +- {vals.std(ddof=1)/math.sqrt(len(vals)):.2f}"
          "  (a few rare samples carry large weights, so this estimate tends to come out slightly low)")
    bmv = 8 * PI**2 * (1 - 0.5 * L2)
    print(f"  BMV's value:  I_3/mu = 8 pi^2 (1 - ln2/2) = {bmv:.4f}, larger than ours by a factor {bmv/I3:.4f};"
          f"  BMV's eq. (43) also has no 1/3!, so the combined factor is {6*bmv/I3:.3f}")
    return {"I3": I3, "I3_quad": 3 * (one + two) + tail, "I3_mc": vals.mean(), "D3": D3_COEFF, "I3_bmv": bmv}


# ======================================================================================
# 7. Other numbers quoted in sections 4, 6 and 8
# ======================================================================================
def run_misc(cfg):
    """Prints the closed-form coefficients and the estimates of section 8."""
    print("\n[misc]")
    print(f"  A   = pi^4/(32 (2pi)^9)          = {A_COEFF:.3e}    (eq. (4.8))")
    print(f"  A_3 = 1/(1024 pi^3)              = {A3_COEFF:.3e}    (eq. (6.3))")
    print(f"  corrections to amplitudes add up to sum_n 1/m_n^2 = pi^2/6 R^2 = {PI**2/6:.4f} R^2  (section 8.1)")
    # The quantity epsilon of eq. (8.2) is the second Renyi entropy that the tower
    # contributes, S_2 = 2 lambda_3^2 V (A_3/R) ln N, divided by the number of accessible
    # modes N_A = V mu^3/(6 pi^2).
    coeff = 2 * A3_COEFF * 6 * PI**2                    # multiplies (lambda_3 R)^2 ln N / (mu R)
    print(f"  epsilon = {coeff:.5f} (lambda_3 R)^2 ln N/(mu R);  3/(256 pi) = {3/(256*PI):.5f};"
          f"  epsilon reaches 1 at (lambda_3 R)^2 ln N = {1/coeff:.0f} mu R")
    # The species bound Lambda <= M_Pl/sqrt(N), with N = Lambda R = Lambda/M_KK levels,
    # gives N = (M_Pl/M_KK)^(2/3)  (section 8.4).
    M_PL = 1.22e19                                       # Planck mass in GeV
    for label, mkk in [("1 TeV", 1e3), ("1e-3 eV", 1e-12)]:
        N = (M_PL / mkk) ** (2 / 3)
        print(f"  species bound with M_KK = {label:8s}:  N = {N:.2e},  ln N = {math.log(N):.1f}")
    print(f"  the quartic coupling becomes strong near level 12 pi^2/lambda = {12*PI**2:.1f}/lambda")
    return {"A": A_COEFF, "A3": A3_COEFF, "eps_coeff": coeff}


# ======================================================================================
# 8. Figure 1 (fig_mc.pdf)
# ======================================================================================
# The panels of figure 1 and the calculations each one needs.
PANELS = {"a": ["cubic"], "b": ["kn", "beta"], "c": ["j13"]}


def make_figure(res):
    """Draws figure 1 from the saved results.

    A panel is drawn only if the calculations it needs have been run (see PANELS).  With
    all three panels the figure is saved as fig_mc.pdf, the file used in the paper;
    otherwise it is saved under the names of the panels it contains, e.g. fig_1c.pdf."""
    panels = [p for p, need in PANELS.items() if all(s in res for s in need)]
    for p, need in PANELS.items():
        if p not in panels:
            print(f"\n[figure]  panel ({p}) skipped: first run  {' '.join(need)}")
    if not panels:
        return
    fname = "fig_mc.pdf" if len(panels) == 3 else "fig_1" + "".join(panels) + ".pdf"

    import matplotlib
    in_notebook = "ipykernel" in sys.modules
    if not in_notebook:                  # outside a notebook, draw without opening a window
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # The style of figure 1 in the paper.
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]    # blue, orange, green: one per cutoff
    marks = ["o", "s", "^"]
    grey = "#8a8984"

    def cutoff_style(i):
        """Marker style for the i-th cutoff: an open circle for the lowest cutoff,
        filled squares and triangles for the others."""
        if i == 0:
            return dict(color=colours[0], marker=marks[0], ms=7.5, lw=0, markerfacecolor="none",
                        markeredgecolor=colours[0], markeredgewidth=1.3, zorder=3)
        return dict(color=colours[i % 3], marker=marks[i % 3], ms=5.5, lw=0,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)

    fig, axs = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.45), squeeze=False)
    ax_of = dict(zip(panels, axs[0]))
    x = np.geomspace(0.002, 0.5, 200)              # n/(Lambda R), for the dashed lines

    # Panel (a): the cubic channel, n R J_n^(3)/A_3.  Monte Carlo points on the
    # quadrature curve, with the asymptotic value 1 as a dashed line.  (The Monte Carlo
    # errors are below 1%, smaller than the points.)
    if "a" in ax_of:
        c = res["cubic"]
        ax = ax_of["a"]
        ax.axhline(1, color=grey, lw=1.0, ls="--", zorder=1)
        ax.plot(c["curve_n"], c["curve"], color=grey, lw=1.4, zorder=2, label="quadrature")
        n = np.array(c["levels"])
        ax.plot(n, n * np.array(c["mc"]) / A3_COEFF, color=colours[0], marker="o", ms=5.5, lw=0,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3, label="Monte Carlo")
        ax.set_xscale("log")
        ax.set_xlim(0.8, 100)
        ax.set_ylim(0.7, 1.04)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_xlabel(r"level $n$")
        ax.set_ylabel(r"$nR\,J_n^{(3)}/A_3$")
        ax.set_title("(a) cubic channel", fontsize=9)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    # Panel (b): the (2,2) and mixed channels, K_n/(Lambda R) against n/(Lambda R) for
    # three cutoffs, with the line a - b n/(Lambda R) of eq. (5.10).  The horizontal axis
    # is logarithmic, which is why the straight line appears curved.
    if "b" in ax_of:
        k = res["kn"]
        a7, b7 = res["beta"]["a"] / 1e-7, res["beta"]["b"] / 1e-7
        ax = ax_of["b"]
        yl = a7 - b7 * x
        ax.plot(x[yl > 0], yl[yl > 0], color=grey, lw=1.2, ls="--", zorder=1,
                label=r"$a-b\,n/\Lambda_{\rm UV}R$")
        for i, L in enumerate(k["Lambda"]):
            ax.plot(np.array(k["levels"]) / L, np.array(k["K"][str(L)]) / L / 1e-7,
                    label=rf"$\Lambda_{{\rm UV}}R={L}$", **cutoff_style(i))
        ax.set_xscale("log")
        ax.set_xlim(0.002, 0.5)
        ax.set_ylim(0, 6.4)
        ax.set_xlabel(r"$n/\Lambda_{\rm UV}R$")
        ax.set_ylabel(r"$R^3K_n/\Lambda_{\rm UV}R\ \ [10^{-7}]$")
        ax.set_title(r"(b) $(2,2)$ and mixed", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower left")

    # Panel (c): the (1,3) channel, J_n^(1,3)/(Lambda R)^2 against n/(Lambda R) for
    # three cutoffs, with the line c - c' n/(Lambda R) of eq. (5.11) fitted at the
    # largest cutoff.
    if "c" in ax_of:
        j = res["j13"]
        Lmax = str(j["Lambda"][-1])
        c7, cp7 = j["c"][Lmax] / 1e-7, j["cp"][Lmax] / 1e-7
        ax = ax_of["c"]
        ax.plot(x, c7 - cp7 * x, color=grey, lw=1.2, ls="--", zorder=1,
                label=r"$c-c'\,n/\Lambda_{\rm UV}R$")
        for i, L in enumerate(j["Lambda"]):
            ax.plot(np.array(j["levels"]) / L, np.array(j["J"][str(L)]) / L**2 / 1e-7,
                    label=rf"$\Lambda_{{\rm UV}}R={L}$", **cutoff_style(i))
        ax.set_xscale("log")
        ax.set_xlim(0.008, 0.5)
        ax.set_ylim(0, 3.2)
        ax.set_xlabel(r"$n/\Lambda_{\rm UV}R$")
        ax.set_ylabel(r"$R^3J^{(1,3)}_n/(\Lambda_{\rm UV}R)^2\ \ [10^{-7}]$")
        ax.set_title(r"(c) $(1,3)$", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower left")

    for ax in axs[0]:
        ax.grid(axis="y", color="#ecebe7", lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(fname)
    print(f"\n[figure]  saved as {fname}")
    if in_notebook:
        plt.show()


# ======================================================================================
# Running the calculations
# ======================================================================================
# Sample sizes and cutoffs.  QUICK gives the paper's numbers within their statistical
# errors in a few minutes; FULL uses the sample sizes of the paper.
QUICK = dict(chunks=8, cubic_samples=200_000, j22_samples=200_000, kn_samples=300_000,
             kn_lambdas=[40, 160, 640], j13_samples=300_000, j13_lambdas=[40, 80, 160],
             beta_lambdas=[50, 200, 800], d3_samples=500_000, d3_chunks=8)
FULL = dict(chunks=10, cubic_samples=2_000_000, j22_samples=2_000_000, kn_samples=2_000_000,
            kn_lambdas=[40, 160, 640], j13_samples=2_000_000, j13_lambdas=[40, 80, 160],
            beta_lambdas=[50, 200, 800, 3200], d3_samples=2_000_000, d3_chunks=20)
SECTIONS = {"cubic": run_cubic, "j22": run_j22, "kn": run_kn, "beta": run_beta,
            "j13": run_j13, "d3": run_d3, "misc": run_misc}


def main(argv=None):
    """Runs the sections named in argv, for example main(["cubic", "--full"]).  With
    argv=None the command-line arguments are used."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sections", nargs="*", default=list(SECTIONS) + ["figure"],
                    metavar="calculation",
                    help="what to run: " + ", ".join(list(SECTIONS) + ["figure"])
                    + "  (see the table at the top of the file)")
    ap.add_argument("--full", action="store_true", help="use the sample sizes of the paper")
    args = ap.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS and s != "figure"]
    if unknown:
        ap.error(f"unknown calculation {', '.join(unknown)}; choose from "
                 + ", ".join(list(SECTIONS) + ["figure"]))
    cfg = FULL if args.full else QUICK
    # Start from earlier results if there are any, so that sections run at different
    # times end up in the same file.
    res = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as fh:
            res = json.load(fh)
    for name in args.sections:
        if name == "figure":
            continue
        t0 = time.time()
        res[name] = SECTIONS[name](cfg)
        print(f"  ({name} took {time.time()-t0:.0f} s)")
        with open(RESULTS_FILE, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    if "figure" in args.sections:
        make_figure(res)


if __name__ == "__main__":
    if os.path.basename(sys.argv[0]).startswith("ipykernel"):
        # The script was pasted into a notebook cell.  Jupyter passes its own arguments,
        # which are not meant for us, so we wait to be told what to run.
        print('Loaded.  Run, for example, main(["j13", "figure"]), or main() for everything.')
    else:
        main()
