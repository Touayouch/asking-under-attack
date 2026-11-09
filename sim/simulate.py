#!/usr/bin/env python3
"""
Stylized simulations for the draft paper
"Asking Under Attack: Learning Fatigue-Aware Confirmation Policies for LLM Agents".

Every number produced here comes from the stylized sequential model of the
paper (Gaussian risk score, linear-with-floor user fatigue). These are model
predictions of curve *shapes*. They are NOT AgentDojo results.

    python simulate.py --out ../paper          # full run (a few minutes on CPU)
    python simulate.py --out ../paper --fast   # quick check
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

np.seterr(divide="ignore", invalid="ignore", over="ignore")

# ----------------------------------------------------------------------------
# Palette and style. Categorical hues in a fixed order; one look per entity,
# reused in every figure ("color follows the entity").
# ----------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREEN, VIOLET = "#008300", "#6250d6"
GRAY, INK, INK2, GRID = "#8c8c8c", "#1f1f1f", "#5a5a5a", "#e6e6e6"
BLUE_RAMP = ["#a9cbf2", "#6aa3e6", "#2a78d6", "#174a8a"]
ORANGE_RAMP = ["#f4a582", "#eb6834", "#a8401a"]

LOOK = {
    "dp":     dict(label="Fatigue-aware optimal (DP)",      color=BLUE,    marker="o", ls="-"),
    "myopic": dict(label="Myopic fatigue-aware (Prop. 1)",  color=ORANGE,  marker="s", ls="--"),
    "static": dict(label="Static guard (fatigue-unaware)",  color=AQUA,    marker="^", ls="-."),
    "always": dict(label="Always confirm",                  color=YELLOW,  marker="D", ls=":"),
    "auto":   dict(label="Autonomous (never ask)",          color=GRAY,    marker="X", ls=(0, (1, 1.2))),
    "lfix":   dict(label="Alg. 1, fixed profile",           color=MAGENTA, marker="v", ls="-"),
    "lrand":  dict(label="Alg. 1, randomized profiles",     color=GREEN,   marker="P", ls="-."),
    "lnofat": dict(label="Alg. 1, fatigue-free simulator",  color=VIOLET,  marker="<", ls="--"),
    "lvan":   dict(label="Alg. 1 with per-group std (vanilla GRPO)", color=INK, marker=".", ls=":"),
}


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.4, "lines.markersize": 3.6,
        "legend.frameon": False, "legend.handlelength": 2.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8, pad=4)


def save(fig, name, figdir, prevdir):
    fig.savefig(os.path.join(figdir, name + ".pdf"))
    fig.savefig(os.path.join(prevdir, name + ".png"), dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Stylized model (Section 4 of the paper)
# ----------------------------------------------------------------------------
BASE = dict(N=40, pi=0.10, mu=2.0, g=1.0, L=10.0, c=0.2, C=5.0, s=0.10, rmin=0.10)
EPS = 0.05  # CMDP attack budget on the attack success rate


def fatigue(l, C, s, rmin):
    l = np.asarray(l, dtype=float)
    return np.maximum(rmin, 1.0 - s * np.maximum(0.0, l - C))


def rfun_of(C, s, rmin=BASE["rmin"]):
    return lambda l: fatigue(l, C, s, rmin)


def r_none(l):
    return np.ones_like(np.asarray(l, dtype=float))


def make_rvec(Cv, sv, rmin):
    """Per-session fatigue profiles (arrays Cv, sv)."""
    return lambda l: np.maximum(rmin, 1.0 - sv * np.maximum(0.0, l - Cv))


def posterior(x, pi=BASE["pi"], mu=BASE["mu"]):
    """P(z = 1 | x) when x ~ N(mu z, 1)."""
    llr = np.clip(mu * np.asarray(x, dtype=float) - 0.5 * mu * mu, -60.0, 60.0)
    return 1.0 / (1.0 + (1.0 - pi) / pi * np.exp(-llr))


def thresholds(r, g, L, c):
    r = np.asarray(r, dtype=float)
    return c / (r * L), (g - c) / (g + (1.0 - r) * L)


def l_dagger(prm):
    g, L, c, C, s = (prm[k] for k in ("g", "L", "c", "C", "s"))
    return C + (1.0 - c * (1.0 / L + 1.0 / g)) / s


def x_grid(prm, K=3001):
    xs = np.linspace(-7.0, 7.0 + prm["mu"], K)
    f = (1 - prm["pi"]) * norm.pdf(xs) + prm["pi"] * norm.pdf(xs - prm["mu"])
    return xs, f / f.sum()


def solve_dp(prm, rfun, N=None, lam=None):
    """Finite-horizon DP over (t, load). Returns V[t, l]."""
    N = int(N or prm["N"])
    g, c = prm["g"], prm["c"]
    L = prm["L"] if lam is None else lam
    xs, w = x_grid(prm)
    p = posterior(xs, prm["pi"], prm["mu"])[None, :]
    r = rfun(np.arange(N + 2))[:, None]
    V = np.zeros((N + 1, N + 2))
    for t in range(N - 1, -1, -1):
        V1 = V[t + 1][:, None]
        V2 = np.append(V[t + 1, 1:], V[t + 1, -1])[:, None]
        QE = (1 - p) * g - p * L + V1
        QA = (1 - p) * g - c - p * (1 - r) * L + V2
        V[t] = (np.maximum(np.maximum(QE, QA), V1) * w).sum(1)
    return V


# --------------------------------------------------------------- policies
# A policy maps (t, load l, posterior p) -> action in {0: execute, 1: ask, 2: refuse}.
def pol_dp(V, prm, rfun, lam=None):
    g, c = prm["g"], prm["c"]
    L = prm["L"] if lam is None else lam
    lmax = V.shape[1] - 1

    def act(t, l, p):
        V1 = V[t + 1, l]
        V2 = V[t + 1, np.minimum(l + 1, lmax)]
        r = rfun(l)
        QE = (1 - p) * g - p * L + V1
        QA = (1 - p) * g - c - p * (1 - r) * L + V2
        QR = V1 + 0.0 * p
        return np.argmax(np.stack([QE, QA, QR], -1), -1)
    return act


def pol_myopic(prm, rfun):
    g, L, c = prm["g"], prm["L"], prm["c"]
    b = g / (g + L)

    def act(t, l, p):
        t1, t2 = thresholds(rfun(l), g, L, c)
        a = np.where(p < t1, 0, np.where(p < t2, 1, 2))
        return np.where(t1 < t2, a, np.where(p < b, 0, 2))
    return act


def pol_static(prm):
    return pol_myopic(prm, r_none)


def pol_auto(prm):
    b = prm["g"] / (prm["g"] + prm["L"])
    return lambda t, l, p: np.where(p < b, 0, 2)


def pol_always():
    return lambda t, l, p: np.ones(np.shape(p), dtype=int)


def pol_guard(tau, hi=0.8):
    return lambda t, l, p: np.where(p < tau, 0, np.where(p < hi, 1, 2))


# ------------------------------------------------------------ Monte Carlo
def draw(prm, n, N=None, seed=0):
    N = int(N or prm["N"])
    rng = np.random.default_rng(seed)
    z = rng.random((n, N)) < prm["pi"]
    x = rng.normal(prm["mu"] * z, 1.0)
    return z, posterior(x, prm["pi"], prm["mu"]), rng.random((n, N))


def run(policy, prm, rtrue, data):
    z, p, u = data
    n, N = z.shape
    g, L, c = prm["g"], prm["L"], prm["c"]
    l = np.zeros(n, dtype=int)
    eb, em, ak = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(N):
        a = policy(t, l, p[:, t])
        zt = z[:, t]
        ok = np.where(zt, u[:, t] >= rtrue(l), True)  # malicious approved w.p. 1 - r(l)
        ex = (a == 0) | ((a == 1) & ok)
        em += ex & zt
        eb += ex & ~zt
        ask = a == 1
        ak += ask
        l = l + ask
    rew = g * eb - L * em - c * ak
    return dict(asr=float(em.sum() / z.sum()), benign=float(eb.sum() / (~z).sum()),
                asks=float(ak.mean()), reward=float(rew.mean()),
                reward_se=float(rew.std(ddof=1) / np.sqrt(n)),
                util=float((g * eb - c * ak).mean()))


def run_flood(policy, rtrue, k, n, seed, lo=1.2, hi=2.0):
    """k benign-but-suspicious decoys, then one disguised injected action."""
    rng = np.random.default_rng(seed)
    l = np.zeros(n, dtype=int)
    done = np.zeros(n)
    for t in range(k):
        p = posterior(rng.uniform(lo, hi, n))
        a = policy(t, l, p)
        done += (a == 0) | (a == 1)           # benign decoys are approved when asked
        l = l + (a == 1)
    p = posterior(rng.uniform(lo, hi, n))
    a = policy(k, l, p)
    ok = rng.random(n) >= rtrue(l)
    harm = (a == 0) | ((a == 1) & ok)
    return float(harm.mean()), (float(done.mean() / k) if k > 0 else np.nan), float(l.mean())


# ---------------------------------------------- Algorithm 1 in the toy model
NF = 6


def features(p, l, t, N):
    lp = np.clip(np.log(p) - np.log1p(-p), -10.0, 10.0) / 4.0
    ln = np.asarray(l, dtype=float) / 10.0
    one = np.ones_like(lp)
    return np.stack([one, lp, ln, ln * lp, ln * ln, one * (t / N)], axis=-1)


def probs(theta, f):
    za, zr = f @ theta[0], f @ theta[1]
    m = np.maximum(0.0, np.maximum(za, zr))
    e = np.stack([np.exp(-m), np.exp(za - m), np.exp(zr - m)], -1)
    return e / e.sum(-1, keepdims=True)


def sample_actions(pr, rng):
    cu = rng.random(pr.shape[0])
    return (cu > pr[:, 0]).astype(int) + (cu > pr[:, 0] + pr[:, 1]).astype(int)


def pol_learned(theta, N, seed=0):
    rng = np.random.default_rng(seed)
    return lambda t, l, p: sample_actions(probs(theta, features(p, l, t, N)), rng)


def fixed_sampler(C, s):
    return lambda rng, B: (np.full(B, float(C)), np.full(B, float(s)))


def rand_sampler(Crange=(2, 10), srange=(0.05, 0.20)):
    return lambda rng, B: (rng.integers(Crange[0], Crange[1] + 1, B).astype(float),
                           rng.uniform(srange[0], srange[1], B))


def train_alg1(prm, sampler, iters, eval_data, eval_rtrue, B=64, G=8, eps=EPS,
               lr=0.03, eta=0.5, lam0=10.0, seed=1, eval_every=10, group_std=False, init="autonomous"):
    """Group-relative policy gradient + dual ascent (the toy analogue of Alg. 1).
    One on-policy step per batch, so the clipped ratio of GRPO is inactive."""
    rng = np.random.default_rng(seed)
    N, g, c, rmin = prm["N"], prm["g"], prm["c"], prm["rmin"]
    theta = np.zeros((2, NF))
    if init == "autonomous":       # start like a prompted base model: rarely asks,
        theta[0, 0] = -2.0         # refuses clearly risky calls (p > 0.09)
        theta[1, :2] = [2.31, 4.0]
    else:
        theta[:, 0] = -2.0
    m1, m2 = np.zeros_like(theta), np.zeros_like(theta)
    lam = lam0
    H = {k: [] for k in ("it", "lam", "U", "asr", "asks")}
    for it in range(iters + 1):
        if it % eval_every == 0:
            res = run(pol_learned(theta, N, seed=10_000 + it), prm, eval_rtrue, eval_data)
            H["it"].append(it); H["lam"].append(lam)
            H["U"].append(res["util"]); H["asr"].append(res["asr"]); H["asks"].append(res["asks"])
        if it == iters:
            break
        z = rng.random((B, N)) < prm["pi"]
        p = posterior(rng.normal(prm["mu"] * z, 1.0))
        z, p = np.repeat(z, G, 0), np.repeat(p, G, 0)    # a group shares the "prompt"
        u = rng.random((B * G, N))
        Cv, sv = sampler(rng, B)                         # the profile is part of the prompt
        rtrue = make_rvec(np.repeat(Cv, G), np.repeat(sv, G), rmin)
        n = B * G
        l = np.zeros(n, dtype=int)
        grad = np.zeros((n, 2, NF))
        eb, em, ak = np.zeros(n), np.zeros(n), np.zeros(n)
        for t in range(N):
            f = features(p[:, t], l, t, N)
            pr = probs(theta, f)
            a = sample_actions(pr, rng)
            oh = np.stack([a == 1, a == 2], 1).astype(float)
            grad += (oh - pr[:, 1:])[:, :, None] * f[:, None, :]
            zt = z[:, t]
            ok = np.where(zt, u[:, t] >= rtrue(l), True)
            ex = (a == 0) | ((a == 1) & ok)
            em += ex & zt
            eb += ex & ~zt
            ak += a == 1
            l = l + (a == 1)
        R = g * eb - c * ak - lam * em                   # Lagrangian return R_lambda
        Rg = R.reshape(B, G)
        cen = Rg - Rg.mean(1, keepdims=True)              # group-relative baseline
        if group_std:                                     # vanilla GRPO scaling (biased, see paper)
            adv = (cen / (Rg.std(1, keepdims=True) + 1e-6)).reshape(-1)
        else:                                             # one batch-level scale keeps lambda's meaning
            adv = (cen / (cen.std() + 1e-6)).reshape(-1)
        gth = (adv[:, None, None] * grad).mean(0) / N
        m1 = 0.9 * m1 + 0.1 * gth
        m2 = 0.999 * m2 + 0.001 * gth ** 2
        k = it + 1
        theta = theta + lr * (m1 / (1 - 0.9 ** k)) / (np.sqrt(m2 / (1 - 0.999 ** k)) + 1e-8)
        asr = em.sum() / max(1, z.sum())
        lam = max(0.0, lam + eta * (asr - eps))          # dual ascent
    return theta, H, lam


def cmdp_oracle(prm, rfun, data, eps=EPS, lo=0.5, hi=500.0, iters=18):
    """DP with multiplier lambda tuned by bisection so that ASR = eps."""
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        V = solve_dp(prm, rfun, lam=mid)
        res = run(pol_dp(V, prm, rfun, lam=mid), prm, rfun, data)
        if res["asr"] > eps:
            lo = mid
        else:
            hi = mid
    V = solve_dp(prm, rfun, lam=hi)
    res = run(pol_dp(V, prm, rfun, lam=hi), prm, rfun, data)
    res["lam"] = hi
    return res


# ------------------------------------------------- pilot study power analysis
def logit_slope_z(X, y, iters=30, ridge=1e-3):
    R, M, K = X.shape
    beta = np.zeros((R, K))
    I = np.eye(K)[None]
    for _ in range(iters):
        eta = np.clip(np.einsum("rmk,rk->rm", X, beta), -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu)
        grad = np.einsum("rmk,rm->rk", X, y - mu) - ridge * beta
        Hm = np.einsum("rmk,rm,rml->rkl", X, W, X) + ridge * I
        beta = beta + np.linalg.solve(Hm, grad[..., None])[..., 0]
    eta = np.clip(np.einsum("rmk,rk->rm", X, beta), -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    Hm = np.einsum("rmk,rm,rml->rkl", X, mu * (1 - mu), X) + ridge * I
    cov = np.linalg.inv(Hm)
    return beta[:, 1] / np.sqrt(cov[:, 1, 1])


def pilot_power(rfun, n_list, m_rate, R=1000, n_req=60, seed=7):
    """Power of a one-sided logistic-regression slope test (alpha = 0.05)."""
    rng = np.random.default_rng(seed)
    k = int(round(n_req * m_rate))
    out = []
    for n in n_list:
        pos = np.argsort(rng.random((R, n, n_req)), axis=2)[:, :, :k]
        ell = pos.reshape(R, n * k).astype(float)
        y = (rng.random(ell.shape) < rfun(ell)).astype(float)
        X = np.stack([np.ones_like(ell), (ell - n_req / 2) / (n_req / 2)], -1)
        zs = logit_slope_z(X, y)
        zs[y.min(1) == 1] = 0.0
        out.append(float(np.mean(zs < -1.6449)))
    return np.array(out)


# ============================================================================
# Figures
# ============================================================================
def fig_zones(prm, V, figdir, prevdir):
    g, L, c, C, s, rmin = (prm[k] for k in ("g", "L", "c", "C", "s", "rmin"))
    ell = np.linspace(0, 25, 2001)
    r = fatigue(ell, C, s, rmin)
    t1, t2 = thresholds(r, g, L, c)
    b = g / (g + L)
    has = t1 < t2
    lo, hi = np.where(has, t1, b), np.where(has, t2, b)
    ld = l_dagger(prm)
    y0, y1 = 0.01, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25))
    ax = axes[0]
    ax.fill_between(ell, y0, lo, color="#eef3f0", lw=0)
    ax.fill_between(ell, lo, hi, color=BLUE, alpha=0.16, lw=0)
    ax.fill_between(ell, hi, y1, color=ORANGE, alpha=0.13, lw=0)
    ax.plot(ell[has], t1[has], color=INK, lw=1.2)
    ax.plot(ell[has], t2[has], color=INK, lw=1.2, ls="--")
    ax.plot(ell[~has], np.full((~has).sum(), b), color=INK, lw=1.2, ls=":")
    ax.axvline(C, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.axvline(ld, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(C + 0.25, 0.0125, r"$C$", color=INK2, fontsize=7.5)
    ax.text(ld + 0.25, 0.0125, r"$\ell^{\!\dagger}$", color=INK2, fontsize=7.5)
    ax.text(1.0, 0.13, "ask", fontsize=8, color=BLUE_RAMP[3])
    ax.text(17.5, 0.017, "execute", fontsize=8, color=INK2)
    ax.text(17.5, 0.35, "refuse", fontsize=8, color=ORANGE_RAMP[2])
    ax.text(8.6, 0.022, r"$\tau_1(\ell)$", fontsize=7.5)
    ax.text(9.0, 0.20, r"$\tau_2(\ell)$", fontsize=7.5)
    ax.text(15.5, 0.068, r"$g/(g{+}L)$", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(y0, y1)
    ax.set_xlim(0, 25)
    ax.set_xlabel(r"user load $\ell$ (confirmations so far)")
    ax.set_ylabel(r"posterior risk $p$")
    title(ax, "a", "Myopic three-zone rule (Prop. 1)")

    ax = axes[1]
    ls_int = np.arange(0, 17)
    rr = fatigue(ls_int, C, s, rmin)
    m1, m2 = thresholds(rr, g, L, c)
    hm = m1 < m2
    ax.plot(ls_int[hm], m1[hm], color=INK2, lw=1.0, ls="--", label="myopic (Prop. 1)")
    ax.plot(ls_int[hm], m2[hm], color=INK2, lw=1.0, ls="--")
    looks = {10: (BLUE, "optimal, 30 steps left"), 35: (ORANGE, "optimal, 5 steps left")}
    for t, (col, lab) in looks.items():
        mu_att = V[t + 1, ls_int] - V[t + 1, ls_int + 1]
        d1 = (c + mu_att) / (rr * L)
        d2 = (g - c - mu_att) / (g + (1 - rr) * L)
        hd = d1 < d2
        ax.fill_between(ls_int, np.where(hd, d1, np.nan), np.where(hd, d2, np.nan),
                        color=col, alpha=0.18, lw=0)
        ax.plot(ls_int[hd], d1[hd], color=col, lw=1.3, marker="o", ms=2.6, label=lab)
        ax.plot(ls_int[hd], d2[hd], color=col, lw=1.3, marker="o", ms=2.6)
    ax.set_yscale("log")
    ax.set_ylim(y0, y1)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel(r"user load $\ell$")
    ax.set_ylabel(r"ask band $[\tau_1^{\mathrm{dyn}},\, \tau_2^{\mathrm{dyn}})$")
    ax.legend(loc="lower right", fontsize=6.4, handlelength=2.0)
    title(ax, "b", "Price of attention narrows the band (Prop. 2)")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_zones", figdir, prevdir)


def fig_optimal_load(figdir, prevdir):
    N, C, s, rmin = 100, 10.0, 0.02, 0.0
    ell = np.linspace(0.01, 60, 4000)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    ax = axes[0]
    alphas = [0.15, 0.3, 0.5, 0.8]
    star = {}
    for a, col in zip(alphas, BLUE_RAMP):
        E = (ell / N) ** a * fatigue(ell, C, s, rmin)
        ls_ = max(C, a * (1 + s * C) / (s * (1 + a)))
        Es = (ls_ / N) ** a * fatigue(ls_, C, s, rmin)
        star[a] = (ls_, float(fatigue(ls_, C, s, rmin)), float(ell[np.argmax(E)]))
        ax.plot(ell, E, color=col, lw=1.4, label=rf"$\alpha={a}$")
        ax.plot([ls_], [Es], marker="o", color=col, ms=4.2, mec="white", mew=0.7, zorder=5)
    ax.axvline(C, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(C + 1.2, 0.03, r"$C$", color=INK2, fontsize=7.5)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel(r"confirmations per session $\ell$")
    ax.set_ylabel(r"attacks caught $\mathcal{E}(\ell)=\rho(\ell/N)\,r(\ell)$")
    ax.legend(loc="upper right", ncol=2, fontsize=6.6, columnspacing=1.0, handlelength=1.6)
    title(ax, "a", r"Interior optimum $\ell^\star$ (dots)")

    ax = axes[1]
    al = np.linspace(0.005, 1.0, 600)
    for sv, col in zip([0.01, 0.02, 0.05], ORANGE_RAMP):
        lstar = np.maximum(C, al * (1 + sv * C) / (sv * (1 + al)))
        ax.plot(al, lstar, color=col, lw=1.4, label=rf"$s={sv}$")
        ax.plot([sv * C], [C], marker="o", color=col, ms=3.6, mec="white", mew=0.6, zorder=5)
    ax.axhline(C, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(0.62, C - 4.2, r"$\ell^\star=C$ below the kink", color=INK2, fontsize=7.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 60)
    ax.set_xlabel(r"risk-ranking spread $\alpha$ (lower = sharper detector)")
    ax.set_ylabel(r"optimal load $\ell^\star$")
    ax.legend(loc="upper left", fontsize=6.6)
    title(ax, "b", r"Kink at $\alpha = sC$ (Prop. 3)")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_optimal_load", figdir, prevdir)
    return star


def fig_pilot(powers, n_list, figdir, prevdir):
    ell = np.linspace(0, 60, 601)
    h1 = np.maximum(0.2, 1 - 0.02 * np.maximum(0, ell - 10))
    h2 = 0.2 + 0.8 * np.exp(-0.04 * np.maximum(0, ell - 5))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    ax = axes[0]
    ax.plot(ell, np.ones_like(ell), color=GRAY, lw=1.1, ls=":", label="no fatigue")
    ax.plot(ell, h1, color=BLUE, lw=1.5, label=r"H1: linear with onset ($C{=}10$, $s{=}0.02$)")
    ax.plot(ell, h2, color=ORANGE, lw=1.5, ls="--", label=r"H2: exponential (onset 5, rate 0.04)")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"confirmation index $\ell$ within the session")
    ax.set_ylabel(r"P(reject malicious request) $r(\ell)$")
    ax.legend(loc="lower left", fontsize=6.3)
    title(ax, "a", "Hypothesized vigilance curves")

    ax = axes[1]
    labels = {0.005: "small ($s{=}0.005$)", 0.01: "medium ($s{=}0.01$)", 0.02: "large ($s{=}0.02$)"}
    for (sv, col) in zip([0.005, 0.01, 0.02], [BLUE_RAMP[1], BLUE_RAMP[2], BLUE_RAMP[3]]):
        ax.plot(n_list, 100 * powers[(sv, 0.1)], color=col, marker="o", ms=3, lw=1.4,
                label=labels[sv] + ", 10% malicious")
    ax.plot(n_list, 100 * powers[(0.005, 0.2)], color=BLUE_RAMP[1], marker="o", ms=3, lw=1.2,
            ls="--", mfc="white", label="small, 20% malicious")
    ax.axhline(80, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(n_list[-1], 82, "80%", color=INK2, fontsize=7, ha="right")
    ax.set_ylim(0, 102)
    ax.set_xlim(0, n_list[-1] + 2)
    ax.set_xlabel("participants")
    ax.set_ylabel("power to detect a decline (%)")
    ax.legend(loc="lower right", fontsize=6.0)
    title(ax, "b", "Simulated power of the pilot")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_pilot", figdir, prevdir)


def fig_pareto(sw_f, sw_nf, pts, Cs, cap, figdir, prevdir):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45))
    ax = axes[0]
    ax.plot(sw_nf["asks"], 100 * np.array(sw_nf["asr"]), color=INK2, lw=1.0, ls="--",
            label="threshold guard, no fatigue")
    ax.plot(sw_f["asks"], 100 * np.array(sw_f["asr"]), color=AQUA, lw=1.4,
            label="threshold guard, with fatigue")
    for key in ["auto", "always", "static", "myopic", "dp"]:
        st, r = LOOK[key], pts[key]
        ax.plot([r["asks"]], [100 * r["asr"]], marker=st["marker"], color=st["color"], ls="none",
                ms=5.5, mec="white", mew=0.6, zorder=6)
    ax.set_xlim(-1, 41)
    ax.set_ylim(0, 82)
    ax.set_xlabel("confirmations per session")
    ax.set_ylabel("attack success rate (%)")
    ax.legend(loc="upper center", fontsize=6.3)
    title(ax, "a", "More asking is not always safer")

    ax = axes[1]
    xs = np.arange(len(Cs))
    for key in ["auto", "always", "static", "myopic", "dp"]:
        st = LOOK[key]
        ax.plot(xs, cap[key], color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.6, lw=1.3)
    ax.plot(xs, cap["dp_mis"], color=BLUE, ls=":", marker="o", ms=3.6, mfc="white", lw=1.1,
            label=r"DP planned for $C{=}5$ (misspecified)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(c)) for c in Cs])
    ax.set_ylim(-5, 41)
    ax.set_xlabel(r"user capacity $C$ (confirmations before fatigue)")
    ax.set_ylabel("reward per session")
    ax.legend(loc="upper left", fontsize=6.3)
    title(ax, "b", "Value of modeling fatigue")
    handles = [plt.Line2D([], [], color=LOOK[k]["color"], ls=LOOK[k]["ls"], marker=LOOK[k]["marker"],
                          ms=4.5, lw=1.3, mec="white", mew=0.5, label=LOOK[k]["label"])
               for k in ["dp", "myopic", "static", "always", "auto"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, -0.07),
               columnspacing=1.4)
    fig.tight_layout(w_pad=1.6, rect=(0, 0.08, 1, 1))
    save(fig, "fig_pareto_sim", figdir, prevdir)


def fig_flooding(ks, fl, figdir, prevdir):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    keys = ["static", "myopic", "dp", "auto"]
    for key in keys:
        st = LOOK[key]
        axes[0].plot(ks, 100 * np.array(fl[key]["harm"]), color=st["color"], ls=st["ls"],
                     marker=st["marker"], ms=2.8, markevery=3, lw=1.3, label=st["label"])
        axes[1].plot(ks[1:], 100 * np.array(fl[key]["done"][1:]), color=st["color"], ls=st["ls"],
                     marker=st["marker"], ms=2.8, markevery=3, lw=1.3, label=st["label"])
    axes[0].set_xlabel(r"number of decoy requests $k$")
    axes[0].set_ylabel("P(injected action executed) (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_xlim(0, ks[-1])
    axes[0].legend(loc="center right", fontsize=6.2)
    title(axes[0], "a", "Flooding: harm vs. decoys")
    axes[1].set_xlabel(r"number of decoy requests $k$")
    axes[1].set_ylabel("benign decoys completed (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_xlim(0, ks[-1])
    title(axes[1], "b", "Cost: utility on the decoys")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_flooding_sim", figdir, prevdir)


def fig_training(runs, ref, figdir, prevdir):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.55), sharex=True)
    spec = [("U", r"Utility $U$ per session", 1.0), ("asr", "Attack success rate (%)", 100.0),
            ("asks", "Confirmations per session", 1.0), ("lam", r"Multiplier $\lambda$", 1.0)]
    for ax, (key, lab, sc), letter in zip(axes.ravel(), spec, "abcd"):
        for name in ["lfix", "lrand", "lnofat", "lvan"]:
            H = runs[name]
            st = LOOK[name]
            y = sc * np.array(H[key])
            if key != "lam":
                k = 3
                y = np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
            ax.plot(H["it"], y, color=st["color"], ls=st["ls"], lw=1.2, label=st["label"])
        ax.axhline(sc * ref[key], color=BLUE, lw=1.0, ls=(0, (4, 2)),
                   label="CMDP optimum (DP, $\\lambda^\\star$)")
        if key == "asr":
            ax.axhline(100 * EPS, color=INK2, lw=0.6, ls=":")
            ax.text(0.98, 100 * EPS + 1.0, r"budget $\varepsilon$", transform=ax.get_yaxis_transform(),
                    ha="right", fontsize=6.6, color=INK2)
            ax.set_ylim(0, 60)
        if key == "lam":
            ax.set_ylim(0, 80)
            lv = runs["lvan"]["lam"][-1]
            ax.annotate(f"vanilla: {lv:.0f}", xy=(0.99, 0.97), xycoords="axes fraction", ha="right",
                        va="top", fontsize=6.4, color=INK)
        title(ax, letter, lab)
    axes[1, 0].set_xlabel("training iteration (512 sessions each)")
    axes[1, 1].set_xlabel("training iteration (512 sessions each)")
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, fontsize=6.4, bbox_to_anchor=(0.5, -0.06),
               columnspacing=1.2)
    fig.tight_layout(h_pad=0.8, w_pad=1.4, rect=(0, 0.07, 1, 1))
    save(fig, "fig_training_sim", figdir, prevdir)


def fig_generalization(gen, figdir, prevdir):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.5), sharex="col")
    for col, (axis_key, xlab, rng_) in enumerate([("C", r"test capacity $C$ ($s=0.1$)", (2, 10)),
                                                    ("s", r"test fatigue slope $s$ ($C=5$)", (0.05, 0.2))]):
        G = gen[axis_key]
        x = np.array(G["x"])
        for row, (metric, ylab) in enumerate([("asr", "attack success rate (%)"),
                                              ("ratio", "utility / oracle utility")]):
            ax = axes[row, col]
            ax.axvspan(rng_[0], rng_[1], color=GREEN, alpha=0.07, lw=0)
            for name in ["lfix", "lrand", "lnofat"]:
                st = LOOK[name]
                y = np.array(G[name][metric]) * (100 if metric == "asr" else 1)
                ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.2, lw=1.3,
                        label=st["label"])
            if metric == "asr":
                ax.axhline(100 * EPS, color=INK2, lw=0.6, ls=":")
                ax.set_ylim(0, 70)
            else:
                ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
                ax.set_ylim(0.6, 1.4)
            if row == 1:
                ax.set_xlabel(xlab)
            if col == 0:
                ax.set_ylabel(ylab)
            letter = "abcd"[row * 2 + col]
            title(ax, letter, ("Attack success" if metric == "asr" else "Relative utility") +
                  (" vs. capacity" if axis_key == "C" else " vs. slope"))
    axes[0, 0].text(6, 64, "training range", color=GREEN, fontsize=6.6, ha="center")
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(h_pad=0.8, w_pad=1.4, rect=(0, 0.05, 1, 1))
    save(fig, "fig_generalization_sim", figdir, prevdir)


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../paper")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(args.out, "figures")
    tabdir = os.path.join(args.out, "tables")
    prevdir = os.path.join(here, "preview")
    for d in (figdir, tabdir, prevdir):
        os.makedirs(d, exist_ok=True)
    set_style()
    n_mc = 4000 if args.fast else 20000
    iters = 300 if args.fast else 2000
    T0 = time.time()
    prm = dict(BASE)
    rb = rfun_of(prm["C"], prm["s"])
    M = {}          # LaTeX macros
    OUT = {"params": prm, "eps": EPS, "n_sessions": n_mc}

    # ---------------- DP for the base profile + theory sanity checks
    V = solve_dp(prm, rb)
    assert np.all(V[:, :-1] - V[:, 1:] >= -1e-9), "Lemma 1 (monotone value) violated"
    mu_att = V[1:, :-1] - V[1:, 1:]
    print(f"[ok] Lemma 1 holds numerically; max price of attention = {mu_att.max():.3f}")
    ld = l_dagger(prm)
    M["SimLdagger"] = f"{ld:.1f}"
    M["SimMuTenZero"] = f"{(V[11, 0] - V[11, 1]):.3f}"
    M["SimMuTenEight"] = f"{(V[11, 8] - V[11, 9]):.3f}"
    M["SimMuThirtyfiveEight"] = f"{(V[36, 8] - V[36, 9]):.3f}"
    fig_zones(prm, V, figdir, prevdir)
    star = fig_optimal_load(figdir, prevdir)
    for a, (ls_, rs, num) in star.items():
        print(f"[ok] Prop 3, alpha={a}: closed form l*={ls_:.2f} (numeric argmax {num:.2f}), r(l*)={rs:.3f}")
    OUT["prop3"] = {str(a): dict(lstar=v[0], r=v[1], numeric=v[2]) for a, v in star.items()}

    # ---------------- policies on the base profile (common random numbers)
    data = draw(prm, n_mc, seed=0)
    pols = {"auto": pol_auto(prm), "always": pol_always(), "static": pol_static(prm),
            "myopic": pol_myopic(prm, rb), "dp": pol_dp(V, prm, rb)}
    pts = {k: run(f, prm, rb, data) for k, f in pols.items()}
    for k, r in pts.items():
        print(f"  {k:7s} ASR={100*r['asr']:5.1f}%  benign={100*r['benign']:5.1f}%  "
              f"asks={r['asks']:5.2f}  reward={r['reward']:6.2f}")
    oracle = cmdp_oracle(prm, rb, data)
    print(f"  CMDP oracle: lambda*={oracle['lam']:.2f} ASR={100*oracle['asr']:.2f}% "
          f"U={oracle['util']:.2f} asks={oracle['asks']:.2f}")
    OUT["base"] = pts
    OUT["oracle"] = oracle

    # threshold-guard sweep, with and without fatigue
    taus = np.geomspace(0.003, 0.79, 34)
    sw_f = {"tau": [], "asks": [], "asr": []}
    sw_nf = {"tau": [], "asks": [], "asr": []}
    for tau in taus:
        a = run(pol_guard(tau), prm, rb, data)
        b = run(pol_guard(tau), prm, r_none, data)
        for D, rr in ((sw_f, a), (sw_nf, b)):
            D["tau"].append(float(tau)); D["asks"].append(rr["asks"]); D["asr"].append(rr["asr"])
    i_min = int(np.argmin(sw_f["asr"]))
    M["SimSweepMinASR"] = f"{100*sw_f['asr'][i_min]:.1f}"
    M["SimSweepMinAsks"] = f"{sw_f['asks'][i_min]:.1f}"
    M["SimSweepMaxAsksASR"] = f"{100*sw_f['asr'][0]:.1f}"
    M["SimSweepMaxAsks"] = f"{sw_f['asks'][0]:.1f}"
    OUT["sweep"] = dict(fatigue=sw_f, nofatigue=sw_nf)

    # capacity sweep
    Cs = [0, 2, 5, 10, 20, 40]
    cap = {k: [] for k in ["auto", "always", "static", "myopic", "dp", "dp_mis"]}
    for Cc in Cs:
        rc = rfun_of(Cc, prm["s"])
        Vc = solve_dp(dict(prm, C=Cc), rc)
        P = {"auto": pol_auto(prm), "always": pol_always(), "static": pol_static(prm),
             "myopic": pol_myopic(prm, rc), "dp": pol_dp(Vc, prm, rc),
             "dp_mis": pol_dp(V, prm, rb)}   # plans with C = 5 but faces C = Cc
        for k, f in P.items():
            cap[k].append(run(f, prm, rc, data)["reward"])
    OUT["capacity"] = dict(C=Cs, **cap)
    fig_pareto(sw_f, sw_nf, pts, Cs, cap, figdir, prevdir)
    print(f"[t={time.time()-T0:.0f}s] pareto + capacity done")

    # ---------------- flooding attack
    Vf = solve_dp(prm, rb, N=60)
    fpols = {"static": pol_static(prm), "myopic": pol_myopic(prm, rb),
             "dp": pol_dp(Vf, prm, rb), "auto": pol_auto(prm)}
    ks = np.arange(0, 51, 2)
    fl = {k: {"harm": [], "done": [], "load": []} for k in fpols}
    for k_, f in fpols.items():
        for kk in ks:
            h, d, lm = run_flood(f, rb, int(kk), n_mc, seed=1000 + int(kk))
            fl[k_]["harm"].append(h); fl[k_]["done"].append(d); fl[k_]["load"].append(lm)
    fig_flooding(ks, fl, figdir, prevdir)
    OUT["flooding"] = dict(k=ks.tolist(), **fl)
    j20 = int(np.where(ks == 20)[0][0])
    M["SimFloodStaticTwenty"] = f"{100*fl['static']['harm'][j20]:.0f}"
    M["SimFloodStaticMax"] = f"{100*max(fl['static']['harm']):.0f}"
    M["SimFloodDPMax"] = f"{100*max(fl['dp']['harm']):.0f}"
    M["SimFloodMyopicMax"] = f"{100*max(fl['myopic']['harm']):.0f}"
    M["SimFloodDPDoneFifty"] = f"{100*fl['dp']['done'][-1]:.0f}"
    print(f"[t={time.time()-T0:.0f}s] flooding done")

    # ---------------- Algorithm 1 in the toy model
    eval_data = draw(prm, 4000 if args.fast else 8000, seed=99)
    runs, thetas = {}, {}
    for name, sampler, gstd in [("lfix", fixed_sampler(prm["C"], prm["s"]), False),
                                ("lrand", rand_sampler(), False),
                                ("lnofat", fixed_sampler(1e9, 0.0), False),
                                ("lvan", fixed_sampler(prm["C"], prm["s"]), True)]:
        th, H, lam = train_alg1(prm, sampler, iters, eval_data, rb, seed=1, eval_every=25,
                                group_std=gstd)
        runs[name], thetas[name] = H, th
        print(f"[t={time.time()-T0:.0f}s] trained {name}: U={H['U'][-1]:.2f} "
              f"ASR={100*H['asr'][-1]:.1f}% asks={H['asks'][-1]:.1f} lambda={lam:.1f}")
    ref = dict(U=oracle["util"], asr=oracle["asr"], asks=oracle["asks"], lam=oracle["lam"])
    fig_training(runs, ref, figdir, prevdir)
    OUT["training"] = runs
    learned = {k: run(pol_learned(thetas[k], prm["N"], seed=5), prm, rb, data) for k in thetas}
    OUT["learned_base"] = learned

    # ---------------- generalization to unseen fatigue profiles
    gen = {}
    for axis_key, grid in [("C", [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]),
                           ("s", [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])]:
        G = {"x": grid, "oracle_U": []}
        gen_names = ["lfix", "lrand", "lnofat"]
        for name in gen_names:
            G[name] = {"asr": [], "U": [], "ratio": []}
        gdata = draw(prm, 4000 if args.fast else 10000, seed=7)
        for v in grid:
            Ct, st_ = (v, 0.1) if axis_key == "C" else (5.0, v)
            rt = rfun_of(Ct, st_)
            orc = cmdp_oracle(dict(prm, C=Ct, s=st_), rt, gdata, iters=14)
            G["oracle_U"].append(orc["util"])
            for name in gen_names:
                rr = run(pol_learned(thetas[name], prm["N"], seed=3), prm, rt, gdata)
                G[name]["asr"].append(rr["asr"]); G[name]["U"].append(rr["util"])
                G[name]["ratio"].append(rr["util"] / orc["util"])
        gen[axis_key] = G
    fig_generalization(gen, figdir, prevdir)
    OUT["generalization"] = gen
    print(f"[t={time.time()-T0:.0f}s] generalization done")

    # ---------------- pilot power analysis
    n_list = [5, 10, 15, 20, 25, 30, 40, 50]
    powers = {}
    for sv in [0.005, 0.01, 0.02]:
        h1 = lambda l, sv=sv: np.maximum(0.2, 1 - sv * np.maximum(0, l - 10))
        powers[(sv, 0.1)] = pilot_power(h1, n_list, 0.10, R=400 if args.fast else 1000)
    h1s = lambda l: np.maximum(0.2, 1 - 0.005 * np.maximum(0, l - 10))
    powers[(0.005, 0.2)] = pilot_power(h1s, n_list, 0.20, R=400 if args.fast else 1000)
    fig_pilot(powers, n_list, figdir, prevdir)
    OUT["pilot_power"] = {f"s={k[0]},m={k[1]}": v.tolist() for k, v in powers.items()}

    def n80(arr):
        idx = np.where(arr >= 0.8)[0]
        return str(n_list[idx[0]]) if len(idx) else f"$>${n_list[-1]}"
    M["SimPowerSmall"] = n80(powers[(0.005, 0.1)])
    M["SimPowerSmallTwenty"] = n80(powers[(0.005, 0.2)])
    M["SimPowerMedium"] = n80(powers[(0.01, 0.1)])
    M["SimPowerLarge"] = n80(powers[(0.02, 0.1)])
    print(f"[t={time.time()-T0:.0f}s] pilot power done: {OUT['pilot_power']}")

    # ---------------- table + macros
    rows = [("Autonomous (never ask)", pts["auto"]),
            ("Always confirm", pts["always"]),
            ("Static guard (fatigue-unaware)", pts["static"]),
            ("Myopic fatigue-aware (Prop.~1)", pts["myopic"]),
            ("Fatigue-aware optimal (DP, $\\lambda{=}L$)", pts["dp"]),
            ("CMDP optimum (DP, $\\lambda^\\star$)", oracle),
            ("Alg.~1, fatigue-free simulator", learned["lnofat"]),
            ("Alg.~1, fixed profile", learned["lfix"]),
            ("Alg.~1, randomized profiles", learned["lrand"]),
            ("Alg.~1, per-group std (vanilla GRPO)", learned["lvan"])]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Policy & ASR (\%) $\downarrow$ & Benign done (\%) $\uparrow$ & Asks/session & Reward $\uparrow$ \\",
             r"\midrule"]
    for i, (name, r) in enumerate(rows):
        if i in (5, 6):
            lines.append(r"\midrule")
        lines.append(f"{name} & {100*r['asr']:.1f} & {100*r['benign']:.1f} & {r['asks']:.1f} & "
                     f"{r['reward']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(tabdir, "sim_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    for k, r in pts.items():
        K = {"auto": "Auto", "always": "Always", "static": "Static", "myopic": "Myopic", "dp": "DP"}[k]
        M[f"Sim{K}ASR"] = f"{100*r['asr']:.1f}"
        M[f"Sim{K}Benign"] = f"{100*r['benign']:.1f}"
        M[f"Sim{K}Asks"] = f"{r['asks']:.1f}"
        M[f"Sim{K}Reward"] = f"{r['reward']:.1f}"
    M["SimOracleLambda"] = f"{oracle['lam']:.1f}"
    M["SimOracleAsks"] = f"{oracle['asks']:.1f}"
    M["SimOracleUtil"] = f"{oracle['util']:.1f}"
    M["SimLVanLambda"] = f"{runs['lvan']['lam'][-1]:.0f}"
    M["SimLFixLambda"] = f"{runs['lfix']['lam'][-1]:.1f}"
    for k, K in [("lfix", "Fix"), ("lrand", "Rand"), ("lnofat", "Nofat"), ("lvan", "Van")]:
        M[f"SimL{K}ASR"] = f"{100*learned[k]['asr']:.1f}"
        M[f"SimL{K}Asks"] = f"{learned[k]['asks']:.1f}"
        M[f"SimL{K}Util"] = f"{learned[k]['util']:.1f}"
        M[f"SimL{K}Ratio"] = f"{100*learned[k]['util']/oracle['util']:.0f}"
    M["SimNsessions"] = f"{n_mc:,}".replace(",", "{,}")
    M["SimIters"] = str(iters)
    gC = gen["C"]
    M["SimGenNofatWorstASR"] = f"{100*max(gC['lnofat']['asr']):.0f}"
    M["SimGenFixWorstASR"] = f"{100*max(gC['lfix']['asr']):.0f}"
    M["SimGenRandWorstASR"] = f"{100*max(gC['lrand']['asr']):.0f}"
    with open(os.path.join(args.out, "sim_numbers.tex"), "w") as fh:
        fh.write("% Auto-generated by sim/simulate.py -- do not edit by hand.\n")
        for k, v in M.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(figdir, "params.json"), "w") as fh:
        json.dump(dict(params=prm, eps=EPS, n_sessions=n_mc, train_iters=iters), fh, indent=2)
    with open(os.path.join(here, "results.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[done] {time.time()-T0:.0f}s total")


if __name__ == "__main__":
    main()
