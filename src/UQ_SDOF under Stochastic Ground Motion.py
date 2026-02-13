# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


# ===============================
# STEP A — Ground motion generator
# ===============================

def kanai_tajimi_psd(w, S0, wg, zeta):
    num = wg**4 + (2*zeta*wg*w)**2
    den = (wg**2 - w**2)**2 + (2*zeta*wg*w)**2
    return S0 * num / den


def envelope(t):
    e = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] < 2:
            e[i] = t[i]/2
        elif t[i] < 12:
            e[i] = 1
        elif t[i] < 20:
            e[i] = 1 - (t[i]-12)/8
    return e


def generate_ag(t, S0, wg, zeta, wmax, Nw, rng):
    dw = wmax/Nw
    k = np.arange(1, Nw+1)
    w = k*dw

    S = kanai_tajimi_psd(w, S0, wg, zeta)
    A = np.sqrt(2*S*dw)

    phi = rng.uniform(0, 2*np.pi, Nw)

    ag = np.sum(A[:,None]*np.cos(w[:,None]*t[None,:] + phi[:,None]), axis=0)

    return envelope(t)*ag


# ===============================
# STEP B — SDOF response
# ===============================

def newmark(ag, t, Tn, zeta, m=1.0):
    """
    Stable Newmark-beta (average acceleration):
    u_ddot + 2*zeta*w*u_dot + w^2*u = -ag(t)
    Returns u, u_dot, u_ddot (relative)
    """
    dt = t[1] - t[0]
    w = 2.0*np.pi / Tn
    k = m*w**2
    c = 2.0*zeta*m*w

    beta = 1/4
    gamma = 1/2

    n = len(t)
    u = np.zeros(n)
    ud = np.zeros(n)
    udd = np.zeros(n)

    # initial acceleration
    udd[0] = (-c*ud[0] - k*u[0] - m*ag[0]) / m

    a0 = 1.0/(beta*dt**2)
    a1 = gamma/(beta*dt)
    a2 = 1.0/(beta*dt)
    a3 = 1.0/(2*beta) - 1.0
    a4 = gamma/beta - 1.0
    a5 = dt*(gamma/(2*beta) - 1.0)

    k_eff = k + a0*m + a1*c

    for i in range(n-1):
        p_next = -m*ag[i+1]

        p_eff = (p_next
                 + m*(a0*u[i] + a2*ud[i] + a3*udd[i])
                 + c*(a1*u[i] + a4*ud[i] + a5*udd[i]))

        u[i+1] = p_eff / k_eff
        ud[i+1] = a1*(u[i+1]-u[i]) - a4*ud[i] - a5*udd[i]
        udd[i+1] = a0*(u[i+1]-u[i]) - a2*ud[i] - a3*udd[i]

    return u, ud, udd
def spectrum(ag, t, Tlist, zeta):
    Sa = []
    for T in Tlist:
        u, ud, udd = newmark(ag, t, T, zeta)
        Sa.append(np.max(np.abs(udd + ag)))   # absolute acceleration
    return np.array(Sa)



# ===============================
# STEP C — Monte Carlo
# ===============================

def monte_carlo(N=200):

    rng = default_rng(1)

    dt = 0.01
    t = np.arange(0,20,dt)

    Tlist = np.arange(0.1,4,0.1)

    Sa_all = []

    for i in range(N):

        ag = generate_ag(t,1,10,0.6,80,1500,rng)

        Sa = spectrum(ag,t,Tlist,0.05)
        Sa_all.append(Sa)

    Sa_all = np.array(Sa_all)

    mean = Sa_all.mean(axis=0)
    std = Sa_all.std(axis=0)

    return t,Tlist,ag,Sa_all,mean,std


# ===============================
# MAIN PROGRAM
# ===============================

t,Tlist,ag,Sa_all,mean,std = monte_carlo(300)

print("Monte Carlo finished")
print("Simulations:", Sa_all.shape[0])


# -------- Step A Plot --------
plt.figure()
plt.plot(t,ag)
plt.title("Ground motion ag(t)")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.grid()
plt.show()


# -------- Step B Plot --------
u,v,a = newmark(ag,t,1.0,0.05)

plt.figure()
plt.plot(t,u)
plt.title("SDOF displacement u(t)")
plt.grid()
plt.show()


# -------- Step C Plot --------
plt.figure()
plt.plot(Tlist,mean,label="Mean spectrum")
plt.plot(Tlist,mean+std,"--",label="Mean+Std")
plt.plot(Tlist,mean-std,"--",label="Mean-Std")
plt.legend()
plt.title("Monte Carlo Mean Spectrum")
plt.grid()
plt.show()


# ============================================================
# Monte Carlo Convergence Check (Mean, Variance, Std, 95% CI)
# ============================================================

track_index = 10   # choose which period to monitor

x = Sa_all[:, track_index]
N = len(x)

running_mean = []
running_var  = []
running_std  = []

running_ci_low  = []
running_ci_high = []

z = 1.96  # 95% confidence interval factor

for i in range(2, N+1):
    sample = x[:i]

    mean_i = np.mean(sample)
    var_i  = np.var(sample, ddof=1)
    std_i  = np.sqrt(var_i)

    # Standard error of mean
    se_i = std_i / np.sqrt(i)

    # Confidence interval
    ci_low  = mean_i - z * se_i
    ci_high = mean_i + z * se_i

    running_mean.append(mean_i)
    running_var.append(var_i)
    running_std.append(std_i)
    running_ci_low.append(ci_low)
    running_ci_high.append(ci_high)

# ---------------- Plot Mean Convergence ----------------
plt.figure()
plt.plot(running_mean)
plt.title("Convergence of Mean")
plt.xlabel("Number of simulations")
plt.ylabel("Running Mean")
plt.grid(True)

# ---------------- Plot Variance Convergence ----------------
plt.figure()
plt.plot(running_var)
plt.title("Convergence of Variance")
plt.xlabel("Number of simulations")
plt.ylabel("Running Variance")
plt.grid(True)

# ---------------- Plot Std Convergence ----------------
plt.figure()
plt.plot(running_std)
plt.title("Convergence of Standard Deviation")
plt.xlabel("Number of simulations")
plt.ylabel("Running Std")
plt.grid(True)

# ---------------- Plot 95% Confidence Interval ----------------
plt.figure()
plt.plot(running_mean, label="Running Mean")
plt.plot(running_ci_low, "--", label="95% CI Lower")
plt.plot(running_ci_high, "--", label="95% CI Upper")
plt.title("Convergence of 95% Confidence Interval")
plt.xlabel("Number of simulations")
plt.ylabel("Mean with 95% CI")
plt.legend()
plt.grid(True)

plt.show()

