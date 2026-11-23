import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, read_csv
from mylib import LinearFit
plt.style.use("ggplot")

def get_p0(t, p, n_ref=100, k=5, n_conferma=3, margine=2):
    """Restituisce (t_pre, p_pre, i_stop): i punti prima dell'aumento repentino.

    n_ref      : punti iniziali usati per stimare baseline e rumore
    k          : soglia in unità di sigma sopra la baseline
    n_conferma : punti consecutivi sopra soglia per validare la salita
    margine    : punti scartati prima della soglia (la salita parte lenta)
    """
    t, p = np.asarray(t, float), np.asarray(p, float)
    base = np.median(p[:n_ref])
    sigma = 1.4826 * np.median(np.abs(p[:n_ref] - base))   # sigma robusta (MAD)
    sigma = max(sigma, 0.005)                               # almeno la risoluzione del sensore
    sopra = p > base + k * sigma
    # primo indice da cui i successivi n_conferma punti sono tutti sopra soglia
    conf = np.convolve(sopra, np.ones(n_conferma, int), mode="valid") == n_conferma
    if not conf.any():
        raise ValueError("Nessuna salita trovata")
    i_stop = max(int(np.argmax(conf)) - margine, 1)
    return t[:i_stop], p[:i_stop], i_stop

def get_max(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    p_max = p.max()
    i_max = np.where(p == p_max)
    t_max = t[i_max][0]
    return t_max, p_max


def get_plateau(t, p, n_ref=100, k=5, n_conferma=3, margine=2):
    t, p = np.asarray(t, float), np.asarray(p, float)
    i_start = len(p) - 1000

    return t[i_start:], p[i_start:], i_start

N_misure = 15

P_0 = []
P_s = []
P_t = []

df = DataFrame({
    "DP_s": [],
    "dDP_s": [],
    "DP_t": [],
    "dDP_t": [],
})

var_P_strum = np.sqrt(0.005)/12
for i in range(N_misure):
    dati = read_csv(f"./misure_pressione/run_{i+1}.csv")

    P = dati["Pressione(kPa)"]
    t = dati["Tempo(s)"]

    # dati.plot("Tempo(s)", "Pressione(kPa)")

    t_iniziali, P_iniziali, i_0 = get_p0(t, P)
    # plt.plot(t_iniziali, P_iniziali)
    P_0 = P_iniziali.mean()
    dP_0 = np.sqrt((P_iniziali.std(ddof=1)**2)/(i_0 + 1) + var_P_strum)

    t_max, P_max = get_max(t, P)
    P_s = P_max
    dP_s = np.sqrt(var_P_strum)
    # plt.plot(t_max, P_max, 'o')

    t_plateau, P_plateau, i_0 = get_plateau(t, P, n_ref=30, k=2.5)
    P_t = P_plateau.mean()
    dP_t = np.sqrt((P_plateau.std(ddof=1)**2)*len(P_plateau) + var_P_strum)
    # plt.plot(t_plateau, P_plateau)
    
    # plt.show()

    df.loc[i] = [P_s - P_0, np.sqrt(dP_s**2 + dP_0**2),P_t - P_0, np.sqrt(dP_t**2 - dP_0**2)]


V = read_csv("./misure_pressione/distanziatori.csv")["dV_mL"]

df["DV"] = V
dV = np.sqrt(2)*1

print(df)

fit_s_old = LinearFit(df, "DV", "DP_s", "dDP_s", True)
# aggiorno le incertezze su P con quelle di V
df["dDP_s"] = np.sqrt(df["dDP_s"]**2 + (fit_s_old.get_fit_params()[0][0] * dV)**2)

fit_s = LinearFit(df, "DV", "DP_s", "dDP_s", True)
print(f"chi2 = {fit_s.get_chi2().chi2}")
m_s = fit_s.get_fit_params()[0][0]
d_m_s = np.sqrt(fit_s.get_fit_params()[1][0][0])
print(f"m_s = {m_s} +- {d_m_s}")
fig, ax = plt.subplots()
fit_s.graph(ax)
plt.show()


fit_t_old = LinearFit(df, "DV", "DP_t", "dDP_t", True)
# aggiorno le incertezze su P con quelle di V
df["dDP_t"] = np.sqrt(df["dDP_t"]**2 + (fit_t_old.get_fit_params()[0][0] * dV)**2)

fit_t = LinearFit(df, "DV", "DP_t", "dDP_t", True)
print(f"chi2 = {fit_t.get_chi2().chi2}")
m_t = fit_t.get_fit_params()[0][0]
d_m_t = np.sqrt(fit_t.get_fit_params()[1][0][0])
print(f"m_t = {m_t} +- {d_m_t}")
fig, ax = plt.subplots()
fit_t.graph(ax)
plt.show()


fit_gamma_old = LinearFit(df, "DP_t", "DP_s", "dDP_s", True)
# aggiorno le incertezze su P con quelle di V
df["dDP_s"] = np.sqrt(df["dDP_s"]**2 + (fit_gamma_old.get_fit_params()[0][0] * df["dDP_t"])**2)

fit_gamma = LinearFit(df, "DP_t", "DP_s", "dDP_s", True)
print(f"chi2 = {fit_gamma.get_chi2().chi2}")
gamma = fit_gamma.get_fit_params()[0][0]
d_gamma = np.sqrt(fit_gamma.get_fit_params()[1][0][0])
print(f"gamma = {gamma} +- {d_gamma}")
fig, ax = plt.subplots()
fit_t.graph(ax)
plt.show()
