import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    t_max = t[i_max]
    return t_max, p_max

dati = pd.read_csv("./misure_pressione/run_01.csv")

d_P_strum = 0.005/np.sqrt(12)

P = dati["Pressione(kPa)"]
t = dati["Tempo(s)"]

dati.plot("Tempo(s)", "Pressione(kPa)")

t_0, P_0, i_0 = get_p0(t, P)
plt.plot(t_0, P_0)

t_max, P_max = get_max(t, P)
plt.plot(t_max, P_max, 'o')


plt.show()
