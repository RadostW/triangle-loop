import final_value
import numpy as np
from tqdm import tqdm


def angle_from_vectors(u, v):
    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)  # cosine of the angle
    return np.arccos(np.clip(c, -1, 1))


if __name__ == "__main__":

    results = dict()

    for phi in tqdm(np.linspace(0, np.pi, 100)):

        qa = np.array([1, 0])
        qb = 1.001*np.array([np.cos(phi), np.sin(phi)])
        qc = -qa - qb

        alphaab = angle_from_vectors(qa, qb)
        alphabc = angle_from_vectors(qb, qc)
        alphaca = angle_from_vectors(qc, qa)

        alphamin = min((alphaab, alphabc, alphaca))

        fin = final_value.final_value(
            l=(1.0, 1.0, 1.0),
            q1=(qa[0], qb[0], qc[0]),
            q2=(qa[1], qb[1], qc[1]),
            gamma=0.60,
            avr="energetic",
            tmax=1000.0,
        )
        results[phi] = {"assymetry": min(fin) / max(fin), "alphamin": alphamin}

    import matplotlib.pyplot as plt

    lists = [(v["alphamin"], v["assymetry"]) for (k, v) in sorted(results.items())]
    x, y = zip(*lists)
    plt.plot(x, y, "o")
    plt.xlabel("smallest angle")
    plt.ylabel("min C / max C")
    plt.ylim([0, 1])
    plt.show()
