import scipy as sc
import scipy.integrate
import numpy as np


def velocity(t, x, l, q1, q2, gamma, avr):
    """
    Rate of change in C_i in terms of C_i and parameters.

    Parameters
    ----------
    t: float
        Time
    x: tuple(float)
        Conductivites
    q1: tuple(float)
        Input flow rates in stage 1
    q2: tuple(float)
        Input flow rates in stage 2
    gamma: float
        Gamma parameter, controls equilibrium conductivity at fixed discharge
    avr: "absolute" or "energetic"
        Stage averaging scheme.

    Returns
    -------
    tuple(float)
        rate of change in each C_i

    """
    (cab, cbc, cca) = x

    (lab, lbc, lca) = l

    (qa1, qb1, qc1) = q1
    (qa2, qb2, qc2) = q2

    if not np.isclose(np.sum(q1), 0.0):
        raise ValueError(f"Flows {q1} have to sum to zero in stage 1")
    if not np.isclose(np.sum(q2), 0.0):
        raise ValueError(f"Flows {q2} have to sum to zero in stage 2")
    if avr != "absolute" and avr != "energetic":
        raise NotImplementedError(f"Averaging scheme {avr} not implemented")

    sigma = cab * cbc * lca + cbc * cca * lab + cca * cab * lbc

    qab1 = (1.0 / sigma) * cab * (qb1 * cca * lbc - qa1 * cbc * lca)
    qbc1 = (1.0 / sigma) * cbc * (qc1 * cab * lca - qb1 * cca * lab)
    qca1 = (1.0 / sigma) * cca * (qa1 * cbc * lab - qc1 * cab * lbc)

    qab2 = (1.0 / sigma) * cab * (qb2 * cca * lbc - qa2 * cbc * lca)
    qbc2 = (1.0 / sigma) * cbc * (qc2 * cab * lca - qb2 * cca * lab)
    qca2 = (1.0 / sigma) * cca * (qa2 * cbc * lab - qc2 * cab * lbc)

    if avr == "absolute":
        qab = 0.5 * (abs(qab1) + abs(qab2))
        qbc = 0.5 * (abs(qbc1) + abs(qbc2))
        qca = 0.5 * (abs(qca1) + abs(qca2))
    if avr == "energetic":
        qab = (0.5 * (abs(qab1) ** 2 + abs(qab2) ** 2)) ** 0.5
        qbc = (0.5 * (abs(qbc1) ** 2 + abs(qbc2) ** 2)) ** 0.5
        qca = (0.5 * (abs(qca1) ** 2 + abs(qca2) ** 2)) ** 0.5

    return [
        -cab + qab ** (2 * gamma),
        -cbc + qbc ** (2 * gamma),
        -cca + qca ** (2 * gamma),
    ]


def trajectory(l, q1, q2, gamma, avr, tmax=100.0, c0=[1.1, 1.2, 1.3]):
    """
    Compute time dependent conductivites

    Parameters
    ----------
    q1: tuple(float)
        Input flow rates in stage 1
    q2: tuple(float)
        Input flow rates in stage 2
    gamma: float
        Gamma parameter, controls equilibrium conductivity at fixed discharge
    avr: "absolute" or "energetic"
        Stage averaging scheme.
    tmax: float, default = 100.
        Time to stop integration at
    c0: list, default = [1.,1.,1.]
        Initial values of conductivites

    Returns
    -------
    Bunch object, result of `solve_ivp`
    """
    traj = sc.integrate.solve_ivp(
        velocity,
        [0, tmax],
        c0,
        args=(l, q1, q2, gamma, avr),
        dense_output=True,
        #rtol=1e-5,
    )
    return traj


def final_value(l, q1, q2, gamma, avr, tmax=100.0, c0=[1.1, 1.2, 1.3]):
    """
    Compute final conductivites

    Parameters
    ----------
    q1: tuple(float)
        Input flow rates in stage 1
    q2: tuple(float)
        Input flow rates in stage 2
    gamma: float
        Gamma parameter, controls equilibrium conductivity at fixed discharge
    avr: "absolute" or "energetic"
        Stage averaging scheme.
    tmax: float, default = 100.
        Time to stop integration at
    c0: list, default = [1.,1.,1.]
        Initial values of conductivites

    Returns
    -------
    array
        Array of length 3 with final conductivities
    """

    traj = trajectory(l, q1, q2, gamma, avr, tmax, c0)
    z = traj.sol(tmax)
    return z


if __name__ == "__main__":
    traj = trajectory(
        l=(1.0, 1.0, 1.0),
        q1=(1.0, 1.0, -2.0),
        q2=(2.0, -1.0, -1.0),
        gamma=0.5,
        avr="energetic",
        tmax=100.0,
    )
    t = np.linspace(0, 100, 300)

    z = traj.sol(t)
    print(f"Final values are: {z[:,-1]}")

    import matplotlib.pyplot as plt

    plt.plot(t, z.T)
    plt.xlabel("time")
    plt.ylabel("conductivity")
    plt.ylim([0, 2])
    plt.legend(["cab", "cbc", "cca"])
    plt.show()
