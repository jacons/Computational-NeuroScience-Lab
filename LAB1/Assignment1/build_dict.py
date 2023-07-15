import json

import numpy as np

check1 = lambda t: 0 if ((t <= 40) or (t >= 40 + 4)) and ((t <= 60) or (t >= 60 + 4)) and \
                        ((t <= 280) or (t >= 280 + 4)) and ((t <= 320) or (t >= 320 + 4)) else 0.65

check2 = lambda t: 0 if ((t <= (100 / 11)) or (t >= (100 / 11) + 2)) and (
        (t <= ((100 / 11) + 5)) or (t >= ((100 / 11) + 5) + 2)) and \
                        ((t <= 70) or (t >= 70 + 2)) and ((t <= 80) or (t >= 80 + 2)) else 9

check3 = lambda t: 1 if ((t >= 10) and (t <= 15)) or ((t >= 80) and (t <= 85)) else -6 if (t >= 70) and (t <= 75) else 0

izhikevich_parameters = dict(
    A=dict(
        title="Tonic spiking (A)",
        length=100,
        tau=0.25,
        params=(0.02, 0.2, -65, 6),  # a, b, c, d
        v0=-70,
        u0=-70 * 0.2,
        current_input=[0 if i <= 10 else 14 for i in np.arange(0, 100 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    B=dict(
        title="Phasic spiking (B)",
        length=200,
        tau=0.25,
        params=(0.02, 0.25, -65, 6),  # a, b, c, d
        v0=-64,
        u0=-64 * 0.25,
        current_input=[0 if i <= 20 else 0.5 for i in np.arange(0, 200 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    C=dict(
        title="Tonic bursting (C)",
        length=220,
        tau=0.25,
        params=(0.02, 0.2, -50, 2),  # a, b, c, d
        v0=-70,
        u0=-70 * 0.2,
        current_input=[0 if i <= 22 else 15 for i in np.arange(0, 220 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    D=dict(
        title="Phasic bursting (D)",
        length=200,
        tau=0.2,
        params=(0.02, 0.25, -55, 0.05),  # a, b, c, d
        v0=-64,
        u0=-64 * 0.25,
        current_input=[0 if i <= 20 else 0.6 for i in np.arange(0, 200 + 0.2, 0.2)],
        coefficients=(5, 140)
    ),
    E=dict(
        title="Mixed mode",
        length=160,
        tau=0.25,
        params=(0.02, 0.2, -55, 4),  # a, b, c, d
        v0=-70,
        u0=-70 * 0.2,
        current_input=[0 if i <= 16 else 10 for i in np.arange(0, 160 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    F=dict(
        title="Spike frequency adaptation (F)",
        length=85,
        tau=0.25,
        params=(0.01, 0.2, -65, 8),  # a, b, c, d
        v0=-70,
        u0=-70 * 0.2,
        current_input=[0 if i <= 8.5 else 30 for i in np.arange(0, 85 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    G=dict(
        title="Class 1 excitable (G)",
        length=300,
        tau=0.25,
        params=(0.02, -0.1, -55, 6),  # a, b, c, d
        v0=-60,
        u0=-70 * -0.1,
        current_input=[0 if i <= 30 else (0.075 * (i - 30)) for i in np.arange(0, 300 + 0.25, 0.25)],
        coefficients=(4.1, 108)
    ),
    H=dict(
        title="Class 2 excitable (H)",
        length=300,
        tau=0.25,
        params=(0.2, 0.26, -65, 0),  # a, b, c, d
        v0=-64,
        u0=-64 * 0.26,
        current_input=[-0.5 if i <= 30 else -0.5 + (0.015 * (i - 30)) for i in np.arange(0, 300 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    I=dict(
        title="Spike latency (I)",
        length=100,
        tau=0.2,
        params=(0.02, 0.2, -65, 6),  # a, b, c, d
        v0=-70,
        u0=-70 * 0.2,
        current_input=[0 if (i <= 10) or (i >= 13) else 7.04 for i in np.arange(0, 100 + 0.2, 0.2)],
        coefficients=(5, 140)
    ),
    J=dict(
        title="Subthreshold oscillator (J)",
        length=200,
        tau=0.25,
        params=(0.05, 0.26, -60, 0),  # a, b, c, d
        v0=-62,
        u0=-62 * 0.26,
        current_input=[0 if (i <= 20) or (i >= 25) else 2 for i in np.arange(0, 200 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    K=dict(
        title="Resonator (K)",
        length=400,
        tau=0.25,
        params=(0.1, 0.26, -60, -1),  # a, b, c, d
        v0=-62,
        u0=-62 * 0.26,
        current_input=[check1(i) for i in np.arange(0, 400 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    L=dict(
        title="Integrator (L)",
        length=100,
        tau=0.25,
        params=(0.02, -0.1, -55, 6),  # a, b, c, d
        v0=-60,
        u0=-60 * -0.1,
        current_input=[check2(i) for i in np.arange(0, 100 + 0.25, 0.25)],
        coefficients=(4.1, 108)
    ),
    M=dict(
        title="Rebound spike (M)",
        length=200,
        tau=0.2,
        params=(0.03, 0.25, -60, 4),  # a, b, c, d
        v0=-64,
        u0=-64 * 0.25,
        current_input=[0 if (i <= 20) or (i >= 25) else -15 for i in np.arange(0, 200 + 0.2, 0.2)],
        coefficients=(5, 140)
    ),
    N=dict(
        title="Rebound burst (N)",
        length=200,
        tau=0.2,
        params=(0.03, 0.25, -52, 0),  # a, b, c, d
        v0=-64,
        u0=-64 * 0.25,
        current_input=[0 if (i <= 20) or (i >= 25) else -15 for i in np.arange(0, 200 + 0.2, 0.2)],
        coefficients=(5, 140)
    ),
    O=dict(
        title="Threshold variability (O)",
        length=100,
        tau=0.25,
        params=(0.03, 0.25, -60, 4),  # a, b, c, d
        v0=-64,
        u0=-64 * 0.25,
        current_input=[check3(i) for i in np.arange(0, 100 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    P=dict(
        title="Bistability (P)",
        length=300,
        tau=0.25,
        params=(0.1, 0.26, -60, 0),  # a, b, c, d
        v0=-61,
        u0=-61 * 0.26,
        current_input=[0.24 if ((i <= 37.5) or (i >= 37.5 + 5)) and ((i <= 216) or (i >= 216 + 5)) else 1.24 for i in
                       np.arange(0, 300 + 0.25, 0.25)],
        coefficients=(5, 140)
    ),
    Q=dict(
        title="DAP (Q)",
        length=50,
        tau=0.1,
        params=(1, 0.2, -60, -21),  # a, b, c, d
        v0=-70,
        u0=-70 * 0.2,
        current_input=[0 if np.abs(i - 10) >= 1 else 20 for i in np.arange(0, 50 + 0.1, 0.1)],
        coefficients=(5, 140)
    ),
    R=dict(
        title="Accommodation (R)",
        length=400,
        tau=0.5,
        params=(0.02, 1, -55, 4),  # a, b, c, d
        v0=-65,
        u0=-16,
        current_input=[x + y for x, y in zip(
            [0 if (i >= 200) else i / 25 for i in np.arange(0, 400 + 0.5, 0.5)],
            [0 if (i <= 300) or (i >= 312.5) else (i - 300) / 12.5 * 4 for i in np.arange(0, 400 + 0.5, 0.5)])],
        coefficients=(5, 140)
    ),
    S=dict(
        title="Inhibition induced spiking (S)",
        length=350,
        tau=0.5,
        params=(-0.02, -1, -60, 8),  # a, b, c, d
        v0=-63.8,
        u0=-63.8 * -1,
        current_input=[75 if (i >= 50) and (i <= 250) else 80 for i in np.arange(0, 350 + 0.5, 0.5)],
        coefficients=(5, 140)
    ),
    T=dict(
        title="Inhibition induced bursting (T)",
        length=350,
        tau=0.5,
        params=(-0.026, -1, -45, -2),  # a, b, c, d
        v0=-63.8,
        u0=-63.8 * -1,
        current_input=[75 if (i >= 50) and (i <= 250) else 80 for i in np.arange(0, 350 + 0.5, 0.5)],
        coefficients=(5, 140)
    ),
)
with open("configurations.json", "w") as f:
    json.dump(izhikevich_parameters, f)
