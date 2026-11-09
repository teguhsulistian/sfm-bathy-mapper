"""
Seawater refractive index - Millard & Seaver (1990) algorithm.

Coefficients: Table 1 of Rezaei-Latifi (2020), J Opt 49(1):32-40.
Equation:     n = n1(T,l) + n2(T,l,S) + n3(P,T,l) + n4(S,P,T)

NOTE on Eq. 2: the paper prints the a1 term as a1*lambda, but the printed units
and numerical checks against known water indices show it must be a1*lambda**2.
That form is the default here. Pass paper_eq2=True to use the printed form,
which reproduces the paper's Table 3 exactly but gives n about 0.0011 too low
versus known pure-water/seawater indices.

Input units
-----------
P          : pressure in dbar  (1 dbar ~ 1 m depth; CTD "Pressure" in m ~ dbar)
S          : practical salinity in psu
T          : temperature in degrees Celsius
wavelength : in NANOMETRES by default (e.g. 470, 540, 667).
             Use wavelength_unit="um" if you have micrometres (0.47, 0.54, 0.667).

Valid range of the model: T 0-30 C, S 0-40 psu, P 0-11000 dbar (110 MPa),
wavelength 500-700 nm. Outside it, a warning is issued but a value is still returned.
"""
import warnings
import numpy as np

# --- Table 1 coefficients -------------------------------------------------
_A = [1.3280657, -0.0045536802, 0.0025471707, 7.501966e-6, 2.802632e-6,
      -5.2883907e-6, -3.0738272e-6, 3.0124687e-8, -2.0863178e-10,
      1.0508621e-5, 2.1282248e-7, -1.705881e-9]
_B = [1.9029121e-4, 2.4239607e-6, -7.396029e-7, 8.9981878e-9,
      1.2078804e-10, -3.589495e-7]
_C = [1.5868363e-6, -1.574074e-11, 1.0712063e-8, -9.4634486e-9,
      1.0100326e-10, 5.8085198e-15]
_D = [-1.1177517e-9, 5.7311268e-11, -1.5460458e-12]


def seawater_refractive_index(P, S, T, wavelength, wavelength_unit="nm",
                              paper_eq2=False):
    """
    Refractive index of seawater (Millard & Seaver, 1990).

    Parameters
    ----------
    P : float or array-like   Pressure [dbar]
    S : float or array-like   Salinity [psu]
    T : float or array-like   Temperature [deg C]
    wavelength : float or array-like   Wavelength [nm by default]
    wavelength_unit : {"nm", "um"}     Unit of `wavelength`
    paper_eq2 : bool   False (default): use a1*lambda**2 (physically consistent).
                       True: use a1*lambda exactly as printed in the paper;
                       this reproduces the paper's Table 3 (about 0.0011 lower).

    Returns
    -------
    n : float or ndarray   Refractive index (dimensionless)
    """
    unit = wavelength_unit.lower().replace("µ", "u")
    if unit == "nm":
        lam_um = np.asarray(wavelength, dtype=float) / 1000.0
    elif unit in ("um", "micron", "micrometer", "micrometre"):
        lam_um = np.asarray(wavelength, dtype=float)
    else:
        raise ValueError("wavelength_unit must be 'nm' or 'um'")

    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    l = lam_um

    # Catch the common unit mistake (e.g. 470 passed with unit="um")
    if np.any(l > 5) or np.any(l < 0.1):
        raise ValueError(
            "Wavelength looks wrong for the chosen unit. Valid model range is "
            "500-700 nm (0.50-0.70 um). Check wavelength_unit.")

    _check_range(P, S, T, l)

    a, b, c, d = _A, _B, _C, _D
    n1 = (a[0] + a[1]*(l if paper_eq2 else l**2) + a[2]/l**2 + a[3]/l**4 + a[4]/l**6
          + a[5]*T + a[6]*T**2 + a[7]*T**3 + a[8]*T**4
          + a[9]*T*l + a[10]*T**2*l + a[11]*l*T**3)
    n2 = (b[0]*S + b[1]*S/l**2 + b[2]*S*T + b[3]*S*T**2
          + b[4]*S*T**3 + b[5]*S*T*l)
    n3 = (c[0]*P + c[1]*P**2 + c[2]*P/l**2 + c[3]*P*T
          + c[4]*P*T**2 + c[5]*P**2*T**2)
    n4 = d[0]*P*S + d[1]*P*T*S + d[2]*P*T**3*S

    n = n1 + n2 + n3 + n4
    return float(n) if np.ndim(n) == 0 else n


def _check_range(P, S, T, l):
    msgs = []
    if np.any((T < 0) | (T > 30)):
        msgs.append("T outside 0-30 C")
    if np.any((S < 0) | (S > 40)):
        msgs.append("S outside 0-40 psu")
    if np.any((P < 0) | (P > 11000)):
        msgs.append("P outside 0-11000 dbar")
    if np.any((l < 0.50) | (l > 0.70)):
        msgs.append("wavelength outside 500-700 nm (extrapolation)")
    if msgs:
        warnings.warn("Outside Millard-Seaver validity range: " + "; ".join(msgs),
                      stacklevel=3)


if __name__ == "__main__":
    # Check against Table 3 (August, lambda = 0.6 um): expected ~1.33728
    print(seawater_refractive_index(0, 39.38, 32.66, 0.6, "um"))
    # Example CTD row, RGB band centres
    for name, wl in [("Blue", 470), ("Green", 540), ("Red", 667)]:
        print(name, wl, "nm:", seawater_refractive_index(0.074, 33.012, 29.108, wl))
