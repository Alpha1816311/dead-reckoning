"""
GNSS + INS Fusion Module

Provides a lightweight GNSS/INS fusion algorithm for
intelligent vehicle dead reckoning.

Inputs:
    - GNSS position
    - INS dead-reckoned position
    - INS velocity
    - GNSS availability

Output:
    - corrected continuous position
"""

import numpy as np


def fuse_gnss_ins(
    gnss_position,
    ins_position,
    ins_velocity,
    gnss_available=None,
    dt=0.1,
    gnss_weight=0.85,
):
    """
    Fuse GNSS position with INS dead-reckoned position.

    Parameters
    ----------
    gnss_position : array-like, shape (N, 2)
        GNSS positions in local Cartesian coordinates [x, y].
        Use NaN for unavailable GNSS measurements.

    ins_position : array-like, shape (N, 2)
        INS dead-reckoned positions [x, y].

    ins_velocity : array-like, shape (N, 2)
        INS velocity [vx, vy] in m/s.

    gnss_available : array-like of bool, optional
        GNSS availability flag for each sample.

    dt : float or array-like
        Sampling interval in seconds.

    gnss_weight : float
        Weight assigned to GNSS during correction.
        Value between 0 and 1.

    Returns
    -------
    corrected_position : ndarray, shape (N, 2)
        Continuous fused position [x, y].
    """

    gnss_position = np.asarray(
        gnss_position,
        dtype=float
    )

    ins_position = np.asarray(
        ins_position,
        dtype=float
    )

    ins_velocity = np.asarray(
        ins_velocity,
        dtype=float
    )

    if gnss_position.ndim != 2 or gnss_position.shape[1] != 2:
        raise ValueError(
            "gnss_position must have shape (N, 2)"
        )

    if ins_position.shape != gnss_position.shape:
        raise ValueError(
            "ins_position must have the same shape as "
            "gnss_position"
        )

    if ins_velocity.shape != gnss_position.shape:
        raise ValueError(
            "ins_velocity must have the same shape as "
            "gnss_position"
        )

    n = len(ins_position)

    if n == 0:
        return np.empty((0, 2))

    # --------------------------------------------------------
    # GNSS availability
    # --------------------------------------------------------

    if gnss_available is None:

        gnss_available = (
            np.isfinite(gnss_position[:, 0]) &
            np.isfinite(gnss_position[:, 1])
        )

    else:

        gnss_available = np.asarray(
            gnss_available,
            dtype=bool
        )

    if gnss_available.ndim != 1 or len(gnss_available) != n:
        raise ValueError(
            "gnss_available must be a one-dimensional array "
            "containing N samples"
        )

    # --------------------------------------------------------
    # DT
    # --------------------------------------------------------

    if np.isscalar(dt):

        dt_array = np.full(n, float(dt))

    else:

        dt_array = np.asarray(
            dt,
            dtype=float
        )

        if dt_array.ndim != 1 or len(dt_array) != n:
            raise ValueError(
                "dt must be a scalar or a one-dimensional array "
                "containing N samples"
            )

    if not np.all(np.isfinite(dt_array)):
        raise ValueError("dt must contain only finite values")

    dt_array = np.clip(
        dt_array,
        1e-4,
        None
    )

    # --------------------------------------------------------
    # GNSS weight
    # --------------------------------------------------------

    gnss_weight = float(gnss_weight)

    if not 0.0 <= gnss_weight <= 1.0:
        raise ValueError(
            "gnss_weight must be between 0 and 1"
        )

    ins_weight = 1.0 - gnss_weight

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    corrected = np.zeros(
        (n, 2),
        dtype=float
    )

    # Initial position
    corrected[0] = ins_position[0]

    if gnss_available[0] and np.all(np.isfinite(gnss_position[0])):
        corrected[0] = (
            gnss_weight * gnss_position[0]
            + ins_weight * corrected[0]
        )

    # --------------------------------------------------------
    # Fusion loop
    # --------------------------------------------------------

    for i in range(1, n):

        # INS prediction from previous fused position
        predicted_position = (
            corrected[i - 1]
            + ins_velocity[i] * dt_array[i]
        )

        # ----------------------------------------------------
        # GNSS available
        # ----------------------------------------------------

        if (
            gnss_available[i]
            and np.all(np.isfinite(gnss_position[i]))
        ):

            corrected[i] = (
                gnss_weight * gnss_position[i]
                + ins_weight * predicted_position
            )

        # ----------------------------------------------------
        # GNSS unavailable
        # ----------------------------------------------------

        else:

            corrected[i] = predicted_position

    return corrected