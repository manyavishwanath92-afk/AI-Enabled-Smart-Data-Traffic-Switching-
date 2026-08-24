"""Transmitted power -> received power (Guide Section 8.2).

P_rx(dBm) = P_tx(dBm) - PL(dB)

This is the ONE place in the radio chain where dBm/dB values are combined
directly (a subtraction in the log domain), and that is correct here:
applying a loss expressed in dB to a power expressed in dBm is exactly
what log-domain subtraction means. From this point onward, anything that
feeds a SINR ratio must first be converted to linear units — see
`network.sinr.dbm_to_linear_mw`. Never subtract or add a dBm value
against a linear-unit value.
"""

from __future__ import annotations


def received_power_dbm(tx_power_dbm: float, path_loss_db: float) -> float:
    """P_rx(dBm) = P_tx(dBm) - PL(dB)."""
    return tx_power_dbm - path_loss_db
