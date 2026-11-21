"""Trajectory decision layer (feature spec F4-F8): per-call flags and session state.

Never reads response content: only ``ResponseMeta`` fields reach this package through the
envelope and the detector scores.
"""
