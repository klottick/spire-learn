"""Dependency-free TorchRL probe check vocabulary."""

from .vector import NATIVE_CHECKS


TORCHRL_CHECKS = NATIVE_CHECKS | frozenset(
    {"observation_mapping", "mask_sampling", "mask_log_prob"}
)
