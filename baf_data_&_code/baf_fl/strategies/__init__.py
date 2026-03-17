"""
Estrategias de agregacao federada.
"""

from .bagging import BaggingStrategy
from .cycling import CyclingStrategy

__all__ = ['BaggingStrategy', 'CyclingStrategy']
