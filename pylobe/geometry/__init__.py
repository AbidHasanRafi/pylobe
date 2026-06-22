"""PyLobe geometry subpackage."""
from pylobe.geometry.base import AntennaGeometry, Material, AIR, FR4, RT5880, PEC
from pylobe.geometry.patch import RectangularPatch, CircularPatch, AnnularRingPatch, ESlotPatch
from pylobe.geometry.dipole import HalfWaveDipole, FoldedDipole, BowTieDipole
from pylobe.geometry.monopole import QuarterWaveMonopole, HelicalMonopole
from pylobe.geometry.slot import SlotAntenna, VivaldiAntenna
from pylobe.geometry.fractal import KochDipole, SierpinskiGasket, MinkowskiPatch
from pylobe.geometry.array import LinearArray, PlanarArray, CircularArray
from pylobe.geometry.yagi import YagiUda
from pylobe.geometry.horn import PyramidalHorn
from pylobe.geometry.loop import SmallLoopAntenna, LargeLoopAntenna
from pylobe.geometry.lpda import LogPeriodicArray
from pylobe.geometry.pifa import PIFA

__all__ = [
    "AntennaGeometry", "Material", "AIR", "FR4", "RT5880", "PEC",
    "RectangularPatch", "CircularPatch", "AnnularRingPatch", "ESlotPatch",
    "HalfWaveDipole", "FoldedDipole", "BowTieDipole",
    "QuarterWaveMonopole", "HelicalMonopole",
    "SlotAntenna", "VivaldiAntenna",
    "KochDipole", "SierpinskiGasket", "MinkowskiPatch",
    "LinearArray", "PlanarArray", "CircularArray",
    "YagiUda",
    "PyramidalHorn",
    "SmallLoopAntenna", "LargeLoopAntenna",
    "LogPeriodicArray",
    "PIFA",
]
