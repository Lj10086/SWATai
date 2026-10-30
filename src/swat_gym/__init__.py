"""swat_gym: a SWAT+ field model of the Kimberly long-term manure trial as an optimisation
environment. The environment is ``swat_gym.ltm``; the experiment driver is
``swat_gym.experiments.ltm``. ``FastRunner`` runs the SWAT+ engine repeatedly on one model."""
from .engine import EngineError, find_engine, run_engine
from .fastrunner import EDITABLE, TXTINOUT, FastRunner
from .manifest import input_files
