import cv2
import random
import numpy as np
from __init__ import *
from typing import Optional


def run_statecontrol(frame: np.ndarray, model_store: ModelStore, sp: SpotifyPlayer):
    screen, confidence = model_store.models['state_detector'].predict(frame)
    print(screen,confidence)
