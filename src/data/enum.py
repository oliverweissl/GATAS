from enum import Enum


class AttackMode(Enum):
    TARGETED = "targeted"  # objectives compare the transcription with a target text
    NOISE_UNTARGETED = "noise-untargeted"  # objectives compare the transcription with the ground truth text
