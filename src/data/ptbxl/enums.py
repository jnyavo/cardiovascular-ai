from enum import IntEnum


class SamplingRate(IntEnum):
    HZ_100 = 100
    HZ_500 = 500

class Lead(IntEnum):
    I = 0
    II = 1
    III = 2
    aVR = 3
    aVL = 4
    aVF = 5
    V1 = 6
    V2 = 7
    V3 = 8
    V4 = 9
    V5 = 10
    V6 = 11