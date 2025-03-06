from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from .types.record import RecordType  # type: ignore
from data.ptbxl.enums import Lead



START_TIME = 0
TIME = 10
END_TIME = START_TIME + TIME

class Record:
  def __init__(self, record: RecordType):
      self.data, self.fields = record
  def draw_ecg(self, lead: Lead = Lead.I):
    """
      Draws a single ECG lead
      :param lead: Lead number
      :return: None
    """

    hz: int = self.fields['fs']
    ecg = self.data.T
    start_length = int(START_TIME * hz)
    sample_length = int(TIME * hz)
    t = np.arange(START_TIME, END_TIME, 1 / hz)
    plt.rcParams["figure.figsize"] = (25, 1.5)
    plt.plot(
        t,
        ecg[lead][start_length: start_length + sample_length],
        linewidth=2,
        color="k",
        alpha=1.0,
        label=Lead(lead).name,
    )
    minimum = min(ecg[lead])
    maximum = max(ecg[lead])
    ylims_candidates = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0 , 1.5, 2.0, 2.5]

    ylims = (
        max([x for x in ylims_candidates if x <= minimum]),
        min([x for x in ylims_candidates if x >= maximum]),
    )
    plt.vlines(np.arange(START_TIME, END_TIME, 0.2), ylims[0], ylims[1], colors="r", alpha=1.0)
    plt.vlines(np.arange(START_TIME, END_TIME, 0.04), ylims[0], ylims[1], colors="r", alpha=0.3)
    plt.hlines(np.arange(ylims[0], ylims[1], 0.5), START_TIME, END_TIME, colors="r", alpha=1.0)
    plt.hlines(np.arange(ylims[0], ylims[1], 0.1), START_TIME, END_TIME, colors="r", alpha=0.3)

    plt.xticks(np.arange(START_TIME, END_TIME + 1, 1.0))
    plt.margins(0.0)
    plt.show()