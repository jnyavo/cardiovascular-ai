"""Inference and visualization script for the imle-net model.
"""
from __future__ import annotations

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import logging
from model.IMLE_net.utils.tf_utils import set_tf_loglevel

from typing import TYPE_CHECKING

set_tf_loglevel(logging.ERROR)

import wfdb, os
import numpy as np
from joblib import load
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px

from model.IMLE_net.preprocessing.preprocess import apply_scaler

if TYPE_CHECKING:
    from .configs.imle_config import Config

def transform_ecg_for_inference(data: np.ndarray, scaler_path: str | None = 'bin/standardization.bin') -> np.asarray:
    """ Preprocessing pipeline

    Parameters
    ----------
    data: np.ndarray
        ECG signal to be preprocessed.
    scaler_path: str
        Path to the scaler file, default is set to bin/standardization.bin
    """
    if scaler_path is not None:
        path = os.path.join(os.getcwd(), scaler_path)
        scaler = load(path)
        data = apply_scaler(data, scaler)
    data = np.transpose(data[..., np.newaxis], (0,2,1,3))
    return data


def build_scores(model: tf.keras.Model, data: np.asarray, config: Config) -> None:
    """ Calculating the attention scores and visualization.

    Parameters
    ----------
    model: tf.keras.Model
        Model to be trained.
    data: np.asarray
        Sample data to be visualized.  
    config:
        Configuration for the imle-net model.

    """

    scores_model = tf.keras.models.Model(inputs = model.input, outputs = [model.get_layer("beat_att").output, 
                                                                           model.get_layer("rhythm_att").output,
                                                                           model.get_layer("channel_att").output])
    beat, rhythm, channel = scores_model(data)
    beat = beat[1].numpy(); rhythm = rhythm[1].numpy(); channel = channel[1].numpy()

    # Beat scores
    lin = np.linspace(0, config.input_channels, num=config.beat_len)
    beat = beat.reshape(240, 13)
    beat_only = np.empty((240, config.beat_len))
    for i in range(beat.shape[0]):
        beat_only[i] = np.interp(lin, np.arange(13), beat[i])

    # Rhythm scores
    rhythm = rhythm.reshape(config.input_channels*int(config.signal_len / config.beat_len))
    
    # Channel scores
    channel = channel.flatten()

    # Beat scores using channel
    beat_channel = np.copy(beat_only.reshape(config.input_channels, config.beat_len * int(config.signal_len / config.beat_len)))     
    for i in range(config.input_channels):
        beat_channel[i] = beat_channel[i] * channel[i]

    beat_normalized = (beat_channel.flatten() - beat_channel.flatten().min(keepdims=True)) / (beat_channel.flatten().max(keepdims=True) - beat_channel.flatten().min(keepdims=True))
    beat_normalized = beat_normalized.reshape(config.input_channels, config.signal_len)
    v_min = np.min(beat_channel.flatten())
    v_max = np.max(beat_channel.flatten())
    
    ch_info = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    results_filepath = os.path.join(os.getcwd(), "results")
    os.makedirs(results_filepath, exist_ok=True)

    # Display beat attention visualization
    fig, axs = plt.subplots(config.input_channels, figsize=(35, 25))
    data = data.squeeze()

    for i, (ax, ch) in enumerate(zip(axs, ch_info)):
        im = ax.scatter(np.arange(len(data[i])), data[i], cmap='Spectral', c=beat_normalized[i], vmin=v_min, vmax=v_max)
        ax.plot(data[i], color=(0.2, 0.68, 1))
        ax.set_yticks([])
        ax.set_title(ch, fontsize=25)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([])
    
    # Save the visualization
    plt.savefig(os.path.join(results_filepath, 'visualization.png'))
    
    # Show the figure in the notebook
    plt.show()

    # Display channel importance scores using matplotlib
    # plt.figure(figsize=(12, 6))
    # bars = plt.bar(ch_info, channel)
    
    # # Add value labels above each bar
    # for i, v in enumerate(channel):
    #     plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
    # plt.title('Channel Importance Scores', fontsize=18)
    # plt.ylabel('Importance Score', fontsize=14)
    # plt.xlabel('ECG Leads', fontsize=14)
    # plt.xticks(rotation=0)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # # Highlight the most important channels
    # threshold = np.mean(channel) + 0.5 * np.std(channel)
    # for i, bar in enumerate(bars):
    #     if channel[i] > threshold:
    #         bar.set_color('orange')
    
    # Create and save the interactive plotly visualization
    fig = px.bar(
        x=ch_info, 
        y=channel, 
        title='Channel Importance Scores',
        labels={'x': 'ECG Leads', 'y': 'Importance Score'}
    )
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=np.arange(config.input_channels), ticktext=ch_info),
        yaxis=dict(title_font=dict(size=14)),
        title_font=dict(size=18)
    )
    
    # Save HTML file
    fig.write_html(os.path.join(results_filepath, 'channel_visualization.html'))
    
    # Display the plotly figure in the notebook
    from IPython.display import display
    display(fig)