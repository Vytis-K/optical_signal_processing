#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def generate_frequency_comb(base_freq, repetition_rate, num_teeth, freq_range):
    """Generate a frequency comb in the frequency domain as a series of delta functions."""
    comb_spectrum = np.zeros(freq_range)
    indices = [int(base_freq + i * repetition_rate) for i in range(num_teeth) if base_freq + i * repetition_rate < freq_range]
    comb_spectrum[indices] = 1
    return comb_spectrum

def generate_sinc_function(freq_range, duration):
    """Generate a sinc function in the frequency domain."""
    t = np.linspace(-duration / 2, duration / 2, freq_range)
    sinc_time = np.sinc(t)
    sinc_freq = np.abs(np.fft.fft(sinc_time))
    sinc_freq = np.fft.fftshift(sinc_freq)
    return sinc_freq

def plot_results(sinc_freq, comb_spectrum):
    """Plot the sinc function, frequency comb, and their product."""
    filtered_signal = sinc_freq * comb_spectrum

    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.plot(sinc_freq, label='Sinc Function')
    plt.title('Sinc Function Frequency Spectrum')
    plt.xlabel('Frequency Index')
    plt.ylabel('Amplitude')

    plt.subplot(132)
    plt.stem(comb_spectrum, linefmt='g-', markerfmt='go', basefmt='g-', use_line_collection=True)
    plt.title('Optical Frequency Comb')
    plt.xlabel('Frequency Index')

    plt.subplot(133)
    plt.plot(filtered_signal, label='Filtered Signal', color='r')
    plt.title('Filtered Signal by Comb')
    plt.xlabel('Frequency Index')
    
    plt.tight_layout()
    plt.show()

# Parameters
base_freq = 50
repetition_rate = 10
num_teeth = 20
freq_range = 1000
duration = 1000

# Generate the frequency domain representations
sinc_freq = generate_sinc_function(freq_range, duration)
comb_spectrum = generate_frequency_comb(base_freq, repetition_rate, num_teeth, freq_range)

# Plot the results
plot_results(sinc_freq, comb_spectrum)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib.animation import FuncAnimation


# In[4]:


def generate_frequency_comb(base_freq, repetition_rate, num_teeth, freq_range):
    """Generate a frequency comb as a series of delta functions."""
    comb_spectrum = np.zeros(freq_range)
    indices = [int(base_freq + i * repetition_rate) % freq_range for i in range(num_teeth)]
    comb_spectrum[indices] = 1
    return comb_spectrum

def generate_sinc_function(freq_range, duration):
    """Generate a sinc function in the frequency domain."""
    t = np.linspace(-duration / 2, duration / 2, freq_range)
    sinc_time = np.sinc(t)
    sinc_freq = np.abs(np.fft.fft(sinc_time))
    sinc_freq = np.fft.fftshift(sinc_freq)
    return sinc_freq

def init():
    line.set_ydata(np.zeros(freq_range))
    markerline.set_ydata(np.zeros(freq_range))
    stemlines.set_segments([[[x, 0], [x, 0]] for x in freq_indices])
    return line, markerline, stemlines

def update(frame):
    base_freq = frame
    comb_spectrum = generate_frequency_comb(base_freq, repetition_rate, num_teeth, freq_range)
    filtered_signal = sinc_freq * comb_spectrum
    line.set_ydata(filtered_signal)
    markerline.set_ydata(comb_spectrum * 1.1 * sinc_freq.max())
    stemlines.set_segments([[[x, 0], [x, y]] for x, y in zip(freq_indices, comb_spectrum * 1.1 * sinc_freq.max())])
    return line, markerline, stemlines

# Parameters
base_freq = 50
repetition_rate = 10
num_teeth = 20
freq_range = 1000
duration = 1000
sinc_freq = generate_sinc_function(freq_range, duration)
freq_indices = np.arange(freq_range)

# Plotting setup
fig, ax = plt.subplots()
line, = ax.plot(freq_indices, sinc_freq, lw=2, label='Filtered Signal')
markerline, stemlines, baseline = ax.stem(freq_indices, np.zeros(freq_range), linefmt='g-', markerfmt='go', basefmt='g-', label='Frequency Comb', use_line_collection=True)
ax.set_xlim(0, freq_range)
ax.set_ylim(0, 1.1 * sinc_freq.max())
ax.legend()

# Animation
ani = FuncAnimation(fig, update, frames=np.arange(0, freq_range, 10), init_func=init, blit=True, interval=50)

plt.show()


# In[6]:


def generate_sinc_wave(t, width):
    """Generate values of a sinc wave based on time array t and width parameter."""
    return np.sinc(t * width)

# Prepare the plot
fig, ax = plt.subplots()
t = np.linspace(-10, 10, 1000)
sinc_wave, = ax.plot(t, generate_sinc_wave(t, 1), 'b-')
ax.set_ylim(-0.3, 1.1)
ax.set_title("Sinc Wave Animation")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")

def init():
    """Initialize the background of the animation."""
    sinc_wave.set_ydata(np.zeros_like(t))
    return sinc_wave,

def update(frame):
    """Update the plot for each frame."""
    width = 1 + frame / 10
    y = generate_sinc_wave(t, width)
    sinc_wave.set_ydata(y)
    return sinc_wave,

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(1, 100), init_func=init, blit=True, interval=100)

plt.show()


# In[13]:


def generate_signals(num_signals, sample_points, signal_amp, spacing):
    """Generates multiple sinc signals at different positions."""
    t = np.linspace(-10, 10, sample_points)
    signals = np.array([signal_amp * np.sinc(t - spacing * i) for i in range(num_signals)])
    return signals, t

def generate_frequency_comb(num_teeth, sample_points):
    """Generates a simple frequency comb."""
    comb = np.zeros(sample_points)
    indices = np.linspace(0, sample_points-1, num_teeth, dtype=int)
    comb[indices] = 1
    return comb

# Prepare the plot
fig, ax1 = plt.subplots(figsize=(10, 5))
sample_points = 1000
num_signals = 5
signal_amp = 1
spacing = 3
signals, t = generate_signals(num_signals, sample_points, signal_amp, spacing)
comb = generate_frequency_comb(20, sample_points)

ax1.set_title("Signal Processing by Agnostic Sampling Transceiver")
lines = [ax1.plot(t, signal, label=f'Signal {i+1}')[0] for i, signal in enumerate(signals)]
line_comb, = ax1.plot(t, comb * signal_amp * 1.2, 'g--', label='Frequency Comb')
ax1.set_ylim(-1.5, 1.5)
ax1.legend()

def init():
    """Initialize the animation."""
    for line in lines:
        line.set_ydata(np.zeros_like(t))
    line_comb.set_ydata(np.zeros_like(t))
    return *lines, line_comb

def update(frame):
    """Update the plot for each frame."""
    comb_rotated = np.roll(comb, frame * 5)
    line_comb.set_ydata(comb_rotated * signal_amp * 1.2)
    for i, line in enumerate(lines):
        processed_signal = signals[i] * comb_rotated
        line.set_ydata(processed_signal)
    return *lines, line_comb

# Create animation
ani = FuncAnimation(fig, update, frames=range(sample_points // 5), init_func=init, blit=True, interval=100)

plt.show()


# In[ ]:




