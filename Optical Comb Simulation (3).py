#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def generate_frequency_comb(base_freq, repetition_rate, num_teeth, freq_range):
    comb_spectrum = np.zeros(freq_range)
    indices = [int(base_freq + i * repetition_rate) for i in range(num_teeth) if base_freq + i * repetition_rate < freq_range]
    comb_spectrum[indices] = 1
    return comb_spectrum

def generate_sinc_function(freq_range, duration):
    t = np.linspace(-duration / 2, duration / 2, freq_range)
    sinc_time = np.sinc(t)
    sinc_freq = np.abs(np.fft.fft(sinc_time))
    sinc_freq = np.fft.fftshift(sinc_freq)
    return sinc_freq

def plot_results(sinc_freq, comb_spectrum):
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

base_freq = 50
repetition_rate = 10
num_teeth = 20
freq_range = 1000
duration = 1000

sinc_freq = generate_sinc_function(freq_range, duration)
comb_spectrum = generate_frequency_comb(base_freq, repetition_rate, num_teeth, freq_range)

plot_results(sinc_freq, comb_spectrum)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib.animation import FuncAnimation


# In[4]:


def generate_frequency_comb(base_freq, repetition_rate, num_teeth, freq_range):
    comb_spectrum = np.zeros(freq_range)
    indices = [int(base_freq + i * repetition_rate) % freq_range for i in range(num_teeth)]
    comb_spectrum[indices] = 1
    return comb_spectrum

def generate_sinc_function(freq_range, duration):
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

base_freq = 50
repetition_rate = 10
num_teeth = 20
freq_range = 1000
duration = 1000
sinc_freq = generate_sinc_function(freq_range, duration)
freq_indices = np.arange(freq_range)

fig, ax = plt.subplots()
line, = ax.plot(freq_indices, sinc_freq, lw=2, label='Filtered Signal')
markerline, stemlines, baseline = ax.stem(freq_indices, np.zeros(freq_range), linefmt='g-', markerfmt='go', basefmt='g-', label='Frequency Comb', use_line_collection=True)
ax.set_xlim(0, freq_range)
ax.set_ylim(0, 1.1 * sinc_freq.max())
ax.legend()

ani = FuncAnimation(fig, update, frames=np.arange(0, freq_range, 10), init_func=init, blit=True, interval=50)

plt.show()


# In[15]:


def generate_sinc_wave(t, width):
    return np.sinc(t * width)

fig, ax = plt.subplots()
t = np.linspace(-10, 10, 1000)
sinc_wave, = ax.plot(t, generate_sinc_wave(t, 1), 'b-')
ax.set_ylim(-0.3, 1.1)
ax.set_title("Sinc Wave Animation")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")

def init():
    sinc_wave.set_ydata(np.zeros_like(t))
    return sinc_wave,

def update(frame):
    width = 1 + frame / 10
    y = generate_sinc_wave(t, width)
    sinc_wave.set_ydata(y)
    return sinc_wave,

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(1, 100), init_func=init, blit=True, interval=100)

plt.show()


# In[16]:


def generate_signals(num_signals, sample_points, signal_amp, spacing):
    t = np.linspace(-10, 10, sample_points)
    signals = np.array([signal_amp * np.sinc(t - spacing * i) for i in range(num_signals)])
    return signals, t

def generate_frequency_comb(num_teeth, sample_points):
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
    for line in lines:
        line.set_ydata(np.zeros_like(t))
    line_comb.set_ydata(np.zeros_like(t))
    return *lines, line_comb

def update(frame):
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





# In[23]:


# Simulation parameters
fs = 1e6  # Sampling frequency in Hz
duration = 1e-3  # Duration of the signal in seconds
t = np.arange(0, duration, 1/fs)  # Time vector

def generate_sine_wave(frequency, amplitude=1):
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_digital_signal(data_rate, amplitude=1):
    num_bits = int(data_rate * duration)
    bit_duration = int(fs / data_rate)
    digital_signal = np.zeros(len(t))
    for i in range(num_bits):
        digital_signal[i * bit_duration:(i + 1) * bit_duration] = amplitude if np.random.rand() > 0.5 else 0
    return digital_signal

def amplitude_modulate(signal, carrier_freq):
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    return signal * carrier

def multiplex_signals(signal1, signal2):
    return np.concatenate((signal1, signal2))

def demultiplex_signals(signal, num_samples):
    return signal[:num_samples], signal[num_samples:2*num_samples]

analog_signal = generate_sine_wave(1e3)
digital_signal = generate_digital_signal(1e3)

modulated_analog = amplitude_modulate(analog_signal, 5e3)
modulated_digital = amplitude_modulate(digital_signal, 5e3)

multiplexed_signal = multiplex_signals(modulated_analog, modulated_digital)
extended_t = np.linspace(0, 2 * duration, len(multiplexed_signal), endpoint=False)
demultiplexed_analog, demultiplexed_digital = demultiplex_signals(multiplexed_signal, len(t))

plt.figure(figsize=(15, 6))
plt.subplot(311)
plt.title("Original Modulated Signals")
plt.plot(t, modulated_analog, label="Modulated Analog Signal")
plt.plot(t, modulated_digital, label="Modulated Digital Signal", linestyle='--')
plt.legend()

plt.subplot(312)
plt.title("Multiplexed Signal")
plt.plot(extended_t, multiplexed_signal, label="Multiplexed Signal")
plt.legend()

plt.subplot(313)
plt.title("Demultiplexed Signals")
plt.plot(t, demultiplexed_analog, label="Demultiplexed Analog Signal")
plt.plot(t, demultiplexed_digital, label="Demultiplexed Digital Signal", linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




