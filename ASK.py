import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000        # Sampling frequency (Hz)
fc = 100         # Carrier frequency (Hz)
data_rate = 10   # Data rate (bits per second)
Ac = 1           # Carrier amplitude

# Time vector
t = np.arange(0, 1, 1/fs)  # 1 second of data

# Generate binary data
data = np.random.randint(0, 2, int(len(t) / (fs / data_rate)))
data_signal = np.repeat(data, fs // data_rate)  # Upsample data to match sampling rate

# Carrier signal
carrier = Ac * np.cos(2 * np.pi * fc * t)

# ASK Modulation
ask_signal = carrier * data_signal

# Plot Modulated Signal
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.title("Binary Data")
plt.plot(t[:500], data_signal[:500], 'b')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 2)
plt.title("Carrier Signal")
plt.plot(t[:500], carrier[:500], 'r')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 3)
plt.title("ASK Modulated Signal")
plt.plot(t[:500], ask_signal[:500], 'g')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

plt.subplot(2, 1, 2)
plt.title("Demodulated Data")
plt.plot(t[:500], recovered_data[:500], 'r')
plt.xlabel("Time (s)")
plt.ylabel("Binary Data")
plt.grid()

plt.tight_layout()
plt.show()
