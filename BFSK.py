import numpy as np
import matplotlib.pyplot as plt

def bfsk_modulation(bit_stream, bit_duration=1, f0=5, f1=10, sampling_rate=1000):
    t = np.arange(0, bit_duration * len(bit_stream), 1/sampling_rate)
    signal = np.array([])
    for bit in bit_stream:
        freq = f1 if bit == 1 else f0
        t_bit = np.arange(0, bit_duration, 1/sampling_rate)
        wave = np.sin(2 * np.pi * freq * t_bit)
        signal = np.concatenate((signal, wave))
    return t, signal

def bfsk_demodulation(signal, bit_duration=1, f0=5, f1=10, sampling_rate=1000):
    num_bits = len(signal) // (bit_duration * sampling_rate)
    bit_stream = []
    for i in range(int(num_bits)):
        t_bit = np.arange(0, bit_duration, 1/sampling_rate)
        segment = signal[i * bit_duration * sampling_rate: (i + 1) * bit_duration * sampling_rate]
        corr_f0 = np.sum(segment * np.sin(2 * np.pi * f0 * t_bit))
        corr_f1 = np.sum(segment * np.sin(2 * np.pi * f1 * t_bit))
        bit_stream.append(1 if corr_f1 > corr_f0 else 0)
    return bit_stream

bit_stream = [1, 0, 0, 1, 0, 0, 1, 0, 1, 1]
t, modulated_signal = bfsk_modulation(bit_stream)
demodulated_data = bfsk_demodulation(modulated_signal)

I_points = [1, 0]
Q_points = [0, 1]

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(range(len(bit_stream)), bit_stream, basefmt=" ")
plt.title("Input Data")
plt.xlabel("Bit Index")
plt.ylabel("Bit Value")
plt.ylim(-0.5, 1.5)
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, modulated_signal)
plt.title("BFSK Modulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 3)
plt.stem(range(len(demodulated_data)), demodulated_data, basefmt=" ", linefmt='r', markerfmt='ro')
plt.title("Demodulated Data")
plt.xlabel("Bit Index")
plt.ylabel("Bit Value")
plt.ylim(-0.5, 1.5)
plt.grid()

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(I_points, Q_points, color=['r', 'b'], s=100)
plt.title("Constellation Diagram")
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.grid()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()
