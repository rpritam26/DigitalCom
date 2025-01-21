# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
bit_rate = 1          # Bits per second
carrier_frequency = 5 # Carrier frequency (Hz)
sample_rate = 1000    # Samples per second
data_bits = np.array([1, 0, 1, 1, 0, 0, 1])  # Input binary data
bit_duration = 1 / bit_rate
t = np.linspace(0, bit_duration, int(sample_rate * bit_duration), endpoint=False)  # Time vector for one bit

# Generate BPSK modulated signal
bpsk_signal = []
for bit in data_bits:
    if bit == 1:
        bpsk_signal.extend(np.cos(2 * np.pi * carrier_frequency * t))  # Binary 1: No phase shift
    else:
        bpsk_signal.extend(-np.cos(2 * np.pi * carrier_frequency * t))  # Binary 0: 180° phase shift

bpsk_signal = np.array(bpsk_signal)

# Simulate noisy channel
noise = np.random.normal(0, 0.5, bpsk_signal.shape)  # Additive white Gaussian noise
noisy_signal = bpsk_signal + noise

# Demodulation
demodulated_bits = []
for i in range(len(data_bits)):
    segment = noisy_signal[i * len(t):(i + 1) * len(t)]
    correlation = np.sum(segment * np.cos(2 * np.pi * carrier_frequency * t))
    if correlation > 0:
        demodulated_bits.append(1)  # Binary 1
    else:
        demodulated_bits.append(0)  # Binary 0

# Convert demodulated bits to numpy array
demodulated_bits = np.array(demodulated_bits)

# Generate binary sequence signals for plotting
time = np.linspace(0, len(data_bits) * bit_duration, len(data_bits) * len(t), endpoint=False)
input_signal = np.repeat(data_bits, len(t))
demodulated_signal = np.repeat(demodulated_bits, len(t))

# Plot results
plt.figure(figsize=(15, 12))

# Input binary sequence
plt.subplot(5, 1, 1)
plt.plot(time, input_signal, label="Input Binary Sequence", drawstyle='steps-post', color='green')
plt.title("Input Binary Sequence")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# BPSK modulated signal
plt.subplot(5, 1, 2)
plt.plot(bpsk_signal, label="BPSK Signal")
plt.title("BPSK Modulated Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid()

# Noisy signal
plt.subplot(5, 1, 3)
plt.plot(noisy_signal, label="Noisy Signal", color='orange')
plt.title("Noisy Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid()

# Demodulated binary sequence
plt.subplot(5, 1, 4)
plt.plot(time, demodulated_signal, label="Demodulated Binary Sequence", drawstyle='steps-post', color='blue')
plt.title("Demodulated Binary Sequence")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# Compare input and demodulated bits
plt.subplot(5, 1, 5)
plt.stem(data_bits, linefmt="g-", markerfmt="go", basefmt="r-", label="Original Data")
plt.stem(demodulated_bits, linefmt="b-", markerfmt="bo", basefmt="r-", label="Demodulated Data")
plt.title("Original vs. Demodulated Data (Bit Index)")
plt.xlabel("Bit Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Print results
print("Original Data: ", data_bits)
print("Demodulated Data: ", demodulated_bits)
