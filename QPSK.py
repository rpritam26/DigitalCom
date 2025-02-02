import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Step 1: Generate predefined binary data
binary_sequence = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
data_bits = np.array(binary_sequence)
num_bits = len(data_bits)

# Step 2: Map bits to QPSK symbols
symbol_map = { 
    (0, 0):  1 + 1j,
    (0, 1): -1 + 1j,
    (1, 1): -1 - 1j,
    (1, 0):  1 - 1j
}

symbols = np.array([symbol_map[tuple(data_bits[i:i+2])] for i in range(0, num_bits, 2)])

# Step 3: Add AWGN noise to the signal
snr_db = 10  # Signal-to-Noise Ratio in dB
snr_linear = 10**(snr_db / 10)
noise_std = np.sqrt(1 / (2 * snr_linear))
noise = noise_std * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
received_symbols = symbols + noise

# Step 4: QPSK Demodulation (Decision Rule)
demodulated_bits = []
demodulated_symbols = []
for sym in received_symbols:
    if np.real(sym) > 0 and np.imag(sym) > 0:
        demodulated_bits.extend([0, 0])
        demodulated_symbols.append(1 + 1j)
    elif np.real(sym) < 0 and np.imag(sym) > 0:
        demodulated_bits.extend([0, 1])
        demodulated_symbols.append(-1 + 1j)
    elif np.real(sym) < 0 and np.imag(sym) < 0:
        demodulated_bits.extend([1, 1])
        demodulated_symbols.append(-1 - 1j)
    else:
        demodulated_bits.extend([1, 0])
        demodulated_symbols.append(1 - 1j)

demodulated_symbols = np.array(demodulated_symbols)

# Step 5: Compute Bit Error Rate (BER)
bit_errors = np.sum(np.array(demodulated_bits) != data_bits)
ber = bit_errors / num_bits
print(f"Bit Error Rate (BER): {ber}")

# Step 6: Plot Constellation and Quadrature Diagrams
plt.figure(figsize=(12, 7))

# Ideal Constellation Diagram
plt.subplot(2, 2, 1)
plt.scatter([1, -1, -1, 1], [1, 1, -1, -1], color='blue', marker='x', s=100)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.title("Ideal QPSK Constellation")
plt.grid(True)

# Received and Demodulated Symbols
plt.subplot(2, 2, 2)
plt.scatter(received_symbols.real, received_symbols.imag, color='red', alpha=0.5, label="Noisy Symbols")
plt.scatter(demodulated_symbols.real, demodulated_symbols.imag, color='green', marker='o', label="Demodulated Symbols")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.title("QPSK Demodulation and Received Signal")
plt.legend()
plt.grid(True)

# Binary Input Data Plot
plt.subplot(2, 2, 3)
plt.plot(data_bits, 'bo-', markersize=2, label="Input Data")
plt.xlabel("Bit Index")
plt.ylabel("Bit Value")
plt.title("Binary Input Data")
plt.grid(True)
plt.legend()



plt.tight_layout()
plt.show()
