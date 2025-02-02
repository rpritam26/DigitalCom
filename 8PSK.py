import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate predefined binary data
binary_sequence = [0,0,0,0,0,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,0,0]
data_bits = np.array(binary_sequence)
num_bits = len(data_bits)

# Ensure the length is a multiple of 3 for 8-PSK
if num_bits % 3 != 0:
    raise ValueError("Number of bits must be a multiple of 3 for 8-PSK.")

# Step 2: Define 8-PSK Symbol Mapping (Gray Coding)
bit_to_phase = {
    (0, 0, 0): 0,
    (0, 0, 1): np.pi / 4,
    (0, 1, 1): np.pi / 2,
    (0, 1, 0): 3 * np.pi / 4,
    (1, 1, 0): np.pi,
    (1, 1, 1): 5 * np.pi / 4,
    (1, 0, 1): 3 * np.pi / 2,
    (1, 0, 0): 7 * np.pi / 4
}

# Step 3: Modulate the symbols
symbols = []
for i in range(0, num_bits, 3):
    bits = tuple(data_bits[i:i+3])
    phase = bit_to_phase[bits]
    symbols.append(np.exp(1j * phase))

symbols = np.array(symbols)

# Step 4: Add AWGN noise
snr_db = 10  # Signal-to-Noise Ratio in dB
snr_linear = 10**(snr_db / 10)
noise_std = np.sqrt(1 / (2 * snr_linear))
noise = noise_std * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
received_symbols = symbols + noise

# Step 5: 8-PSK Demodulation (Decision Rule)
demodulated_bits = []
phases = np.array(list(bit_to_phase.values()))
for sym in received_symbols:
    received_phase = np.angle(sym)
    closest_phase = min(phases, key=lambda p: abs(p - received_phase))
    decoded_bits = [key for key, value in bit_to_phase.items() if value == closest_phase][0]
    demodulated_bits.extend(decoded_bits)

# Step 6: Compute Bit Error Rate (BER)
bit_errors = np.sum(np.array(demodulated_bits) != data_bits)
ber = bit_errors / num_bits
print(f"Bit Error Rate (BER): {ber}")

# Step 7: Plot Constellation Diagrams
plt.figure(figsize=(12,5))

# Plot ideal constellation
plt.subplot(1, 2, 1)
plt.scatter(np.cos(phases), np.sin(phases), color='blue', marker='x', label="Ideal Symbols")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.title("Ideal 8-PSK Constellation")
plt.legend()
plt.grid(True)

# Plot received constellation
plt.subplot(1, 2, 2)
plt.scatter(received_symbols.real, received_symbols.imag, color='red', alpha=0.5, label="Noisy Symbols")
plt.scatter(symbols.real, symbols.imag, color='blue', marker='x', label="Ideal Symbols")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.title("8-PSK Constellation with Noise")
plt.legend()
plt.grid(True)

plt.show()

# Step 8: Plot Binary Sequence
plt.figure(figsize=(8, 2))
plt.step(range(len(binary_sequence)), binary_sequence, where='mid', color='black')
plt.ylim(-0.5, 1.5)
plt.yticks([0, 1])
plt.xlabel("Bit Index")
plt.ylabel("Bit Value")
plt.title("Input Binary Sequence")
plt.grid(True)
plt.show()
