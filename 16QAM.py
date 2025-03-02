import numpy as np
import matplotlib.pyplot as plt

# Define 16-QAM constellation points
qam16_symbols = {
    0: (-3-3j),  1: (-3-1j),  2: (-3+3j),  3: (-3+1j),
    4: (-1-3j),  5: (-1-1j),  6: (-1+3j),  7: (-1+1j),
    8: (3-3j),   9: (3-1j),  10: (3+3j), 11: (3+1j),
    12: (1-3j), 13: (1-1j), 14: (1+3j), 15: (1+1j)
}

# Generate random data symbols
num_symbols = 1000
data = np.random.randint(0, 16, num_symbols)

# Modulate the data
modulated_signal = np.array([qam16_symbols[symbol] for symbol in data])

# Plot the constellation diagram
plt.figure(figsize=(6, 6))
plt.scatter(modulated_signal.real, modulated_signal.imag, color='b', alpha=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('16-QAM Constellation Diagram')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.show()
