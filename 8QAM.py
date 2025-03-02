import numpy as np
import matplotlib.pyplot as plt

# Define 8-QAM constellation points
qam8_symbols = {
    0: (-1-1j), 1: (-1+1j), 2: (1-1j), 3: (1+1j),
    4: (-3-3j), 5: (-3+3j), 6: (3-3j), 7: (3+3j)
}

# Generate random data symbols
num_symbols = 1000
data = np.random.randint(0, 8, num_symbols)

# Modulate the data
modulated_signal = np.array([qam8_symbols[symbol] for symbol in data])

# Plot the constellation diagram
plt.figure(figsize=(6, 6))
plt.scatter(modulated_signal.real, modulated_signal.imag, color='b', alpha=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('8-QAM Constellation Diagram')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.show()
