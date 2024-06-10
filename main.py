# Testing Sample 
import numpy as np
import matplotlib.pyplot as plt

# Initialize the lattice with random spins (+1 or -1)
def initialize_lattice(L):
    return np.random.choice([-1, 1], size=(L, L))

# Calculate the total energy of the lattice
def calculate_energy(lattice):
    L = lattice.shape[0]
    energy = 0
    # Sum over all pairs of neighboring spins
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            # Neighbors with periodic boundary conditions
            neighbors = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            energy += -S * neighbors
    return energy / 4.0  # Each pair counted twice, so divide by 4

# Calculate the change in energy if the spin at (i, j) is flipped
def delta_energy(lattice, i, j):
    L = lattice.shape[0]
    S = lattice[i, j]
    # Neighbors with periodic boundary conditions
    neighbors = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
    return 2 * S * neighbors

# Perform the Metropolis algorithm for one Monte Carlo step
def metropolis(lattice, beta):
    L = lattice.shape[0]
    # Attempt to flip each spin in the lattice once per step
    for _ in range(L * L):
        i, j = np.random.randint(0, L, size=2)  # Randomly choose a spin
        dE = delta_energy(lattice, i, j)        # Calculate energy change
        # Flip the spin with Metropolis criterion
        if dE < 0 or np.random.rand() < np.exp(-dE * beta):
            lattice[i, j] *= -1

# Simulate the Ising model using the Metropolis algorithm
def simulate(L, temperature, steps):
    beta = 1.0 / temperature  # Inverse temperature
    lattice = initialize_lattice(L)  # Initialize lattice
    energies = []  # To store energy values
    
    # Perform Metropolis steps
    for step in range(steps):
        metropolis(lattice, beta)
        if step % 100 == 0:
            energies.append(calculate_energy(lattice))
            # Plot the lattice configuration every 1000 steps
            if step % 1000 == 0:
                plot_lattice(lattice, step)
    
    return lattice, energies

# Plot the lattice of spins
def plot_lattice(lattice, step):
    plt.imshow(lattice, cmap='coolwarm')
    plt.title(f'Ising Model Lattice at Step {step}')
    plt.colorbar()
    plt.show()

# Plot the energy as a function of Monte Carlo steps
def plot_energy(energies):
    plt.plot(energies)
    plt.title('Energy vs. Monte Carlo steps')
    plt.xlabel('Steps (x100)')
    plt.ylabel('Energy')
    plt.show()

# Parameters for the simulation
L = 20           # Lattice size (LxL)
temperature = 2.0  # Temperature (T)
steps = 100000      # Number of Monte Carlo steps

# Run the simulation
lattice, energies = simulate(L, temperature, steps)

# Plot the final lattice configuration and energy plot
plot_lattice(lattice, steps)
plot_energy(energies)
