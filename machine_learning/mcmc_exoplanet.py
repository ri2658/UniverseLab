import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def generate_synthetic_rv_data():
    """Generates synthetic Radial Velocity (RV) data."""
    TRUE_MASS_SINI = 50.0  
    PERIOD = 365.25        
    np.random.seed(42)  
    t_observations = np.sort(np.random.uniform(0, 1095, 50))
    true_velocities = TRUE_MASS_SINI * np.sin(2 * np.pi * t_observations / PERIOD)
    instrument_error = 10.0 
    uncertainties = np.full(50, instrument_error)
    noisy_velocities = true_velocities + np.random.normal(0, instrument_error, 50)
    
    output_path = Path(__file__).resolve().parents[1] / "datasets" / "synthetic_exoplanet_rv.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_days", "radial_velocity_m_s", "uncertainty_m_s"])
        for t, v, err in zip(t_observations, noisy_velocities, uncertainties):
            writer.writerow([f"{t:.2f}", f"{v:.2f}", f"{err:.1f}"])
            
    return t_observations, noisy_velocities, uncertainties

def log_likelihood(mass_guess, t, rv_data, rv_err): # rv = radial velocity
    """
    Calculates how well our guessed mass fits the data.
    This is mathematically equivalent to the Energy in a Boltzmann distribution.
    """
    PERIOD = 365.25

    # Our model for the radial velocity based on the guessed mass
    model_rv = mass_guess * np.sin(2 * np.pi * t / PERIOD)
    
    # Chi-squared represents the sum of the squared errors
    chisq = np.sum(((rv_data - model_rv) / rv_err) ** 2)
    
    # Our likelihood function is based on the assumption that the errors are Gaussian.
    # So, the log-likelihood is proportional to the negative of 1/2 the chi-squared statistic.
    return -0.5 * chisq

def run_mcmc(t, rv_data, rv_err, steps=10000):
    """
    The Metropolis-Hastings Algorithm.
    A random walk through parameter space seeking high probability.
    """
    # 1. Start with a completely wrong, random guess
    current_mass = 10.0
    current_ll = log_likelihood(current_mass, t, rv_data, rv_err)
    
    samples = []
    
    for _ in range(steps):
        # 2. Propose a new mass (Take a random step. Step size is roughly 2.0 m/s)
        proposed_mass = current_mass + np.random.normal(0, 2.0)
        
        # 3. Calculate new likelihood
        proposed_ll = log_likelihood(proposed_mass, t, rv_data, rv_err)
        
        # 4. Calculate Acceptance Probability
        # If proposed_ll > current_ll, the ratio is > 1 (automatically accepted).
        # If it's worse, the ratio is < 1, so we MIGHT accept it.
        # We use np.exp because we are working in Log-Likelihoods.
        if proposed_ll > current_ll:
            acceptance_prob = 1.0
        else:
            # We might still accept a worse guess because MH seeks to generate
            # samples from the posterior distribution, not just the maximum likelihood estimate.
            # As such, we must visit low-density areas LESS OFTEN, but still proportionally
            acceptance_prob = np.exp(proposed_ll - current_ll)
        
        # 5. Accept or Reject
        if np.random.uniform(0, 1) < acceptance_prob:
            current_mass = proposed_mass
            current_ll = proposed_ll
            
        samples.append(current_mass)
        
    return samples

def plot_posterior(samples):
    """Plots a histogram of all the masses the MCMC walker visited."""
    # Burn-in: We discard the first 1000 steps where the walker was still walking
    # towards the "high probability" valley from its random starting point.
    burn_in = 1000
    valid_samples = samples[burn_in:]
    
    plt.figure(figsize=(10, 6))
    plt.hist(valid_samples, bins=50, color='#8b5cf6', alpha=0.7, density=True)
    plt.axvline(np.mean(valid_samples), color='#facc15', linestyle='dashed', linewidth=3, label=f'MCMC Guess: {np.mean(valid_samples):.2f}')
    plt.axvline(50.0, color='#ef4444', linestyle='solid', linewidth=3, label='True Mass: 50.00')
    
    # Modern styling
    plt.style.use('dark_background')
    plt.gca().set_facecolor('#050b18')
    plt.gcf().set_facecolor('#050b18')
    
    plt.title('MCMC Posterior Distribution of Exoplanet Mass', color='white', pad=20, fontsize=16)
    plt.xlabel('Mass Amplitude (m/s)', color='white', fontsize=12)
    plt.ylabel('Probability Density', color='white', fontsize=12)
    plt.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    
    out_path = Path(__file__).resolve().parents[1] / "visualizations" / "mcmc_posterior.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved posterior visualization to {out_path}")

if __name__ == "__main__":
    # Generate data
    t, rv_data, rv_err = generate_synthetic_rv_data()
    
    # Run MCMC
    print("Running MCMC Metropolis-Hastings (10,000 steps)...")
    samples = run_mcmc(t, rv_data, rv_err, steps=10000)
    
    # Plot Results
    plot_posterior(samples)
