import numpy as np
import pandas as pd
import joblib
from scipy.stats import boxcox, poisson, norm
import os
import matplotlib.pyplot as plt

class MonteCarloCapacitySimulator:
    def __init__(self, csv_path="data/bor_2324_data.csv", hospital_capacities=None, time_steps=12, lambda_rate=0.8, cumulative_limit=0.045, save_path="models/hospital_capacity_simulations.joblib", csv_save_path="models"):
        """
        Initialize the Monte Carlo capacity simulator with hospital data from bor_2324_data.csv.

        :param csv_path: Path to the CSV file containing hospital capacity information.
        :param hospital_capacities: Optional dictionary to customize hospital capacities.
        :param time_steps: Number of 2-hour intervals in the day.
        :param lambda_rate: Average rate of change per 2-hour interval (Poisson distribution).
        :param cumulative_limit: Maximum cumulative percentage change allowed over the day (±4.5%).
        :param save_path: Path to save or load the simulation results (Joblib).
        :param csv_save_path: Path to save simulation results in CSV format.
        """
        self.csv_path = csv_path
        self.hospital_capacities = hospital_capacities or {}
        self.time_steps = time_steps
        self.lambda_rate = lambda_rate
        self.cumulative_limit = cumulative_limit
        self.save_path = save_path
        self.csv_save_path = csv_save_path
        self.hospital_names = []
        self.lambdas = {}
        
        # Load hospital names and initialize capacities
        self.hospital_data = self.load_and_initialize_hospitals()

    def load_and_initialize_hospitals(self):
        """
        Load capacity data for each hospital from the CSV file, ensure data is positive,
        apply Box-Cox transformation, and store the lambda values for each hospital.
        
        :return: Dictionary containing Box-Cox transformed data and lambda for each hospital.
        """
        # Load the CSV data
        df = pd.read_csv(self.csv_path)
        
        # Ensure 'Date' column is treated as datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # List of hospitals to process
        hospitals = ['AH', 'CGH', 'KTPH', 'NTFGH', 'NUH(A)', 'SGH', 'SKH', 'TTSH', 'WH']
        
        # Initialize dictionaries to store capacities and transformation results
        self.hospital_capacities = {hospital: 500 for hospital in hospitals}  # Default capacities set to 500
        boxcox_results = {}

        # Process each hospital's data
        for hospital in hospitals:
            # Ensure data is numeric and remove NaN values
            data = pd.to_numeric(df[hospital], errors='coerce').dropna()
            data = data[data > 0]  # Box-Cox transformation requires positive values
            
            # Apply Box-Cox transformation
            transformed_data, lambda_val = boxcox(data)
            boxcox_results[hospital] = {'transformed_data': transformed_data, 'lambda': lambda_val}
            self.lambdas[hospital] = lambda_val  # Store lambda for future reference

        return boxcox_results

    import numpy as np

    def run_simulation(self, num_simulations=10000):
        """
        Run Monte Carlo simulation by back-transforming Box-Cox data, 
        generating capped normal samples as percentages.

        :param num_simulations: Number of simulation runs to create capacity projections.
        :return: Dictionary of simulated capacity trajectories in percentage terms for each hospital.
        """
        hospital_capacity_simulation = {}

        # Define time step labels as 2-hour intervals (2, 4, 6, ..., 24)
        time_labels = time_labels = ["Simulation"] + [str(2 * i) for i in range(self.time_steps)]

        for hospital, data_info in self.hospital_data.items():
            lambda_val = data_info['lambda']
            transformed_data = data_info['transformed_data']

            # Step 1: Back-transform the Box-Cox data to the original scale (percentage format)
            if lambda_val != 0:
                original_data = (transformed_data * lambda_val + 1) ** (1 / lambda_val)
            else:
                original_data = np.exp(transformed_data)

            # Step 2: Calculate mean and variance on the original scale
            mean_capacity = np.mean(original_data)
            std_dev_capacity = np.sqrt(np.var(original_data))

            # Step 3: Generate capped normal samples in percentage form
            uniform_randoms = np.random.rand(num_simulations)
            normal_samples = mean_capacity + std_dev_capacity * norm.ppf(uniform_randoms)
            normal_samples = np.clip(normal_samples, 0, 1)  # Cap values to between 0 and 1

            hospital_simulations = []

            for initial_sample in normal_samples:
                # Use the initial sample as a starting capacity percentage
                capacity = initial_sample
                capacity_over_time = [capacity]
                cumulative_change = 0

                for t in range(self.time_steps):
                    delta_capacity = poisson.rvs(self.lambda_rate) / 100.0
                    change_direction = np.random.choice([-1, 1])
                    delta_capacity *= change_direction

                    # Ensure cumulative change stays within the cumulative limit
                    if abs(cumulative_change + delta_capacity) > self.cumulative_limit:
                        delta_capacity = -np.sign(cumulative_change) * min(abs(delta_capacity), self.cumulative_limit - abs(cumulative_change))

                    # Update capacity and cap to [0, 1]
                    capacity = np.clip(capacity + delta_capacity, 0, 1)
                    cumulative_change += delta_capacity

                    capacity_over_time.append(capacity)

                # Store simulation with time labels
                hospital_simulations.append(pd.Series(capacity_over_time, index=time_labels))

            hospital_capacity_simulation[hospital] = hospital_simulations

        # Save simulations to both joblib and CSV
        self.save_simulation_to_joblib(hospital_capacity_simulation)
        self.save_simulation_to_csv(hospital_capacity_simulation)

        simulation_indices = [123, 285, 341, 578, 635, 709, 811, 980, 67]

        # Loop over each hospital in the simulation results
        for hospital, simulations in hospital_capacity_simulation.items():
            plt.figure(figsize=(10, 6))
            
            # Plot each specified simulation on the same plot
            for index in simulation_indices:
                # Select the specific simulation by index (adjust for 0-based indexing)
                simulation = simulations[index - 1]
                
                # Drop the first column (label 'Simulation') for plotting
                simulation_without_first = simulation.drop("Simulation")
                
                # Plot the simulation with a label
                simulation_without_first.plot(label=f"Simulation {index}")
            
            # Configure plot aesthetics
            plt.title(f"Selected Simulations for {hospital}")
            plt.xlabel("Time (hours)")
            plt.ylabel("Capacity (%)")
            plt.ylim(0.6, 1)  # Ensure y-axis stays between 0 and 1
            plt.legend()
            plt.show()

        return hospital_capacity_simulation


    def save_simulation_to_joblib(self, simulation_data):
        """Save the simulation data to a Joblib file for efficient reloading."""
        joblib.dump(simulation_data, self.save_path)
        print(f"Simulation saved to {self.save_path}")

    def save_simulation_to_csv(self, simulation_data):
        """Save simulation data to a CSV format, with each hospital in a separate file."""
        for hospital, data in simulation_data.items():
            df = pd.DataFrame(data)
            csv_file_path = os.path.join(self.csv_save_path, f"{hospital}_capacity_simulations.csv")
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            df.to_csv(csv_file_path, index_label="Time Step")
            print(f"Simulation for {hospital} saved to {csv_file_path}")

    def load_simulation(self):
        """
        Load precomputed simulations from a Joblib file.

        :return: Loaded simulation dictionary in absolute terms, or None if file not found.
        """
        if os.path.exists(self.save_path):
            hospital_capacity_simulation = joblib.load(self.save_path)
            print(f"Simulation loaded from {self.save_path}")
            return hospital_capacity_simulation
        else:
            print(f"No saved simulation found at {self.save_path}. Please run a simulation first.")
            return None

    def check_capacity_at_time(self, hospital_capacity_simulation, hospital_name, time_step, simulation_index=0):
        """
        Retrieve the capacity of a specified hospital at a given 2-hour time step in absolute terms.

        :param hospital_capacity_simulation: Dictionary of all simulation results.
        :param hospital_name: Name of the hospital to check.
        :param time_step: Time step (2-hour intervals) to check (0-based index).
        :param simulation_index: Index of the simulation run to reference (default is the first simulation).
        :return: Absolute capacity at the given time for the specified hospital.
        """
        if hospital_name not in hospital_capacity_simulation:
            raise ValueError(f"Hospital '{hospital_name}' not found in simulation data.")
        
        if time_step < 0 or time_step > self.time_steps:
            raise ValueError(f"Invalid time step {time_step}. Must be between 0 and {self.time_steps}.")
        
        # Retrieve capacity from the specified simulation and time step
        absolute_capacity = hospital_capacity_simulation[hospital_name][simulation_index][time_step]
        
        return absolute_capacity

if __name__ == "__main__":
    # Define actual capacities for each hospital
    actual_capacities = {
        "AH": 326,
        "CGH": 1054,
        "KTPH": 795,
        "NTFGH": 700,
        "NUH(A)": 1289,
        "SGH": 1939,
        "SKH": 799,
        "TTSH": 2000,
        "WH": 1000,
        # https://www.singhealth.com.sg/about-singhealth/newsroom/Documents/SingHealth%20Duke-NUS%20Annual%20Report%20(FY2223)%20Annual%20Overview%20-%20Final.pdf
        # https://www.nuh.com.sg/about-nuh/who-we-are
    }
    
    # Initialize the MonteCarloCapacitySimulator with the actual hospital capacities
    simulator = MonteCarloCapacitySimulator(
        csv_path="data/bor_2324_data.csv",
        hospital_capacities=actual_capacities,
        save_path="models/hospital_capacity_simulations.joblib",
        csv_save_path="models"
    )

    # Run the simulation and save in both Joblib and CSV formats
    hospital_capacity_simulation = simulator.run_simulation(num_simulations=1000)