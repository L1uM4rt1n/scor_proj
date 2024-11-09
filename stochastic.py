import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

class EmergencyResponseOptimizer:
    def __init__(self, hospital_capacity, hospital_locations, postal_code_coords, lambda_rate=2, discharge_rate=0.1, max_distance=10, C_a=10, C_d=2, C_h=5, hours=24):
        # Parameters for hospitals and Monte Carlo simulation
        self.hospital_capacity = hospital_capacity
        self.current_capacity = np.copy(hospital_capacity)
        self.hospital_locations = hospital_locations
        self.postal_code_coords = postal_code_coords
        self.lambda_rate = lambda_rate
        self.discharge_rate = discharge_rate
        self.max_distance = max_distance
        self.C_a = C_a
        self.C_d = C_d
        self.C_h = C_h
        self.hours = hours
        
        # Run Monte Carlo simulation in advance for projected capacity
        self.projected_capacity = self.monte_carlo_simulation()

        # Initialize logistic regression model
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def monte_carlo_simulation(self):
        """Simulate external admissions and discharges for each hospital over time."""
        capacities = [np.copy(self.hospital_capacity)]
        for hour in range(self.hours):
            change = poisson.rvs(self.lambda_rate, size=len(self.hospital_capacity)) - poisson.rvs(self.discharge_rate, size=len(self.hospital_capacity))
            new_capacity = np.maximum(capacities[-1] + change, 0)  # Ensure capacity is non-negative
            capacities.append(new_capacity)
        return capacities

    def train_logistic_model(self, df, target_column):
        """Train logistic regression model on historical data for emergency prediction."""
        # Incorporate code from logistic_regression.py
        df["arrtime"] = pd.to_datetime(df["arrtime"])
        df["hour"] = df["arrtime"].dt.hour
        df["2_hour_interval"] = (df["hour"] // 2) * 2
        df["consecutive_calls"] = df["call_counts"].rolling(window=15, min_periods=1).sum()
        
        # Prepare features and target
        X = df[["rssi", "hour", "2_hour_interval", "consecutive_calls"]]
        y = df[target_column]
        
        # Split and scale data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_emergency_probability(self, rssi, hour, consecutive_calls):
        """Predict the probability of a genuine emergency based on real-time inputs."""
        time_interval = (hour // 2) * 2
        X = np.array([[rssi, hour, time_interval, consecutive_calls]])
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1][0]  # Probability of genuine emergency

    def get_coordinates(self, postal_code):
        """Retrieve the geographic coordinates for a given postal code."""
        return self.postal_code_coords.get(postal_code, None)

    def calculate_distances(self, patient_coords):
        """Calculate Euclidean distances between a patient's location and each hospital."""
        # Incorporate distance calculation from Euclidean distance.py
        return np.array([euclidean(patient_coords, hospital) for hospital in self.hospital_locations])

    def transition_function(self, current_hour, assigned_hospital):
        """Update the capacity state of the assigned hospital based on projected capacity and admissions."""
        # Use Monte Carlo-projected capacity for the current hour as reference
        self.current_capacity = np.copy(self.projected_capacity[current_hour])
        
        # Adjust capacity immediately for the hospital that received the new admission
        if self.current_capacity[assigned_hospital] > 0:
            self.current_capacity[assigned_hospital] -= 1

    def optimize_assignment(self, emergency_prob, distances):
        """Optimize the assignment of an emergency to a hospital to minimize total cost."""
        num_hospitals = len(self.hospital_capacity)

        # Initialize Gurobi model
        model = gp.Model("EmergencyResponseOptimization")
        
        # First-stage decision variable: respond to emergency or not
        u1 = model.addVar(vtype=GRB.BINARY, name="u1")

        # Second-stage decision variable: assign emergency to specific hospital
        u2 = model.addVars(num_hospitals, vtype=GRB.BINARY, name="u2")

        # Define cost function
        cost_expr = self.C_a * (1 - emergency_prob) * u1
        for j in range(num_hospitals):
            if distances[j] <= self.max_distance:
                cost_expr += (self.C_d * distances[j] + self.C_h / self.hospital_capacity[j]) * u2[j]
        model.setObjective(cost_expr, GRB.MINIMIZE)

        # Constraints: Assignment and capacity constraints
        model.addConstr(gp.quicksum(u2[j] for j in range(num_hospitals)) == u1, "Assignment")
        for j in range(num_hospitals):
            model.addConstr(u2[j] <= self.current_capacity[j], f"Capacity_{j}")
            if distances[j] > self.max_distance:
                model.addConstr(u2[j] == 0, f"Distance_{j}")

        # Optimize the model
        model.optimize()

        # Retrieve optimal solution
        if model.status == GRB.OPTIMAL:
            hospital_assignment = None
            if u1.x > 0.5:  # Respond to emergency if u1 is 1
                for j in range(num_hospitals):
                    if u2[j].x > 0.5:  # Assigned hospital
                        hospital_assignment = j
            return model.objVal, hospital_assignment
        else:
            return float('inf'), None  # No feasible solution

    def make_decision(self, postal_code, rssi, hour, consecutive_calls, current_hour):
        """Full decision-making pipeline from input features to optimal hospital assignment."""
        # Step 1: Get patient coordinates from postal code
        patient_coords = self.get_coordinates(postal_code)
        if patient_coords is None:
            raise ValueError("Invalid postal code or missing coordinates.")
        
        # Step 2: Predict probability of genuine emergency
        emergency_prob = self.predict_emergency_probability(rssi, hour, consecutive_calls)

        # Step 3: Calculate distances to hospitals
        distances = self.calculate_distances(patient_coords)

        # Step 4: Optimize assignment
        optimal_cost, hospital_assignment = self.optimize_assignment(emergency_prob, distances)

        # Step 5: Apply transition function to update the assigned hospital's capacity
        if hospital_assignment is not None:
            self.transition_function(current_hour, hospital_assignment)

        # Output decision
        return {"is_emergency": emergency_prob >= 0.5, "assigned_hospital": hospital_assignment, "total_cost": optimal_cost}
