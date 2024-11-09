import joblib
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Import classes from provided utility files
from euclidean import HospitalDistanceCalculator
from monte_carlo import MonteCarloCapacitySimulator
from regression_model import LogisticRegressionUtility

# Define actual hospital bed capacities
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
}

# Load utility objects
distance_calculator = HospitalDistanceCalculator(output_csv="models/distances_to_hospitals.csv")
monte_carlo_simulator = MonteCarloCapacitySimulator(csv_path="data/bor_2324_data.csv")
logistic_util = LogisticRegressionUtility()

# Load pre-trained logistic regression model and scaler
logistic_util.load_model()

# Load pre-simulated hospital capacities from joblib
hospital_capacity_simulation = monte_carlo_simulator.load_simulation()

def initialize_percentage_capacity(time_step):
    """Initialize hospital capacities as percentages at the start of each time step."""
    initial_percentages = {hospital: monte_carlo_simulator.check_capacity_at_time(
        hospital_capacity_simulation, hospital, time_step) for hospital in hospital_capacity_simulation.keys()}
    return initial_percentages

def update_percentage_capacity(current_percentage_capacity, assigned_hospital):
    """Update the capacity percentage for the assigned hospital after making an assignment."""
    # Convert percentage to absolute capacity for the assigned hospital
    absolute_capacity = current_percentage_capacity[assigned_hospital] * actual_capacities[assigned_hospital]
    absolute_capacity -= 1  # Decrement by 1 bed
    current_percentage_capacity[assigned_hospital] = max(0, absolute_capacity / actual_capacities[assigned_hospital])
    return current_percentage_capacity

def real_time_multi_stage_optimization(emergency_data, p_threshold, max_distance, time_horizon=12):
    # Initialize the Gurobi model
    model = gp.Model("RealTimeMultiStageOptimization")

    # First- and second-stage decision variables
    u_i1 = {}
    u_ij2 = {}

    # Loop through each time step for multi-stage optimization
    for t in range(time_horizon):
        current_percentage_capacity = initialize_percentage_capacity(t)

        # Create variables and constraints dynamically for each call
        for call in emergency_data.itertuples():
            p_real_emergency = logistic_util.predict_probability(call.rssi, call.hour)
            u_i1[call.call_id] = model.addVar(vtype=GRB.BINARY, name=f"u_i1_{call.call_id}_t{t}")

            for hospital in hospital_capacity_simulation.keys():
                u_ij2[(call.call_id, hospital, t)] = model.addVar(vtype=GRB.BINARY, name=f"u_ij2_{call.call_id}_{hospital}_t{t}")
                
                # Distance constraint
                distances = distance_calculator.get_distances_from_postal_code(call.postal_code)
                distance_row = distances[distances['hospital'] == hospital]
                if not distance_row.empty and distance_row['distance_km'].values[0] > max_distance:
                    model.addConstr(u_ij2[(call.call_id, hospital, t)] == 0, f"MaxDistance_{call.call_id}_{hospital}_t{t}")

            # Capacity constraints
            for hospital in hospital_capacity_simulation.keys():
                absolute_capacity = current_percentage_capacity[hospital] * actual_capacities[hospital]
                model.addConstr(
                    gp.quicksum(u_ij2[(call.call_id, hospital, t)] for call in emergency_data.itertuples()) <= absolute_capacity,
                    f"Capacity_{hospital}_t{t}"
                )

            # Single assignment per emergency constraint
            model.addConstr(
                gp.quicksum(u_ij2[(call.call_id, hospital, t)] for hospital in hospital_capacity_simulation.keys()) == u_i1[call.call_id],
                f"SingleAssignment_{call.call_id}_t{t}"
            )

            # Update capacity within the same time step
            for hospital in hospital_capacity_simulation.keys():
                for call in emergency_data.itertuples():
                    if u_ij2[(call.call_id, hospital, t)].x > 0.5:
                        current_percentage_capacity = update_percentage_capacity(current_percentage_capacity, hospital)

    # Objective function
    objective_terms = []
    for t in range(time_horizon):
        for call in emergency_data.itertuples():
            p_real_emergency = logistic_util.predict_probability(call.rssi, call.hour)
            non_emergency_cost = (1 - p_real_emergency) * u_i1[call.call_id]
            objective_terms.append(non_emergency_cost)

            for hospital in hospital_capacity_simulation.keys():
                distances = distance_calculator.get_distances_from_postal_code(call.postal_code)
                distance_row = distances[distances['hospital'] == hospital]
                if not distance_row.empty:
                    distance_km = distance_row['distance_km'].values[0]
                    distance_cost = distance_weight(distance_km) * u_ij2[(call.call_id, hospital, t)]
                    capacity_cost = capacity_weight(current_percentage_capacity[hospital]) * u_ij2[(call.call_id, hospital, t)]
                    objective_terms.append(distance_cost + capacity_cost)

    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
    model.optimize()

    assignments = {}
    for call in emergency_data.itertuples():
        for hospital in hospital_capacity_simulation.keys():
            for t in range(time_horizon):
                if u_ij2[(call.call_id, hospital, t)].x > 0.5:
                    assignments[(call.call_id, t)] = hospital

    total_cost = model.objVal
    return assignments, total_cost

