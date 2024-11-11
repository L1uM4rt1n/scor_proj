import joblib
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import csv
from datetime import datetime
import multiprocessing

from euclidean import HospitalDistanceCalculator
from monte_carlo import MonteCarloCapacitySimulator

distance_calculator = HospitalDistanceCalculator()
monte_carlo_simulator = MonteCarloCapacitySimulator(csv_path="data/bor_2324_data.csv")

hospital_capacity_simulation = monte_carlo_simulator.load_simulation()

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

def initialize_percentage_capacity(time_step):
    initial_percentages = {
        hospital: monte_carlo_simulator.check_capacity_at_time(
            hospital_capacity_simulation, hospital, time_step
        ) for hospital in hospital_capacity_simulation.keys()
    }
    return initial_percentages

def update_percentage_capacity(current_percentage_capacity, assigned_hospital):

    # Define actual hospital bed capacities
    print(f"Old occupancy {assigned_hospital}: {current_percentage_capacity[assigned_hospital]}")
    absolute_occupancy = current_percentage_capacity[assigned_hospital] * actual_capacities[assigned_hospital]
    absolute_occupancy += 1
    current_percentage_capacity[assigned_hospital] = min(1, absolute_occupancy / actual_capacities[assigned_hospital])
    print(f"New occupancy {assigned_hospital}: {current_percentage_capacity[assigned_hospital]}")
    return current_percentage_capacity

# Cost functions with scaling factors for balanced optimization
NON_EMERGENCY_WEIGHT = 200
DISTANCE_WEIGHT = 2
CAPACITY_WEIGHT = 50

def distance_weight(distance):
    return DISTANCE_WEIGHT * (distance ** 2)

def capacity_weight(percentage_capacity):
    return CAPACITY_WEIGHT * (percentage_capacity ** 2)

def real_time_multi_stage_optimization(emergency_data, max_distance, time_horizon=12):
    model = gp.Model("RealTimeMultiStageOptimization")

    # Performance parameters for Gurobi
    num_cores = multiprocessing.cpu_count()
    model.setParam('Threads', num_cores)
    model.setParam('MIPFocus', 1)
    model.setParam('Presolve', 2)
    model.setParam('Heuristics', 0.5)
    model.setParam('TimeLimit', 600)

    u_i1 = {}
    u_ij2 = {}
    p_real_emergency_dict = {}  # Dictionary to store precomputed emergency probabilities
    M = 1  # Big-M constant (adjustable if needed)

    # Iterate over actual time intervals (0, 2, ..., 22)
    for t in range(0, time_horizon * 2, 2):
        time_step_index = t // 2
        current_percentage_capacity = initialize_percentage_capacity(time_step_index)

        for call in emergency_data:
            iot_lora_id = call['iot_lora_id']
            call_index = call['call_index']
            two_hour = call['two_hour']
            post_code = call['post_code']
            rssi = call['rssi']
            call_id = (iot_lora_id, call_index)
            recording = int(call['recording'])

            # Calculate and store the probability of an emergency only if the incident's `two_hour` matches `t`
            if two_hour == t:
                p_real_emergency_dict[call_id] = recording

            # Define decision variable for emergency assignment
            u_i1[call_id] = model.addVar(vtype=GRB.BINARY, name=f"u_i1_{iot_lora_id}_{call_index}_t{time_step_index}")

                        # Big-M constraint: if p_real_emergency > 0.5, enforce u_i1 = 1
            if call_id in p_real_emergency_dict:
                p_real_emergency = p_real_emergency_dict[call_id]
                model.addConstr(u_i1[call_id] * M >= p_real_emergency - 0.5, name=f"EmergencyAssign_{call_id}")

            for hospital in hospital_capacity_simulation.keys():
                u_ij2[(call_id, hospital, time_step_index)] = model.addVar(
                    vtype=GRB.BINARY, name=f"u_ij2_{iot_lora_id}_{call_index}_{hospital}_t{time_step_index}")

                # Distance constraint
                distances = distance_calculator.get_distances_from_postal_code(post_code)
                distance_row = distances[distances['hospital'] == hospital]
                if not distance_row.empty:
                    distance_km = distance_row['distance_km'].values[0]
                    if distance_km > max_distance:
                        model.addConstr(u_ij2[(call_id, hospital, time_step_index)] == 0,
                                        f"MaxDistance_{iot_lora_id}_{call_index}_{hospital}_t{time_step_index}")

            # Capacity constraint based on absolute values
            for hospital in hospital_capacity_simulation.keys():
                absolute_capacity = current_percentage_capacity[hospital] * actual_capacities[hospital]
                model.addConstr(
                    gp.quicksum(u_ij2[(call_id, hospital, time_step_index)]
                                for call in emergency_data if call['two_hour'] == two_hour) <= absolute_capacity,
                    f"Capacity_{hospital}_t{time_step_index}"
                )

            # Enforce that if `u_i1 = 1`, exactly one hospital must be assigned
            model.addConstr(
                gp.quicksum(u_ij2[(call_id, hospital, time_step_index)]
                            for hospital in hospital_capacity_simulation.keys()) == u_i1[call_id],
                f"SingleAssignment_{iot_lora_id}_{call_index}_t{time_step_index}"
            )

    # Objective function
    objective_terms = [] 
    for t in range(0, time_horizon * 2, 2):
        time_step_index = t // 2
        for call in emergency_data:
            iot_lora_id = call['iot_lora_id']
            call_index = call['call_index']
            call_id = (iot_lora_id, call_index)
            two_hour = call['two_hour']

            # Only evaluate this call if it occurred at the current time step
            if two_hour != t:
                continue

            # Use the precomputed probability of an emergency if available
            p_real_emergency = p_real_emergency_dict.get(call_id, 0)
            non_emergency_cost = (1 - p_real_emergency) * u_i1[call_id] * NON_EMERGENCY_WEIGHT
            print(call_id, p_real_emergency, t)
            objective_terms.append(non_emergency_cost)

            post_area = call['post_area']
            distances = distance_calculator.get_distances_from_postal_code(post_area)
            print(f"ui1 probability: {u_i1[call_id]}")
            print(f"postarea: {post_area}")
            print(distances)
            for hospital in hospital_capacity_simulation.keys():
                distance_row = distances[distances['hospital'] == hospital]
                if not distance_row.empty:
                    distance_km = distance_row['distance_km'].values[0]
                    distance_cost = distance_weight(distance_km) * u_ij2[(call_id, hospital, time_step_index)]
                    print(f"Hospital: {hospital}: Occupancy {current_percentage_capacity[hospital]}")
                    capacity_cost = capacity_weight(current_percentage_capacity[hospital]) * u_ij2[(call_id, hospital, time_step_index)]
                    objective_terms.append(distance_cost + capacity_cost)

    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        assignments = {}
        for call in emergency_data:
            iot_lora_id = call['iot_lora_id']
            call_index = call['call_index']
            call_id = (iot_lora_id, call_index)
            two_hour = call['two_hour']  # Retrieve the two-hour timing of the call

            for hospital in hospital_capacity_simulation.keys():
                for t in range(0, time_horizon * 2, 2):
                    time_step_index = t // 2

                    # Only make an assignment if the call's two-hour timing matches the current time step `t`
                    if two_hour == t and u_ij2[(call_id, hospital, time_step_index)].x > 0.5:
                        # Save the assignment result along with the precomputed emergency probability
                        assignments[(call_id, t)] = (hospital, p_real_emergency_dict.get(call_id, 0))

                        # Transition function
                        current_percentage_capacity = update_percentage_capacity(current_percentage_capacity, hospital)
                        
        return assignments, model.objVal
    else:
        print("No optimal solution found.")
        return None, None



if __name__ == "__main__":
    import csv
    import random
    from datetime import datetime

    # Initialize an empty list to store each call as a dictionary
    emergency_data = []

    # Load emergency data from CSV file
    with open("data/final_calls_data.csv", mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)  # Read all rows into a list
        random.shuffle(rows)  # Shuffle the rows to randomize

        for idx, row in enumerate(rows[:10]):  # Get the first 10 after shuffling
            # Parse and process `arrtime` to get the hour, rounding down to the nearest even hour
            arrtime = datetime.strptime(row['arrtime'], "%Y-%m-%d %H:%M:%S")
            two_hour = arrtime.hour - (arrtime.hour % 2)

            # Append each row as a separate entry in the list with a unique call index
            emergency_data.append({
                'iot_lora_id': int(row['iot_lora_id']),
                'call_index': idx,
                'two_hour': two_hour,
                'post_code': row['post_code'],
                'rssi': float(row['rssi']),
                'post_area': row['district'],
                'recording': row['recording']
            })

    print(emergency_data)

    # Run the optimization model with the list of calls
    max_distance = 20
    time_horizon = 12

    assignments, total_cost = real_time_multi_stage_optimization(emergency_data, max_distance, time_horizon)

    if assignments is not None:
        print("Assignments:", assignments)
        print("Total cost:", total_cost)



