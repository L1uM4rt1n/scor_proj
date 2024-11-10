import pandas as pd
from scipy.spatial.distance import euclidean
import csv

class HospitalDistanceCalculator:
    def __init__(self, output_csv='models/distances_to_hospitals.csv', input_midpoint_csv='data/postal_midpoints.csv'):
        # Set the output path for the CSV file
        self.output_csv = output_csv
        self.input_csv = input_midpoint_csv
        self.midpoints = ""


        # Approximate coordinates for hospitals in Singapore
        self.hospital_coordinates = {
            "AH": (1.2889, 103.8030),
            "CGH": (1.3417, 103.9491),
            "KTPH": (1.4244, 103.8389),
            "NTFGH": (1.3330, 103.7468),
            "NUH(A)": (1.2958, 103.7832),
            "SKH": (1.3911, 103.8930),
            "SGH": (1.2780, 103.8345),
            "TTSH": (1.3214, 103.8454),
            "WH": (1.4250, 103.7946)
        }

    def get_midpoints(self, input_midpoint_csv):
        midpoints = {}
        with open(input_midpoint_csv, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                midpoints[row['postal_prefix']] = [float(row['LATITUDE']), float(row['LONGITUDE'])]
            self.midpoints = midpoints


    def calculate_distances(self):
        # Calculate Euclidean distances from each district midpoint to each hospital
        distances = []
        for postal_area, postal_coords in self.midpoints.items():
            for hospital, hospital_coords in self.hospital_coordinates.items():
                # Calculate Euclidean distance and convert to kilometers
                distance_km = euclidean(postal_coords, hospital_coords) * 111.11  # in kilometers
                distances.append({
                    'postal_district': postal_area,
                    'hospital': hospital,
                    'distance_km': distance_km
                })
        
        # Save distances to CSV
        distances_df = pd.DataFrame(distances)
        distances_df.to_csv(self.output_csv, index=False)
        print(f"Distances saved to {self.output_csv}")
        
    def get_distances_from_postal_code(self, postal_code):
        # Convert postal code to district
        sector = int(str(postal_code)[:2])
        
        if not sector:
            print(f"No sector found for postal code: {postal_code}")
            return None
        
        # Load distances from CSV
        distances_df = pd.read_csv(self.output_csv)
        print(distances_df.head())
        print(distances_df.info())
        
        # Fetch distances for the specific district
        district_distances = distances_df[distances_df['postal_district'] == sector]
        return district_distances

# Testing in main block
if __name__ == "__main__":
    # Output CSV file path
    output_csv = 'models/distances_to_hospitals.csv'
    
    # Initialize the HospitalDistanceCalculator
    calculator = HospitalDistanceCalculator(output_csv=output_csv, input_midpoint_csv='data/postal_midpoints.csv')
    calculator.get_midpoints('data/postal_midpoints.csv')
    print(calculator.midpoints)
    
    # Calculate and save distances to CSV
    calculator.calculate_distances()
    
    # Test get_distances_from_postal_code with postal code 68
    postal_code = 68
    distances = calculator.get_distances_from_postal_code(postal_code)
    
    # Print results
    if distances is not None:
        print(f"Distances from district of postal code {postal_code} to each hospital:")
        print(distances)
    else:
        print(f"No distances found for postal code {postal_code}.")