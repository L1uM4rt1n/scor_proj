import pandas as pd
from scipy.spatial.distance import euclidean

class HospitalDistanceCalculator:
    def __init__(self, output_csv):
        # Set the output path for the CSV file
        self.output_csv = output_csv

        # Approximate coordinates (latitude, longitude) for postal districts
        self.postal_district_coords = {
            "D01": (1.2834, 103.8515), "D02": (1.2776, 103.8433), "D03": (1.2925, 103.7875),
            "D04": (1.2693, 103.8183), "D05": (1.2921, 103.7665), "D06": (1.2897, 103.8510),
            "D07": (1.2997, 103.8546), "D08": (1.3126, 103.8531), "D09": (1.3048, 103.8318),
            "D10": (1.3157, 103.8079), "D11": (1.3278, 103.8409), "D12": (1.3260, 103.8640),
            "D13": (1.3312, 103.8794), "D14": (1.3197, 103.8922), "D15": (1.3028, 103.9063),
            "D16": (1.3201, 103.9555), "D17": (1.3644, 103.9915), "D18": (1.3521, 103.9439),
            "D19": (1.3700, 103.8967), "D20": (1.3516, 103.8399), "D21": (1.3382, 103.7764),
            "D22": (1.3321, 103.7430), "D23": (1.3773, 103.7639), "D24": (1.4031, 103.7114),
            "D25": (1.4383, 103.7857), "D26": (1.4003, 103.8251), "D27": (1.4194, 103.8273),
            "D28": (1.3968, 103.8735)
        }

        # Map postal sectors (first 2 digits of postal code) to districts
        self.postal_sector_to_district = {
            '01': 'D01', '02': 'D01', '03': 'D01', '04': 'D01', '05': 'D01', '06': 'D01',
            '07': 'D02', '08': 'D02', '14': 'D03', '15': 'D03', '16': 'D03',
            '09': 'D04', '10': 'D04', '11': 'D05', '12': 'D05', '13': 'D05',
            '17': 'D06', '18': 'D07', '19': 'D07', '20': 'D08', '21': 'D08',
            '22': 'D09', '23': 'D09', '24': 'D10', '25': 'D10', '26': 'D10', '27': 'D10',
            '28': 'D11', '29': 'D11', '30': 'D11', '31': 'D12', '32': 'D12', '33': 'D12',
            '34': 'D13', '35': 'D13', '36': 'D13', '37': 'D13', '38': 'D14', '39': 'D14',
            '40': 'D14', '41': 'D14', '42': 'D15', '43': 'D15', '44': 'D15', '45': 'D15',
            '46': 'D16', '47': 'D16', '48': 'D16', '49': 'D17', '50': 'D17', '81': 'D17',
            '51': 'D18', '52': 'D18', '53': 'D19', '54': 'D19', '55': 'D19', '82': 'D19',
            '56': 'D20', '57': 'D20', '58': 'D21', '59': 'D21', '60': 'D22', '61': 'D22',
            '62': 'D22', '63': 'D22', '64': 'D22', '65': 'D23', '66': 'D23', '67': 'D23',
            '68': 'D23', '69': 'D24', '70': 'D24', '71': 'D24', '72': 'D25', '73': 'D25',
            '77': 'D26', '78': 'D26', '75': 'D27', '76': 'D27', '79': 'D28', '80': 'D28'
        }

        # Approximate coordinates for hospitals in Singapore
        self.hospital_coordinates = {
            "Alexandra Hospital": (1.2889, 103.8030),
            "Changi General Hospital": (1.3417, 103.9491),
            "Jurong Community Hospital": (1.3328, 103.7467),
            "Khoo Teck Puat Hospital": (1.4244, 103.8389),
            "Ng Teng Fong General Hospital": (1.3330, 103.7468),
            "National University Hospital": (1.2958, 103.7832),
            "Sengkang General Hospital": (1.3911, 103.8930),
            "Singapore General Hospital": (1.2780, 103.8345),
            "Tan Tock Seng Hospital": (1.3214, 103.8454),
            "Woodlands Health": (1.4250, 103.7946)
        }

    def calculate_distances(self):
        # Calculate Euclidean distances from each district midpoint to each hospital
        distances = []
        for district, district_coords in self.postal_district_coords.items():
            for hospital, hospital_coords in self.hospital_coordinates.items():
                # Calculate Euclidean distance and convert to kilometers
                distance_km = euclidean(district_coords, hospital_coords) * 111.11  # in kilometers
                distances.append({
                    'postal_district': district,
                    'hospital': hospital,
                    'distance_km': distance_km
                })
        
        # Save distances to CSV
        distances_df = pd.DataFrame(distances)
        distances_df.to_csv(self.output_csv, index=False)
        print(f"Distances saved to {self.output_csv}")
        
    def get_distances_from_postal_code(self, postal_code):
        # Convert postal code to district
        sector = str(postal_code)[:2]
        district = self.postal_sector_to_district.get(sector)
        
        if not district:
            print(f"No district found for postal code: {postal_code}")
            return None
        
        # Load distances from CSV
        distances_df = pd.read_csv(self.output_csv)
        
        # Fetch distances for the specific district
        district_distances = distances_df[distances_df['postal_district'] == district]
        return district_distances

# Testing in main block
if __name__ == "__main__":
    # Output CSV file path
    output_csv = 'models/distances_to_hospitals.csv'
    
    # Initialize the HospitalDistanceCalculator
    calculator = HospitalDistanceCalculator(output_csv=output_csv)
    
    # Calculate and save distances to CSV
    calculator.calculate_distances()
    
    # Test get_distances_from_postal_code with postal code 68
    postal_code = '68'
    distances = calculator.get_distances_from_postal_code(postal_code)
    
    # Print results
    if distances is not None:
        print(f"Distances from district of postal code {postal_code} to each hospital:")
        print(distances)
    else:
        print(f"No distances found for postal code {postal_code}.")