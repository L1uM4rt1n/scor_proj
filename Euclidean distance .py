
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np

# Load the datasets
calls_df =  pd.read_csv(r'C:\Users\User\calls.csv')
hospitals_df = pd.read_excel(r'C:\Users\User\hospital_data.xlsx', sheet_name='Sheet3')

# Approximate coordinates (latitude, longitude) for postal districts
postal_district_coords = {
    "D01": (1.2834, 103.8515),  # Marina Bay, Raffles Place
    "D02": (1.2776, 103.8433),  # Chinatown, Tanjong Pagar
    "D03": (1.2925, 103.7875),  # Queenstown, Tiong Bahru
    "D04": (1.2693, 103.8183),  # Harbourfront, Telok Blangah
    "D05": (1.2921, 103.7665),  # Clementi, Buona Vista
    "D06": (1.2897, 103.8510),  # City Hall, Clarke Quay
    "D07": (1.2997, 103.8546),  # Bugis, Rochor
    "D08": (1.3126, 103.8531),  # Little India, Serangoon Road
    "D09": (1.3048, 103.8318),  # Orchard, River Valley
    "D10": (1.3157, 103.8079),  # Holland, Bukit Timah
    "D11": (1.3278, 103.8409),  # Newton, Novena
    "D12": (1.3260, 103.8640),  # Balestier, Toa Payoh
    "D13": (1.3312, 103.8794),  # Macpherson, Potong Pasir
    "D14": (1.3197, 103.8922),  # Geylang, Paya Lebar
    "D15": (1.3028, 103.9063),  # East Coast, Marine Parade
    "D16": (1.3201, 103.9555),  # Bedok, Upper East Coast
    "D17": (1.3644, 103.9915),  # Changi
    "D18": (1.3521, 103.9439),  # Tampines, Pasir Ris
    "D19": (1.3700, 103.8967),  # Hougang, Punggol
    "D20": (1.3516, 103.8399),  # Ang Mo Kio, Bishan
    "D21": (1.3382, 103.7764),  # Clementi Park, Bukit Timah
    "D22": (1.3321, 103.7430),  # Jurong
    "D23": (1.3773, 103.7639),  # Bukit Panjang, Choa Chu Kang
    "D24": (1.4031, 103.7114),  # Lim Chu Kang
    "D25": (1.4383, 103.7857),  # Woodlands
    "D26": (1.4003, 103.8251),  # Upper Thomson
    "D27": (1.4194, 103.8273),  # Yishun, Sembawang
    "D28": (1.3968, 103.8735),  # Seletar, Yio Chu Kang
}

# Map postal sectors (first 2 digits of postal code) to districts
postal_sector_to_district = {
    '01': 'D01', '02': 'D01', '03': 'D01', '04': 'D01', '05': 'D01', '06': 'D01',
    '07': 'D02', '08': 'D02',
    '14': 'D03', '15': 'D03', '16': 'D03',
    '09': 'D04', '10': 'D04',
    '11': 'D05', '12': 'D05', '13': 'D05',
    '17': 'D06',
    '18': 'D07', '19': 'D07',
    '20': 'D08', '21': 'D08',
    '22': 'D09', '23': 'D09',
    '24': 'D10', '25': 'D10', '26': 'D10', '27': 'D10',
    '28': 'D11', '29': 'D11', '30': 'D11',
    '31': 'D12', '32': 'D12', '33': 'D12',
    '34': 'D13', '35': 'D13', '36': 'D13', '37': 'D13',
    '38': 'D14', '39': 'D14', '40': 'D14', '41': 'D14',
    '42': 'D15', '43': 'D15', '44': 'D15', '45': 'D15',
    '46': 'D16', '47': 'D16', '48': 'D16',
    '49': 'D17', '50': 'D17', '81': 'D17',
    '51': 'D18', '52': 'D18',
    '53': 'D19', '54': 'D19', '55': 'D19', '82': 'D19',
    '56': 'D20', '57': 'D20',
    '58': 'D21', '59': 'D21',
    '60': 'D22', '61': 'D22', '62': 'D22', '63': 'D22', '64': 'D22',
    '65': 'D23', '66': 'D23', '67': 'D23', '68': 'D23',
    '69': 'D24', '70': 'D24', '71': 'D24',
    '72': 'D25', '73': 'D25',
    '77': 'D26', '78': 'D26',
    '75': 'D27', '76': 'D27',
    '79': 'D28', '80': 'D28',
}

# Extract first 2 digits of postal code to identify sector
calls_df['postal_sector'] = calls_df['post_code'].astype(str).str[:2]
calls_df['postal_district'] = calls_df['postal_sector'].map(postal_sector_to_district)

# Map postal district to coordinates
calls_df['latitude'] = calls_df['postal_district'].map(lambda x: postal_district_coords[x][0] if x in postal_district_coords else np.nan)
calls_df['longitude'] = calls_df['postal_district'].map(lambda x: postal_district_coords[x][1] if x in postal_district_coords else np.nan)

# Drop rows where coordinates could not be assigned
calls_df = calls_df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

# Display the first few rows to confirm assignments
calls_df[['post_code', 'postal_sector', 'postal_district', 'latitude', 'longitude']].head()

# Checking the column names and sample data of both DataFrames for consistency.

# Display column names and first few rows of `calls_df` to verify structure
calls_df_info = calls_df.head(), calls_df.columns

# Display column names and first few rows of `hospital_data_df` to verify structure
hospital_data_df_info = hospitals_df.head(), hospitals_df.columns

calls_df_info, hospital_data_df_info

district_midpoints = (
    calls_df
    .groupby('postal_district')[['latitude', 'longitude']]
    .mean()
    .reset_index()
    .rename(columns={'latitude': 'mid_latitude', 'longitude': 'mid_longitude'})
)

# Display the calculated midpoints
print("Midpoints for each district:")
print(district_midpoints)

# Approximate latitude and longitude values for hospitals in Singapore for distance calculations.
# Rough estimates to represent the general locations of each hospital.

hospital_coordinates = {
    "Alexandra Hospital": (1.2889, 103.8030),
    "Singapore General Hospital": (1.2780, 103.8345),
    "National University Hospital": (1.2958, 103.7832),
    "Raffles Hospital": (1.3076, 103.8607),
    "Farrer Park Hospital": (1.3122, 103.8496),
    "Tan Tock Seng Hospital": (1.3214, 103.8454),
    "Mount Elizabeth Hospital": (1.3051, 103.8355),
    "Gleneagles Hospital": (1.3075, 103.8194),
    "KK Women's and Children's Hospital": (1.3112, 103.8451),
    "Changi General Hospital": (1.3417, 103.9491),
    "Khoo Teck Puat Hospital": (1.4244, 103.8389),
    "Ng Teng Fong General Hospital": (1.3330, 103.7468),
    "Jurong Community Hospital": (1.3328, 103.7467),
    "Sengkang General Hospital": (1.3911, 103.8930),
    "Yishun Community Hospital": (1.4267, 103.8365),
    "Mount Alvernia Hospital": (1.3413, 103.8363),
    "Parkway East Hospital": (1.3205, 103.9123),
    "Bright Vision Hospital": (1.3695, 103.8742),
    "St. Luke's Hospital": (1.3459, 103.7374),
    "Thomson Medical Centre": (1.3203, 103.8435)
}


hospitals_df = pd.DataFrame([
    {'hospital': name, 'latitude': coords[0], 'longitude': coords[1]} 
    for name, coords in hospital_coordinates.items()
])

#Calculate Euclidean distances between each district midpoint and each hospital
distances = []

for _, district in district_midpoints.iterrows():
    district_coords = (district['mid_latitude'], district['mid_longitude'])
    
    for _, hospital in hospitals_df.iterrows():
        hospital_coords = (hospital['latitude'], hospital['longitude'])
        
        # Calculate Euclidean distance and convert to meters (approximately 111,000 meters per degree)
        distance_km = euclidean(district_coords, hospital_coords) * 111  # now in kilometers
        distances.append({
            'postal_district': district['postal_district'],
            'hospital': hospital['hospital'],
            'distance_km': distance_km
        })

# Convert distances list to DataFrame for further analysis
distances_df = pd.DataFrame(distances)

# Display the first few rows of distances DataFrame
print("Distances from district midpoints to hospitals:")
print(distances_df.head())