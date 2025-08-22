# Valid towns in Singapore
VALID_TOWNS = [
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
    'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
    'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
    'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
    'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
    'TOA PAYOH', 'WOODLANDS', 'YISHUN'
]

# Valid flat types
VALID_FLAT_TYPES = [
    '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 
    'EXECUTIVE', 'MULTI-GENERATION'
]

# Valid storey ranges
VALID_STOREY_RANGES = [
    '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15',
    '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30',
    '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45',
    '46 TO 48', '49 TO 51'
]

# Valid flat models
VALID_FLAT_MODELS = [
    'Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
    'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
    'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
    'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
    'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'
]

# Default floor areas by flat type
DEFAULT_FLOOR_AREAS = {
    '1 ROOM': 35,
    '2 ROOM': 45,
    '3 ROOM': 70,
    '4 ROOM': 90,
    '5 ROOM': 110,
    'EXECUTIVE': 140,
    'MULTI-GENERATION': 115
}

def get_hdb_context():
    return f"""
Singapore HDB Context:

VALID TOWNS ({len(VALID_TOWNS)} total):
{', '.join(VALID_TOWNS)}

FLAT TYPES:
{', '.join(VALID_FLAT_TYPES)}

TYPICAL FLOOR AREAS:
{', '.join([f"{k}: {v} sqm" for k, v in DEFAULT_FLOOR_AREAS.items()])}

STOREY RANGES:
{', '.join(VALID_STOREY_RANGES)}

COMMON FLAT MODELS:
{', '.join(VALID_FLAT_MODELS)}

TYPICAL CHARACTERISTICS:
- Remaining lease: 60-90+ years (average: 65 years)
- Lease commenced: 1960s-2020s (common: 1980s-1990s)
- Most flats built: 1980-2000

FOR NEW BTO ENQUIRIES:
New BTO flats are newly constructed and typically have:
- Remaining lease: 100 years
- Lease commenced: Current year
""" 

