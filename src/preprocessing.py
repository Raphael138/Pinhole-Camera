import numpy as np

def label_to_color(labels):
    # Define a color map for each label (RGB values between 0 and 1)
    color_map = {
        0: [1.0, 0.0, 0.0],          # camera - hard red
        1000: [0.0, 0.0, 0.0],       # to_label - black
        1001: [0.5, 0.5, 0.5],       # undet - gray
        1002: [0.7, 0.0, 0.0],       # linear_misc - light red
        1003: [0.0, 1.0, 0.0],       # surf_misc - green
        1004: [0.0, 0.0, 1.0],       # scatter_misc - blue
        1005: [1.0, 1.0, 0.0],       # artifact - yellow
        1100: [1.0, 0.5, 0.0],       # default_wire - orange
        1101: [1.0, 0.0, 1.0],       # wire_bundle - magenta
        1102: [0.5, 0.0, 1.0],       # isolated_wire - purple
        1103: [0.0, 1.0, 1.0],       # utility_pole - cyan
        1104: [0.5, 1.0, 0.0],       # crossarm - lime green
        1105: [1.0, 0.5, 0.5],       # support_wire - pink
        1106: [0.0, 0.5, 1.0],       # support_pole - sky blue
        1107: [0.5, 0.5, 1.0],       # lamp_support - light blue
        1108: [1.0, 0.0, 0.5],       # transformer - crimson
        1109: [0.5, 1.0, 1.0],       # fire_hydrant - pale cyan
        1110: [0.0, 0.5, 0.0],       # post - dark green
        1111: [0.5, 0.5, 0.0],       # sign - olive
        1112: [0.5, 0.0, 0.0],       # pylon - brown
        1113: [1.0, 0.75, 0.0],      # bench - gold
        1114: [0.75, 0.75, 0.75],    # lamp - silver
        1115: [1.0, 0.25, 0.25],     # traffic_lights - light red
        1116: [0.75, 0.25, 0.75],    # traffic_lights_support - orchid
        1117: [0.25, 0.5, 0.25],     # garbage - forest green
        1118: [0.25, 0.75, 0.5],     # crosswalk_light - sea green
        1119: [0.5, 0.25, 0.75],     # parking_meter - plum
        1200: [1.0, 0.6, 0.0],       # load_bearing - dark orange
        1201: [0.8, 0.4, 0.2],       # cliff - copper
        1202: [0.6, 0.3, 0.0],       # ground - bronze
        1203: [0.4, 0.2, 0.1],       # paved_road - deep brown
        1204: [0.8, 0.8, 0.0],       # trail - mustard yellow
        1205: [0.9, 0.9, 0.7],       # curb - beige
        1206: [0.3, 0.6, 0.3],       # walkway - pastel green
        1207: [0.0, 0.3, 0.5],       # guardrail - deep blue
        1300: [0.0, 1.0, 0.5],       # foliage - light green
        1301: [0.4, 0.8, 0.0],       # grass - grass green
        1302: [0.6, 0.4, 0.2],       # small_trunk - light brown
        1303: [0.4, 0.2, 0.1],       # large_trunk - dark brown
        1304: [0.2, 0.6, 0.2],       # thin_branch - fern green
        1305: [0.1, 0.4, 0.1],       # thick_branch - forest brown
        1306: [0.5, 1.0, 0.0],       # shrub - lime
        1400: [0.5, 0.5, 1.0],       # facade - light blue
        1401: [0.3, 0.3, 0.3],       # wall - dark gray
        1402: [0.7, 0.7, 0.7],       # stairs - light gray
        1403: [1.0, 0.7, 0.5],       # door - light orange
        1404: [0.5, 0.8, 1.0],       # window - pale blue
        1405: [0.8, 0.5, 0.3],       # chimney - brick red
        1406: [0.9, 0.8, 0.6],       # roof - sand
        1407: [0.4, 0.6, 0.8],       # chainlinkfence - steel blue
        1408: [0.6, 0.4, 0.2],       # fence - wooden brown
        1409: [0.9, 0.9, 0.7],       # gate - light beige
        1410: [0.7, 0.7, 0.7],       # ceiling - light gray
        1411: [0.9, 0.9, 0.6],       # facade_ledge - sandstone
        1412: [0.3, 0.2, 0.5],       # column - violet
        1413: [0.8, 0.2, 0.0],       # mailbox - bright red
        1500: [1.0, 0.8, 0.8],       # human - light pink
        1501: [0.5, 0.0, 0.5],       # vehicle - purple
        1600: [0.6, 0.4, 0.3],       # rock - earthy brown
        1601: [0.9, 0.7, 0.3]        # concertina_wire - goldenrod
    }
    
    # Get colors for each label
    colors = np.array([color_map.get(label, [0.5, 0.5, 0.5]) for label in labels])  # Default to gray if label not in map
    return colors

# Function to load the point cloud file
def load_oakland_data_with_labels(file_path):
    # Load the point cloud data
    data = np.loadtxt(file_path)
    points = data[:, :3] 
    labels = data[:, 3].astype(int)
    return points, labels

# Function to load and append multiple files
def load_multiple_oakland_files(file_paths):
    all_points = []
    all_labels = []
    
    for file_path in file_paths:
        points, labels = load_oakland_data_with_labels(file_path)
        all_points.append(points)
        all_labels.append(labels)
    
    # Concatenate all points and labels
    all_points = np.vstack(all_points)
    all_labels = np.concatenate(all_labels)
    
    return all_points, all_labels