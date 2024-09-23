import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1)) # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9000.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.vx = (self.Z1[0] - self.Z2[0]) / dt
            self.vy = (self.Z1[1] - self.Z2[1]) / dt
            self.vz = (self.Z1[2] - self.Z2[2]) / dt
            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt*dt)/2.0
        T_3 = (dt*dt*dt)/3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt              
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            measurements.append((r, az, el, mt))
    return measurements

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el

def cart2sph2(x: float, y: float, z: float, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))

        if x[i] > 0.0:                
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]       

        az[i] = az[i] * 180 / 3.14 

        if az[i] < 0.0:
            az[i] = 360 + az[i]

        if az[i] > 360:
            az[i] = az[i] - 360   
      
    return r, az, el

def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def generate_hypotheses(tracks, reports):
    hypotheses = []
    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            if i == j:
                hypotheses.append([(i, j)])
    return hypotheses

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            track = tracks[track_idx]
            report = reports[report_idx]
            residual = np.array(report) - np.array(track)
            prob *= np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
        probabilities.append(prob)
    return probabilities

def get_association_weights(hypotheses, probabilities, num_tracks):
    # Initialize the association_weights list with empty lists for each track
    association_weights = [[] for _ in range(num_tracks)]

    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if track_idx < len(association_weights):
                association_weights[track_idx].append((report_idx, prob))
            else:
                print(f"Error: track_idx {track_idx} is out of range.")
    
    return association_weights

# New form_clusters_via_association function
def mahalanobis_distance(track, report, cov_inv):
    residual = np.array(report) - np.array(track)
    distance = np.dot(np.dot(residual.T, cov_inv), residual)
    return distance

def form_clusters_via_association(tracks, reports, kalman_filter, chi2_threshold):
    association_list = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])  # 3x3 covariance matrix for position only

    # Step 1: Create associations based on Mahalanobis distance or chi-square test
    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            distance = mahalanobis_distance(track, report, cov_inv)
            print("check the distance",distance)
            if distance < chi2_threshold:
                association_list.append((i, j))
                print(f"Track {i} associated with Report {j}, Mahalanobis distance: {distance:.4f}")

    # Step 2: Form clusters of tracks and reports
    clusters = []
    while association_list:
        cluster_tracks = set()
        cluster_reports = set()
        stack = [association_list.pop(0)]
        
        while stack:
            track_idx, report_idx = stack.pop()
            cluster_tracks.add(track_idx)
            cluster_reports.add(report_idx)
            new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
            for assoc in new_assoc:
                if assoc not in stack:
                    stack.append(assoc)
            association_list = [assoc for assoc in association_list if assoc not in new_assoc]
        
        # Append clusters as a tuple of two lists (tracks, reports)
        clusters.append((list(cluster_tracks), [reports[r] for r in cluster_reports]))

    print("Clusters formed:", clusters)
    return clusters

def select_best_report(cluster_tracks, cluster_reports, kalman_filter):
    """
    Selects the best report based on the highest association weight.
    Returns the selected report and its corresponding track index.
    """
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])  # 3x3 covariance matrix for position only

    best_report = None
    best_track_idx = None
    max_weight = -np.inf

    # Calculate association weights
    for i, track in enumerate(cluster_tracks):
        for j, report in enumerate(cluster_reports):
            residual = np.array(report) - np.array(track)
            weight = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
            if weight > max_weight:
                max_weight = weight
                best_report = report
                best_track_idx = i

    return best_track_idx, best_report

# Check if the Doppler correlation is within the threshold
def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    return abs(doppler_1 - doppler_2) < doppler_threshold

# Check if the distance is within the range threshold
def range_gate(distance, range_threshold):
    return distance < range_threshold

# Get the next available track ID from the track list
def get_next_track_id(track_id_list):
    for idx, track in enumerate(track_id_list):
        if track['state'] == 'free':
            track_id_list[idx]['state'] = 'occupied'
            return track['id'], idx
    new_id = len(track_id_list) + 1
    track_id_list.append({'id': new_id, 'state': 'occupied'})
    return new_id, len(track_id_list) - 1

# Release a track ID by marking it as free
def release_track_id(track_id_list, idx):
    track_id_list[idx]['state'] = 'free'

# Main function for initializing tracks with updated state logic
def initialize_tracks(measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold, mode):
    tracks = []
    track_id_list = []
    miss_counts = {}
    hit_counts = {}
    tentative_ids = {}
    firm_ids = set()
    state_map = {}  # Keeps track of the state of each track ID

    # Define the state progression based on the firm_threshold
    state_progression = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']    }
    progression_states = state_progression[firm_threshold]
    # Helper function to determine miss threshold based on the current state
    def get_miss_threshold(current_state, firm_threshold):
        if current_state is None:
            return firm_threshold  # Fallback, though this should not happen
        if current_state.startswith('Poss'):
            return 1  # For Poss states, delete after 1 miss
        elif current_state.startswith('Tentative'):
            return 2 if firm_threshold == 3 else 3  # Tentative, delete after 2 or 3 misses
        elif current_state == 'Firm':
            return 3 if firm_threshold == 3 else 5  # Firm, delete after 3 or 5 misses
    for i, measurement in enumerate(measurements):
        measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
        measurement_doppler = measurement[3]
        measurement_time = measurement[4]
        assigned = False
        # Iterate through existing tracks and attempt to assign the new measurement
        for track_id, track in enumerate(tracks):
            if not track:  # Skip tracks that have been released (set to None)
                continue
            last_measurement = track['measurements'][-1][0]  # Only extract the measurement part
            last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
            last_doppler = last_measurement[3]
            last_time = last_measurement[4]
            distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
            doppler_correlated = doppler_correlation(measurement_doppler, last_doppler, doppler_threshold)
            range_satisfied = range_gate(distance, range_threshold)
            time_diff = measurement_time - last_time
            if doppler_correlated and range_satisfied and time_diff <= time_threshold:
                # Track ID assignment logic
                if track_id not in firm_ids:
                    if track_id in tentative_ids:
                        hit_counts[track_id] += 1
                        miss_counts[track_id] = 0  # Reset miss count on hit                        
                        # Update the state based on hit counts
                        if hit_counts[track_id] < firm_threshold:
                            state_map[track_id] = progression_states[hit_counts[track_id] - 1]
                        if hit_counts[track_id] >= firm_threshold:
                            firm_ids.add(track_id)
                            state_map[track_id] = 'Firm'
                    else:
                        tentative_ids[track_id] = True
                        hit_counts[track_id] = 1
                        miss_counts[track_id] = 0
                        state_map[track_id] = progression_states[0]                
                # Append measurement to the track along with its current state
                track['measurements'].append((measurement, state_map[track_id]))
                assigned = True
                break

        if not assigned:
            # Create a new track if no existing track was assigned
            new_track_id, new_track_idx = get_next_track_id(track_id_list)
            tracks.append({
                'track_id': new_track_id,
                'measurements': [(measurement, progression_states[0])]  # Start with the first state
            })
            miss_counts[new_track_idx] = 0
            hit_counts[new_track_idx] = 1
            tentative_ids[new_track_idx] = True
            state_map[new_track_idx] = progression_states[0]

        # Check miss counts and release the track if necessary
        for track_id, track in enumerate(tracks):
            if not track:  # Skip over already released tracks
                continue
            # Get current state of the track
            current_state = state_map.get(track_id)
            # Ensure current_state is valid before checking miss threshold
            if current_state is None:
                continue  # Skip if there's no valid state

            # Determine miss threshold based on current state
            miss_threshold = get_miss_threshold(current_state, firm_threshold)
            # If this track wasn't assigned this time, increment its miss count
            if not assigned and track['track_id'] == new_track_id:
                miss_counts[track_id] += 1
            # Release the track if the miss count exceeds the threshold for its state
            if miss_counts[track_id] >= miss_threshold:
                tracks[track_id] = None  # Clear the track
                release_track_id(track_id_list, track_id)
                state_map.pop(track_id, None)  # Remove state from state_map
    return tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states


# Select initiation mode based on user input
def select_initiation_mode(mode):
    if mode == '3-state':
        return 3
    elif mode == '5-state':
        return 5
    elif mode == '7-state':
        return 7
    else:
        raise ValueError("Invalid initiation mode. Choose '3-state', '5-state', or '7-state'.")

def main():
    # File path for measurements CSV
    file_path = 'ttk.csv'

    # Read measurements from CSV
    measurements = read_measurements_from_csv(file_path)
    
    kalman_filter = CVFilter()
    
    csv_file_predicted = "ttk.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values
    measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

    A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)
    number = 1000

    result = np.divide(A[0], number)            

    # Form measurement groups based on time intervals less than 50 milliseconds
    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)
    t = []
    rnge = []
    azme = []
    elem = []

    # List to store the filter states
    filter_states = []

    # Define track initiation parameters
    doppler_threshold = 100  # Doppler correlation threshold
    range_threshold = 100  # Range gate threshold
    firm_threshold = 3  # Number of hits required for a firm track
    time_threshold = 0.1  # Time difference threshold for track association
    mode = '3-state'  # Track initiation mode

    # Select initiation mode based on user input
    firm_threshold = select_initiation_mode(mode)

    # Initialize tracks with updated state logic
    tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states = initialize_tracks(measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold, mode)

    # Process each group of measurements
    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")

        # List of tracks and reports (in Cartesian coordinates)
        tracks_in_group = []
        reports = []

        for i, (rng, azm, ele, mt) in enumerate(group):
            print(f"\nMeasurement {i + 1}: (az={azm}, el={ele}, r={rng}, t={mt})\n")
            x, y, z = sph2cart(azm, ele, rng)
            if not kalman_filter.first_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                print("Initialized Filter state:", kalman_filter.Sf.flatten())
                continue
            elif kalman_filter.first_rep_flag and not kalman_filter.second_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                print("Initialized Filter state 2nd M:", kalman_filter.Sf.flatten())
            else:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)                
                kalman_filter.predict_step(mt)
                reports.append((x, y, z))  # Append as a tuple of Cartesian coordinates
                # Use the Kalman filter state as the track
                tracks_in_group.append(kalman_filter.Sf[:3].flatten())
        # Form clusters using the association logic
        clusters = form_clusters_via_association(tracks_in_group, reports, kalman_filter, chi2_threshold=kalman_filter.gate_threshold)
        print("Clusters formed:", clusters)  # Print clusters
        print("Number of clusters:", len(clusters))
        # Process each cluster and select the best report for updating
        for cluster_tracks, cluster_reports in clusters:
            if cluster_tracks and cluster_reports:
                best_track_idx, best_report = select_best_report(cluster_tracks, cluster_reports, kalman_filter)
                if best_report is not None:
                    print(f"Selected Best Report for Track {best_track_idx + 1}: {best_report}")
                    # Prepare the measurement vector Z for the Kalman filter update
                    Z = np.array([[best_report[0]], [best_report[1]], [best_report[2]]])
                    print("Measurement Vector Z:", Z)
                    # Perform the Kalman filter update step with the selected report
                    kalman_filter.update_step(Z)
                    print("Updated filter state:", kalman_filter.Sf.flatten())
                    # Save the updated state for plotting or further processing
                    r_val, az_val, el_val = cart2sph(kalman_filter.Sf[0], kalman_filter.Sf[1], kalman_filter.Sf[2])
                    t.append(mt)
                    rnge.append(r_val)
                    azme.append(az_val)
                    elem.append(el_val)
                    # Append the updated state to the list
                    filter_states.append(kalman_filter.Sf.flatten())
                else:
                    print("No valid report found for this cluster.")
    # Save the filter states to a CSV file
    with open('updated_filter_states.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'Z', 'VX', 'VY', 'VZ'])  # Header
        writer.writerows(filter_states)
if __name__ == "__main__":
    main()


