Certainly! I'll modify the code according to your requirements. Here's the updated version incorporating your changes:

```python
import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2

# ... (Keep all the existing imports and class definitions)

def initialize_tracks(measurement_groups, doppler_threshold, range_threshold, firm_threshold, time_threshold, mode):
    tracks = []
    track_id_list = []
    miss_counts = {}
    hit_counts = {}
    tentative_ids = {}
    firm_ids = set()
    state_map = {}

    state_progression = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
    }
    progression_states = state_progression[firm_threshold]

    def get_miss_threshold(current_state, firm_threshold):
        if current_state.startswith('Poss'):
            return 1
        elif current_state.startswith('Tentative'):
            return 2 if firm_threshold == 3 else 3
        elif current_state == 'Firm':
            return 3 if firm_threshold == 3 else 5

    for group_idx, group in enumerate(measurement_groups):
        for measurement in group:
            measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
            measurement_time = measurement[3]
            assigned = False

            for track_id, track in enumerate(tracks):
                if not track:
                    continue

                last_measurement = track['measurements'][-1][0]
                last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
                distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
                time_diff = measurement_time - last_measurement[3]

                range_satisfied = distance < range_threshold
                if range_satisfied and time_diff <= time_threshold:
                    if track_id not in firm_ids:
                        if track_id in tentative_ids:
                            hit_counts[track_id] += 1
                            miss_counts[track_id] = 0
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

                    track['measurements'].append((measurement, state_map[track_id]))
                    assigned = True
                    break

            if not assigned:
                new_track_id = len(track_id_list) + 1
                tracks.append({
                    'track_id': new_track_id,
                    'measurements': [(measurement, progression_states[0])]
                })
                track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                miss_counts[new_track_id] = 0
                hit_counts[new_track_id] = 1
                tentative_ids[new_track_id] = True
                state_map[new_track_id] = progression_states[0]

    return tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states

def gating_check(track, measurement, doppler_threshold, range_threshold):
    last_measurement = track['measurements'][-1][0]
    last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
    measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
    
    distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
    
    # Implement Doppler check if you have Doppler information
    # For now, we'll just use the range check
    return distance < range_threshold

def jpda_filter(track, validated_measurements, kalman_filter):
    # Implement JPDA filtering here
    # This is a placeholder implementation
    if validated_measurements:
        avg_measurement = np.mean(validated_measurements, axis=0)
        x, y, z = sph2cart(avg_measurement[0], avg_measurement[1], avg_measurement[2])
        kalman_filter.predict_step(avg_measurement[3])
        Z = np.array([[x], [y], [z]])
        kalman_filter.update_step(Z)
    return kalman_filter

def main():
    file_path = 'ttk.csv'

    measurements = read_measurements_from_csv(file_path)
    
    kalman_filter = CVFilter()

    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)
    t = []
    rnge = []
    azme = []
    elem = []

    filter_states = []

    doppler_threshold = 100
    range_threshold = 100
    firm_threshold = 3
    time_threshold = 0.1
    mode = '3-state'

    firm_threshold = select_initiation_mode(mode)

    tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states = initialize_tracks(measurement_groups, doppler_threshold, range_threshold, firm_threshold, time_threshold, mode)

    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")

        for track_id, track in enumerate(tracks):
            if not track:
                continue

            validated_measurements = []
            for measurement in group:
                if gating_check(track, measurement, doppler_threshold, range_threshold):
                    validated_measurements.append(measurement)

            if validated_measurements:
                current_state = state_map.get(track_id, None)
                print(f"Track {track_id} is in state: {current_state}")

                if current_state == 'Firm':
                    kalman_filter = jpda_filter(track, validated_measurements, kalman_filter)
                    print("Updated filter state:", kalman_filter.Sf.flatten())
                    r_val, az_val, el_val = cart2sph(kalman_filter.Sf[0], kalman_filter.Sf[1], kalman_filter.Sf[2])
                    t.append(validated_measurements[0][3])  # Using the time from the first validated measurement
                    rnge.append(r_val)
                    azme.append(az_val)
                    elem.append(el_val)
                    filter_states.append(kalman_filter.Sf.flatten())

    with open('updated_filter_states.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
        writer.writerows(filter_states)

if __name__ == "__main__":
    main()
```

Key changes made:

1. Modified `initialize_tracks` to work with measurement groups instead of individual measurements.

2. Added a `gating_check` function to perform the doppler and range check for each measurement.

3. Created a placeholder `jpda_filter` function to implement JPDA filtering. You'll need to replace this with a proper JPDA implementation.

4. Updated the `main` function to:
   - Process measurement groups
   - Perform gating checks for each track and measurement
   - Apply JPDA filtering for tracks in the 'Firm' state with validated measurements

Note that this implementation is a framework, and you'll need to flesh out some parts:

- The `gating_check` function currently only implements the range check. You'll need to add the Doppler check if you have that information available.
- The `jpda_filter` function is a placeholder. You'll need to implement the full Joint Probabilistic Data Association algorithm here.

This structure allows you to process measurement groups, perform gating checks, and apply JPDA filtering for firm tracks. You can further refine and optimize this code based on your specific requirements and the details of your JPDA implementation.