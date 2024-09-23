# Process each group of measurements
for group_idx, group in enumerate(measurement_groups):
    print(f"Processing measurement group {group_idx + 1}...")

    # List of tracks and reports (in Cartesian coordinates)
    tracks_in_group = []
    reports = []

    for i, (rng, azm, ele, mt) in enumerate(group):
        print(f"\nMeasurement {i + 1}: (az={azm}, el={ele}, r={rng}, t={mt})\n")
        x, y, z = sph2cart(azm, ele, rng)

        # Perform gating and checks based on track state and mode
        for track_id, track in enumerate(tracks):
            state = state_map.get(track_id, None)  # Get the current state of the track
            if state == 'Poss1' and not kalman_filter.first_rep_flag:
                # If state is 'Poss1' and the Kalman filter is not yet initialized
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                print(f"Initialized Filter state for Track {track_id} in Poss1 state:", kalman_filter.Sf.flatten())
                continue

            elif state == 'Tentative1' and not kalman_filter.second_rep_flag:
                # If state is 'Tentative1' and the second representation flag is not set, initialize again
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                print(f"Initialized Filter state 2nd M for Track {track_id} in Tentative1 state:", kalman_filter.Sf.flatten())

            elif state == 'Firm':
                # If the state is 'Firm', perform the prediction and update step
                kalman_filter.predict_step(mt)
                Z = np.array([[x], [y], [z]])  # Measurement vector in Cartesian coordinates
                kalman_filter.update_step(Z)
                print(f"Updated Filter state for Track {track_id} in Firm state:", kalman_filter.Sf.flatten())

            else:
                # Otherwise, just store the measurement for later association
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)

            # Store the reports in Cartesian coordinates
            reports.append((x, y, z))  # Cartesian coordinates for the report
            tracks_in_group.append(kalman_filter.Sf[:3].flatten())  # Use the Kalman filter state as the track

    # Perform the association logic
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
