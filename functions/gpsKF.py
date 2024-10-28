import numpy as np


def gps_kf(params_dict, vehicle_state_gt, gps_meas, cv_measure_gps):
    vehicle_estimate_kf = np.zeros((4*params_dict['num_vehicles'], params_dict['timeSteps']))*np.nan
    vehicle_estimate_kf_cov = np.zeros((4*params_dict['num_vehicles'], 4*params_dict['num_vehicles'],
                                        params_dict['timeSteps']))*np.nan

    for time in range(params_dict['timeSteps']):
        veh_state = np.reshape(vehicle_state_gt[:, time], (params_dict['num_vehicles'], 4))
        [vehicle_state_prior, vehicle_state_prior_cov] = generate_vehicle_prior(params_dict, veh_state)

        Ht = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        if time == 0:
            for i in range(params_dict['num_vehicles']):
                cv_gps_diag = cv_measure_gps[:, i * 2:(i + 1) * 2, time]
                gain = np.matmul(vehicle_state_prior_cov[:, i*4:(i+1)*4], np.matmul(np.transpose(Ht), np.linalg.inv(
                    np.matmul(Ht, np.matmul(vehicle_state_prior_cov[:, i*4:(i+1)*4], np.transpose(Ht))) + cv_gps_diag)))
                vehicle_estimate_kf_cov[i*4:(i + 1)*4, i*4:(i + 1)*4, time] = vehicle_state_prior_cov[:, i*4:(i+1)*4] - np.matmul(gain, np.matmul(Ht, vehicle_state_prior_cov[:, i*4:(i+1)*4]))
                vehicle_estimate_kf[i*4:(i + 1)*4, time] = vehicle_state_prior[i, :] + np.matmul(gain, np.transpose(
                    gps_meas[i, :, time]) - np.matmul(Ht, vehicle_state_prior[i,:]))
        else:
            for i in range(params_dict['num_vehicles']):
                if not np.isnan(vehicle_estimate_kf[i*4, time-1]):
                    # cv = np.array([[params_dict['std_v_x']**2*np.sin(np.deg2rad(vehicle_heading_gt[i, time]))**2 +
                    #                params_dict['std_v_y']**2*np.cos(np.deg2rad(vehicle_heading_gt[i, time]))**2, 0],
                    #               [0, params_dict['std_v_x']**2*np.cos(np.deg2rad(vehicle_heading_gt[i, time]))**2 +
                    #                params_dict['std_v_y']**2*np.sin(np.deg2rad(vehicle_heading_gt[i, time]))**2]])
                    cv = np.array([[params_dict['std_v_x']**2, 0], [0, params_dict['std_v_y']**2]])

                    # Prediction
                    cv_acc = np.array([[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
                    vehicle_gps_prediction_cov = np.matmul(params_dict['motionMatrixA'], np.matmul(vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time-1], np.transpose(params_dict['motionMatrixA']))) + np.matmul(params_dict['motionMatrixB'], np.matmul(cv, np.transpose(params_dict['motionMatrixB'])))
                    vehicle_gps_prediction = np.matmul(params_dict['motionMatrixA'], vehicle_estimate_kf[i*4:(i+1)*4, time-1])

                    # Update
                    cv_gps_diag = cv_measure_gps[:, i*2:(i+1)*2, time]
                    gain = np.matmul(vehicle_gps_prediction_cov, np.matmul(np.transpose(Ht), np.linalg.inv(np.matmul(Ht, np.matmul(vehicle_gps_prediction_cov, np.transpose(Ht))) + cv_gps_diag)))
                    vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time] = vehicle_gps_prediction_cov - np.matmul(gain, np.matmul(Ht, vehicle_gps_prediction_cov))
                    vehicle_estimate_kf[i*4:(i+1)*4, time] = vehicle_gps_prediction + np.matmul(gain, np.transpose(gps_meas[i, :, time]) - np.matmul(Ht, vehicle_gps_prediction))

    return vehicle_estimate_kf, vehicle_estimate_kf_cov


def generate_vehicle_prior(params_dict, veh_state):
    vehicle_state_prior = np.zeros((params_dict['num_vehicles'], 4))
    vehicle_state_prior_cov = np.zeros((4, params_dict['num_vehicles']*4))

    for i in range(params_dict['num_vehicles']):
        vehicle_state_prior[i, :] = veh_state[i, :]
        vehicle_state_prior_cov[:, i*4:(i+1)*4] = np.vstack((np.hstack((params_dict['GPSPriorPostd']*np.eye(2), np.zeros((2, 2)))),
                                                            np.hstack((np.zeros((2, 2)), params_dict['GPSPriorVelStd']*np.eye(2)))))
    return vehicle_state_prior, vehicle_state_prior_cov
