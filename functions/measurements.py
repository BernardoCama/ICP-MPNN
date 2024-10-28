import numpy as np


def generate_gps_meas(params_dict, vehicle_state_gt):
    cv_measure_gps = np.zeros((2, 2*params_dict['num_vehicles'], params_dict['timeSteps']))
    gps_meas = np.zeros((params_dict['num_vehicles'], 2, params_dict['timeSteps']))

    for time in range(params_dict['timeSteps']):
        veh_state = np.reshape(vehicle_state_gt[:, time], (params_dict['num_vehicles'], 4))
        for veh in range(params_dict['num_vehicles']):
            cv_measure_gps[:, veh*2:(veh+1)*2, time] = np.array([[params_dict['GPSaccuracyPos_x']**2, 0],
                                                                [0, params_dict['GPSaccuracyPos_y']**2]])
            r_mat = np.linalg.cholesky(cv_measure_gps[:, veh*2:(veh+1)*2, time])
            curr_meas = np.transpose(np.transpose(veh_state[veh, 0:2])) + np.transpose(np.matmul(np.transpose(r_mat), np.random.randn(2, 1)))
            gps_meas[veh, :, time] = curr_meas
    return cv_measure_gps, gps_meas


def generate_v2f_meas(params_dict, fea_vect, fea_vect_boxes, vehicle_state_gt, conn_features):
    v2fcov_fv = np.zeros((2 * params_dict['num_vehicles'], 2 * params_dict['num_targets'], params_dict['timeSteps']))
    z_fv = np.zeros((2 * params_dict['num_vehicles'], params_dict['num_targets'], params_dict['timeSteps']))
    z_fv_boxes = np.zeros((params_dict['num_targets'], 3, 8, params_dict['timeSteps'], params_dict['num_vehicles']))
    connectivity_matrix_gtall = np.zeros((params_dict['num_vehicles'], params_dict['num_targets'], params_dict['timeSteps']))

    for time in range(params_dict['timeSteps']):
        for veh in range(params_dict['num_vehicles']):
            mu_mea_z_boxes = np.zeros((params_dict['num_targets'], 3, 8))

            fea = np.reshape(fea_vect[:, time, veh], (params_dict['num_targets'], 4))
            fea_boxes = fea_vect_boxes[:, :, :, time, veh]
            veh_state = np.reshape(vehicle_state_gt[:, time], (params_dict['num_vehicles'], 4))
            mu_mea_z = np.tile(fea[:, 0:2], (1, params_dict['num_vehicles'])) - np.tile(np.transpose(np.reshape(veh_state[:, 0:2], (-1, 1))), (params_dict['num_targets'], 1))

            mu_mea_z_boxes[:, 0, :] = fea_boxes[:, 0, :] - veh_state[veh, 0]
            mu_mea_z_boxes[:, 1, :] = fea_boxes[:, 1, :] - veh_state[veh, 1]
            mu_mea_z_boxes[:, 2, :] = fea_boxes[:, 2, :]

            connectivity_matrix_gtall[veh, :, time] = conn_features[veh, :, time]

            for f in range(params_dict['num_targets']):
                v2fcov_fv[veh*2:(veh+1)*2, f*2:(f+1)*2, time] = params_dict['V2FCov']
                if connectivity_matrix_gtall[veh, f, time]:
                    z_fv[veh*2:(veh+1)*2, f, time] = mu_mea_z[f, veh*2:(veh+1)*2]  # + np.random.randn(1, 2) * params_dict['V2Fstd']
                    z_fv_boxes[f, :, :, time, veh] = mu_mea_z_boxes[f, :, :]  # + np.random.randn(3, 8) * params_dict['V2Fstd']
    return z_fv, z_fv_boxes, v2fcov_fv, connectivity_matrix_gtall
