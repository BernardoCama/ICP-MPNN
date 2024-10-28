import numpy as np
from scipy.linalg import block_diag
import scipy.io as sio
# from .GNN_data_association import GNN_data_association


def icp_with_data_association(params_dict, fea_vect,  connectivity_matrix_gt_all, z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, gps_meas,
                              cv_measure_gps, vehicle_estimate_kf, vehicle_estimate_kf_cov):
    estimate_icpda_cov = np.zeros((4*params_dict['num_vehicles'] + 2*params_dict['num_targets'],
                           4*params_dict['num_vehicles'] + 2*params_dict['num_targets'], params_dict['timeSteps']))*np.nan
    estimateICPDA = np.zeros((4*params_dict['num_vehicles'] + 2*params_dict['num_targets'], params_dict['timeSteps']))*np.nan
    numberOfDetection = np.zeros((params_dict['num_targets'], params_dict['timeSteps']))
    vehicleNumberOfMeasGT = np.zeros((params_dict['num_vehicles'], params_dict['timeSteps']))
    incorrectAssociation = np.zeros((params_dict['timeSteps'], ))
    number_of_total_v2f = np.zeros((params_dict['timeSteps'], 1))

    # Added
    previous_z_fv_boxes_vehicles = np.reshape(fea_vect[:, params_dict['start_time']-1], (params_dict['num_targets'], 4))[:,:2]

    for time in range(params_dict['start_time']-1, params_dict['timeSteps']):
        print(f"Timestep {time+1}/{params_dict['timeSteps']}")
        fea = np.reshape(fea_vect[:, time], (params_dict['num_targets'], 4))
        z_fv = z_fv_all[:, :, time]
        z_fv_boxes = z_fv_all_boxes[:, :, :, time, :]

        v2f_cov_fv = v2f_cov_fv_all[:,:, time]
        connectivity_matrix_gt = connectivity_matrix_gt_all[:,:, time]

        [featureStatePrior, featureStatePriorCov] = generate_target_prior(params_dict, fea)

        numberOfDetection[0:params_dict['num_targets'], time] = np.sum(connectivity_matrix_gt, 0)
        vehicleNumberOfMeasGT[:,time] = np.sum(connectivity_matrix_gt,1)
        number_of_total_v2f[time] = np.sum(connectivity_matrix_gt)

        # Centralized cooperative localization
        targetPrediction = np.zeros((params_dict['num_targets'], 2))
        targetPredictionCov = np.zeros((2, 2*params_dict['num_targets']))

        # Targets prediction
        if number_of_total_v2f[time] == 0 or time == params_dict['start_time']-1:
            estimateICPDA[:, time] = np.hstack((vehicle_estimate_kf[:, time], np.reshape(featureStatePrior, (2*params_dict['num_targets'], ))))
            for i in range(params_dict['num_vehicles']):
                estimate_icpda_cov[i*4:(i+1)*4, i*4:(i+1)*4, time] = vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time]
            for f in range(params_dict['num_targets']):
                estimate_icpda_cov[f*2+4*params_dict['num_vehicles']:(f+1)*2+4*params_dict['num_vehicles'], f*2+4*params_dict['num_vehicles']:(f+1)*2+4*params_dict['num_vehicles'],time] = featureStatePriorCov[:, f*2:(f+1)*2]
        elif number_of_total_v2f[time] != 0:
            for f in range(params_dict['num_targets']):
                if numberOfDetection[f, time] >= 1 and numberOfDetection[f, time - 1] >= 1:
                    targetPrediction[f,:] = np.transpose(estimateICPDA[f*2+4*params_dict['num_vehicles']:(f+1)*2+4*params_dict['num_vehicles'], time-1])
                    targetPredictionCov[:, f*2:(f+1)*2] = estimate_icpda_cov[f*2+4*params_dict['num_vehicles']:(f+1)*2+4*params_dict['num_vehicles'], f*2+4*params_dict['num_vehicles']:(f+1)*2+4*params_dict['num_vehicles'], time-1]
                else:
                    targetPredictionCov[:, f*2:(f+1)*2] = featureStatePriorCov[:, f*2:(f+1)*2]
                    targetPrediction[f, :] = featureStatePrior[f, :]

            # Vehicles prediction
            vehicle_prediction = np.zeros((4*params_dict['num_vehicles'], ))
            vehicle_prediction_cov = np.zeros((4, 4*params_dict['num_vehicles']))

            for veh in range(params_dict['num_vehicles']):
                if not np.isnan(all(estimateICPDA[veh*4:(veh+1)*4,time-1])):
                    cv = np.array([[params_dict['std_v_x']**2, 0],
                                  [0, params_dict['std_v_y']**2]])
                    vehicle_prediction_cov[:, veh*4:(veh+1)*4] = np.matmul(params_dict['motionMatrixA'], np.matmul(estimate_icpda_cov[veh*4:(veh+1)*4, veh*4:(veh+1)*4, time-1], np.transpose(params_dict['motionMatrixA']))) + np.matmul(params_dict['motionMatrixB'], np.matmul(cv, np.transpose(params_dict['motionMatrixB'])))
                    vehicle_prediction[veh*4:(veh+1)*4] = np.matmul(params_dict['motionMatrixA'], estimateICPDA[veh*4:(veh+1)*4, time-1])
                else:
                    vehicle_prediction[veh*4:(veh+1)*4] = vehicle_estimate_kf[veh*4:(veh+1)*4, time]
                    vehicle_prediction_cov[:, veh*4:(veh+1)*4] = vehicle_estimate_kf_cov[veh*4:(veh+1)*4, veh*4:(veh+1)*4, time]

            # Reorder boxes for data association
            z_fv_boxes_vehicles = list()
            id_target = []
            curr_veh_pos_list = []
            for curr_veh in range(params_dict['num_vehicles']):
                sensed_targets = list()
                for f in range(params_dict['num_targets']):
                    if connectivity_matrix_gt[curr_veh, f]:
                        sensed_targets.append(f)
                curr_veh_pos = np.reshape(np.tile(np.hstack((vehicle_prediction[curr_veh*4:curr_veh*4+2], 0)), (8, )), (8, 3))
                curr_veh_pos_list.append(curr_veh_pos)
                z_fv_boxes_vehicles.append(z_fv_boxes[sensed_targets, :, :, curr_veh] + np.transpose(curr_veh_pos))
                id_target.append(sensed_targets)

            # Todo: association                           
            # connectivityMatrixICPDA, misura_FV, association_errors = GNN_data_association(params_dict,
            #                                                 z_fv_boxes_vehicles,  # NUM_VEH X (NUM_MES X 3 X 8)
            #                                                 id_target, # NUM_VEH X NUM_MES
            #                                                 connectivity_matrix_gt, # NUM_VEH X NUM_MES
            #                                                 previous_z_fv_boxes_vehicles, # NUM_MES X 2 (centroid 2D)
            #                                                 curr_veh_pos_list,    # NUM_VEH X 1
            #                                                 num_vehicles = params_dict['num_vehicles'],
            #                                                 num_features = params_dict['num_targets'])
            # previous_z_fv_boxes_vehicles = fea[:, :2]
            # incorrectAssociation[time] = association_errors

            # Perfect DA
            connectivityMatrixICPDA = connectivity_matrix_gt # NUM_VEH X NUM_FEATURES
            misura_FV = z_fv                                 # 2*NUM_VEH X NUM_FEATURES

            # ICP estimate
            [estimate_icpda_cov[:, :, time], estimateICPDA[:, time]] = cooperative_localization(params_dict,
                                                                                               connectivityMatrixICPDA,
                                                                                               gps_meas[:, :, time],
                                                                                               misura_FV, np.sum(vehicleNumberOfMeasGT[:,time]), v2f_cov_fv, cv_measure_gps[:, :, time], targetPredictionCov, vehicle_prediction_cov, targetPrediction, vehicle_prediction)
            # sio.savemat(f'estimateICPDA_time{time}.mat',estimateICPDA)
    return estimateICPDA, estimate_icpda_cov, numberOfDetection, incorrectAssociation


def cooperative_localization(params_dict, VFconnect, y_v, z_FV, M_z, C_z_FV, Cv_measure_gps, Cf_b, Cv_b, mu_f_b, mu_v_b):

    Nf = params_dict['num_targets']
    Nv = params_dict['num_vehicles']

    y_v = np.reshape(y_v, -1)

    y_v_tot = np.zeros((2*Nv,))
    Cv_mea_gps = np.zeros((2, 2*Nv))
    C_v_b = np.zeros((4, 4*Nv))
    for i in range(Nv):
        # y_v_tot[i*4:(i+1)*4] = np.hstack((y_v[i*2:(i+1)*2], np.zeros(2, )))
        y_v_tot[i*2:(i+1)*2] = y_v[i*2:(i+1)*2]
        Cv_mea_gps[:, i*2:(i+1)*2] = Cv_measure_gps[:, i*2:(i+1)*2]
        C_v_b[:, i*4:(i+1)*4] = Cv_b[:, i*4:(i+1)*4]

    Cv_gps_diag = np.zeros((2 * Nv, 2*Nv))
    # to build V2F measurements
    meas_z_vec = np.zeros((2 * int(M_z),))
    R_all = np.zeros((2*int(M_z), 2*int(M_z)))
    # to build H
    M_v = np.zeros((2 * int(M_z), 4 * Nv))
    M_f = np.zeros((2 * int(M_z), 2 * Nf))

    # build covariance
    D = np.zeros((4 * Nv, 4 * Nv))
    E = np.zeros((4 * Nv, 2 * Nf))
    G = np.zeros((2 * Nf, 2 * Nf))

    theta_f_prev_t = np.zeros((2 * Nf,))
    theta_v_prev_t = np.zeros((4 * Nv,))
    theta = np.zeros((4 * Nv + 2 * Nf,))*np.nan
    C_theta = np.zeros((4 * Nv + 2 * Nf, 4 * Nv + 2 * Nf))*np.nan

    P = np.hstack((np.eye(2), np.zeros((2, 2))))

    m = 0
    for i in range(Nv):
        for f in range(Nf):
            if VFconnect[i, f] == 1: # for subset of features
                conn = 1
                # build complete set of V2F measurements
                m += 1
                meas_z_vec[(m - 1)*2: m*2] = z_FV[i*2: (i+1)*2, f] # prendo tutte le mis di tutte le fea di tutti e le ordino una sotto l'altra
                # V2F complete  covariance
                R_all[(m - 1)*2: m*2, (m - 1)*2:m*2] = C_z_FV[i*2:(i+1)*2, f*2:(f+1)*2]
                # build H
                M_v[(m - 1)*2:m*2, i*4:(i+1)*4] = -P
                M_f[(m - 1)*2:m*2, f*2:(f+1)*2] = np.eye(2)

                # matrice cov(only measurements are considered now)
                D[i*4:(i+1)*4, i*4:(i+1)*4] = D[i*4:(i+1)*4, i*4:(i+1)*4] + block_diag(np.linalg.inv(C_z_FV[i*2:(i+1)*2, f*2:(f+1)*2]), np.zeros((2, 2)))
                E[i*4:(i+1)*4, f*2:(f+1)*2] = np.vstack((-np.linalg.inv(C_z_FV[i*2:(i+1)*2, f*2:(f+1)*2]), np.zeros((2, 2))))
                G[f*2:(f+1)*2, f*2:(f+1)*2] = G[f*2:(f+1)*2, f*2:(f+1)*2] + np.linalg.inv(C_z_FV[i*2:(i+1)*2, f*2:(f+1)*2])

            # a priori on feature
            if i == 0:
                # belief on feature at previous time
                G[f*2:(f+1)*2, f*2:(f+1)*2] = G[f*2:(f+1)*2, f*2:(f+1)*2] + np.linalg.inv(Cf_b[:, f*2:(f+1)*2])
                theta_f_prev_t[f*2:(f+1)*2] = np.transpose(mu_f_b[f, :])

        # GPS vehicle likelihood + belief on vehicle at previous time
        D[i*4:(i+1)*4, i*4:(i+1)*4] = D[i*4:(i+1)*4, i*4:(i+1)*4] + np.linalg.inv(Cv_b[:, i*4:(i+1)*4]) + np.matmul(np.transpose(P), np.matmul(np.linalg.inv(Cv_measure_gps[:, i*2:(i+1)*2]), P))  # block_diag(np.linalg.inv(Cv_measure_gps[:, i*2:(i+1)*2]), np.zeros((2, 2)))
        theta_v_prev_t[i*4:(i+1)*4] = np.transpose(mu_v_b[i*4:(i+1)*4])

        # GPS complete covariance
        Cv_gps_diag[i*2:(i+1)*2, i*2:(i+1)*2] = Cv_measure_gps[:, i*2:(i+1)*2]

    # Covariance Estimate
    C_theta_inv = np.vstack((np.hstack((D, E)), np.hstack((np.transpose(E), G))))
    C_theta_centr = np.linalg.inv(C_theta_inv)

    # Mean Estimate
    H = np.vstack((np.hstack((np.kron(np.eye(Nv), P), np.zeros((2*Nv, 2*Nf)))), np.hstack((M_v, M_f))))

    rho = np.hstack((y_v_tot, meas_z_vec)) # GPS meas + V2F meas
    theta_prev = np.hstack((theta_v_prev_t, theta_f_prev_t))
    Q_all = block_diag(Cv_gps_diag, R_all)

    theta_centr = theta_prev + np.matmul(C_theta_centr, np.matmul(np.transpose(H), np.matmul(np.linalg.inv(Q_all), (rho - np.matmul(H, theta_prev)))))
    # sio.savemat('prova_icp.mat', {'theta_centr_p': theta_centr, 'C_theta_centr_p': C_theta_centr})

    return C_theta_centr, theta_centr


def generate_target_prior(params_dict, fea):
    featureStatePriorCov = np.tile(params_dict['featurePriorPosStd']**2*np.eye(2), (1, params_dict['num_targets']))
    featureStatePrior = fea[:, 0:2]
    return featureStatePrior, featureStatePriorCov


