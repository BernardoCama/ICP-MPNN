import numpy as np
from scipy.linalg import block_diag
import scipy.io as sio
# from .GNN_data_association import GNN_data_association, GNN_data_association_adaptive
from copy import deepcopy


def icp_with_data_association_adaptive(params_dict, fea_vect,  connectivity_matrix_gt_all, z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, gps_meas,
                              cv_measure_gps, vehicle_estimate_kf, vehicle_estimate_kf_cov):
    estimate_icpda_cov = dict()
    estimateICPDA = dict()
    numberOfDetection = dict()

    for i in range(params_dict['num_vehicles']):
        estimateICPDA[f'veh_{i}'] = list()
        estimate_icpda_cov[f'veh_{i}'] = list()
        numberOfDetection[f'veh_{i}'] = list()
        for time in range(params_dict['timeSteps']):
            estimateICPDA[f'veh_{i}'].append([])
            estimate_icpda_cov[f'veh_{i}'].append([])
            numberOfDetection[f'veh_{i}'].append(np.zeros(params_dict['num_targets'], ))

    vehicleNumberOfMeasGT = np.zeros((params_dict['num_vehicles'], params_dict['timeSteps']))
    incorrectAssociation = np.zeros((params_dict['timeSteps'], ))
    number_of_total_v2f = np.zeros((params_dict['timeSteps'], 1))

    # Added
    # previous_z_fv_boxes_vehicles = np.reshape(fea_vect[:, params_dict['start_time']-1], (params_dict['num_targets'], 4))[:,:2]
    previous_z_fv_boxes_vehicles = [{} for vehicle in range(params_dict['num_targets'])]

    unique_targets = list()
    vehicles_target_list = dict()
    target_history = list()
    for time in range(params_dict['start_time']-1, params_dict['timeSteps']):
        print(f"Timestep {time+1}/{params_dict['timeSteps']}")
        fea = np.reshape(fea_vect[:, time], (params_dict['num_targets'], 4))
        z_fv = z_fv_all[:, :, time]
        z_fv_boxes = z_fv_all_boxes[:, :, :, time, :]

        v2f_cov_fv = v2f_cov_fv_all[:, :, time]
        connectivity_matrix_gt = connectivity_matrix_gt_all[:,:, time]

        # Targets at vehicles & a-priori
        z_fv_t = list()
        z_fv_boxes_t = list()
        featureStatePrior = list()
        featureStatePriorCov = list()
        targetPrediction = list()
        targetPredictionCov = list()
        id_targets_true = list()

        for i in range(params_dict['num_vehicles']):
            sensed_targets = []
            for f in range(params_dict['num_targets']):
                if connectivity_matrix_gt[i, f]:
                    sensed_targets.append(f)
                    if f not in unique_targets:
                        unique_targets.append(f)
            z_fv_t.append(z_fv[i * 2:(i + 1) * 2, sensed_targets])
            z_fv_boxes_t.append(z_fv_boxes[sensed_targets, :, :, i])

            if time == params_dict['start_time']-1:
                featureStatePriorVeh, featureStatePriorCovVeh = generate_target_prior(params_dict, fea[sensed_targets, :])
                featureStatePrior.append(featureStatePriorVeh)
                featureStatePriorCov.append(featureStatePriorCovVeh)
                targetPrediction.append(np.zeros((len(sensed_targets), 2)))
                targetPredictionCov.append(np.zeros((2, 2*len(sensed_targets))))
            else:
                featureStatePriorVeh, featureStatePriorCovVeh = generate_target_prior(params_dict, fea[unique_targets, :])
                featureStatePrior.append(featureStatePriorVeh)
                featureStatePriorCov.append(featureStatePriorCovVeh)
                targetPrediction.append(np.zeros((len(unique_targets), 2)))
                targetPredictionCov.append(np.zeros((2, 2*len(unique_targets))))
            id_targets_true.append(sensed_targets)

        # Build new connectivity_matrix
        connectivity_matrix_t = np.zeros((params_dict['num_vehicles'], len(unique_targets)))
        id_targets = [list() for i in range(params_dict['num_vehicles'])]
        # if time == params_dict['start_time']-1:
        #     # Assign first row of connectivity matrix to first vehicle
        #     connectivity_matrix_t[0, 0:len(id_targets_true[0])] = 1
        #     for id, elem in enumerate(id_targets_true[0]):
        #         unique_targets.append(elem)
        #         id_targets[0].append(len(unique_targets)-1)
        #     start = 1
        # else:
        #     start = 0

        for i in range(params_dict['num_vehicles']):
            for num, f in enumerate(id_targets_true[i]):
                for id, elem in enumerate(unique_targets):
                    if elem == f:
                        connectivity_matrix_t[i, id] = 1
                        id_targets[i].append(id)
            if time > params_dict['start_time'] - 1:
                numberOfDetection[f'veh_{i}'][time][0:len(unique_targets)] = connectivity_matrix_t[i, :]

        # Target & vehicle prediction
        vehicle_prediction = np.zeros((4 * params_dict['num_vehicles'],))
        vehicle_prediction_cov = np.zeros((4, 4 * params_dict['num_vehicles']))
        for i in range(params_dict['num_vehicles']):
            if time == params_dict['start_time']-1:
                estimateICPDA[f'veh_{i}'][time] = np.hstack((vehicle_estimate_kf[i*4:(i+1)*4, time], np.reshape(featureStatePrior[i], (np.size(featureStatePrior[i]), ))))
                estimate_icpda_cov[f'veh_{i}'][time] = np.zeros((4 + 2*featureStatePrior[i].shape[0], 4 + 2*featureStatePrior[i].shape[0]))
                estimate_icpda_cov[f'veh_{i}'][time][0:4, 0:4] = vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time]
                for f in range(featureStatePrior[i].shape[0]):
                    estimate_icpda_cov[f'veh_{i}'][time][f*2+4:(f+1)*2+4, f*2+4:(f+1)*2+4] = featureStatePriorCov[i][:, f*2:(f+1)*2]
            else:
                for f in range(featureStatePrior[i].shape[0]):
                    if numberOfDetection[f'veh_{i}'][time][f] == 1 and numberOfDetection[f'veh_{i}'][time-1][f] == 1:
                        targetPrediction[i][f, :] = np.transpose(estimateICPDA[f'veh_{i}'][time-1][f*2+4:(f+1)*2+4])
                        targetPredictionCov[i][:, f*2:(f+1)*2] = estimate_icpda_cov[f'veh_{i}'][time-1][f*2+4:(f+1)*2+4, f*2+4:(f+1)*2+4]
                    else:
                        targetPrediction[i][f, :] = featureStatePrior[i][f, :]
                        targetPredictionCov[i][:, f*2:(f+1)*2] = featureStatePriorCov[i][:, f*2:(f+1)*2]

                # Vehicle prediction
                if estimateICPDA[f'veh_{i}'][time-1] is not None:
                    cv = np.array([[params_dict['std_v_x'] ** 2, 0],
                                   [0, params_dict['std_v_y'] ** 2]])
                    vehicle_prediction_cov[:, i*4:(i+1)*4] = np.matmul(params_dict['motionMatrixA'], np.matmul(estimate_icpda_cov[f'veh_{i}'][time-1][0:4, 0:4], np.transpose(params_dict['motionMatrixA']))) + np.matmul(params_dict['motionMatrixB'], np.matmul(cv, np.transpose(params_dict['motionMatrixB'])))
                    vehicle_prediction[i*4:(i+1)*4] = np.matmul(params_dict['motionMatrixA'], estimateICPDA[f'veh_{i}'][time-1][0:4])
                else:
                    vehicle_prediction[i*4:(i+1)*4] = vehicle_estimate_kf[i*4:(i+1)*4, time]
                    vehicle_prediction_cov[:, i*4:(i+1)*4] = vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time]

        # Data association
        if time > params_dict['start_time'] - 1:

            # numberOfDetection[0:params_dict['num_targets'], time] = np.sum(connectivity_matrix_gt, 0)
            vehicleNumberOfMeasGT[:, time] = np.sum(connectivity_matrix_gt,1)
            # number_of_total_v2f[time] = np.sum(connectivity_matrix_gt)

            # Reorder boxes for data association
            z_fv_boxes_vehicles = list()
            curr_veh_pos_list = []
            for curr_veh in range(params_dict['num_vehicles']):
                curr_veh_pos = np.reshape(np.tile(np.hstack((vehicle_prediction[curr_veh*4:curr_veh*4 + 2], 0)), (8,)),
                                        (8, 3))
                curr_veh_pos_list.append(curr_veh_pos)
                z_fv_boxes_vehicles.append(z_fv_boxes_t[curr_veh] + np.transpose(curr_veh_pos))

            # Todo: association
            connectivityMatrixICPDA, misura_FV, previous_z_fv_boxes_vehicles, association_errors = GNN_data_association_adaptive(params_dict,
                                                                                      z_fv_boxes_vehicles,
                                                                                      # NUM_VEH X (NUM_MES X 3 X 8)
                                                                                      id_targets,  # NUM_VEH X NUM_MES
                                                                                      connectivity_matrix_t,
                                                                                      # NUM_VEH X NUM_MES
                                                                                      previous_z_fv_boxes_vehicles,
                                                                                      # NUM_MES X 2 (centroid 2D)
                                                                                      curr_veh_pos_list,  # NUM_VEH X 1
                                                                                      num_vehicles=params_dict[
                                                                                          'num_vehicles'],
                                                                                      num_features=len(unique_targets))

            # previous_z_fv_boxes_vehicles = fea[:, :2]
            incorrectAssociation[time] = association_errors

            if association_errors > 0:
                ciao = 1

            # Collect target positions & cov
            targetPrediction_new = np.zeros((len(unique_targets), 2))
            targetPrediction_cov_new = np.zeros((2, 2*len(unique_targets), ))
            for i in range(params_dict['num_vehicles']):
                for f in range(connectivityMatrixICPDA.shape[1]):
                    if connectivityMatrixICPDA[i, f]:
                        curr_target = np.where(np.asarray(id_targets[i]) == f)[0][0]
                        targetPrediction_new[f, :] = targetPrediction[i][curr_target, :]
                        targetPrediction_cov_new[:, 2*f:(f+1)*2] = targetPredictionCov[i][:, curr_target*2:(curr_target+1)*2]
            # Check if target have disappeared or are not associated with any other targets
            for f in range(connectivityMatrixICPDA.shape[1]):
                if np.sum(connectivityMatrixICPDA[:, f]) == 0:
                    targetPrediction_new[f, :] = target_history[2*f:2*(f+1)]
                    targetPrediction_cov_new[:, 2*f:(f+1)*2] = params_dict['featurePriorPosStd']**2*np.eye(2)

            # Perfect DA
            # connectivityMatrixICPDA = connectivity_matrix_gt # NUM_VEH X NUM_FEATURES
            # misura_FV = z_fv                                 # 2*NUM_VEH X NUM_FEATURES

            # ICP estimate
            [curr_estimate_icp_cov, curr_estimate_icp] = cooperative_localization(params_dict, connectivityMatrixICPDA,
                                                                                                gps_meas[:, :, time],
                                                                                                misura_FV, np.sum(connectivityMatrixICPDA), v2f_cov_fv, cv_measure_gps[:, :, time], targetPrediction_cov_new, vehicle_prediction_cov, targetPrediction_new, vehicle_prediction, len(unique_targets))
            # Reorder structure for next step
            for i in range(params_dict['num_vehicles']):
                estimateICPDA[f'veh_{i}'][time] = np.hstack((curr_estimate_icp[i*4:(i+1)*4], curr_estimate_icp[4*params_dict['num_vehicles']:len(curr_estimate_icp)]))
                estimate_icpda_cov[f'veh_{i}'][time] = block_diag(curr_estimate_icp_cov[i*4:(i+1)*4, i*4:(i+1)*4],
                                                                  curr_estimate_icp_cov[4*params_dict['num_vehicles']:4*params_dict['num_vehicles']+2*len(unique_targets),
                                                                  4*params_dict['num_vehicles']:4*params_dict['num_vehicles']+2*len(unique_targets)])
                target_history = curr_estimate_icp[4*params_dict['num_vehicles']:2*len(curr_estimate_icp)]
    return estimateICPDA, estimate_icpda_cov, numberOfDetection, incorrectAssociation


def cooperative_localization(params_dict, VFconnect, y_v, z_FV, M_z, C_z_FV, Cv_measure_gps, Cf_b, Cv_b, mu_f_b, mu_v_b, num_targets):

    Nf = num_targets
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
    featureStatePriorCov = np.tile(params_dict['featurePriorPosStd']**2*np.eye(2), (1, fea.shape[0]))
    featureStatePrior = fea[:, 0:2]
    return featureStatePrior, featureStatePriorCov


