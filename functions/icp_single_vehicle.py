import numpy as np
from scipy.linalg import block_diag
import scipy.io as sio


def icp_with_data_association_single(params_dict, fea_vect, connectivity_matrix_gt_all, z_fv_all, z_fv_all_boxes,
                                    v2f_cov_fv_all, gps_meas,
                                    cv_measure_gps, vehicle_estimate_kf, vehicle_estimate_kf_cov):
    estimate_icpda_cov = []
    estimateICPDA = []
    vehicleNumberOfMeasGT = []
    incorrectAssociation = []
    for i in range(params_dict['num_vehicles']):
        estimate_icpda_cov.append(np.zeros((4 + 2 * params_dict['num_targets'], 4 + 2 * params_dict['num_targets'],
                                           params_dict['timeSteps'])) * np.nan)
        estimateICPDA.append(np.zeros((4 + 2 * params_dict['num_targets'], params_dict['timeSteps']))*np.nan)
        vehicleNumberOfMeasGT.append(np.zeros((params_dict['timeSteps'],)))
        incorrectAssociation.append(np.zeros((params_dict['timeSteps'],)))

    for time in range(params_dict['start_time'] - 1, params_dict['timeSteps']):
        print(f"Timestep {time + 1}/{params_dict['timeSteps']}")
        fea = np.reshape(fea_vect[:, time], (params_dict['num_targets'], 4))
        z_fv = z_fv_all[:, :, time]
        v2f_cov_fv = v2f_cov_fv_all[:, :, time]
        connectivity_matrix_gt = connectivity_matrix_gt_all[:, :, time]

        [featureStatePrior, featureStatePriorCov] = generate_target_prior(params_dict, fea)

        # Individual localization with ICP
        for i in range(params_dict['num_vehicles']):
            vehicleNumberOfMeasGT[i][time] = np.sum(connectivity_matrix_gt[i, :])
            v2f_cov_fv = v2f_cov_fv_all[0:2, :, time]

            targetPrediction = np.zeros((params_dict['num_targets'], 2))
            targetPredictionCov = np.zeros((2, 2 * params_dict['num_targets']))

            # Targets prediction
            if vehicleNumberOfMeasGT[i][time] == 0 or time == params_dict['start_time'] - 1:
                estimateICPDA[i][:, time] = np.hstack((vehicle_estimate_kf[i*4:(i+1)*4, time], np.reshape(featureStatePrior, (2 * params_dict['num_targets'],))))
                estimate_icpda_cov[i][0:4, 0:4, time] = vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time]
                for f in range(params_dict['num_targets']):
                    estimate_icpda_cov[i][f*2+4:(f+1)*2+4, f*2+4:(f+1)*2+4, time] = featureStatePriorCov[:, f*2:(f+1)*2]
            elif vehicleNumberOfMeasGT[i][time] != 0:
                for f in range(params_dict['num_targets']):
                    if connectivity_matrix_gt_all[i, f, time] == 1 and connectivity_matrix_gt_all[i, f, time-1] == 1:
                        targetPrediction[f, :] = np.transpose(estimateICPDA[i][f*2+4:(f+1)*2+4, time-1])
                        targetPredictionCov[:, f*2:(f+1)*2] = estimate_icpda_cov[i][f*2+4:(f+1)*2+4, f*2+4:(f+1)*2+4,time - 1]
                    else:
                        targetPredictionCov[:, f*2:(f+1)*2] = featureStatePriorCov[:, f*2:(f+1)*2]
                        targetPrediction[f, :] = featureStatePrior[f, :]

                if not np.isnan(all(estimateICPDA[i][0:4, time-1])):
                    cv = np.array([[params_dict['std_v_x'] ** 2, 0],
                                   [0, params_dict['std_v_y'] ** 2]])
                    vehicle_prediction_cov = np.matmul(params_dict['motionMatrixA'], np.matmul(estimate_icpda_cov[i][0:4, 0:4, time-1], np.transpose(params_dict['motionMatrixA']))) + np.matmul(params_dict['motionMatrixB'], np.matmul(cv, np.transpose(params_dict['motionMatrixB'])))
                    vehicle_prediction = np.matmul(params_dict['motionMatrixA'], estimateICPDA[i][0:4, time-1])
                else:
                    vehicle_prediction = vehicle_estimate_kf[i*4:(i+1)*4, time]
                    vehicle_prediction_cov = vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time]

                # Reorder V2F measurements & association
                misura_FV = np.zeros((params_dict['num_targets'], 2))
                conn_stimata = np.zeros((params_dict['num_targets'], ))

                # metti matrice likelihood
                sensed_targets = []
                for f in range(z_fv.shape[1]):
                    if connectivity_matrix_gt[i, f]:
                        distances = np.sqrt(np.sum((fea[:, 0:2] - (np.transpose(z_fv[i*2:(i+1)*2, f]) + vehicle_prediction[0:2]))**2, 1))
                        id_targets = np.argsort(distances)

                        for elem in id_targets:
                            if elem not in sensed_targets:
                                break
                        id_target = elem
                        # misura_FV[id_target, :] = fea[f, 0:2] - vehicle_prediction[0:2]
                        misura_FV[id_target, :] = z_fv[i*2:(i+1)*2, f]
                        conn_stimata[id_target] = 1
                        sensed_targets.append(id_target)

                incorrectAssociation[i][time] = np.sum(conn_stimata != connectivity_matrix_gt[i, :])

                # print(f'Association errors vehicle {i}: {incorrectAssociation[i][time]}')
                # Perfect DA
                # connectivityMatrixICPDA = connectivity_matrix_gt # NUM_VEH X NUM_FEATURES
                # misura_FV = z_fv                                 # 2*NUM_VEH X NUM_FEATURES

                # ICP estimate
                [estimate_icpda_cov[i][:, :, time], estimateICPDA[i][:, time]] = single_localization(params_dict, conn_stimata, gps_meas[i, :, time], misura_FV, np.sum(conn_stimata),
                                                                                                     v2f_cov_fv, cv_measure_gps[:, i*2:(i+1)*2, time], targetPredictionCov, vehicle_prediction_cov, targetPrediction, vehicle_prediction)
    return estimateICPDA, estimate_icpda_cov, vehicleNumberOfMeasGT, incorrectAssociation


def single_localization(params_dict, VFconnect, y_v, z_FV, M_z, C_z_FV, Cv_measure_gps, Cf_b, Cv_b, mu_f_b, mu_v_b):
    Nf = params_dict['num_targets']
    Nv = 1

    y_v = np.reshape(y_v, -1)

    y_v_tot = np.zeros((2 * Nv,))
    Cv_mea_gps = np.zeros((2, 2 * Nv))
    C_v_b = np.zeros((4, 4 * Nv))
    for i in range(Nv):
        # y_v_tot[i*4:(i+1)*4] = np.hstack((y_v[i*2:(i+1)*2], np.zeros(2, )))
        y_v_tot[i * 2:(i + 1) * 2] = y_v[i * 2:(i + 1) * 2]
        Cv_mea_gps[:, i * 2:(i + 1) * 2] = Cv_measure_gps[:, i * 2:(i + 1) * 2]
        C_v_b[:, i * 4:(i + 1) * 4] = Cv_b[:, i * 4:(i + 1) * 4]

    Cv_gps_diag = np.zeros((2 * Nv, 2 * Nv))
    # to build V2F measurements
    meas_z_vec = np.zeros((2 * int(M_z),))
    R_all = np.zeros((2 * int(M_z), 2 * int(M_z)))
    # to build H
    M_v = np.zeros((2 * int(M_z), 4 * Nv))
    M_f = np.zeros((2 * int(M_z), 2 * Nf))

    # build covariance
    D = np.zeros((4 * Nv, 4 * Nv))
    E = np.zeros((4 * Nv, 2 * Nf))
    G = np.zeros((2 * Nf, 2 * Nf))

    theta_f_prev_t = np.zeros((2 * Nf,))
    theta_v_prev_t = np.zeros((4 * Nv,))
    theta = np.zeros((4 * Nv + 2 * Nf,)) * np.nan
    C_theta = np.zeros((4 * Nv + 2 * Nf, 4 * Nv + 2 * Nf)) * np.nan

    P = np.hstack((np.eye(2), np.zeros((2, 2))))

    m = 0
    for f in range(Nf):
        # belief on feature at previous time
        G[f*2:(f+1)*2, f*2:(f+1)*2] = G[f*2:(f+1)*2, f*2:(f+1)*2] + np.linalg.inv(Cf_b[:, f * 2:(f + 1) * 2])
        theta_f_prev_t[f * 2:(f + 1) * 2] = np.transpose(mu_f_b[f, :])

        if VFconnect[f] == 1:  # for subset of features
            conn = 1
            # build complete set of V2F measurements
            m += 1
            meas_z_vec[(m - 1) * 2: m * 2] = z_FV[f, :]  # prendo tutte le mis di tutte le fea di tutti e le ordino una sotto l'altra
            # V2F complete  covariance
            R_all[(m-1)*2:m*2, (m-1)*2:m*2] = C_z_FV[:, f*2:(f+1)*2]
            # build H
            M_v[(m-1)*2:m*2, :] = -P
            M_f[(m-1)*2:m*2, f*2:(f+1)*2] = np.eye(2)

            # matrice cov(only measurements are considered now)
            D = D + np.matmul(np.transpose(P), np.matmul(np.linalg.inv(C_z_FV[:, f*2:(f+1)*2]), P))
            E[0:4, f*2:(f+1)*2] = np.vstack((-np.linalg.inv(C_z_FV[:, f*2:(f+1)*2]), np.zeros((2, 2))))
            G[f*2:(f+1)*2, f*2:(f+1)*2] = G[f*2:(f+1)*2, f*2:(f+1)*2] + np.linalg.inv(C_z_FV[:, f*2:(f+1)*2])

    # GPS vehicle likelihood + belief on vehicle at previous time
    D = D + np.linalg.inv(Cv_b) + np.matmul(np.transpose(P), np.matmul(np.linalg.inv(Cv_measure_gps), P))  # block_diag(np.linalg.inv(Cv_measure_gps[:, i*2:(i+1)*2]), np.zeros((2, 2)))
    theta_v_prev_t = np.transpose(mu_v_b[0:4])

    # GPS complete covariance
    Cv_gps_diag = Cv_measure_gps

    # Covariance Estimate
    C_theta_inv = np.vstack((np.hstack((D, E)), np.hstack((np.transpose(E), G))))
    C_theta_centr = np.linalg.inv(C_theta_inv)

    # Mean Estimate
    H = np.vstack((np.hstack((np.kron(np.eye(Nv), P), np.zeros((2 * Nv, 2 * Nf)))), np.hstack((M_v, M_f))))

    rho = np.hstack((y_v_tot, meas_z_vec))  # GPS meas + V2F meas
    theta_prev = np.hstack((theta_v_prev_t, theta_f_prev_t))
    Q_all = block_diag(Cv_gps_diag, R_all)

    if np.linalg.det(Q_all) == 0:
        ciao = 1

    theta_centr = theta_prev + np.matmul(C_theta_centr, np.matmul(np.transpose(H), np.matmul(np.linalg.inv(Q_all), (
                rho - np.matmul(H, theta_prev)))))
    return C_theta_centr, theta_centr


def generate_target_prior(params_dict, fea):
    featureStatePriorCov = np.tile(params_dict['featurePriorPosStd'] ** 2 * np.eye(2), (1, params_dict['num_targets']))
    featureStatePrior = fea[:, 0:2]
    return featureStatePrior, featureStatePriorCov
