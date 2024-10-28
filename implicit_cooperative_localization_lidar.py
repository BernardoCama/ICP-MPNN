import numpy as np
import scipy.io as sio
from functions.measurements import generate_gps_meas, generate_v2f_meas
from functions.gpsKF import gps_kf
from functions.icp_data_association import icp_with_data_association
from functions.icp_single_vehicle import icp_with_data_association_single
from functions.icp_adaptive import icp_with_data_association_adaptive
from functions.errors import compute_errors, compute_errors_single
import matplotlib.pyplot as plt
import time


def simulation_parameters():
    # Parameters for simulation
    params_dict = dict()
    params_dict['samplingTime'] = 0.2
    params_dict['timeSteps'] = 1500
    params_dict['start_time'] = 100
    params_dict['motionMatrixA'] = np.vstack((np.hstack((np.eye(2), params_dict['samplingTime']*np.eye(2))),
                                              np.hstack((np.zeros((2, 2)), np.eye(2)))))
    params_dict['motionMatrixB'] = np.vstack((params_dict['samplingTime']**2/2*np.eye(2), params_dict['samplingTime']*np.eye(2)))
    params_dict['num_vehicles'] = 20
    params_dict['num_targets'] = 72
    params_dict['n_mc'] = 1
    params_dict['featurePriorPosStd'] = 100
    params_dict['featurePriorPosVel'] = 100
    params_dict['GPSaccuracyPos_x'] = 1
    params_dict['GPSaccuracyPos_y'] = 1
    # params_dict['GPSaccuracyVel_x'] = 0.45
    # params_dict['GPSaccuracyVel_y'] = 0.45
    params_dict['GPSPriorPostd'] = 100
    params_dict['GPSPriorVelStd'] = 100
    params_dict['V2Fstd'] = 0.05
    params_dict['V2FCov'] = params_dict['V2Fstd']**2*np.eye(2)
    params_dict['V2FCovInv'] = np.linalg.inv(params_dict['V2FCov'])
    params_dict['std_v_x'] = 1
    params_dict['std_v_y'] = 1

    # GNN
    params_dict['model_folder'] = 'gaussian_noise'
    params_dict['model_name'] = 'model_50_noise_1.0'

    return params_dict


def cooperative_localization_framework():
    data = sio.loadmat('CARLADatasetPoles.mat')
    params_dict = simulation_parameters()

    estimateICPDA = np.zeros((4*params_dict['num_vehicles']+2*params_dict['num_targets'], params_dict['timeSteps']))
    err_square_GPS = np.zeros((params_dict['num_vehicles'],params_dict['timeSteps'], params_dict['n_mc']))
    err_square_ICPDA = np.zeros((params_dict['num_vehicles'],params_dict['timeSteps'], params_dict['n_mc']))
    errorICPDA_featurePos = np.zeros((params_dict['num_targets'],params_dict['timeSteps'], params_dict['n_mc']))
    incorrectAssociation_mc = np.zeros((params_dict['timeSteps']))

    start_time = time.time()
    for mc in range(params_dict['n_mc']):
        print(f"Monte carlo run {mc+1}/{params_dict['n_mc']}")

        # Generate GPS measurements
        # cv_measure_gps: 2 x 2*num_agents x timesteps 
        # gps_meas: num_agents x 2 x timesteps
        # Covariances, gps meas
        cv_measure_gps, gps_meas = generate_gps_meas(params_dict, data['vehicleStateGT'])

        # Generate V2F measurements
        # z_fv_all: 2*num_agents x num_features x timesteps
        # z_fv_all_boxes: num_features x 3 x 8 x timesteps x num_agents
        # v2f_cov_fv_all: 2*num_agents, 2*num_features, timesteps
        # connectivity_matrix_gt_all: num_agents x num_features x timesteps
        z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, connectivity_matrix_gt_all = generate_v2f_meas(params_dict, data['Fea_vect'], data['Fea_vect_boxes'], data['vehicleStateGT'], data['conn_features'])

        # GPS tracking
        # vehicle_estimate_kf:  num_inputs (4) * num_agents x timesteps
        # vehicle_estimate_kf_cov:  num_inputs (4) * num_agents x num_inputs (4) * num_agents x timesteps
        vehicle_estimate_kf, vehicle_estimate_kf_cov = gps_kf(params_dict, data['vehicleStateGT'], gps_meas, cv_measure_gps)
 
        # ICP-DA solution
        estimateICPDA, estimateICPDACov, numberOfDetection, incorrectAssociation = icp_with_data_association(params_dict, data['Fea_vect_true'], connectivity_matrix_gt_all,
                                                                                                             z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, gps_meas, cv_measure_gps,
                                                                                                             vehicle_estimate_kf, vehicle_estimate_kf_cov)

        # ICP-DA single vehicle
        # estimateICPDA_single, estimateICPDACov_single, numberOfDetection_single, incorrectAssociation_single = icp_with_data_association_single(params_dict, data['Fea_vect_true'], connectivity_matrix_gt_all,
        #                                                                                                      z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, gps_meas, cv_measure_gps,
        #                                                                                                      vehicle_estimate_kf, vehicle_estimate_kf_cov)
        # estimatedICPDA, estimateICPDACov, _, _ = icp_with_data_association_adaptive(params_dict, data['Fea_vect_true'], connectivity_matrix_gt_all,
        #                                                                                                      z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, gps_meas, cv_measure_gps,
        #                                                                                                      vehicle_estimate_kf, vehicle_estimate_kf_cov)
        # sio.savemat('prova_gps.mat', {'pred': vehicle_estimate_kf})
        # incorrectAssociation_mc += incorrectAssociation_single

        # Compute errors
        # err_square_GPS[:, :, mc], err_square_ICPDA[:, :, mc], errorICPDA_featurePos[:, :, mc] = compute_errors(params_dict, data['vehicleStateGT'], data['Fea_vect_true'], vehicle_estimate_kf, estimateICPDA, connectivity_matrix_gt_all)

        # Compute errors single
        # err_square_GPS[:, :, mc], err_square_ICPDA[:, :, mc], errorICPDA_featurePos[:, :, mc] = compute_errors_single(params_dict, data['vehicleStateGT'], data['Fea_vect_true'], vehicle_estimate_kf, estimateICPDA_single, connectivity_matrix_gt_all)

        print(f'Time: {time.time() - start_time}')
    # Plot results
    data_out = {'rmse_GPS': np.sqrt(np.mean(np.mean(err_square_GPS, 2), 0)),
                'rmse_ICPDA': np.sqrt(np.mean(np.mean(err_square_ICPDA, 2), 0)),
                'rmse_features': np.sqrt(np.mean(np.mean(errorICPDA_featurePos, 2), 0)),
                'incorrectAssociation': incorrectAssociation_mc/params_dict['n_mc']}
    txt = f'rmse_gps_acc_{params_dict["GPSaccuracyPos_x"]}.mat'
    sio.savemat(txt, data_out)

    plt.plot(np.sqrt(np.mean(np.mean(err_square_GPS, 2), 0)))
    plt.plot(np.sqrt(np.mean(np.mean(err_square_ICPDA, 2), 0)))
    plt.show()

    for i in range(params_dict['num_vehicles']):
        plt.plot(incorrectAssociation_single[i])
    plt.show()


if __name__ == '__main__':
    cooperative_localization_framework()