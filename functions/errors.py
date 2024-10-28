import numpy as np

def compute_errors(params_dict, vehicleStateGT, Fea_vect, vehicleEstimateKF, estimateICPDA, connectivity_matrix_gt):
    err_square_GPS = np.zeros((params_dict['num_vehicles'],params_dict['timeSteps']))
    err_square_ICPDA = np.zeros((params_dict['num_vehicles'],params_dict['timeSteps']))
    err_square_ICPDA_fea = np.zeros((params_dict['num_targets'],params_dict['timeSteps']))

    for time in range(params_dict['timeSteps']):
        for i in range(params_dict['num_vehicles']):
            # x and y
            err_square_GPS[i, time] = (vehicleEstimateKF[i*4, time] - vehicleStateGT[i*4, time])**2 + (vehicleEstimateKF[i*4+1, time]-vehicleStateGT[i*4+1, time])**2
            err_square_ICPDA[i, time] = (estimateICPDA[i*4, time] - vehicleStateGT[i*4, time])**2 + (estimateICPDA[i*4+1, time]-vehicleStateGT[i*4+1, time])**2

        for f in range(params_dict['num_targets']):
            err_square_ICPDA_fea[f, time] = (estimateICPDA[f*2+4*params_dict['num_vehicles'], time] - Fea_vect[f*4,time])**2 + \
                                            (estimateICPDA[f*2+4*params_dict['num_vehicles']+1, time] - Fea_vect[f*4+1,time])**2
            if err_square_ICPDA_fea[f, time] == 0:
                err_square_ICPDA_fea[f, time] = np.nan
    return err_square_GPS, err_square_ICPDA, err_square_ICPDA_fea


def compute_errors_single(params_dict, vehicleStateGT, Fea_vect, vehicleEstimateKF, estimateICPDA, connectivity_matrix_gt):
    err_square_GPS = np.zeros((params_dict['num_vehicles'],params_dict['timeSteps']))
    err_square_ICPDA = np.zeros((params_dict['num_vehicles'],params_dict['timeSteps']))
    err_square_ICPDA_fea = np.zeros((params_dict['num_targets'],params_dict['timeSteps']))

    for time in range(params_dict['timeSteps']):
        for i in range(params_dict['num_vehicles']):
            err_square_GPS[i, time] = (vehicleEstimateKF[i*4, time] - vehicleStateGT[i*4, time])**2 + (vehicleEstimateKF[i*4+1, time]-vehicleStateGT[i*4+1, time])**2
            err_square_ICPDA[i, time] = (estimateICPDA[i][0, time] - vehicleStateGT[i*4, time])**2 + (estimateICPDA[i][1, time]-vehicleStateGT[i*4+1, time])**2

        # for f in range(params_dict['num_targets']):
        #    err_square_ICPDA_fea[f, time] = (estimateICPDA[i][f*2+4*params_dict['num_vehicles'], time] - Fea_vect[f*4,time])**2 + \
        #                                    (estimateICPDA[f*2+4*params_dict['num_vehicles']+1, time] - Fea_vect[f*4+1,time])**2
        #    if err_square_ICPDA_fea[f, time] == 0:
        #        err_square_ICPDA_fea[f, time] = np.nan
    return err_square_GPS, err_square_ICPDA, err_square_ICPDA_fea