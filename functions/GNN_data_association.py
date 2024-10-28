import numpy as np
import torch
import os
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops, to_networkx
import networkx as nx

from .GNN.Classes.train_validate_fun import * 
from .GNN.Classes.model import Net

MODELS_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'GNN', 'Models')

def create_graph (boxes_vehicles, id_target, connectivityMatrixICPDA, predicted_id_target):

    number_vehicles = len(boxes_vehicles)

    connectivity_matrix_dict = []
    i = 0
    FP_count = 0
    for vehicle in range(20):
        connectivity_matrix_dict.append({}) 
        for name in id_target[vehicle]:
            connectivity_matrix_dict[vehicle][name] = i
            i += 1


    edge_index = []   # indexes of edges
    edge_labels = []  # labels of edges
    edge_attr = []    # attributes of edges
    source_nodes = []
    dest_nodes = []
    x = {}            
    node_id = {}      # id at the node
    feature_name = {} # name of the feature
    measure_id = {}   # id of the measurement
    track_id = {}     # identificative of the tracking
    node_labels = {}

    node = 0

    for vehicle in range(number_vehicles):

        feature = 0

        num_measurements = boxes_vehicles[vehicle].shape[0]

        for name in id_target[vehicle]:

            box = boxes_vehicles[vehicle][feature,:,:].reshape([3,8,-1])
            
            centroid = np.mean(box, 1)
            
            add_source_node = 0

            for vehicle2 in range(vehicle+1, number_vehicles):

                feature2 = 0

                num_measurements2 = boxes_vehicles[vehicle2].shape[0]

                for name2 in id_target[vehicle2]:

                    box2 = boxes_vehicles[vehicle2][feature2,:,:].reshape([3,8,-1])
                    
                    centroid2 = np.mean(box2, 1)
                    
                    if np.linalg.norm(centroid-centroid2) < 10:  # m

                        # print(vehicle, name, vehicle2, name2)
                    
                        add_source_node = 1

                        node2 = connectivity_matrix_dict[vehicle2][name2]
                        
                        x[node] = [box.flatten().tolist(), vehicle]
                        x[node2] = [box2.flatten().tolist(), vehicle2]

                        node_labels[node] = 0
                        node_labels[node2] = 0

                        node_id[node] = node
                        node_id[node2] = node2 
                        
                        feature_name[node] = name
                        feature_name[node2] = name2   

                        measure_id[node] = feature
                        measure_id[node2] = feature2
                        
                        track_id[node] = name
                        track_id[node2] = name2      
                        
                        source_nodes.append(node)

                        dest_nodes.append(node2)

                        edge_labels.append(1 if (name == name2) else 0)

                        edge_attr.append([box[0,0]-box2[0,0],
                                            box[1,0]-box2[1,0], 
                                            box[2,0]-box2[2,0],
                                            box[0,7]-box2[0,7],
                                            box[1,7]-box2[1,7],
                                            box[2,7]-box2[2,7]])

                    feature2 += 1           

            # Account for measurements that are clearly distinguishable 
            if (not add_source_node) and (node not in [v for k,v in node_id.items()]):
                connectivityMatrixICPDA[vehicle][name] = 1   
                predicted_id_target[vehicle][feature] = name         
                
            node += 1
            
            feature += 1

    # Node features
    x = OrderedDict(sorted(x.items()))
    # Node vehicle (correspondence node_ith - vehicle, order of nodes is the same of x)
    node_vehicle = torch.tensor([v[1] for k, v in x.items()], dtype=torch.float)  
    x = torch.tensor([v[0] for k, v in x.items()], dtype=torch.float)
    # Node labels
    node_labels = OrderedDict(sorted(node_labels.items()))
    node_labels = torch.tensor([v for k, v in node_labels.items()], dtype=torch.long)
    # Edge indexes
    encoder = LabelEncoder().fit(source_nodes + dest_nodes)
    edge_index = torch.tensor([encoder.transform(source_nodes).tolist(), encoder.transform(dest_nodes).tolist()], dtype=torch.long)
    # Edge features
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    # Edge labels
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)
    # Time instant
    time_instant = torch.tensor(-1, dtype=torch.long)
    # Id of each node
    node_id = torch.tensor(encoder.transform([v for k, v in node_id.items()]), dtype=torch.float)
    # Name of each node
    feature_name = OrderedDict(sorted(feature_name.items()))
    feature_name = tuple([v for k, v in feature_name.items()])
    # Id for tracking data association for each node
    track_id = OrderedDict(sorted(track_id.items()))
    track_id = tuple([v for k, v in track_id.items()])
    # Id of the measurement of each vehicle
    measure_id = OrderedDict(sorted(measure_id.items()))
    measure_id = tuple([v for k, v in measure_id.items()])

    data = Data (x = x,
                edge_attr = torch.cat((edge_attr, edge_attr), dim = 0).squeeze(),
                edge_index = torch.cat((edge_index, torch.stack((edge_index[1], edge_index[0]))), dim=1))
    data.node_labels = node_labels
    data.edge_labels = torch.cat((edge_labels, edge_labels), dim = 0)
    data.time_instant = time_instant
    data.node_vehicle = node_vehicle
    data.node_id = node_id
    data.feature_name = feature_name
    data.track_id = track_id
    data.measure_id = measure_id

    return data, connectivityMatrixICPDA, predicted_id_target

def GNN_data_association(params_dict, boxes_vehicles, id_target, connectivity_matrix_gt, previous_boxes_vehicles, curr_veh_pos_list, num_vehicles = 20, num_features = 72):

    connectivityMatrixICPDA = np.zeros((num_vehicles, num_features))
    predicted_id_target = [[-1]*len(boxes_vehicles[v]) for v in range(num_vehicles)]
    misura_FV = np.zeros((2*num_vehicles, num_features))

    # create input graph
    input, connectivityMatrixICPDA, predicted_id_target = create_graph(boxes_vehicles, id_target, connectivityMatrixICPDA, predicted_id_target)

    # load model
    model_dir = os.path.join(MODELS_DIR, params_dict['model_folder'], params_dict['model_name'])

    device = torch.device('cpu')#('cuda')
    model = Net({'num_enc_steps': 8, 'num_class_steps': 7}).to(device)
    model.load_state_dict(torch.load(model_dir))

    # Association First step
    with torch.no_grad():
        output = model(input)
        output = torch.sigmoid(output['classified_edges'][-1])
        output_fractionary = output
        output = (output>0.5).view(-1).float()

    # compute connected components
    edges = input.edge_index.t()
    edgelist_ones = edges[np.where(output==1)[0]].numpy() # does not preserve order
    edgelist_zeros = edges[np.where(output==0)[0]].numpy()   
        
    # Create graph
    G = to_networkx(input, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False) 
    # Consider only edges classified as 1
    G.remove_edges_from([tuple(x) for x in edgelist_zeros]) 
    # Compute connected components
    G_undirected = G.to_undirected(reciprocal=True)
    connected_components = nx.connected_components(G_undirected)
    connected_components = sorted(connected_components)
    # print(f'Number connected components: {len(connected_components)}')
    G_components = [G_undirected.subgraph(c).copy() for c in connected_components]

    # Recover correspondence node-vehicle
    node_vehicle = input.node_vehicle
    # Encode node_vehicle to be used as color mapping
    encoder = LabelEncoder().fit(node_vehicle)
    node_vehicle = encoder.transform(node_vehicle)
    # Recover real id of vehicles
    node_vehicle = encoder.inverse_transform(node_vehicle).astype(int)
    # Recover feature_name associated to the node
    feature_name = input.feature_name
    # Recover number of measurement associated to the node
    measure_id = input.measure_id
    # Node attributes
    x = input.x.reshape(-1, 3, 8).numpy()

    # Recover correspondence node-vehicle
    connected_components_vehicles = []
    # Which vehicles are in each connected component
    connected_components_unique_vehicles = []
    # Avg position of the target bewteen vehicles
    avg_pos = {v:[-1]*len(boxes_vehicles[v]) for v in range(num_vehicles)}
    for el in connected_components:
        el = sorted(el)
        # check association
        # print('Node id:', el, 'Meas. ID:', np.array([measure_id[int(node)] for node in el]), 'Feature name:', np.array([feature_name[int(node)] for node in el]), 'Vehicles:', np.array([node_vehicle[int(node)] for node in el]))
        connected_components_vehicles.append(np.array([node_vehicle[int(node)] for node in el])) 
        connected_components_unique_vehicles.append(np.unique(np.array([node_vehicle[int(node)] for node in el])))

        for node in el:
            mean_box = np.mean([x[int(node)] for node in el], 0)
            # try:
            #     avg_pos[node_vehicle[int(node)]].append(mean_box)
            # except:
            avg_pos[node_vehicle[int(node)]][measure_id[int(node)]] = mean_box

    
    map(connected_components_unique_vehicles.extend, connected_components_unique_vehicles)


    # Association Second step
    for vehicle in range(num_vehicles): 
        j_bb = 0
        j_feat = 0
        for target_avg_pos in avg_pos[vehicle]:

            if isinstance(target_avg_pos, np.ndarray):

                target_avg_pos = np.mean(target_avg_pos, 1)[:2]
                ind_min = 0
                min_ = 100
                i = 0
                for old_target in previous_boxes_vehicles:
                    # old_target_avg_pos = np.mean(old_target, 1)
                    if (np.linalg.norm(target_avg_pos-old_target) < min_) and (i not in predicted_id_target[vehicle]):
                        ind_min = i
                        min_ = np.linalg.norm(target_avg_pos-old_target)
                    i += 1
                if predicted_id_target[vehicle][j_feat] != -1:
                    while (j_feat<len(predicted_id_target[vehicle])) and (predicted_id_target[vehicle][j_feat] != -1):
                        j_feat += 1
                
                try:
                    target_name = ind_min
                    connectivityMatrixICPDA[vehicle][target_name] = 1
                    # if target_name not in predicted_id_target[vehicle]:
                    predicted_id_target[vehicle][j_feat] = target_name
                except:
                    pass

                j_feat = min(j_feat, j_bb)
                j_bb += 1
                j_feat += 1

    association_errors = np.sum(connectivityMatrixICPDA != connectivity_matrix_gt)
    print(f'Association errors:', association_errors)

    # Compute misura_FV
    for vehicle in range(num_vehicles): 
        for target in range(num_features): 
            if connectivityMatrixICPDA[vehicle][target] == 1:
                
                pos_ = np.mean(boxes_vehicles[vehicle][predicted_id_target[vehicle].index(target)], 1)[:2]
                pos_vehicle = curr_veh_pos_list[vehicle][0][:2]
                pos_ = pos_ - pos_vehicle
                misura_FV[2*vehicle][target] = pos_[0]
                misura_FV[2*vehicle+1][target] =  pos_[1]

    return connectivityMatrixICPDA, misura_FV, association_errors



def GNN_data_association_adaptive(params_dict, boxes_vehicles, id_target, connectivity_matrix_gt, previous_boxes_vehicles, curr_veh_pos_list, num_vehicles = 20, num_features = 72):

    connectivityMatrixICPDA = np.zeros((num_vehicles, num_features))
    predicted_id_target = [[-1]*len(boxes_vehicles[v]) for v in range(num_vehicles)]
    misura_FV = np.zeros((2*num_vehicles, num_features))

    # create input graph
    input, connectivityMatrixICPDA, predicted_id_target = create_graph(boxes_vehicles, id_target, connectivityMatrixICPDA, predicted_id_target)

    # load model
    model_dir = os.path.join(MODELS_DIR, params_dict['model_folder'], params_dict['model_name'])

    device = torch.device('cpu')#('cuda')
    model = Net({'num_enc_steps': 8, 'num_class_steps': 7}).to(device)
    model.load_state_dict(torch.load(model_dir))

    # Association First step
    with torch.no_grad():
        output = model(input)
        output = torch.sigmoid(output['classified_edges'][-1])
        output_fractionary = output
        output = (output>0.5).view(-1).float()

    # compute connected components
    edges = input.edge_index.t()
    edgelist_ones = edges[np.where(output==1)[0]].numpy() # does not preserve order
    edgelist_zeros = edges[np.where(output==0)[0]].numpy()   
        
    # Create graph
    G = to_networkx(input, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False) 
    # Consider only edges classified as 1
    G.remove_edges_from([tuple(x) for x in edgelist_zeros]) 
    # Compute connected components
    G_undirected = G.to_undirected(reciprocal=True)
    connected_components = nx.connected_components(G_undirected)
    connected_components = sorted(connected_components)
    # print(f'Number connected components: {len(connected_components)}')
    G_components = [G_undirected.subgraph(c).copy() for c in connected_components]

    # Recover correspondence node-vehicle
    node_vehicle = input.node_vehicle
    # Encode node_vehicle to be used as color mapping
    encoder = LabelEncoder().fit(node_vehicle)
    node_vehicle = encoder.transform(node_vehicle)
    # Recover real id of vehicles
    node_vehicle = encoder.inverse_transform(node_vehicle).astype(int)
    # Recover feature_name associated to the node
    feature_name = input.feature_name
    # Recover number of measurement associated to the node
    measure_id = input.measure_id
    # Node attributes
    x = input.x.reshape(-1, 3, 8).numpy()

    # Recover correspondence node-vehicle
    connected_components_vehicles = []
    # Which vehicles are in each connected component
    connected_components_unique_vehicles = []
    # Avg position of the target bewteen vehicles
    avg_pos = {v:[-1]*len(boxes_vehicles[v]) for v in range(num_vehicles)}
    for el in connected_components:
        el = sorted(el)
        # check association
        # print('Node id:', el, 'Meas. ID:', np.array([measure_id[int(node)] for node in el]), 'Feature name:', np.array([feature_name[int(node)] for node in el]), 'Vehicles:', np.array([node_vehicle[int(node)] for node in el]))
        connected_components_vehicles.append(np.array([node_vehicle[int(node)] for node in el])) 
        connected_components_unique_vehicles.append(np.unique(np.array([node_vehicle[int(node)] for node in el])))

        for node in el:
            mean_box = np.mean([x[int(node)] for node in el], 0)
            # try:
            #     avg_pos[node_vehicle[int(node)]].append(mean_box)
            # except:
            avg_pos[node_vehicle[int(node)]][measure_id[int(node)]] = mean_box

    
    map(connected_components_unique_vehicles.extend, connected_components_unique_vehicles)


    # Association Second step
    for vehicle in range(num_vehicles): 


        # If vehicle has no recorded targets
        if len(previous_boxes_vehicles[vehicle]) == 0:

            j_bb = 0
            for target_avg_pos in avg_pos[vehicle]:

                if isinstance(target_avg_pos, np.ndarray):
        
                    target_avg_pos = np.mean(target_avg_pos, 1)[:2]      

                    # choose as first target name the real name 
                    target_name = id_target[vehicle][j_bb]
                    previous_boxes_vehicles[vehicle][target_name] = target_avg_pos

                    connectivityMatrixICPDA[vehicle][target_name] = 1

                    predicted_id_target[vehicle][j_bb] = target_name

                j_bb += 1

        else:
                        
            j_bb = 0
            j_feat = 0

            for target_avg_pos in avg_pos[vehicle]:

                if isinstance(target_avg_pos, np.ndarray):

                    target_avg_pos = np.mean(target_avg_pos, 1)[:2]
                    ind_min = 0
                    min_ = 100
                    i = 0
                    for name, old_target in previous_boxes_vehicles[vehicle].items():
                        # old_target_avg_pos = np.mean(old_target, 1)
                        if (np.linalg.norm(target_avg_pos-old_target) < min_) and (name not in predicted_id_target[vehicle]):
                            ind_min = name
                            min_ = np.linalg.norm(target_avg_pos-old_target)
                        i += 1
                    
                    # if new measurement is close to the old measurements -> new target belongs to the old targets set
                    if min_ < 5:
                        if predicted_id_target[vehicle][j_feat] != -1:
                            while (j_feat<len(predicted_id_target[vehicle])) and (predicted_id_target[vehicle][j_feat] != -1):
                                j_feat += 1
                        
                        try:
                            target_name = ind_min
                            connectivityMatrixICPDA[vehicle][target_name] = 1
                            # if target_name not in predicted_id_target[vehicle]:
                            predicted_id_target[vehicle][j_feat] = target_name

                            # NO
                            # previous_boxes_vehicles[vehicle][target_name] = target_avg_pos
                        except:
                            pass
                    
                    # it is a new target
                    else:

                        target_name = id_target[vehicle][j_bb]

                        connectivityMatrixICPDA[vehicle][target_name] = 1

                        predicted_id_target[vehicle][j_bb] = target_name

                        # NO
                        # previous_boxes_vehicles[vehicle][target_name] = target_avg_pos

                    j_feat = min(j_feat, j_bb)
                    j_feat += 1
                j_bb += 1


    association_errors = np.sum(connectivityMatrixICPDA != connectivity_matrix_gt)
    print(f'Association errors:', association_errors)

    # Compute misura_FV
    for vehicle in range(num_vehicles): 
        for target in range(num_features): 
            if connectivityMatrixICPDA[vehicle][target] == 1:
                
                pos_ = np.mean(boxes_vehicles[vehicle][predicted_id_target[vehicle].index(target)], 1)[:2]
                pos_vehicle = curr_veh_pos_list[vehicle][0][:2]
                pos_ = pos_ - pos_vehicle
                misura_FV[2*vehicle][target] = pos_[0]
                misura_FV[2*vehicle+1][target] =  pos_[1]

    return connectivityMatrixICPDA, misura_FV, previous_boxes_vehicles, association_errors
