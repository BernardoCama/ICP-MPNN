import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_max, scatter_add

class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(MLP, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)# Models



# Building models
# 2 encoder MLPs (1 for nodes (only second part), 1 for edges) that provide the initial node and edge embeddings 
# final binary classifier for edges
class MLPGraphIndependent(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing.
    It consists of two MLPs, one for nodes and one for edges, and they are applied independently to node and edge
    features, respectively.
    """

    def __init__(self, edge_in_dim = None, node_in_dim = None, edge_out_dim = None, node_out_dim = None,
                 node_fc_dims = None, edge_fc_dims = None, dropout_p = None, use_batchnorm = None):
        super(MLPGraphIndependent, self).__init__()

        if node_in_dim is not None :
            self.node_mlp = MLP(input_dim=node_in_dim, fc_dims=list(node_fc_dims) + [node_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.node_mlp = None

        if edge_in_dim is not None :
            self.edge_mlp = MLP(input_dim=edge_in_dim, fc_dims=list(edge_fc_dims) + [edge_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.edge_mlp = None

    def forward(self, edge_feats = None, nodes_feats = None):

        if self.node_mlp is not None:
            out_node_feats = self.node_mlp(nodes_feats)

        else:
            out_node_feats = nodes_feats

        if self.edge_mlp is not None:
            out_edge_feats = self.edge_mlp(edge_feats)

        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats

################################################################################################################
# To perform forward of edge model used in the 'core' Message Passing Network
class EdgeModel(nn.Module):
    """
    Class used to peform the edge update during Neural message passing
    """
    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        # E X 96 = (E X 32) (E X 32) (E X 16)
        out = torch.cat([source, target, edge_attr], dim=1)
        
        # return      E X 16
        return self.edge_mlp(out)

################################################################################################################
# To perform forward of node models:
# 2 node update (fut, past) + 1 node update model
class TimeAwareNodeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self, flow_in_mlp, flow_out_mlp, node_mlp, node_agg_fn):
        super(TimeAwareNodeModel, self).__init__()

        self.flow_in_mlp = flow_in_mlp   # in 48, out 32
        self.flow_out_mlp = flow_out_mlp # in 48, out 32
        self.node_mlp = node_mlp         # in 64, out 32
        self.node_agg_fn = node_agg_fn # 'sum'

                    # (N X 32)  (2 X E) (E X 16)
    def forward(self, x, edge_index, edge_attr): # eg N=915, E=93772, FEAT_EDGES=16, FEAT_NODES=32
        row, col = edge_index
        flow_out_mask = row < col
        flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]
        #  (E/2 X 48)           # (E/2 X 32)         (E/2 X 16)
        flow_out_input = torch.cat([x[flow_out_col], edge_attr[flow_out_mask]], dim=1)                                
        flow_out = self.flow_out_mlp(flow_out_input) # eg flow_out_input = E/2 X 48
        # N X 32                  # (E/2 X 32)     E/2          N 
        flow_out = self.node_agg_fn(flow_out, flow_out_row, x.size(0))

        flow_in_mask = row > col
        flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]
        flow_in_input = torch.cat([x[flow_in_col], edge_attr[flow_in_mask]], dim=1)
        flow_in = self.flow_in_mlp(flow_in_input)
        flow_in = self.node_agg_fn(flow_in, flow_in_row, x.size(0))
        
        # (N X 64)        (N X 32) (N X 32)
        flow = torch.cat((flow_in, flow_out), dim=1)
        
                # return (N X 32)
        return self.node_mlp(flow)

################################################################################################################
class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)

        Returns: Updated Node and Edge Feature matrices

        """
        row, col = edge_index  # row = first row, col = second row

        # Edge Update
        if self.edge_model is not None:#(E X 32) (E X 32)      (E X 32)
            edge_attr = self.edge_model(x[row],        x[col],       edge_attr)

        # Node Update
        if self.node_model is not None:
             #                (N X 32)  (2 X E)     (E X 16)
            x = self.node_model(x,      edge_index, edge_attr)

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)


class Net(torch.nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder MLPs (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update MLPs (3 for nodes, 1 per edges) used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output.

    This class was initially based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """
    def __init__(self, params= None):
        super(Net, self).__init__()
        

        self.node_cnn = None # CNN used to encode bounding box apperance information.

        # Assign paramters
        self.params = params
        if isinstance(self.params, dict):
            for key, value in self.params.items():
                setattr(self, key, value)
 
        # Define Encoder and Classifier Networks
        # parameters
   
        # Change to encode directly positions of 8 angles
        # encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 2048, 'node_fc_dims': [128], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}
        encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 3*8, 'node_fc_dims': [72], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}
        classifier_feats_dict = {'edge_in_dim': 16, 'edge_fc_dims': [8], 'edge_out_dim': 1, 'dropout_p': 0, 'use_batchnorm': False}

        # 2 encoder MLPs (1 for nodes (only second part), 1 for edges) that provide the initial node and edge embeddings 
        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        # final binary classifier for edges
        self.classifier = MLPGraphIndependent(**classifier_feats_dict)

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(encoder_feats_dict=encoder_feats_dict)

        if (self.params is None) or ("num_enc_steps" not in self.params):  
            self.num_enc_steps = 1#12 #  Number of message passing steps
            self.num_class_steps = 1#11 #  Number of message passing steps during feature vectors are classified (after Message Passing)


    # Building core network, called in __init__
    # - 4 update MLPs (3 for nodes, 1 per edges) used in the 'core' Message Passing Network
    def _build_core_MPNet(self, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = 'sum'
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all multi-layer perceptrons (MLPs) involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = False # Determines whether initially encoded node feats are used during node updates
        self.reattach_initial_edges = True  # Determines whether initially encoded edge feats are used during node updates

        edge_factor = 2 if self.reattach_initial_edges else 1  # 2
        node_factor = 2 if self.reattach_initial_nodes else 1  # 1
                          # 1 * 2 * 32 + 2 * 16
        # 96
        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict[
            'edge_out_dim'] 
        # 48
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']


        # Define all MLPs used within the MPN
        edge_mlp = MLP(input_dim=96,
                       fc_dims=[80, 16],
                       dropout_p=0,
                       use_batchnorm=0)

        flow_in_mlp = MLP(input_dim=48,
                          fc_dims=[56, 32],
                          dropout_p=0,
                          use_batchnorm=0)

        flow_out_mlp = MLP(input_dim=48,
                           fc_dims=[56, 32],
                           dropout_p=0,
                           use_batchnorm=0)

        node_mlp = nn.Sequential(*[nn.Linear(2 * encoder_feats_dict['node_out_dim'], # 64
                                 encoder_feats_dict['node_out_dim']), # 32
                                   nn.ReLU(inplace=True)])

        # Define all MLPs used within the MPN
        return MetaLayer(edge_model=EdgeModel(edge_mlp = edge_mlp), # edge update model
                                                                     # 2 node update (fut, past) + 1 node update model
                         node_model=TimeAwareNodeModel(flow_in_mlp = flow_in_mlp,  # in 48, out 32
                                                       flow_out_mlp = flow_out_mlp, # in 48, out 32
                                                       node_mlp = node_mlp, # in 64, out 32
                                                       node_agg_fn = node_agg_fn)) # aggregation function (sum)
        

    def forward(self, data):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix
              - edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index) #edges X #edge_features (6)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First encoding of node images of boxes/features
        x_is_img = len(x.shape) == 4
        if self.node_cnn is not None and x_is_img:
            x = self.node_cnn(x)

            emb_dists = nn.functional.pairwise_distance(x[edge_index[0]], x[edge_index[1]]).view(-1, 1)
            edge_attr = torch.cat((edge_attr, emb_dists), dim = 1)

        # Encoding features step
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x) # edge_attr (E X 6), x are the node features (N X 2048) -> change to (N X 8)
        # latent_edge_feats (E X 16)
        # latent_node_feats (N X 32)
        
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.num_enc_steps - self.num_class_steps + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
        outputs_dict = {'classified_edges': []}
        for step in range(1, self.num_enc_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges: # True 
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes: # False 
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)
 
            # Message Passing Step                            (N X 32)           (2 X E)      (E X 32)
            #  (N X 32)        (E X 16)
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            if step >= first_class_step:
                # Classification Step               (E X 16)
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict['classified_edges'].append(dec_edge_feats)

        if self.num_enc_steps == 0:
            dec_edge_feats, _ = self.classifier(latent_edge_feats)
            outputs_dict['classified_edges'].append(dec_edge_feats)

        return outputs_dict
        





class Net_FP(torch.nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder MLPs (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update MLPs (3 for nodes, 1 per edges) used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output to classify active edges.
    - 1 node classifier MLP that performs binary classification over the Message Passing Network's output to classify FP.

    """
    def __init__(self, params= None):
        super(Net_FP, self).__init__()
        

        self.node_cnn = None # CNN used to encode bounding box apperance information.

        # Assign paramters
        self.params = params
        if isinstance(self.params, dict):
            for key, value in self.params.items():
                setattr(self, key, value)
 
        # Define tasks
        if (self.params is None) or ("edge_classification_task" not in self.params):  
            self.edge_classification_task = 1 # default solve assocoaition
        if (self.params is None) or ("node_classification_task" not in self.params):  
            self.node_classification_task = 0


        # Define Encoder and Classifier Networks
        # parameters
   
        # Change to encode directly positions of 8 angles
        # encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 2048, 'node_fc_dims': [128], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}
        encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 3*8, 'node_fc_dims': [72], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}

        if self.edge_classification_task:
            classifier_feats_dict = {'edge_in_dim': 16, 'edge_fc_dims': [8], 'edge_out_dim': 1, 'dropout_p': 0, 'use_batchnorm': False}
            # final binary classifier for edges
            self.classifier = MLPGraphIndependent(**classifier_feats_dict)

        if self.node_classification_task:
            node_classifier_feats_dict = {'edge_in_dim': 32, 'edge_fc_dims': [16], 'edge_out_dim': 1, 'dropout_p': 0, 'use_batchnorm': False}
            # final binary classifier for nodes FP
            self.node_classifier = MLPGraphIndependent(**node_classifier_feats_dict)

        # 2 encoder MLPs (1 for nodes (only second part), 1 for edges) that provide the initial node and edge embeddings 
        self.encoder = MLPGraphIndependent(**encoder_feats_dict)


        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(encoder_feats_dict=encoder_feats_dict)

        if (self.params is None) or ("num_enc_steps" not in self.params):  
            self.num_enc_steps = 1#12 #  Number of message passing steps
            self.num_class_steps = 1#11 #  Number of message passing steps during feature vectors are classified (after Message Passing)


    # Building core network, called in __init__
    # - 4 update MLPs (3 for nodes, 1 per edges) used in the 'core' Message Passing Network
    def _build_core_MPNet(self, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = 'sum'
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all multi-layer perceptrons (MLPs) involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = 0 # Determines whether initially encoded node feats are used during node updates     (skip-connection)
        self.reattach_initial_edges = True  # Determines whether initially encoded edge feats are used during node updates (skip-connection)

        edge_factor = 2 if self.reattach_initial_edges else 1  # 2
        node_factor = 2 if self.reattach_initial_nodes else 1  # 1
                          # 1 * 2 * 32 + 2 * 16
        # 96
        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict[
            'edge_out_dim'] 
        # 48
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']


        # Define all MLPs used within the MPN
        edge_mlp = MLP(input_dim=96,
                       fc_dims=[80, 16],
                       dropout_p=0,
                       use_batchnorm=0)

        flow_in_mlp = MLP(input_dim=48,
                          fc_dims=[56, 32],
                          dropout_p=0,
                          use_batchnorm=0)

        flow_out_mlp = MLP(input_dim=48,
                           fc_dims=[56, 32],
                           dropout_p=0,
                           use_batchnorm=0)

        node_mlp = nn.Sequential(*[nn.Linear(2 * encoder_feats_dict['node_out_dim'], # 64
                                 encoder_feats_dict['node_out_dim']), # 32
                                   nn.ReLU(inplace=True)])

        # Define all MLPs used within the MPN
        return MetaLayer(edge_model=EdgeModel(edge_mlp = edge_mlp), # edge update model
                                                                     # 2 node update (fut, past) + 1 node update model
                         node_model=TimeAwareNodeModel(flow_in_mlp = flow_in_mlp,  # in 48, out 32
                                                       flow_out_mlp = flow_out_mlp, # in 48, out 32
                                                       node_mlp = node_mlp, # in 64, out 32
                                                       node_agg_fn = node_agg_fn)) # aggregation function (sum)
        

    def forward(self, data):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix
              - edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index) #edges X #edge_features (6)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First encoding of node images of boxes/features
        x_is_img = len(x.shape) == 4
        if self.node_cnn is not None and x_is_img:
            x = self.node_cnn(x)

            emb_dists = nn.functional.pairwise_distance(x[edge_index[0]], x[edge_index[1]]).view(-1, 1)
            edge_attr = torch.cat((edge_attr, emb_dists), dim = 1)

        # Encoding features step
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x) # edge_attr (E X 6), x are the node features (N X 2048) -> change to (N X 8)
        # latent_edge_feats (E X 16)
        # latent_node_feats (N X 32)
        
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.num_enc_steps - self.num_class_steps + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
        
        if self.edge_classification_task and self.node_classification_task:
            outputs_dict = {'classified_edges': [], 'classified_nodes':[]}
        elif self.edge_classification_task:
            outputs_dict = {'classified_edges': []}
        elif self.node_classification_task: 
            outputs_dict = {'classified_nodes':[]}
        
        for step in range(1, self.num_enc_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges: # True 
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes: # True 
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)
 
            # Message Passing Step                            (N X 32)           (2 X E)      (E X 32)
            #  (N X 32)        (E X 16)
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            if step >= first_class_step:

                if self.edge_classification_task:
                    # Edge Classification Step               (E X 16)
                    dec_edge_feats, _ = self.classifier(latent_edge_feats)
                    outputs_dict['classified_edges'].append(dec_edge_feats)

                if self.node_classification_task:
                    # Node Classification Step
                    dec_node_feats, _ = self.node_classifier(latent_node_feats)
                    outputs_dict['classified_nodes'].append(dec_node_feats)

        if self.num_enc_steps == 0:

            if self.edge_classification_task:
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict['classified_edges'].append(dec_edge_feats)

            if self.node_classification_task:
                dec_node_feats, _ = self.node_classifier(latent_node_feats)
                outputs_dict['classified_nodes'].append(dec_node_feats)

        return outputs_dict

























































































































































































































































































































































































































































































































































