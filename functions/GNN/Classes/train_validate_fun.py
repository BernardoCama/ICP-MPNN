import torch
import pandas as pd
import torch.nn.functional as F


def train(device, model, optimizer, train_loader):
    
    type_ = 'train'
    
    model.train()

    logs_all = []
    
    for data in train_loader:
        
        data = data.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = _compute_loss(output, data)
        
        metrics, _ = compute_perform_metrics(output, data)
        
        logs = {**metrics, **{'loss': loss.item()}}
        
        log = {key + f'/{type_}': val for key, val in logs.items()}
        
        loss.backward()

        logs_all.append(log)
        
        optimizer.step()
    
    return epoch_end(logs_all)


def _compute_loss(outputs, batch):
    
    # Define Balancing weight
    positive_vals = batch.edge_labels.sum()

    if positive_vals:
        pos_weight = torch.tensor([(batch.edge_labels.shape[0] - positive_vals) / positive_vals], dtype=torch.float)

    else: # If there are no positives labels, avoid dividing by zero
        pos_weight = torch.tensor([0], dtype=torch.float)

    # Compute Weighted BCE:
    loss = 0
    num_steps = len(outputs['classified_edges'])
    for step in range(num_steps):
        loss += F.binary_cross_entropy_with_logits(outputs['classified_edges'][step].view(-1).float(),
                                                        batch.edge_labels.view(-1).float(), #reshape
                                                        pos_weight= pos_weight)
    return loss




def evaluate(device, model, loader, thr=None):
    type_ = 'val'
    model.eval()

    logs_all = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)        
            
            output = model(data)
            
            loss = _compute_loss(output, data)
            
            metrics, _ = compute_perform_metrics(output, data, thr)
            
            logs = {**metrics, **{'loss': loss.item()}}
            
            log = {key + f'/{type_}': val for key, val in logs.items()}

            logs_all.append(log)            
            
    return epoch_end(logs_all)
    
def epoch_end(outputs):
    metrics = pd.DataFrame(outputs).mean(axis=0).to_dict()
    metrics = {metric_name: torch.as_tensor(metric).item() for metric_name, metric in metrics.items()}
    return metrics


# In[678]:


def compute_perform_metrics(graph_out, graph_obj, thr=None):
    """
    Computes both classification metrics and constraint satisfaction rate
    Args:
        graph_out: output of MPN, dict with key 'classified' edges, and val a list torch.Tensor of unnormalized loggits for
        every edge, at every messagepassing step.
        graph_obj: Graph Object

    Returns:
        dictionary with metrics summary
    """
    if thr is None:
        thr = 0.5
    
    edges_out = graph_out['classified_edges'][-1]
    with torch.no_grad():
        edges_out = (torch.sigmoid(edges_out.view(-1)) > thr).float()

    # Compute Classification Metrics
    class_metrics, confusion_matrix = fast_compute_class_metric(edges_out, graph_obj.edge_labels)

    return class_metrics, confusion_matrix

def fast_compute_class_metric(test_preds, test_sols, class_metrics = ('accuracy', 'recall', 'precision')):
    """
    Computes manually (i.e. without sklearn functions) accuracy, recall and predicision.

    Args:
        test_preds: numpy array/ torch tensor of size N with discrete output vals
        test_sols: numpy array/torch tensor of size N with binary labels
        class_metrics: tuple with a subset of values from ('accuracy', 'recall', 'precision') indicating which
        metrics to report

    Returns:
        dictionary with values of accuracy, recall and precision
    """
    with torch.no_grad():

        TP = ((test_sols == 1) & (test_preds == 1)).sum().float()
        FP = ((test_sols == 0) & (test_preds == 1)).sum().float()
        TN = ((test_sols == 0) & (test_preds == 0)).sum().float()
        FN = ((test_sols == 1) & (test_preds == 0)).sum().float()

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0)
        precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0)

    class_metrics_dict =  {'accuracy': accuracy.item(), 'recall': recall.item(), 'precision': precision.item()}
    class_metrics_dict = {met_name: class_metrics_dict[met_name] for met_name in class_metrics}
    
    return class_metrics_dict, [TP.tolist(), FP.tolist(),  FN.tolist(), TN.tolist()]





##################################################################################################################################################################





def train_FP(device, model, optimizer, train_loader):

    node_classification_task = model.node_classification_task
    edge_classification_task = model.edge_classification_task
    
    type_ = 'train'
    
    model.train()

    logs_all = []
    
    for data in train_loader:
        
        data = data.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = _compute_loss_FP(output, data, node_classification_task=node_classification_task, edge_classification_task=edge_classification_task)
        
        metrics, _  = compute_perform_metrics_FP(output, data, node_classification_task=node_classification_task, edge_classification_task=edge_classification_task)
        
        logs = {**metrics, **{'loss': loss.item()}}
        
        log = {key + f'/{type_}': val for key, val in logs.items()}
        
        loss.backward()

        logs_all.append(log)
        
        optimizer.step()
    
    return epoch_end(logs_all)


def _compute_loss_FP(outputs, batch, node_classification_task=None, edge_classification_task=None):
    
    if edge_classification_task:
        # Define Balancing weight for edges
        positive_vals = batch.edge_labels.sum()
        if positive_vals:
            pos_weight_edges = torch.tensor([(batch.edge_labels.shape[0] - positive_vals) / positive_vals], dtype=torch.float)
        else: # If there are no positives labels, avoid dividing by zero
            pos_weight_edges = torch.tensor([0], dtype=torch.float)

        # Compute Weighted BCE: 
        loss_edges = 0
        num_steps = len(outputs['classified_edges'])
        for step in range(num_steps):
            loss_edges += F.binary_cross_entropy_with_logits(outputs['classified_edges'][step].view(-1).float(),
                                                            batch.edge_labels.view(-1).float(), #reshape
                                                            pos_weight= pos_weight_edges)

    if node_classification_task:
        # Define Balancing weight for nodes
        positive_vals = batch.node_labels.sum()
        if positive_vals:
            pos_weight_nodes = torch.tensor([(batch.node_labels.shape[0] - positive_vals) / positive_vals], dtype=torch.float)
        else: # If there are no positives labels, avoid dividing by zero
            pos_weight_nodes = torch.tensor([0], dtype=torch.float)

        loss_nodes = 0
        num_steps = len(outputs['classified_nodes'])
        for step in range(num_steps):
            loss_nodes += F.binary_cross_entropy_with_logits(outputs['classified_nodes'][step].view(-1).float(),
                                                            batch.node_labels.view(-1).float(), #reshape
                                                            pos_weight= pos_weight_nodes)

    if edge_classification_task and node_classification_task:
        # loss = loss_edges + loss_nodes
        loss = torch.sqrt(loss_edges * loss_nodes)
    elif edge_classification_task:
        loss = loss_edges
    elif node_classification_task:
        loss = loss_nodes

    return loss




def evaluate_FP(device, model, loader, thr=None):
    type_ = 'val'

    node_classification_task = model.node_classification_task
    edge_classification_task = model.edge_classification_task

    model.eval()

    logs_all = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)        
            
            output = model(data)
            
            loss = _compute_loss_FP(output, data, node_classification_task=node_classification_task, edge_classification_task=edge_classification_task)
            
            metrics, _  = compute_perform_metrics_FP(output, data, thr=thr, node_classification_task=node_classification_task, edge_classification_task=edge_classification_task)
            
            logs = {**metrics, **{'loss': loss.item()}}
            
            log = {key + f'/{type_}': val for key, val in logs.items()}

            logs_all.append(log)            
            
    return epoch_end(logs_all)



def compute_perform_metrics_FP(graph_out, graph_obj, thr=None, node_classification_task=None, edge_classification_task=None):
    """
    Computes both classification metrics and constraint satisfaction rate
    Args:
        graph_out: output of MPN, dict with key 'classified' edges and nodes, and val a list torch.Tensor of unnormalized loggits for
        every edge, at every messagepassing step.
        graph_obj: Graph Object

    Returns:
        dictionary with metrics summary
    """
    if thr is None:
        thr = 0.5
    
    if edge_classification_task:
        edges_out = graph_out['classified_edges'][-1]
        with torch.no_grad():
            edges_out = (torch.sigmoid(edges_out.view(-1)) > thr).float()
        # Compute Classification Metrics for edges
        class_metrics_edges, confusion_matrix_edges = fast_compute_class_metric(edges_out, graph_obj.edge_labels)

        # Rename metrics
        class_metrics_edges['accuracy_edges'] = class_metrics_edges.pop('accuracy')
        class_metrics_edges['precision_edges'] = class_metrics_edges.pop('precision')
        class_metrics_edges['recall_edges'] = class_metrics_edges.pop('recall')


    if node_classification_task:
        nodes_out = graph_out['classified_nodes'][-1]
        with torch.no_grad():
            nodes_out = (torch.sigmoid(nodes_out.view(-1)) > thr).float()
        # Compute Classification Metrics for edges
        class_metrics_nodes, confusion_matrix_nodes = fast_compute_class_metric(nodes_out, graph_obj.node_labels)

        # Rename metrics
        class_metrics_nodes['accuracy_nodes'] = class_metrics_nodes.pop('accuracy')
        class_metrics_nodes['precision_nodes'] = class_metrics_nodes.pop('precision')
        class_metrics_nodes['recall_nodes'] = class_metrics_nodes.pop('recall')


    if edge_classification_task and node_classification_task:
        return {**class_metrics_edges, **class_metrics_nodes}, [confusion_matrix_edges, confusion_matrix_nodes]
    elif edge_classification_task:
        return class_metrics_edges, confusion_matrix_edges
    elif node_classification_task:
        return class_metrics_nodes, confusion_matrix_nodes

























































































































































































































































































































































































































































































































































