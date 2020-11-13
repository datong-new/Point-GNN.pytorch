import torch
from torch import nn
from torch_scatter import scatter_max


def multi_layer_neural_network_fn(Ks):
    linears = []
    for i in range(1, len(Ks)):
        linears += [
        nn.Linear(Ks[i-1], Ks[i]),
        nn.ReLU(),
        nn.BatchNorm1d(Ks[i])]
    return nn.Sequential(*linears)

def multi_layer_fc_fn(Ks=[300, 64, 32, 64], num_classes=4, is_logits=False, num_layers=4):
    assert len(Ks) == num_layers
    linears = []
    for i in range(1, len(Ks)):
        linears += [
                nn.Linear(Ks[i-1], Ks[i]),
                nn.ReLU(),
                nn.BatchNorm1d(Ks[i])
                ]

    if is_logits:
        linears += [
                nn.Linear(Ks[-1], num_classes)]
    else:
        linears += [
                nn.Linear(Ks[-1], num_classes),
                nn.ReLU(),
                nn.BatchNorm1d(num_classes)
                ]
    return nn.Sequential(*linears)

def max_aggregation_fn(features, index, l):
    """
    Arg: features: N x dim
    index: N x 1, e.g.  [0,0,0,1,1,...l,l]
    l: lenght of keypoints

    """
    index = index.unsqueeze(-1).expand(-1, features.shape[-1]) # N x 64
    set_features = torch.zeros((l, features.shape[-1]), device=features.device).permute(1,0).contiguous() # len x 64
    set_features, argmax = scatter_max(features.permute(1,0), index.permute(1,0), out=set_features)
    set_features = set_features.permute(1,0)
    return set_features

def focal_loss_sigmoid(labels, logits, alpha=0.5, gamma=2):
     """
     github.com/tensorflow/models/blob/master/\
         research/object_detection/core/losses.py
     Computer focal loss for binary classification
     Args:
       labels: A int32 tensor of shape [batch_size]. N x 1
       logits: A float32 tensor of shape [batch_size]. N x C
       alpha: A scalar for focal loss alpha hyper-parameter.
       If positive samples number > negtive samples number,
       alpha < 0.5 and vice versa.
       gamma: A scalar for focal loss gamma hyper-parameter.
     Returns:
       A tensor of the same shape as `labels`
     """

     prob = logits.sigmoid()
     labels = torch.nn.functional.one_hot(labels.squeeze().long(), num_classes=prob.shape[1])

     cross_ent = torch.clamp(logits, min=0) - logits * labels + torch.log(1+torch.exp(-torch.abs(logits)))
     prob_t = (labels*prob) + (1-labels) * (1-prob)
     modulating = torch.pow(1-prob_t, gamma)
     alpha_weight = (labels*alpha)+(1-labels)*(1-alpha)

     focal_cross_entropy = modulating * alpha_weight * cross_ent
     return focal_cross_entropy

class PointSetPooling(nn.Module):
    def __init__(self, point_MLP_depth_list=[4, 32, 64, 128, 300], output_MLP_depth_list=[300, 300, 300]):
        super(PointSetPooling, self).__init__()

        Ks = list(point_MLP_depth_list)
        self.point_linears = multi_layer_neural_network_fn(Ks)
        
        Ks = list(output_MLP_depth_list)
        self.out_linears = multi_layer_neural_network_fn(Ks)

    def forward(self, 
            point_features,
            point_coordinates,
            keypoint_indices,
            set_indices):
        """apply a features extraction from point sets.
        Args:
            point_features: a [N, M] tensor. N is the number of points.
            M is the length of the features.
            point_coordinates: a [N, D] tensor. N is the number of points.
            D is the dimension of the coordinates.
            keypoint_indices: a [K, 1] tensor. Indices of K keypoints.
            set_indices: a [S, 2] tensor. S pairs of (point_index, set_index).
            i.e. (i, j) indicates point[i] belongs to the point set created by
            grouping around keypoint[j].
        returns: a [K, output_depth] tensor as the set feature.
        Output_depth depends on the feature extraction options that
        are selected.
        """

        #print(f"point_features: {point_features.shape}")
        #print(f"point_coordinates: {point_coordinates.shape}")
        #print(f"keypoint_indices: {keypoint_indices.shape}")
        #print(f"set_indices: {set_indices.shape}")

        # Gather the points in a set
        point_set_features = point_features[set_indices[:, 0]]
        point_set_coordinates = point_coordinates[set_indices[:, 0]]
        point_set_keypoint_indices = keypoint_indices[set_indices[:, 1]]

        #point_set_keypoint_coordinates_1 = point_features[point_set_keypoint_indices[:, 0]]
        point_set_keypoint_coordinates = point_coordinates[point_set_keypoint_indices[:, 0]]

        point_set_coordinates = point_set_coordinates - point_set_keypoint_coordinates
        point_set_features = torch.cat([point_set_features, point_set_coordinates], axis=-1)

        # Step 1: Extract all vertex_features
        extracted_features = self.point_linears(point_set_features) # N x 64

        # Step 2: Aggerate features using scatter max method.
        #index = set_indices[:, 1].unsqueeze(-1).expand(-1, extracted_features.shape[-1]) # N x 64
        #set_features = torch.zeros((len(keypoint_indices), extracted_features.shape[-1]), device=extracted_features.device).permute(1,0).contiguous() # len x 64
        #set_features, argmax = scatter_max(extracted_features.permute(1,0), index.permute(1,0), out=set_features)
        #set_features = set_features.permute(1,0)

        set_features = max_aggregation_fn(extracted_features, set_indices[:, 1], len(keypoint_indices))

        # Step 3: MLP for set_features
        set_features = self.out_linears(set_features)
        return set_features

class GraphNetAutoCenter(nn.Module):
    def __init__(self, auto_offset=True, auto_offset_MLP_depth_list=[300, 64, 3], edge_MLP_depth_list=[303, 300, 300], update_MLP_depth_list=[300, 300, 300]):
        super(GraphNetAutoCenter, self).__init__()
        self.auto_offset = auto_offset
        self.auto_offset_fn = multi_layer_neural_network_fn(auto_offset_MLP_depth_list)
        self.edge_feature_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.update_fn = multi_layer_neural_network_fn(update_MLP_depth_list)


    def forward(self, input_vertex_features,
        input_vertex_coordinates,
        keypoint_indices,
        edges):
        """apply one layer graph network on a graph. .
        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
        returns: a [N, M] tensor. Updated vertex features.
        """
        #print(f"input_vertex_features: {input_vertex_features.shape}")
        #print(f"input_vertex_coordinates: {input_vertex_coordinates.shape}")
        #print(NOT_USED)
        #print(f"edges: {edges.shape}")

        # Gather the source vertex of the edges
        s_vertex_features = input_vertex_features[edges[:, 0]]
        s_vertex_coordinates = input_vertex_coordinates[edges[:, 0]]

        if self.auto_offset:
            offset = self.auto_offset_fn(input_vertex_features)
            input_vertex_coordinates = input_vertex_coordinates + offset

        # Gather the destination vertex of the edges
        d_vertex_coordinates = input_vertex_coordinates[edges[:, 1]]

        # Prepare initial edge features
        edge_features = torch.cat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], dim=-1)
        
        # Extract edge features
        edge_features = self.edge_feature_fn(edge_features)

        # Aggregate edge features
        aggregated_edge_features = max_aggregation_fn(edge_features, edges[:,1], len(keypoint_indices))

        # Update vertex features
        update_features = self.update_fn(aggregated_edge_features)
        output_vertex_features  = update_features + input_vertex_features
        return output_vertex_features 

class ClassAwarePredictor(nn.Module):
    def __init__(self, num_classes, box_encoding_len):
        super(ClassAwarePredictor, self).__init__()
        self.cls_fn = multi_layer_fc_fn(Ks=[300, 64], num_layers=2, num_classes=num_classes, is_logits=True)
        self.loc_fns = nn.ModuleList()
        self.num_classes = num_classes
        self.box_encoding_len = box_encoding_len

        for i in range(num_classes):
            self.loc_fns += [
                    multi_layer_fc_fn(Ks=[300, 300, 64], num_layers=3, num_classes=box_encoding_len, is_logits=True)]

    def forward(self, features):
        logits = self.cls_fn(features)
        box_encodings_list = []
        for loc_fn in self.loc_fns:
            box_encodings = loc_fn(features).unsqueeze(1)
            box_encodings_list += [box_encodings]

        box_encodings = torch.cat(box_encodings_list, dim=1)
        return logits, box_encodings

class MultiLayerFastLocalGraphModelV2(nn.Module):
    def __init__(self, num_classes, box_encoding_len, regularizer_type=None, \
            regularizer_kwargs=None, layer_configs=None, mode=None, graph_net_layers=3):
        super(MultiLayerFastLocalGraphModelV2, self).__init__()
        self.num_classes = num_classes
        self.box_encoding_len = box_encoding_len
        self.point_set_pooling = PointSetPooling()

        self.graph_nets = nn.ModuleList()
        for i in range(graph_net_layers):
            self.graph_nets.append(GraphNetAutoCenter())

        self.predictor = ClassAwarePredictor(num_classes, box_encoding_len)


    def forward(self, batch, is_training):
        input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
            cls_labels, encoded_boxes, valid_boxes = batch

        point_features, point_coordinates, keypoint_indices, set_indices = input_v, vertex_coord_list[0], keypoint_indices_list[0], edges_list[0]
        point_features = self.point_set_pooling(point_features, point_coordinates, keypoint_indices, set_indices)


        point_coordinates, keypoint_indices, set_indices = vertex_coord_list[1], keypoint_indices_list[1], edges_list[1]
        for i, graph_net in enumerate(self.graph_nets):
            point_features = graph_net(point_features, point_coordinates, keypoint_indices, set_indices)
        logits, box_encodings = self.predictor(point_features)
        return logits, box_encodings

    def postprocess(self, logits):
        softmax = nn.Softmax(dim=1)
        prob = softmax(logits)
        return prob

    def loss(self, logits, labels, pred_box, gt_box, valid_box, 
            cls_loss_type="focal_sigmoid", loc_loss_type='huber_loss', loc_loss_weight=1.0, cls_loss_weight=1.0):
        """Output loss value.
        Args:
            logits: [N, num_classes] tensor. The classification logits from
            predict method.
            labels: [N] tensor. The one hot class labels.
            pred_box: [N, num_classes, box_encoding_len] tensor. The encoded
            bounding boxes from the predict method.
            gt_box: [N, 1, box_encoding_len] tensor. The ground truth encoded
            bounding boxes.
            valid_box: [N] tensor. An indicator of whether the vertex is from
            an object of interest (whether it has a valid bounding box).
            cls_loss_type: string, the type of classification loss function.
            cls_loss_kwargs: dict, keyword args to the classifcation loss.
            loc_loss_type: string, the type of localization loss function.
            loc_loss_kwargs: dict, keyword args to the localization loss.
            loc_loss_weight: scalar, weight on localization loss.
            cls_loss_weight: scalar, weight on the classifcation loss.
        returns: a dict of cls_loss, loc_loss, reg_loss, num_endpoint,
        num_valid_endpoint. num_endpoint is the number of output vertices.
        num_valid_endpoint is the number of output vertices that have a valid
        bounding box. Those numbers are useful for weighting during batching.
        """
        """
        print(f"logits: {logits.shape}")
        print(f"labels: {labels.shape}")
        print(f"pred_box: {pred_box.shape}")
        print(f"gt_box: {gt_box.shape}")
        print(f"valid_box: {valid_box.shape}")
        """

        point_loss = focal_loss_sigmoid(labels,logits) # same shape as logits, N x C
        num_endpoint = point_loss.shape[0]
        cls_loss = cls_loss_weight * point_loss.mean()

        batch_idx = torch.arange(0, pred_box.shape[0])
        batch_idx = batch_idx.unsqueeze(1).to(labels.device)
        batch_idx = torch.cat([batch_idx, labels], dim=1)
        pred_box = pred_box[batch_idx[:, 0], batch_idx[:, 1]]
        huger_loss = nn.SmoothL1Loss(reduction="none")
        all_loc_loss = huger_loss(pred_box, gt_box.squeeze())
        all_loc_loss = all_loc_loss * valid_box.squeeze(1)

        num_valid_endpoint = valid_box.sum()
        mean_loc_loss = all_loc_loss.mean(dim=1)

        if num_valid_endpoint==0:
            loc_loss = 0
        else: loc_loss = mean_loc_loss.sum() / num_valid_endpoint
        classwise_loc_loss = []


        for class_idx in range(self.num_classes):
            class_mask = torch.nonzero(labels==int(class_idx), as_tuple=False)
            l = mean_loc_loss[class_mask]
            classwise_loc_loss += [l]
        loss_dict = {}
        loss_dict['classwise_loc_loss'] = classwise_loc_loss

        params = torch.cat([x.view(-1) for x in self.parameters()])
        reg_loss = torch.mean(params.abs())

        loss_dict.update({'cls_loss': cls_loss, 'loc_loss': loc_loss,
            'reg_loss': reg_loss,
            'num_end_point': num_endpoint,
            'num_valid_endpoint': num_valid_endpoint
            })
        return loss_dict
