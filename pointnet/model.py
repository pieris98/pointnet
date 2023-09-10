import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        '''I implement conv1d because shared MLP shares weights between neurons. We could use Linear layer with permuted inputs.
        Link explaining equivalence: https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-vs-linear-layer'''
        self.pointmlp1_1 = nn.Sequential(nn.Conv1d(3,64,1),nn.BatchNorm1d(64))
        self.pointmlp1_2 = nn.Sequential(nn.Conv1d(64,64,1),nn.BatchNorm1d(64))

        self.pointmlp2_1 = nn.Sequential(nn.Conv1d(64,64,1),nn.BatchNorm1d(64))
        self.pointmlp2_2 = nn.Sequential(nn.Conv1d(64,128,1),nn.BatchNorm1d(128))
        self.pointmlp2_3 = nn.Sequential(nn.Conv1d(128,1024,1),nn.BatchNorm1d(1024))
        


    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """
        # TODO : Implement forward function.
        ''' need to permute again for the next layer (t-net OR shared mlp/conv1d)
        since they expect [B,C,N]'''
        pointcloud = pointcloud.permute(0,2,1)
        # print('pointcloud shape',pointcloud.shape)
        if (self.input_transform):
            tnet_input = self.stn3(pointcloud)
            # print('tnet_input shape',tnet_input.shape)
            # print('pointcloud shape',pointcloud.shape)
            '''need to permute the input shape [B,3,N]->[B,N,3] 
            for the batch matrix mul for tnet (spatial transformation)'''
            pointcloud = pointcloud.permute(0,2,1)
            # print('permuted pointcloud shape',pointcloud.shape)
            pointcloud = torch.matmul(pointcloud, tnet_input)
            # print('after spatial transform shape',pointcloud.shape)
            ''' need to permute back again for the next layers (shared mlp/conv1d)
            since they expect [B,C,N]'''
            pointcloud = pointcloud.permute(0,2,1)
            # print('re-permute shape',pointcloud.shape)
        
        '''pass through 1st MLP'''
        pointcloud = F.relu(self.pointmlp1_1(pointcloud))
        pointcloud = F.relu(self.pointmlp1_2(pointcloud))
        # print('after mlp1 shape',pointcloud.shape)
        
        if(self.feature_transform):
            tnet_feature = self.stn64(pointcloud)
            ''' store this in a variable intermediate_features for regularization term(orthogonal loss)'''
            feature_transform = tnet_feature.clone()
            # print('tnet_feature shape',tnet_feature.shape)
            pointcloud = torch.matmul(pointcloud.permute(0,2,1), tnet_feature)
            # print('after feature spatial transform shape',pointcloud.shape)
            ''' store this in a variable intermediate_features for part segmentation'''
            intermediate_features = pointcloud.clone()
            ''' need to permute again for the next layers (shared mlp/conv1d)
            since they expect [B,C,N]'''
            pointcloud = pointcloud.permute(0,2,1)

        '''pass through 2nd MLP'''
        pointcloud = F.relu(self.pointmlp2_1(pointcloud))
        pointcloud = F.relu(self.pointmlp2_2(pointcloud))
        pointcloud = F.relu(self.pointmlp2_3(pointcloud))

        ''' max pooling across features (columns) to get global feature vector'''
        glob_features = torch.max(pointcloud, 2)[0]
        # print('global feature shape:',pointcloud.shape)
        if (self.feature_transform and self.input_transform):
            return glob_features, intermediate_features, feature_transform
        elif (not self.feature_transform):
            return glob_features, None, None
'''Debugging'''
# pointnetfeat=PointNetFeat(input_transform=True, feature_transform=True)
# points=Variable(torch.rand(size=(10,2048,3)))
# # print('input shape',points.shape)
# out_feat, _ = pointnetfeat(points)
# print('out feat shape',out_feat.shape)

class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        ''' no shared (Conv1d) mlp, no ReLU and no BatchNorm for final classifier head MLP'''
        self.mlp3_1 = nn.Sequential(nn.Linear(1024,512))
        '''dropout p=0.3 = 1- keep ratio (=0.7 from the paper)'''
        self.mlp3_2 = nn.Sequential(nn.Linear(512,256),nn.Dropout(0.3)) 
        self.mlp3_3 = nn.Linear(256,num_classes)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        
        pointcloud , _ , feature_transform= self.pointnet_feat(pointcloud)
        pointcloud = F.relu(self.mlp3_1(pointcloud))
        pointcloud = F.relu(self.mlp3_2(pointcloud))
        logits = self.mlp3_3(pointcloud) # no relu and no softmax yet here, we want logits
        return logits,feature_transform

'''Debugging'''
# pointnetcls=PointNetCls(num_classes=40,input_transform=True, feature_transform=True)
# points=Variable(torch.rand(size=(10,2048,3)))
# classes_logits,_=pointnetcls(points)
# print('Shape of class OUTPUT',classes_logits.shape)

class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        
        self.pointnetfeat = PointNetFeat(input_transform=True, feature_transform=True)
        self.pointmlp1_1 = nn.Sequential(nn.Conv1d(1088,512,1),nn.BatchNorm1d(512))
        self.pointmlp1_2 = nn.Sequential(nn.Conv1d(512,256,1),nn.BatchNorm1d(256))
        self.pointmlp1_3 = nn.Sequential(nn.Conv1d(256,128,1),nn.BatchNorm1d(128))
        
        self.pointmlp2_1 = nn.Sequential(nn.Conv1d(128,128,1),nn.BatchNorm1d(128))
        self.pointmlp2_2 = nn.Sequential(nn.Conv1d(128,m,1),nn.BatchNorm1d(m))
        """
        self.stn3 = STNKd(k=3)
        
        self.pointfc1 = nn.Sequential(nn.Conv1d(3,64,1),nn.BatchNorm1d(64))
        self.pointfc2 = nn.Sequential(nn.Conv1d(64,128,1),nn.BatchNorm1d(128))
        self.pointfc3 = nn.Sequential(nn.Conv1d(128,128,1),nn.BatchNorm1d(128))
        
        self.stn128 = STNKd(k=128)
        
        self.pointfc4 = nn.Sequential(nn.Conv1d(128,512,1),nn.BatchNorm1d(512))
        self.pointfc5 = nn.Sequential(nn.Conv1d(512,2048,1),nn.BatchNorm1d(2048))

        self.mlp_1 = nn.Sequential(nn.Linear(3024,256),nn.BatchNorm1d(256))
        self.mlp_2 = nn.Sequential(nn.Linear(256,256),nn.BatchNorm1d(256))
        self.mlp_3 = nn.Sequential(nn.Linear(256,128),nn.BatchNorm1d(128))

        self.out_linear = nn.Linear(128,m)
        """
    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        """
        '''need to permute for the T-Net layer 
        [B,N,3]->[B,3,N] for spatial transformation
        '''
        pointcloud = pointcloud.permute(0,2,1)

        tnet_input = self.stn3(pointcloud)
        
        '''need to permute the back to [B,3,N]->[B,N,3] 
        for the batch matrix mul for tnet (spatial transformation)'''
        pointcloud = pointcloud.permute(0,2,1)
        # print('permuted pointcloud shape',pointcloud.shape)
        pointcloud = torch.matmul(pointcloud, tnet_input)
        # print('after spatial transform shape',pointcloud.shape)
        ''' need to permute back again for the next layers (point FC/conv1d)
        since they expect [B,C,N]'''
        pointcloud = pointcloud.permute(0,2,1)
        # print('re-permute shape',pointcloud.shape)
  
        fc1 = F.relu(self.pointfc1(pointcloud))
        fc2 = F.relu(self.pointfc2(fc1))
        fc3 = F.relu(self.pointfc3(fc2))

        ''' second t-net spatial transformation with 128 features'''
        feature_transform = self.stn128(fc3.permute(0,2,1))
        '''multiply this with fc3'''
        t2 = torch.matmul(fc3, feature_transform)

        '''need to permute back again for the next layers (point FC/conv1d)
        since they expect [B,C,N]'''
        fc4 = F.relu(self.pointfc4(t2.permute(0,2,1)))
        fc5 = F.relu(self.pointfc5(fc4))

        '''maxpooling for features (columns) to get global feature vector'''
        global_features = torch.max(fc5,2)[0]
        print('global features shape:',global_features.shape)

        '''concat skip links as in Figure 9 of PointNet paper'''
        broadcast_global_features = torch.stack([global_features]*fc5.shape[1],dim=1)
        print('broadcasted global features shape:',broadcast_global_features.shape)
        concat_features = torch.cat([fc1,fc2,fc3,t2,fc4,broadcast_global_features],dim=-1)
        print('concatenated features shape:',concat_features.shape)
        
        '''need to permute back again for the next layers (point FC/conv1d)
        since they expect [B,C,N]'''
        concat_features = concat_features.permute(0,2,1)
        #print('re-permute shape',concat_features.shape)
        concat_features = F.relu(self.mlp_1(concat_features))
        concat_features = F.relu(self.mlp_2(concat_features))
        logits = self.out_linear(concat_features)

        """
        
        global_features, intermediate_features,feature_transform = self.pointnetfeat(pointcloud)
        
        #print('global features shape:',global_features.shape)
        #print('intermediate features shape:',intermediate_features.shape)
        
        '''broadcast global features to match intermediate features B and N dimensions [10,1024]->[10,2048,1024]'''
        broadcast_global_features = torch.stack([global_features]*intermediate_features.shape[1],dim=1)
        #print('broadcasted global features shape:',broadcast_global_features.shape)
        '''concatenate intermediate features with broadcasted global features over C dimension
        [10,2048,64]+[10,2048,1024]->[10,2048,1088]'''
        features = torch.cat([intermediate_features, broadcast_global_features], dim=-1)
        
        ''' need to permute for the next layers (shared mlp/conv1d)
        since they expect [B,C,N]'''
        features = features.permute(0,2,1)
        features = F.relu(self.pointmlp1_1(features))
        features = F.relu(self.pointmlp1_2(features))
        features = F.relu(self.pointmlp1_3(features))

        features = F.relu(self.pointmlp2_1(features))
        features = self.pointmlp2_2(features)
        #print('features shape:',features.shape)
        return features, feature_transform
                
'''Debugging'''
# partseg_net = PointNetPartSeg()
# points=Variable(torch.rand(size=(10,2048,3)))
# seg=partseg_net(points)
# print('Shape of Seg OUTPUT',seg.shape)

class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.fc1 = nn.Sequential(nn.Linear(1024, int(num_points/4)),nn.BatchNorm1d(int(num_points/4)))
        self.fc2 = nn.Sequential(nn.Linear(int(num_points/4), int(num_points/2)),nn.BatchNorm1d(int(num_points/2)))
        self.fc3 = nn.Sequential(nn.Linear(int(num_points/2), num_points),nn.Dropout(),nn.BatchNorm1d(num_points))
        self.fc4 = nn.Linear(num_points, num_points*3)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        num_points = pointcloud.shape[1]
        #print('num points:',num_points)
        '''???Maybe permute not needed for only FC layers???'''
        #pointcloud = pointcloud.permute(0,2,1)

        pointcloud,_,_ = self.pointnet_feat(pointcloud)
        pointcloud = F.relu(self.fc1(pointcloud))
        pointcloud = F.relu(self.fc2(pointcloud))
        pointcloud = F.relu(self.fc3(pointcloud))
        pointcloud = self.fc4(pointcloud)
        pointcloud = pointcloud.reshape(-1, num_points, 3)
        return pointcloud


'''Debugging'''
# ae_net = PointNetAutoEncoder(num_points=2048)
# points=Variable(torch.rand(size=(10,2048,3)))
# seg=ae_net(points)
# print('Shape of AE OUTPUT',seg.shape)

def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
