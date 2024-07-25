import torch
import torch.nn.functional as F

threshold = 0.9

class DenseMatcher(nn.Module):
    def __init__(self, feature_query, feature_projection):
        super().__init__()
        self.feature_query = feature_query
        self.feature_projection = feature_projection
    
    def __preprocess():
        #Norm the channels dimension 
        #The feature map size is [channel, height, width], the channels dimension represent the feature for each pixel in feature map 
        feature_query_norm = self.feature_query / self.feature_query.norm(dim=0)[:,None, None]
        feature_projection_norm = self.feature_projection / self.feature_projection.norm(dim=0)[:,None, None]

        # resize to [channel, height x width]
        feature_query = torch.reshape(feature_query_norm, (feature_query_norm.size(0), -1))
        feature_projection = torch.reshape(feature_projection_norm, (feature_projection_norm.size(0), -1))

        #cosine similarity
        res = torch.randn(feature_query.size)
        for i in range(feature_query.size(1)):
            for j in range(feature_projection.size(1)):
                res[i][j] = F.cosine_similarity(a[:,i], b[:,j], dim=0)
        
        #get the max indice
        col_max_indices = torch.argmax(res, dim=0)
        row_max_indices = torch.argmax(res, dim=1)

        #find the indices matching 
        matching_points = [(row_max_indices[i], col_max_indices[row_max_indices[i]]) for i in range(feature_query.size(1) if i == col_max_indices[row_max_indices[i]] and res[row_max_indices[i]][col_max_indices[row_max_indices[i]]] > thresdhold)]
        

        


    def forward()