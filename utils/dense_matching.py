import torch
import torch.nn.functional as F

threshold = 0.9


class DenseMatcher(nn.Module):
    def __init__(self, feature_query, feature_projection):
        super().__init__()
        self.feature_query = feature_query
        self.feature_projection = feature_projection
        self.height = feature_query.size(0)
        self.width = feature_query.size(1)


    def oneDim2TwoDim(self, index):
        div = index // self.width
        mod = index % self.width
        if div + 1 < h:
            return (div, mod) 
        else:
            return None
    
    def __preprocess():

        #Norm the channels dimension 
        #The feature map size is [channel, height, width], the channels dimension represent the feature for each pixel in feature map 
        feature_query_norm = self.feature_query / self.feature_query.norm(dim=0)[:,None, None]
        feature_projection_norm = self.feature_projection / self.feature_projection.norm(dim=0)[:,None, None]

        # resize to [channel, height x width]
        feature_query = torch.reshape(feature_query_norm, (feature_query_norm.size(0), -1))
        feature_projection = torch.reshape(feature_projection_norm, (feature_projection_norm.size(0), -1))

        return feature_query, feature_projection

    def __computeSimilarity(self, feature_query, feature_projection):
        #cosine similarity
        res = torch.randn(feature_query.size)
        for i in range(feature_query.size(1)):
            for j in range(feature_projection.size(1)):
                res[i][j] = F.cosine_similarity(a[:,i], b[:,j], dim=0)
        
        #get the max indice
        col_max_indices = torch.argmax(res, dim=0).tolist()
        row_max_indices = torch.argmax(res, dim=1).tolist()

        return res, col_max_indices, row_max_indices
    
    def __getMatchingPoints(self, res, col_max_indices, row_max_indices):
        query_matching_points = None
        projection_matching_points = None
         #find the indices matching 
        matching_points = [(col_max_indices[i], row_max_indices[col_max_indices[i]]) for i in range(len(col_max_indices)) if i == row_max_indices[col_max_indices[i]] and res[col_max_indices[i]][row_max_indices[col_max_indices[i]]] > thresdhold]
        
        #one dimension point to two dimension points 
        #for example for a 4x4 image one dimension point 8 = (2,0)
        for points in matching_points:
            query_matching_points.append(self.oneDim2TwoDim(points[0]))
            projection_matching_points.append(self.oneDim2TwoDim(points[1]))
        
        return query_matching_points, projection_matching_points

    def forward(self):
        # Get the preprocessing features
        feature_query, feature_projection = self.__preprocessing()

        # Compute the similarity and get the max index for col and row
        res, col_max_indices, row_max_indices = self.__computeSimilarity(feature_query, feature_projection)

        #Get the matching points
        query_matching_points, projection_matching_points = self.__getMatchingPoints(res, col_max_indices, row_max_indices)

        return query_matching_points, projection_matching_points 