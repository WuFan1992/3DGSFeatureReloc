import torch
import torch.nn.functional as F
import time 

threshold = 0.9


class DenseMatcher():
    def __init__(self, feature_query, feature_projection):
        super().__init__()
        self.feature_query = feature_query
        self.feature_projection = feature_projection
        self.height = feature_query.size(1)
        self.width = feature_query.size(2)


    def oneDim2TwoDim(self, index):
        div = index // self.width
        mod = index % self.width
        if div + 1 < self.height:
            return (div, mod) 
        else:
            return None
    
    def preprocess(self):
        #Norm the channels dimension 
        #The feature map size is [channel, height, width], the channels dimension represent the feature for each pixel in feature map 
        feature_query_norm = self.feature_query / self.feature_query.norm(dim=0)
        feature_projection_norm = self.feature_projection / self.feature_projection.norm(dim=0)


        # resize to [channel, height x width]
        feature_query = torch.reshape(feature_query_norm, (feature_query_norm.size(0), -1))
        feature_projection = torch.reshape(feature_projection_norm, (feature_projection_norm.size(0), -1))

        print("feature query size in preprocess is ", feature_query.shape)

        return feature_query, feature_projection

    def computeSimilarity(self, feature_query, feature_projection):
        #cosine similarity
        res = torch.randn(feature_query.size(1), feature_projection.size(1))
        ct = 1
        start_time = time.time()

        
        for i in range(feature_query.size(1)):
            for j in range(feature_projection.size(1)):
                ct= ct+1
                res[i][j] = F.cosine_similarity(feature_query[:,i], feature_projection[:,j], dim=0).item()
                print("i = ", i, "j = ", j, "res = ", res[i][j] , " ct = ", ct)
        print("--- %s seconds ---" % (time.time() - start_time))
              
        
        #get the max indice
        col_max_indices = torch.argmax(res, dim=0).tolist()
        row_max_indices = torch.argmax(res, dim=1).tolist()

        return res, col_max_indices, row_max_indices


    
    def getMatchingPoints(self, res, col_max_indices, row_max_indices):
        query_matching_points = []
        projection_matching_points = []
         #find the indices matching 
        matching_points = [(col_max_indices[i], row_max_indices[col_max_indices[i]]) for i in range(len(col_max_indices)) if i == row_max_indices[col_max_indices[i]] and res[col_max_indices[i]][row_max_indices[col_max_indices[i]]] > threshold]
        
        #one dimension point to two dimension points 
        #for example for a 4x4 image one dimension point 8 = (2,0)
        for points in matching_points:
            query_matching_points.append(self.oneDim2TwoDim(points[0]))
            projection_matching_points.append(self.oneDim2TwoDim(points[1]))
        
        return query_matching_points, projection_matching_points

    def matching(self):
        # Get the preprocessing features
        feature_query, feature_projection = self.preprocess()

        # Compute the similarity and get the max index for col and row
        res, col_max_indices, row_max_indices = self.computeSimilarity(feature_query, feature_projection)

        #Get the matching points
        query_matching_points, projection_matching_points = self.getMatchingPoints(res, col_max_indices, row_max_indices)

        return query_matching_points, projection_matching_points 
    
 
        