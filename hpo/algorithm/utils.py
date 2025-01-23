from hpo.core.SearchSpace import SearchSpace

class leaf :
    def __init__(self,
                 space : SearchSpace,
                 depth,
                 loop=0,
                 depth_id=0,
                 score = None,
                 score_state = "unknown") :
        self.global_id = str(depth) + "_" + str(depth_id)
        self.space = space
        self.depth = depth
        self.depth_id = depth_id
        self.score = score
        self.state= True
        self.loop = loop
        self.score_state = score_state