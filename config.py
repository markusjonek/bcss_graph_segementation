

class Config:
    def __init__(self):
        self.gnn_layers = [1024]*4
        self.classifier_hidden_dim = 512
        self.dropout = 0
        self.learning_rate = 0.0001
        self.weight_decay = 1e-5
        self.num_epochs = 1000

        self.dataset_dir = 'datasets/full_graph_data_edge'

        self.save_dir = 'saved_models'
        self.result_dir = 'results'
        self.loss_log_dir = 'log'
        
        self.class_names = ["boundary", "tumor", "stroma", "inflammatory", "necrosis", "other"]
    
    def __str__(self):
        return f"GNN layers: {self.gnn_layers} \nClassifier hidden dim: {self.classifier_hidden_dim} \nDropout: {self.dropout} \nLearning rate: {self.learning_rate} \nWeight decay: {self.weight_decay} \nEpochs: {self.num_epochs}"

