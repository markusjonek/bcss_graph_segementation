

class Config:
    def __init__(self):
        self.gnn_layers = [64]*4
        self.classifier_hidden_dim = 64
        self.dropout = 0.1
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.num_epochs = 50
        self.val_every = 5

        self.dataset_dir = 'datasets/bm_dataset_pt'

        self.model_save_dir = 'saved_models'

    def __str__(self):
        return f"GNN layers: {self.gnn_layers} \
            \nClassifier hidden dim: {self.classifier_hidden_dim} \
            \nDropout: {self.dropout} \
            \nLearning rate: {self.learning_rate} \
            \nWeight decay: {self.weight_decay} \
            \nEpochs: {self.num_epochs}"

