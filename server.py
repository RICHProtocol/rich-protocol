import torch
import copy

class ServerAgent:
    def __init__(self, model_class, model_args, device="cpu"):
        """
        Initialize the server with the global model.
        :param model_class: The class of the model (e.g., ResNet).
        :param model_args: Arguments to initialize the model.
        :param device: Device for computation (e.g., 'cpu' or 'cuda').
        """
        self.device = device
        self.global_model = model_class(**model_args).to(self.device)  # Initialize the global model
        self.client_models = []  # To store client models
        self.client_weights = []  # To store client model weights
        print(f"ServerAgent initialized with model: {model_class.__name__}")

    def get_global_model_state(self):
        """
        Get the state of the global model to distribute to clients.
        :return: The state_dict of the global model.
        """
        return self.global_model.state_dict()

    def aggregate_models(self):
        """
        Aggregate client models into a new global model using weighted averaging.
        This accounts for heterogeneous client performance and resources.
        """
        if not self.client_models or not self.client_weights:
            print("No client models to aggregate.")
            return

        # Initialize an empty state_dict for aggregation
        aggregated_state_dict = copy.deepcopy(self.global_model.state_dict())
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] = torch.zeros_like(aggregated_state_dict[key], device=self.device)

        # Weighted sum of client models
        total_weight = sum(self.client_weights)
        for client_model, weight in zip(self.client_models, self.client_weights):
            for key in aggregated_state_dict.keys():
                aggregated_state_dict[key] += (weight / total_weight) * client_model[key].to(self.device)

        # Load aggregated weights into the global model
        self.global_model.load_state_dict(aggregated_state_dict)
        print("Global model updated after aggregating client models.")

        # Clear client models and weights for the next round
        self.client_models = []
        self.client_weights = []

    def evaluate_global_model(self):
        """
        Evaluate the global model on a validation dataset.
        """
        print("Evaluating global model on validation/test data.")


