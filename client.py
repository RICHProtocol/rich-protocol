import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.utils.prune as prune


class CustomDataset(Dataset):
    """
    Custom dataset to load images and labels from a directory.
    """

    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset with the directory containing images.
        :param data_dir: Path to the dataset directory.
        :param transform: Optional transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)  # Assuming all files are images
        self.labels = [0] * len(self.image_files)  # Placeholder for labels, modify based on your dataset

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load an image and its label from the dataset.
        :param idx: Index of the sample.
        :return: Processed image and its label.
        """
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]  # Placeholder label, update accordingly
        return image, label


class ClientAgent:
    def __init__(self, model_class, model_args, device="cpu", memory_limit=512, data_dir=None, pruning_threshold=0.2):
        """
        Initialize the client with local data and model settings.
        :param model_class: The model class (e.g., ResNet, Transformer).
        :param model_args: Arguments to initialize the model.
        :param device: Device for computation (e.g., 'cpu' or 'cuda').
        :param memory_limit: Limit on memory usage for model pruning.
        :param data_dir: Path to the local dataset directory.
        """
        self.device = device
        self.model = model_class(**model_args).to(self.device)  # Initialize the model
        self.memory_limit = memory_limit  # Maximum memory available for pruning
        self.data_dir = data_dir  # Directory path for local data
        self.local_data = None  # Placeholder for local dataset
        self.local_loader = None  # DataLoader for local training
        self.model_state = None  # Placeholder for model state after training
        self.client_weight = None  # Placeholder for client weight based on local resources
        self.data_size = 0  # Initialize data_size attribute
        self.pruning_threshold = pruning_threshold  # Pruning threshold for model compression
        print(
            f"ClientAgent initialized with model: {model_class.__name__}, memory limit: {self.memory_limit}, data directory: {self.data_dir}")

    def load_local_data(self):
        """
        Load local data for training. Assumes data is stored as images in the specified directory.
        """
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist.")
            return None

        # Define the necessary transformations (e.g., resize, normalize)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to the input size of the model
            transforms.ToTensor(),  # Convert images to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
        ])

        # Load the dataset using the custom dataset class
        dataset = CustomDataset(data_dir=self.data_dir, transform=transform)
        self.local_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Example batch size

        self.data_size = len(dataset)  # Set data_size to the number of samples in the dataset

        print(f"Local data loaded from {self.data_dir}. Number of samples: {self.data_size}")
        return self.local_loader

    def load_global_model(self, global_model_state_dict):
        """
        Load the global model parameters into the local model.

        :param global_model_state_dict: A dictionary containing the global model parameters.
        """
        print("Loading global model parameters...")
        try:
            self.model.load_state_dict(global_model_state_dict)
            self.model.to(self.device)
            print("Global model loaded successfully.")
        except Exception as e:
            print(f"Error loading global model: {e}")

    def prune_and_compress(self):
        """
        Prune and compress the model based on available memory and the pruning threshold.
        Adjust the model's architecture to fit the client’s resources.
        """
        # Example pruning strategy (this can be expanded based on memory limit and device capabilities)
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params > self.memory_limit:
            self.model = self._prune_model(self.model, self.pruning_threshold)  # Prune model using the threshold
            print(f"Model pruned to fit memory constraints with pruning threshold {self.pruning_threshold}.")
        else:
            print(f"Model fits within memory constraints with no pruning needed.")

    def _prune_model(self, model, pruning_threshold):
        """
        Apply pruning to reduce the model size based on the pruning threshold.
        This implementation uses weight magnitude pruning to zero out a fraction of the smallest weights.

        :param model: The model to be pruned.
        :param pruning_threshold: The fraction of weights to prune (threshold).
        :return: The pruned model.
        """
        # Calculate the number of parameters to prune based on the pruning threshold
        print(f"Pruning model using a threshold of {pruning_threshold}.")

        # Apply pruning to each layer (weight magnitude pruning)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                # Compute the pruning amount based on the threshold
                prune_amount = pruning_threshold  # Fraction of weights to prune

                print(f"Pruning {name} layer with pruning amount: {prune_amount:.4f}")

                # Prune the weights based on their magnitude using L1 unstructured pruning
                prune.l1_unstructured(module, name='weight', amount=prune_amount)

        print("Pruning complete.")
        return model

    def train_local_model(self):
        """
        Train the local model using client data.
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # Local training loop using DataLoader
        self.model.train()

        # Initialize loss variable to avoid referencing before assignment
        loss = None

        for epoch in range(2):  # Example for a few epochs of local training
            for inputs, labels in self.local_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

        self.model_state = self.model.state_dict()

        # Calculate client weight based on data size (simplified)
        self.client_weight = self.data_size / (self.data_size + 1)  # Simple weight based on data size
        return self.model_state, self.client_weight

    def upload_model(self):
        """
        Upload the trained model and the associated weight to the server.
        """
        return self.model_state, self.client_weight
