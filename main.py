import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
from server import ServerAgent
from client import ClientAgent
from models.ResNet_agent import ResNetAgent

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
PRUNING_THRESHOLD = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ImageNet dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Replace with the actual path to ImageNet
train_dataset = datasets.ImageFolder(root='path_to_imagenet/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Simulate multiple edge devices (clients)
NUM_CLIENTS = 5
clients = []

# Initialize clients with a ResNet model and local data
for i in range(NUM_CLIENTS):
    client = ClientAgent(
        model_class=ResNetAgent,
        model_args={'input_channels': 3, 'num_classes': 1000, 'width': 64, 'depth': 3},
        device=DEVICE,
        data_dir='path_to_client_data',  # Replace with path to each client's local data
        pruning_threshold=PRUNING_THRESHOLD
    )
    clients.append(client)

# Initialize server with a ResNet model
server = ServerAgent(
    model_class=ResNetAgent,
    model_args={'input_channels': 3, 'num_classes': 1000, 'width': 64, 'depth': 3},
    device=DEVICE
)


def main():
    """
    Federated learning process: simulate distributed model training, pruning, and aggregation.
    """
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Step 1: Distribute global model to clients
        global_model_state = server.get_global_model_state()  # Get the global model state from the server
        for client in clients:
            client.load_global_model(global_model_state)  # Load the global model state into the client

        # Step 2: Clients perform local training, pruning, and retraining
        for client in clients:
            print(f"Client {clients.index(client) + 1} starts local training")

            # Load local data for the client
            client.load_local_data()

            # Perform local pruning and training
            client.prune_and_compress()  # Prune model to fit memory constraints
            final_model_state, client_weight = client.train_local_model()  # Local training

            # Upload model to server
            server.client_models.append(final_model_state)  # Collect the model state after training
            server.client_weights.append(client_weight)  # Collect the model weight

        # Step 3: Aggregate models on the server
        print("Server is aggregating the models from clients.")
        server.aggregate_models()  # Aggregate the model updates from all clients

        # Step 4: Evaluate the global model (optional, can be done after each epoch)
        server.evaluate_global_model()  # Evaluate the global model on validation/test data (if desired)

        print(f"End of Epoch {epoch + 1}/{EPOCHS}")


if __name__ == "__main__":
    main()
