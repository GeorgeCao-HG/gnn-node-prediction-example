# main.py
# A complete, runnable script to demonstrate a GNN for a node-level prediction task.

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# --- 1. GNN Model Definition ---
# This is the model architecture we discussed. It's designed to process
# graphs and make predictions for each node.

class MyGNNModel(torch.nn.Module):
    """
    A simple Graph Neural Network with two GCNConv layers.
    """
    def __init__(self, num_node_features, num_output_classes):
        super(MyGNNModel, self).__init__()
        
        hidden_channels = 16
        
        # Define the GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Define the final output layer
        self.output_layer = Linear(hidden_channels, num_output_classes)

    def forward(self, x, edge_index):
        """
        Defines the forward pass of the model.
        
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, num_node_features].
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
        
        Returns:
            Tensor: The output predictions for each node, shape [num_nodes, num_output_classes].
        """
        # 1. First GNN layer + activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 2. Second GNN layer
        x = self.conv2(x, edge_index)
        
        # 3. Final prediction layer
        output = self.output_layer(x)
        
        return output

# --- 2. Synthetic Dataset Generation ---
# In a real system, this data would come from your pre-processed simulation results.
# Here, we create a dummy dataset to make the script runnable.

def create_dummy_dataset(num_graphs, num_nodes_per_graph, num_features, num_classes):
    """
    Generates a list of random graph Data objects for demonstration.
    """
    dataset = []
    for _ in range(num_graphs):
        # Generate random node features
        x = torch.randn((num_nodes_per_graph, num_features))
        
        # Generate random edges (a simple way to create a connected graph)
        edge_source = torch.randint(0, num_nodes_per_graph, (int(num_nodes_per_graph * 1.5),))
        edge_target = torch.randint(0, num_nodes_per_graph, (int(num_nodes_per_graph * 1.5),))
        edge_index = torch.stack([edge_source, edge_target], dim=0)
        
        # Generate a dummy target 'y' for each node.
        # We'll make the target a simple transformation of the input features
        # so the model has a pattern to learn.
        y = x[:, :num_classes] * 2 + 1 # Example transformation
        
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
        
    return dataset

# --- 3. Training and Evaluation Logic ---

def train(model, loader, optimizer, loss_fn):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        # Perform forward pass
        predicted_y = model(batch.x, batch.edge_index)
        # Calculate loss
        loss = loss_fn(predicted_y, batch.y)
        # Perform backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader, loss_fn):
    """
    Evaluates the model on a dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            predicted_y = model(batch.x, batch.edge_index)
            loss = loss_fn(predicted_y, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


# --- 4. Main Execution Block ---

if __name__ == "__main__":
    # Hyperparameters and Configuration
    NUM_GRAPHS = 100
    AVG_NODES_PER_GRAPH = 20
    NUM_NODE_FEATURES = 8
    NUM_OUTPUT_CLASSES = 4
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 20
    BATCH_SIZE = 32

    print("--- GNN Node Prediction Demo ---")
    
    # 1. Create the dataset
    print("Generating synthetic dataset...")
    full_dataset = create_dummy_dataset(NUM_GRAPHS, AVG_NODES_PER_GRAPH, NUM_NODE_FEATURES, NUM_OUTPUT_CLASSES)
    
    # Split dataset into training and testing
    train_size = int(0.8 * len(full_dataset))
    train_dataset = full_dataset[:train_size]
    test_dataset = full_dataset[train_size:]
    
    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"Dataset created with {len(train_dataset)} training graphs and {len(test_dataset)} test graphs.")
    
    # 2. Initialize the model, optimizer, and loss function
    model = MyGNNModel(num_node_features=NUM_NODE_FEATURES, num_output_classes=NUM_OUTPUT_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss() # Mean Squared Error is suitable for this regression task
    
    print(f"Model initialized:\n{model}")
    
    # 3. Run the training loop
    print("\n--- Starting Training ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, loss_fn)
        test_loss = test(model, test_loader, loss_fn)
        print(f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
    print("--- Training Finished ---")

