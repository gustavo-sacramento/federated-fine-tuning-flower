"""vitexample: A Flower / PyTorch app with Vision Transformers."""

from logging import INFO

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from vitexample.task import apply_eval_transforms
from vitexample.task import get_model, get_yolo_model, set_params, set_yolo_params, test, get_params, get_yolo_params

from flwr.common import Context, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg


def get_evaluate_fn(
    centralized_testset: Dataset,
    num_classes: int,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round, parameters, config):
        """Use the entire Oxford Flowers-102 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Instantiate model and apply current global parameters
        model = get_model(num_classes)
        set_params(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_eval_transforms)

        testloader = DataLoader(testset, batch_size=128)
        # Run evaluation
        loss, accuracy = test(model, testloader, device=device)
        log(INFO, f"round: {server_round} -> acc: {accuracy:.4f}, loss: {loss: .4f}")

        return loss, {"accuracy": accuracy}

    return evaluate

def get_evaluate_yolo_fn(data_yaml: str, num_classes):
    """Return an evaluation function for centralized evaluation."""
    def evaluate(server_round, parameters, config):
        """Use YOLO's validation function for evaluation."""
        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Instantiate model and apply current global parameters
        model = get_yolo_model(num_classes, data_yaml)
        set_yolo_params(model, parameters)
        model.to(device)

        # Run validation using YOLO's built-in validation function
        results = model.val()
        log(
            INFO,
            f"round: {server_round} -> box_loss: {results.box_loss:.4f}, "
            f"mAP@0.5: {results.map50:.4f}, mAP@0.5:0.95: {results.map:.4f}"
        )

        return results.box_loss, {
            "map50": results.map50,
            "map": results.map
        }

    return evaluate

def yolo_server_fn(context: Context):
    """Server function defining the federated learning setup."""
    
    # Dataset and model configuration from context
    data_yaml = context.run_config["data-yaml"]  # Path to `data.yaml`
    num_classes = context.run_config["num-classes"]  # Number of classes
    num_server_rounds = context.run_config["num-server-rounds"]

    # Initialize YOLO model
    model = get_yolo_model(num_classes, data_yaml)
    model.overrides["data"] = data_yaml  # Load dataset configuration
    model.model.nc = num_classes  # Set number of classes

    # Extract initial model parameters
    ndarrays = get_yolo_params(model)
    init_parameters = ndarrays_to_parameters(ndarrays)

    # Configure the federated strategy
    strategy = FedAvg(
        fraction_fit=0.5,  # Sample 50% of available clients
        fraction_evaluate=0.0,  # No federated evaluation
        evaluate_fn=get_evaluate_yolo_fn(data_yaml, num_classes),  # Centralized evaluation
        initial_parameters=init_parameters,
    )

    # Construct ServerConfig
    config = ServerConfig(num_rounds=num_server_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


def server_fn(context: Context):

    # Define tested for central evaluation
    dataset_name = context.run_config["dataset-name"]
    dataset = load_dataset(dataset_name)
    test_set = dataset["test"]

    # Set initial global model
    num_classes = context.run_config["num-classes"]
    ndarrays = get_params(get_model(num_classes))
    init_parameters = ndarrays_to_parameters(ndarrays)

    # Configure the strategy
    strategy = FedAvg(
        fraction_fit=0.5,  # Sample 50% of available clients
        fraction_evaluate=0.0,  # No federated evaluation
        evaluate_fn=get_evaluate_fn(
            test_set, num_classes
        ),  # Global evaluation function
        initial_parameters=init_parameters,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# app = ServerApp(server_fn=server_fn)
app = ServerApp(server_fn=yolo_server_fn)
