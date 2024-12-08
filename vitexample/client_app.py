"""vitexample: A Flower / PyTorch app with Vision Transformers."""

import torch
from torch.utils.data import DataLoader

from flwr.common import Context
from flwr.client import NumPyClient, ClientApp


from vitexample.task import apply_train_transforms, get_dataset_partition
from vitexample.task import get_model, get_yolo_model, set_params, set_yolo_params, get_params, get_yolo_params, train, train_yolo_ultralytics

# class FedViTClient(NumPyClient):
#     def __init__(self, trainloader, learning_rate, num_classes):
#         self.trainloader = trainloader
#         self.learning_rate = learning_rate
#         self.model = get_model(num_classes)

#         # Determine device
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)  # send model to device

#     def fit(self, parameters, config):
#         set_params(self.model, parameters)

#         # Set optimizer
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         # Train locally
#         avg_train_loss = train(
#             self.model, self.trainloader, optimizer, epochs=1, device=self.device
#         )
#         # Return locally-finetuned part of the model
#         return (
#             get_params(self.model),
#             len(self.trainloader.dataset),
#             {"train_loss": avg_train_loss},
#         )


# def client_fn(context: Context):
#     """Return a FedViTClient."""

#     # Read the node_config to fetch data partition associated to this node
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     dataset_name = context.run_config["dataset-name"]
#     trainpartition = get_dataset_partition(num_partitions, partition_id, dataset_name)

#     batch_size = context.run_config["batch-size"]
#     lr = context.run_config["learning-rate"]
#     num_classes = context.run_config["num-classes"]
#     trainset = trainpartition.with_transform(apply_train_transforms)

#     trainloader = DataLoader(
#         trainset, batch_size=batch_size, num_workers=2, shuffle=True
#     )

#     return FedViTClient(trainloader, lr, num_classes).to_client()


# app = ClientApp(client_fn=client_fn)

class FedViTClient(NumPyClient):
    def __init__(self, learning_rate, data_yaml, num_classes):
        self.learning_rate = learning_rate
        self.model = get_yolo_model(num_classes, data_yaml)
        self.data_yaml = data_yaml

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def fit(self, parameters, config):
        set_yolo_params(self.model, parameters)

        # Train the model locally using YOLOs built-in training function
        results = train_yolo_ultralytics(self.model, self.data_yaml, 1, self.device)

        # Extract loss from results
        train_loss = results.box_loss + results.cls_loss + results.obj_loss

        parameters = get_yolo_params(self.model)

        # Return locally-finetuned part of the model
        return (
            parameters,
            results.dataset.size,
            {"train_loss": train_loss.item()},
        )


def client_fn(context: Context):
    """Return a FedViTClient."""
    learning_rate = context.run_config["learning-rate"]
    num_classes = context.run_config["num-classes"]
    data_yaml = context.run_config["data-yaml"]

    return FedViTClient(learning_rate, data_yaml, num_classes).to_client()


app = ClientApp(client_fn=client_fn)

