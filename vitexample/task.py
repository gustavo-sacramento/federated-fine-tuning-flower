"""vitexample: A Flower / PyTorch app with Vision Transformers."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomResizedCrop,
    Resize,
    CenterCrop,
)

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def get_model(num_classes: int):
    """Return a pretrained ViT with all layers frozen except output head."""

    # Instantiate a pre-trained ViT-B on ImageNet
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # We're going to federated the finetuning of this model
    # using (by default) the Oxford Flowers-102 dataset. One easy way
    # to achieve this is by re-initializing the output block of the
    # ViT so it outputs 102 clases instead of the default 1k
    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, num_classes)

    # Disable gradients for everything
    model.requires_grad_(False)
    # Now enable just for output head
    model.heads.requires_grad_(True)

    return model

def get_yolo_model(num_classes: int, data_yaml):
    """Return a pretrained YOLO with all layers frozen except Detection head."""
    model = YOLO('yolo11n.pt')
    model.overrides["data"] = data_yaml

    # # Replace last layer with our layer
    # model.model._modules['model']._modules['23'] = Detect(num_classes)

    # 3. Freeze all layers except for the Detect layer
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    # Unfreeze the parameters of the Detect layer
    for param in model.model._modules['model']._modules['23'].parameters():
        param.requires_grad = True  # Unfreeze the custom Detect layer
    
    return model
    

def set_params(model, parameters):
    """Apply the parameters to model head."""
    finetune_layers = model.heads
    params_dict = zip(finetune_layers.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    finetune_layers.load_state_dict(state_dict, strict=True)

def set_yolo_params(model, parameters):
    finetune_layers = model.model._modules['model']._modules['23']
    params_dict = zip(finetune_layers.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    finetune_layers.load_state_dict(state_dict, strict=False)

def get_params(model):
    """Get parameters from model head as ndarrays."""
    finetune_layers = model.heads
    return [val.cpu().numpy() for _, val in finetune_layers.state_dict().items()]

def get_yolo_params(model):
    """Get parameters from model Detection head as ndarrays"""
    finetune_layers = model.model._modules['model']._modules['23']
    return [val.cpu().numpy() for _, val in finetune_layers.state_dict().items()]

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    avg_loss = 0
    # A very standard training loop for image classification
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()

    return avg_loss / len(trainloader)


def train_yolo(net, trainloader, optimizer, loss_fn, epochs, device):
    """Train the YOLO model on the training set."""
    net.train()
    net.to(device)
    avg_loss = 0
    
    for _ in range(epochs):
        for batch in trainloader:
            # Extract images and targets from the batch
            images = batch["image"].to(device)  # Shape: (B, C, H, W)
            targets = batch["target"].to(device)  # YOLO targets (bounding boxes, labels, etc.)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = net(images)  # Model outputs YOLO-specific predictions
            
            # Compute loss
            loss = loss_fn(predictions, targets)
            avg_loss += loss.item() / len(targets)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
    
    return avg_loss / len(trainloader)


def train_yolo_ultralytics(model, data_path, epochs, device):
    # Train the model
    results = model.train(
        data=data_path,   # Path to data.yaml defining dataset and labels
        epochs=epochs,    # Number of training epochs
        device=device,    # Device for training
        imgsz=640,        # Image size
        batch=16,         # Batch size
        verbose=True      # Print training details
    )

    return results


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def test_yolo_ultralytics(model, dataloader, device):
    """
    Validate the YOLO model on the test set.

    Args:
        model (YOLO): Ultralytics YOLO model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (str): Device to run the evaluation ('cpu' or 'cuda').

    Returns:
        dict: Evaluation results including loss, mAP, precision, recall, etc.
    """
    # Move model to the specified device
    model.to(device)
    model.eval()
    
    results = []
    with torch.no_grad():
        for batch in dataloader:
            # Load data
            images = batch["image"].to(device)
            targets = batch["label"]  # Expected targets (bounding boxes and classes)
            
            # Run inference
            predictions = model(images)
            
            # Collect predictions and targets for evaluation
            results.append((predictions, targets))
    
    # Calculate evaluation metrics
    metrics = model.val(results=results)
    
    return metrics


fds = None


def get_dataset_partition(num_partitions: int, partition_id: int, dataset_name: str):
    """Get Oxford Flowers datasets and partition it."""
    global fds
    if fds is None:
        # Get dataset (by default Oxford Flowers-102) and create IID partitions
        partitioner = IidPartitioner(num_partitions)
        fds = FederatedDataset(
            dataset=dataset_name, partitioners={"train": partitioner}
        )

    return fds.load_partition(partition_id)


def apply_eval_transforms(batch):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch

def apply_yolo_eval_transforms(batch, img_size=640):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    batch["image"] = [transforms(img) for img in batch["image"]]

    # Scale bounding boxes if labels are present
    if "label" in batch:
        for i, bboxes in enumerate(batch["label"]):
            # Scale bounding boxes to the resized image dimensions
            bboxes[:, [1, 3]] *= img_size / batch["image"][i].shape[1]  # Scale x-coordinates
            bboxes[:, [2, 4]] *= img_size / batch["image"][i].shape[2]  # Scale y-coordinates
            batch["label"][i] = bboxes

    return batch


def apply_train_transforms(batch):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            RandomResizedCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch

def apply_train_transforms(batch, img_size=640):
    """
    Apply training transforms for YOLO models, including data augmentation and bounding box scaling.

    Args:
        batch (dict): Batch containing images and optionally labels (bounding boxes).
        img_size (int): Target size for resizing the images (default is 640 for YOLO).

    Returns:
        dict: Batch with transformed images and updated bounding boxes.
    """
    # Define transformations for training
    transforms = Compose(
        [
            RandomResizedCrop((img_size, img_size)),  # Random crop and resize to target size
            ToTensor(),  # Convert images to tensors
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ]
    )

    # Apply transforms to images
    batch["image"] = [transforms(img) for img in batch["image"]]

    # Update bounding boxes if labels are present
    if "label" in batch:
        for i, bboxes in enumerate(batch["label"]):
            # Scale bounding boxes to the resized image dimensions
            bboxes[:, [1, 3]] *= img_size / batch["image"][i].shape[1]  # Scale x-coordinates
            bboxes[:, [2, 4]] *= img_size / batch["image"][i].shape[2]  # Scale y-coordinates
            batch["label"][i] = bboxes

    return batch



# Define the original Detect head
class Detect(nn.Module):
    def __init__(self, num_classes: int):
        super(Detect, self).__init__()

        # cv2 (module list with sequential conv blocks)
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=1)
            ),
        ])
        
        # cv3 (module list with depthwise conv blocks)
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 80, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(80),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1, groups=80, bias=False),
                nn.BatchNorm2d(80),
                nn.SiLU(inplace=True),
                nn.Conv2d(80, 80, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(80),
                nn.SiLU(inplace=True)
            ),
            nn.Conv2d(80, 80, kernel_size=1, stride=1)
        ])
        
        # DFL (for final detection predictions)
        self.dfl = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

        # Assuming num_anchors = 3 for YOLO head, final layer is for bounding box (4), confidence (1), and num_classes
        num_anchors = 3
        self.final_conv = nn.Conv2d(80, num_anchors * (4 + 1 + num_classes), kernel_size=1, stride=1)

    def forward(self, x):
        # Pass through cv2 and cv3 layers
        for layer in self.cv2:
            x = layer(x)
        for layer in self.cv3:
            x = layer(x)
        
        # Apply the final conv layer for detection predictions
        x = self.final_conv(x)
        
        return x
