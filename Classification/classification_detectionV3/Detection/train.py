import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from model import device


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def train_one_epoch_detection(model, data_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(data_loader, desc=f"Epoch {epoch}", ncols=100)

    for images, targets in progress:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Faster R-CNN returns a dict of losses
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        avg_loss = total_loss / total_batches
        progress.set_postfix(loss=f"{avg_loss:.4f}")

    return avg_loss


def evaluate_detection_map(model, data_loader):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)  # list of dicts

            preds_cpu = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

            metric.update(preds_cpu, targets_cpu)

    results = metric.compute()
    return results
