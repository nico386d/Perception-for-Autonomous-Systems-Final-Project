import torch
from sklearn.metrics import accuracy_score
from model import device


def train_one_epoch(model, data_loader, optimizer, loss_function, epoch):
    """
    One training epoch.
    Returns: (average_loss, train_accuracy)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # training accuracy
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    train_acc = correct / total if total > 0 else 0.0

    return average_loss, train_acc


def evaluate_classifier(model, data_loader):
    """
    Evaluate accuracy on a given loader (validation set in your case).
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted_labels = outputs.argmax(dim=1)

            all_predictions.extend(predicted_labels.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy
