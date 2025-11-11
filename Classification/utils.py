import random, torch, torchvision as tv

def set_random_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def create_training_transforms(image_size):
    return tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
    ])

def create_validation_transforms(image_size):
    return tv.transforms.Compose([
        tv.transforms.Resize(int(image_size * 1.15)),
        tv.transforms.CenterCrop(image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
    ])

try:
    from sklearn.metrics import f1_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def evaluate_predictions(logit_batches, label_batches):
    probabilities = torch.cat([batch.softmax(1) for batch in logit_batches])
    labels = torch.cat(label_batches)
    predictions = probabilities.argmax(1)
    accuracy = (predictions == labels).float().mean().item()
    results = {"accuracy": accuracy}
    if SKLEARN_AVAILABLE:
        results["macro_f1"] = float(f1_score(labels, predictions, average="macro"))
        results["confusion_matrix"] = confusion_matrix(labels, predictions)
    return results
