import torch
from utils import evaluate_predictions

class ModelTrainer:
    def __init__(self, model, optimizer, scheduler, device, output_directory, use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
        self.best_score = -1.0
        self.output_path = output_directory / "best_model.pt"
        output_directory.mkdir(parents=True, exist_ok=True)

    def fit(self, training_loader, validation_loader, loss_function, total_epochs):
        for epoch in range(1, total_epochs + 1):
            training_loss = self.train_one_epoch(training_loader, loss_function)
            metrics = self.validate(validation_loader)
            accuracy = metrics["accuracy"]
            macro_f1 = metrics.get("macro_f1", 0.0)
            print(f"Epoch {epoch:02d} | loss={training_loss:.4f} | val_acc={accuracy:.4f} | val_f1={macro_f1:.4f}")
            self.scheduler.step()

            score = macro_f1 if macro_f1 else accuracy
            if score > self.best_score:
                self.best_score = score
                torch.save(self.model.state_dict(), self.output_path)
                print("Saved best model")

    def train_one_epoch(self, data_loader, loss_function):
        self.model.train()
        total_loss, total_samples = 0.0, 0
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(images)
                loss = loss_function(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
        return total_loss / max(1, total_samples)

    @torch.no_grad()
    def validate(self, data_loader):
        self.model.eval()
        all_outputs, all_labels = [], []
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            all_outputs.append(self.model(images).cpu())
            all_labels.append(labels.cpu())
        return evaluate_predictions(all_outputs, all_labels)
