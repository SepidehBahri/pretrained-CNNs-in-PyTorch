import torch
import json
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10,
                full_model_path=None, weights_path=None, logs_path=None):

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = 100.0 * correct_val / total_val

        # --- Save logs ---
        if logs_path:
            os.makedirs(os.path.dirname(logs_path), exist_ok=True)
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
            with open(logs_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # --- Save best model ---
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            if full_model_path:
                torch.save(model, full_model_path)
            if weights_path:
                torch.save(model.state_dict(), weights_path)
            print(f"Epoch {epoch+1}: 🔥 New best val_acc = {val_acc:.2f}%, model saved.")

        print(f"[{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return model
