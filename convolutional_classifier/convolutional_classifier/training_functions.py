import time
import torch
import json
import logging
import torch.nn.functional as F


def train_network(n_epochs, model, optimizer, criterion, train_dataloader, valid_dataloader, device, params_save_path, logger):

    # Save all parameters
    params = {
        "n_epochs": n_epochs,
        "optimizer": str(optimizer),
        "criterion": str(criterion),
        "train_dataloader_length": len(train_dataloader.dataset),
        "valid_dataloader_length": len(valid_dataloader.dataset),
        "device": str(device),
    }

    train_losses = []
    valid_losses = []
    start_time = time.time()

    for epoch in range(1, n_epochs+1):
        epoch_start_time = time.time()

        # Training phase
        train_loss = 0.0
        model.train()
        for i, batch in enumerate(train_dataloader, 1):
            sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(sequence_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequence_data.size(0)

            # Log the progress
            curr_time = time.time()
            elapsed_time = curr_time - epoch_start_time
            processed_sequences = i * batch['sequence'].size(0)
            percentage_complete = (processed_sequences / len(train_dataloader.dataset)) * 100
            logger.info(f"Epoch {epoch}/{n_epochs} | Processed {processed_sequences}/{len(train_dataloader.dataset)} sequences ({percentage_complete:.2f}%) | Training Loss: {train_loss/processed_sequences:.6f} | Elapsed Time: {elapsed_time:.2f}s")

        train_loss /= len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # valid phase
        valid_loss = 0.0
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader, 1):
                sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
                labels = batch["label"].to(device)
            
                outputs = model(sequence_data)
                _, preds = torch.max(outputs, 1)

                y_pred.extend(preds.tolist())
                y_true.extend(labels.tolist())

                loss = criterion(outputs, labels)
                valid_loss += loss.item() * sequence_data.size(0)

        valid_loss /= len(valid_dataloader.dataset)
        valid_losses.append(valid_loss)

        correct_preds = torch.eq(torch.max(F.softmax(outputs, dim=-1), dim=-1)[1], labels).float().sum()
        params['correct_preds'] = correct_preds.item()
        total_preds = torch.FloatTensor([labels.size(0)])
        params['total_preds'] = total_preds.item()
        correct_preds = correct_preds.to(device)
        total_preds = total_preds.to(device)
        accuracy = correct_preds / total_preds
        params['accuracy'] = accuracy.item()
        logger.info(f'Accuracy: {accuracy.item():.4f}')

        with open(f"{params_save_path}/parameters.json", 'w') as f:
            json.dump(params, f)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_duration = epoch_end_time - start_time

        logger.info(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tvalid Loss: {valid_loss:.6f} \tEpoch Time: {epoch_duration:.2f}s \tTotal Time: {total_duration:.2f}s")

    # Save the final model
    torch.save(model.state_dict(), f"{params_save_path}/final_model.pt")

    total_duration = time.time() - start_time
    logger.info(f"Total training time: {total_duration:.2f}s")

    return train_losses, valid_losses, y_true, y_pred
