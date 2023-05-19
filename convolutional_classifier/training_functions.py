import time
import torch
import logging
import json

def train_network(n_epochs, model, optimizer, criterion, train_dataloader, test_dataloader, device, model_save_path, params_save_path):

    logging.basicConfig(filename=f"{params_save_path}/training.log", level=logging.INFO)



    # Save all parameters
    params = {
        "n_epochs": n_epochs,
        "optimizer": str(optimizer),
        "criterion": str(criterion),
        "train_dataloader_length": len(train_dataloader.dataset),
        "test_dataloader_length": len(test_dataloader.dataset),
        "device": str(device),
    }

    with open(f"{params_save_path}/parameters.json", 'w') as f:
        json.dump(params, f)
        

    train_losses = []
    test_losses = []
    start_time = time.time()

    for epoch in range(1, n_epochs+1):
        epoch_start_time = time.time()

        # Training phase
        train_loss = 0.0
        model.train()
        for batch in train_dataloader:
            sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(sequence_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequence_data.size(0)

        train_loss /= len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # Test phase
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
                labels = batch["label"].to(device)

                outputs = model(sequence_data)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * sequence_data.size(0)

        test_loss /= len(test_dataloader.dataset)
        test_losses.append(test_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_duration = epoch_end_time - start_time

        logging.info(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTest Loss: {test_loss:.6f} \tEpoch Time: {epoch_duration:.2f}s \tTotal Time: {total_duration:.2f}s")

    # Save the final model
    torch.save(model.state_dict(), f"{params_save_path}/final_model.pt")

    total_duration = time.time() - start_time
    logging.info(f"Total training time: {total_duration:.2f}s")

    return train_losses, test_losses
