import time
import torch
import json
import logging
import torch.nn.functional as F


def train_vae_network(n_epochs, model, optimizer, train_dataloader, valid_dataloader, device, params_save_path, logger):
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

            optimizer.zero_grad()

            # VAE outputs both reconstructions and latent variable parameters (mean and log var)
            reconstructions, mu, logvar = model(sequence_data)
            reconstruction_loss = F.binary_cross_entropy(reconstructions, sequence_data, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss is the sum of reconstruction loss and KLD
            loss = reconstruction_loss + KLD

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

        # Validation phase
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader, 1):
                sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
            
                reconstructions, mu, logvar = model(sequence_data)
                reconstruction_loss = F.binary_cross_entropy(reconstructions, sequence_data, reduction='sum')
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = reconstruction_loss + KLD
                valid_loss += loss.item() * sequence_data.size(0)

        valid_loss /= len(valid_dataloader.dataset)
        valid_losses.append(valid_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_duration = epoch_end_time - start_time

        logger.info(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f} \tEpoch Time: {epoch_duration:.2f}s \tTotal Time: {total_duration:.2f}s")

    # Save the final model
    torch.save(model.state_dict(), f"{params_save_path}/final_model.pt")

    total_duration = time.time() - start_time
    logger.info(f"Total training time: {total_duration:.2f}s")

    return train_losses, valid_losses