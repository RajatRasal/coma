import time

from tqdm import tqdm


def run_svi(svi, model, train_loader, test_loader, epochs, scheduler, device, output_particles, writer):
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train_eval_svi(svi, train_loader, device, train=True)
        t_duration = time.time() - t
        test_loss = train_eval_svi(svi, test_loader, device, train=False)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }
        writer.print_info(info)
        # writer.save_model_checkpoint(model, epoch)
        for batch in test_loader:
            pred = model.generate(batch.x.to(device)[0].view(1, 642, 3), output_particles)
            print(batch.x[0])
            print(pred)
            print()
            break


def train_eval_svi(svi, loader, device, train=True):
    total_loss = 0.
    for data in tqdm(loader):
        x = data.x.to(device)
        if train:
            total_loss += svi.step(x)
        else:
            total_loss += svi.evaluate_loss(x)

    size = len(loader.dataset)
    total_epoch_loss = total_loss / size
    return total_epoch_loss
