import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

def evaluate_reconstruction(model, test_loader, device):

    model.eval()

    psnr_metric = PeakSignalNoiseRatio(
        data_range=1.0,
        dim=(1, 2, 3), # PSNR per image (over C, H, W), then averaged across the test set
        reduction="elementwise_mean"
    ).to(device)

    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0
    ).to(device)

    total_squared_error = 0.0
    total_pixels = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            outputs = model(images)

            if isinstance(outputs, (tuple, list)):
                reconstruction = outputs[0]
            else:
                reconstruction = outputs

            squared_error = (reconstruction - images).pow(2)

            total_squared_error += squared_error.sum().item()
            total_pixels += images.numel()

            psnr_metric.update(reconstruction, images)
            ssim_metric.update(reconstruction, images)

    test_mse = total_squared_error / total_pixels
    test_psnr = psnr_metric.compute().item()
    test_ssim = ssim_metric.compute().item()

    return {
        "test_mse": test_mse,
        "test_psnr": test_psnr,
        "test_ssim": test_ssim,
    }


def evaluate_hidden_sparsity(
    model,
    test_loader,
    device,
    threshold=0.01
):
    """
    Measure the fraction of hidden-layer activations
    below the specified threshold.
    """

    model.eval()

    inactive_units = 0
    total_units = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            outputs = model(images)

            # Third output is the 256-dimensional hidden layer h
            h = outputs[2]

            inactive_units += (h < threshold).sum().item()
            total_units += h.numel()

    return inactive_units / total_units
