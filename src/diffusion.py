import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cuda"
    ):
        self.noise_steps = noise_steps
        self.device = torch.device(device)

        # Linear noise schedule
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            noise_steps,
            device=self.device
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(
            self.alphas,
            dim=0
        )

        # Precomputed coefficients
        self.sqrt_alpha_bar = torch.sqrt(
            self.alpha_bar
        )

        self.sqrt_one_minus_alpha_bar = torch.sqrt(
            1.0 - self.alpha_bar
        )

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion process.

        x_t = sqrt(alpha_bar_t) * x0
              + sqrt(1 - alpha_bar_t) * noise

        x0:    (B, 1, 28, 28)
        t:     (B,)
        noise: (B, 1, 28, 28)
        """

        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(
            -1, 1, 1, 1
        )

        sqrt_one_minus_alpha_bar = (
            self.sqrt_one_minus_alpha_bar[t].view(
                -1, 1, 1, 1
            )
        )

        x_t = (
            sqrt_alpha_bar * x0
            + sqrt_one_minus_alpha_bar * noise
        )

        return x_t

    @torch.no_grad()
    def sample(
        self,
        model,
        n,
        return_trajectory=False,
        trajectory_steps=(999, 750, 500, 250, 100)
    ):
        """
        Reverse diffusion process.

        Starts with Gaussian noise and progressively
        removes predicted noise.

        Returns images in [0, 1].
        """

        model.eval()

        # Start from pure Gaussian noise
        x = torch.randn(
            n, 1, 28, 28,
            device=self.device
        )

        trajectory = {}

        for t in reversed(range(self.noise_steps)):
            if return_trajectory and t in trajectory_steps:
                trajectory[t] = (
                    ((x.clamp(-1, 1) + 1) / 2)
                    .detach()
                    .cpu()
                )
                
            # Current timestep for every image
            timestep = torch.full(
                (n,),
                t,
                device=self.device,
                dtype=torch.long
            )

            # Predict noise
            predicted_noise = model(x, timestep)

            alpha = self.alphas[t]
            beta = self.betas[t]
            alpha_bar = self.alpha_bar[t]

            # Reverse diffusion mean
            x = (1.0 / torch.sqrt(alpha)) * (
                x
                - (beta / torch.sqrt(1.0 - alpha_bar))
                * predicted_noise
            )

            # Add random noise except at final step
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise

            # Save selected intermediate states
            if return_trajectory and t in trajectory_steps:
                trajectory[t] = (
                    ((x.clamp(-1, 1) + 1) / 2)
                    .detach()
                    .cpu()
                )

        # Final generated x0
        x = x.clamp(-1, 1)
        images = (x + 1) / 2

        if return_trajectory:
            trajectory[0] = images.detach().cpu()
            return images, trajectory

        return images
