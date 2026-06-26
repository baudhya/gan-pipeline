from gan_pipeline.training.pix2pix_trainer import Pix2PixTrainer


class Pix2PixHDTrainer(Pix2PixTrainer):
    """
    pix2pixHD trainer: ResNet generator + 3-scale PatchGAN + VGG + FM losses.

    Identical training loop to Pix2PixTrainer; differs only in which
    model architecture params are recorded to MLflow.
    """

    def _log_params(self) -> dict[str, object]:
        return {
            "model": self.cfg.model.name,
            "loss_type": self.cfg.training.loss_type,
            "lambda_vgg": self.lambda_vgg,
            "lambda_fm": self.lambda_fm,
            "lambda_gp": self.lambda_gp,
            "n_scales": len(self._ms_disc.discriminators),
            "ngf": self.cfg.model.generator.ngf,
            "n_downsampling": self.cfg.model.generator.n_downsampling,
            "n_blocks": self.cfg.model.generator.n_blocks,
            "lr_g": self.cfg.training.lr_generator,
            "lr_d": self.cfg.training.lr_discriminator,
            "batch_size": self.cfg.training.batch_size,
        }
