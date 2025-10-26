import torch
import torch.nn as nn
import numpy as np
from encoder import Encoder
from decoder import *
from quantizer import VectorQuantizer
from typing import Optional, Tuple

# assumes your Encoder, Decoder, and the updated VectorQuantizer (with embedding_init) are defined

class VQVAE(nn.Module):
    """
    VQ-VAE with MLP encoder/decoder and customizable codebook init.

    Args
    ----
    in_dim:        int or None. If None, Encoder infers at first forward.
    x_dim:         int, if output is flat (use exactly one of x_dim or out_shape).
    out_shape:     (C,H,W), if you want image-shaped output.
    h_dim:         hidden width for both encoder/decoder MLPs.
    n_layers_enc:  hidden layers in encoder.
    n_layers_dec:  hidden layers in decoder.
    n_embeddings:  codebook size K.
    embedding_dim: code dimension D (must equal encoder out_dim & decoder z_dim).
    beta:          commitment cost.
    embedding_init: optional (K, D) numpy/torch array to init codebook.
    normalize_init: L2-normalize code vectors after init.
    trainable_codes: whether codebook updates with grad.
    out_activation: None | 'sigmoid' | 'tanh' for decoder output.
    save_img_embedding_map: if True, track code usage counts.
    """

    def __init__(
        self,
        in_dim: Optional[int],
        x_dim: Optional[int] = None,
        out_shape: Optional[Tuple[int,int,int]] = None,
        h_dim: int = 256,
        n_layers_enc: int = 2,
        n_layers_dec: int = 2,
        n_embeddings: int = 512,
        embedding_dim: int = 128,
        beta: float = 0.25,
        embedding_init=None,
        normalize_init: bool = False,
        trainable_codes: bool = True,
        out_activation: Optional[str] = None,
        save_img_embedding_map: bool = False,
        dropout: float = 0.0,
        likelihood: str = None,
        **kwargs,
    ):
        super().__init__()
        assert (x_dim is not None) ^ (out_shape is not None), \
            "Specify exactly one of x_dim or out_shape."

        # Encoder: outputs (B, embedding_dim)
        self.encoder = Encoder(
            in_dim=in_dim, h_dim=h_dim, n_layers=n_layers_enc,
            out_dim=embedding_dim, dropout=dropout
        )

        # Vector quantizer with optional custom init
        self.vector_quantization = VectorQuantizer(
            n_e=n_embeddings,
            e_dim=embedding_dim,
            beta=beta,
            embedding_init=embedding_init,
            #normalize_init=normalize_init,
            trainable=trainable_codes,
            **kwargs,
        )

        # Decoder: takes (B, embedding_dim) -> x
        self.likelihood = likelihood
        if likelihood == None:
            self.decoder = Decoder(
                z_dim=embedding_dim,
                h_dim=h_dim,
                n_layers=n_layers_dec,
                x_dim=x_dim,
                out_shape=out_shape,
                out_activation=out_activation,
                dropout=dropout,
            )
        elif likelihood == "nb":
            self.decoder = NBDecoder(
                z_dim=embedding_dim,
                h_dim=h_dim,
                n_layers=n_layers_dec,
                x_dim=x_dim,
                dropout=dropout,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported likelihood: {likelihood}")

        # (Optional) track code usage counts
        self.track_usage = save_img_embedding_map
        if self.track_usage:
            self.register_buffer("code_usage_counts", torch.zeros(n_embeddings, dtype=torch.long))

    def forward(self, x):
        """
        Returns:
          embedding_loss: scalar VQ loss
          x_hat: reconstruction (flat or image-shaped)
          perplexity: codebook perplexity
          (optional) indices: flattened code indices selected (N,) where N=B for MLP
        """
        # Encoder -> (B, D)
        z_e = self.encoder(x)

        # Vector quantization
        vq_loss, z_q, perplexity, one_hot, indices = self.vector_quantization(z_e)

        # Optional code usage accounting (MLP case: N == B)
        if self.track_usage:
            with torch.no_grad():
                self.code_usage_counts += torch.bincount(
                    indices, minlength=self.vector_quantization.n_e
                )

        # Decoder
        if self.likelihood == None:
            x_hat = self.decoder(z_q)
            return vq_loss, x_hat, perplexity, indices
        elif self.likelihood == "nb":
            mu,theta = self.decoder(z_q)
            nll = nb_nll_from_mu_theta(x, mu, theta)
            return vq_loss, nll, perplexity, indices
        '''
        if return_indices:
            return vq_loss, x_hat, perplexity, indices
        return vq_loss, x_hat, perplexity
        '''