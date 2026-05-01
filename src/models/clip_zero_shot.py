"""
CLIP Zero-Shot Anomaly Detector
================================
Wraps OpenCLIP for zero-shot anomaly detection.
Computes anomaly scores by comparing image embeddings against
normal/abnormal text prompt ensembles via cosine similarity.

Reference:
    WinCLIP (Jeong et al., CVPR 2023) — compositional prompt ensembles
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class CLIPZeroShotDetector:
    """Zero-shot anomaly detector using CLIP.

    Scores images based on cosine similarity between image features
    and text prompt embeddings for normal vs. abnormal states.

    Args:
        model_name: OpenCLIP model name (e.g., 'ViT-L-14').
        pretrained: Pretrained weights identifier (e.g., 'laion2b_s32b_b82k').
        device: 'cuda' or 'cpu'. Auto-detects if None.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: Optional[str] = None,
    ):
        import open_clip

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading OpenCLIP {model_name} ({pretrained}) on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"Model loaded successfully.")

    @torch.no_grad()
    def encode_text_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode a list of text prompts and return the mean-pooled, normalized embedding.

        This implements prompt ensembling — averaging embeddings from multiple
        prompt templates for a more robust text representation.

        Args:
            prompts: List of text strings.

        Returns:
            Normalized mean text embedding, shape (1, embed_dim).
        """
        tokens = self.tokenizer(prompts).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        # Mean-pool across all prompts (ensemble)
        mean_feature = text_features.mean(dim=0, keepdim=True)
        mean_feature = F.normalize(mean_feature, dim=-1)
        return mean_feature

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed image tensor.

        Args:
            image: Preprocessed image tensor, shape (C, H, W) or (B, C, H, W).

        Returns:
            Normalized image embedding, shape (B, embed_dim).
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        image_features = self.model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    @torch.no_grad()
    def compute_anomaly_scores(
        self,
        dataloader,
        normal_prompts: List[str],
        abnormal_prompts: List[str],
    ) -> tuple:
        """Compute anomaly scores for an entire dataloader.

        The anomaly score for each image is the softmax probability
        of the abnormal class:
            score = softmax([sim_normal, sim_abnormal])[1]

        Args:
            dataloader: PyTorch DataLoader yielding (image, label, category, path).
            normal_prompts: List of text prompts for normal state.
            abnormal_prompts: List of text prompts for abnormal state.

        Returns:
            (scores, labels) — numpy arrays of anomaly scores and ground truth labels.
        """
        # Pre-compute text embeddings (once)
        normal_embed = self.encode_text_prompts(normal_prompts)   # (1, D)
        abnormal_embed = self.encode_text_prompts(abnormal_prompts)  # (1, D)
        text_embeds = torch.cat([normal_embed, abnormal_embed], dim=0)  # (2, D)

        all_scores = []
        all_labels = []

        for images, labels, _, _ in tqdm(dataloader, desc="Computing scores", leave=False):
            image_features = self.encode_image(images)  # (B, D)

            # Cosine similarity with both prompts
            similarity = image_features @ text_embeds.T  # (B, 2)

            # Apply temperature scaling (CLIP uses 100.0 logit scale typically)
            logit_scale = self.model.logit_scale.exp()
            similarity = similarity * logit_scale

            # Softmax to get probability of abnormal class
            probs = similarity.softmax(dim=-1)
            anomaly_scores = probs[:, 1]  # probability of abnormal

            all_scores.append(anomaly_scores.cpu().numpy())
            all_labels.append(labels.numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        return scores, labels

    def get_preprocess(self):
        """Return the image preprocessing transform for this model."""
        return self.preprocess
