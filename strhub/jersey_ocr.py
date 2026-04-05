import string

import torch
from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


def _restrict_logits_to_digits(logits, tokenizer):
    allowed_ids = {tokenizer._stoi[d] for d in string.digits if d in tokenizer._stoi}
    if hasattr(tokenizer, 'eos_id'):
        allowed_ids.add(tokenizer.eos_id)
    if hasattr(tokenizer, 'blank_id'):
        allowed_ids.add(tokenizer.blank_id)
    masked_logits = torch.full_like(logits, -torch.inf)
    masked_logits[..., sorted(allowed_ids)] = logits[..., sorted(allowed_ids)]
    return masked_logits


class JerseyOCR:
    """Inference wrapper for jersey number recognition.

    Usage:
        ocr = JerseyOCR()
        number, confidence = ocr.read(pil_image)
    """

    def __init__(self, checkpoint: str = 'pretrained=parseq', device: str = 'cuda'):
        self.device = device
        self.model = load_from_checkpoint(checkpoint).eval().to(device)
        self.transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)

    @torch.inference_mode()
    def read(self, image: Image.Image) -> tuple[str, float]:
        """Read a jersey number from a cropped PIL image.

        Args:
            image: PIL Image of the jersey number crop.

        Returns:
            Tuple of (predicted number string, mean confidence score 0-1).
        """
        x = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        logits = _restrict_logits_to_digits(self.model(x), self.model.tokenizer)
        p = logits.softmax(-1)
        pred, confidence = self.model.tokenizer.decode(p)
        return pred[0], confidence[0].mean().item()

    @torch.inference_mode()
    def read_batch(self, images: list[Image.Image]) -> list[tuple[str, float]]:
        """Read jersey numbers from a batch of cropped PIL images.

        Args:
            images: List of PIL Images of jersey number crops.

        Returns:
            List of (predicted number string, mean confidence score 0-1) tuples.
        """
        batch = torch.stack([self.transform(img.convert('RGB')) for img in images]).to(self.device)
        logits = _restrict_logits_to_digits(self.model(batch), self.model.tokenizer)
        p = logits.softmax(-1)
        preds, confidences = self.model.tokenizer.decode(p)
        return [(pred, conf.mean().item()) for pred, conf in zip(preds, confidences)]
