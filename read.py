#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import string

from PIL import Image

import torch

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


def restrict_logits_to_digits(logits, tokenizer):
    allowed_ids = {tokenizer._stoi[d] for d in string.digits if d in tokenizer._stoi}
    if hasattr(tokenizer, 'eos_id'):
        allowed_ids.add(tokenizer.eos_id)
    if hasattr(tokenizer, 'blank_id'):
        allowed_ids.add(tokenizer.blank_id)
    if not allowed_ids:
        raise ValueError('Tokenizer does not contain digit tokens.')

    masked_logits = torch.full_like(logits, -torch.inf)
    keep = sorted(allowed_ids)
    masked_logits[..., keep] = logits[..., keep]
    return masked_logits


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images', nargs='+', help='Images to read')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    for fname in args.images:
        # Load image and prepare for input
        image = Image.open(fname).convert('RGB')
        image = img_transform(image).unsqueeze(0).to(args.device)

        logits = restrict_logits_to_digits(model(image), model.tokenizer)
        p = logits.softmax(-1)
        pred, p = model.tokenizer.decode(p)
        print(f'{fname}: {pred[0]}')


if __name__ == '__main__':
    main()
