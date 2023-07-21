import torch
from torch import nn, einsum
import torch.nn.functional as F
import open_clip

from einops import rearrange

from beartype import beartype
from beartype.typing import List, Optional

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1)

class OpenClipAdapter(nn.Module):
    @beartype
    def __init__(
        self,
        name = 'ViT-B/32',
        pretrained = 'laion400m_e32',
        tokenizer_name = 'ViT-B-32-quickgelu',
        eos_id = 49407
    ):
        super().__init__()

        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained = pretrained)
        tokenizer = open_clip.get_tokenizer(tokenizer_name)

        self.clip = clip
        self.tokenizer = tokenizer
        self.eos_id = eos_id

        # hook for getting final text representation

        text_attention_final = self.find_layer('ln_final')
        self._dim_latent = text_attention_final.weight.shape[0]
        self.text_handle = text_attention_final.register_forward_hook(self._text_hook)

        # hook for getting final image representation
        # this is for vision-aided gan loss

        self._dim_image_latent = self.find_layer('visual.ln_post').weight.shape[0]

        num_visual_layers = len(clip.visual.transformer.resblocks)
        self.image_handles = []

        for visual_layer in range(num_visual_layers):
            image_attention_final = self.find_layer(f'visual.transformer.resblocks.{visual_layer}')

            handle = image_attention_final.register_forward_hook(self._image_hook)
            self.image_handles.append(handle)

        # normalize fn

        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    @property
    def device(self):
        return next(self.parameters()).device

    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.text_handle()
        self.image_handle()

    def _text_hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    def _image_hook(self, _, inputs, outputs):
        if not hasattr(self, 'image_encodings'):
            self.image_encodings = []

        self.image_encodings.append(outputs)

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def image_size(self):
        image_size = self.clip.visual.image_size
        if isinstance(image_size, tuple):
            return max(image_size)
        return image_size

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.positional_embedding.shape[0]

    @beartype
    def embed_texts(
        self,
        texts: List[str]
    ):
        ids = self.tokenizer(texts)
        ids = ids.to(self.device)
        ids = ids[..., :self.max_text_len]

        is_eos_id = (ids == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        text_mask = text_mask & (ids != 0)
        assert not self.cleared

        text_embed = self.clip.encode_text(ids)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        del self.text_encodings
        return l2norm(text_embed.float()), text_encodings.float()

    def embed_images(self, images):
        if images.shape[-1] != self.image_size:
            images = F.interpolate(images, self.image_size)

        assert not self.cleared
        images = self.clip_normalize(images)
        image_embeds = self.clip.encode_image(images)

        image_encodings = rearrange(self.image_encodings, 'l n b d -> l b n d')
        del self.image_encodings

        return l2norm(image_embeds.float()), image_encodings.float()

    @beartype
    def contrastive_loss(
        self,
        images,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None
    ):
        assert exists(texts) ^ exists(text_embeds)

        if not exists(text_embeds):
            text_embeds, _ = self.embed_texts(texts)

        image_embeds, _ = self.embed_images(images)

        n = text_embeds.shape[0]

        temperature = self.clip.logit_scale.exp()
        sim = einsum('i d, j d -> i j', text_embeds, image_embeds) * temperature

        labels = torch.arange(n, device = sim.device)

        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
