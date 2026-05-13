# -*- coding:utf-8 -*-
import torch

from downstream.model import PhysioMEClassifier
from models.dp_neuronet.model import BiosignalEncoder
from models.physiome.model import PhysioME
from models.transformer import apply_lora


def load_pretrained_to_classifier(ckpt_path: str, n_classes: int):
    """Load a PhysioME-Hetero checkpoint and wrap it in a classification head.

    Returns ``(PhysioMEClassifier, (unimodal_param, multimodal_param))`` so that
    callers can inspect the architecture parameters if needed.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ch_names = ckpt['ch_names']
    unimodal_param = ckpt['modality_backbone_param']
    multimodal_param = ckpt['entire_model_param']
    model_state = ckpt['model_state']
    hyperparameter = ckpt['hyperparameter']

    # Build one LoRA-wrapped BiosignalEncoder per modality (hand-rolled LoRA — see
    # ``models/transformer/lora.py``). The wrap order matches the trainers'
    # ``_load_pretrained_unimodal``: build encoder, then ``apply_lora``.
    backbone_networks = {}
    for ch_name in ch_names:
        encoder = BiosignalEncoder(**unimodal_param)
        encoder = apply_lora(
            encoder,
            target_attrs=('out_proj',),
            r=hyperparameter['lora_r'],
            alpha=hyperparameter['lora_alpha'],
            dropout=hyperparameter['lora_dropout'],
            rslora=True,
        )
        backbone_networks[ch_name] = encoder

    # Reconstruct full PhysioME and load pretrained weights.
    physio_me = PhysioME(
        backbone_networks=backbone_networks,
        backbone_embed_dim=multimodal_param['backbone_embed_dim'],
        num_backbone_frames=multimodal_param['backbone_num_frames'],
        encoder_embed_dim=multimodal_param['encoder_embed_dim'],
        encoder_heads=multimodal_param['encoder_heads'],
        encoder_depths=multimodal_param['encoder_depths'],
        decoder_embed_dim=multimodal_param['decoder_embed_dim'],
        decoder_heads=multimodal_param['decoder_heads'],
        decoder_depths=multimodal_param['decoder_depths'],
        decoder_recon_depths=multimodal_param['decoder_recon_depths'],
        projection_hidden=multimodal_param['projection_hidden'],
        temperature=multimodal_param['temperature'],
        # Pre-grouping checkpoints (per-modal decoders) won't carry this key;
        # PhysioME then falls back to DEFAULT_MODAL_TO_GROUP. Old ckpts must
        # be re-trained — their state_dict keys (multimodal_decoder_dict.*)
        # are incompatible with the new (multimodal_decoder_body_dict.*) layout.
        modal_to_group=multimodal_param.get('modal_to_group'),
    )
    physio_me.load_state_dict(model_state)

    classifier = PhysioMEClassifier(physio_me=physio_me, n_classes=n_classes)
    return classifier, (unimodal_param, multimodal_param)
