"""Batch adaptation specific to the BraTS 2021 example pipeline."""

from torchmanager_core import torch

from reg_bbrg.data import MedicalTranslationDataset
from reg_bbrg.data.protocols import MedicalTranslationData


class BraTS2021TranslationDataset(MedicalTranslationDataset):
    """Accept the collated list of crops produced by the BraTS 2021 loader."""

    @staticmethod
    def unpack_data(batch: MedicalTranslationData | list[MedicalTranslationData]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return MedicalTranslationDataset.unpack_data(batch)
        if not batch:
            raise ValueError("Cannot unpack an empty list of BraTS 2021 crops.")

        # Each crop contains [batch, slices, channels, ...] tensors.
        pairs = [MedicalTranslationDataset.unpack_data(crop) for crop in batch]
        return pairs[0] if len(pairs) == 1 else (torch.cat([pair[0] for pair in pairs], dim=0), torch.cat([pair[1] for pair in pairs], dim=0))
