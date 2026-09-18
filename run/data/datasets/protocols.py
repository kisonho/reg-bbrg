from enum import Enum


class SupportedMedicalDatasets(Enum):
    """Enum for the paired datasets."""
    BRATS2018 = "brats2018"
    """The BraTS 2018 dataset."""
    BRATS2021 = "brats2021"
    """The BraTS 2021 dataset with separate NIfTI files per subject."""
    ISEG = "iseg"
    """The iSeg 2017 dataset."""
    IXI = "ixi"
    """The IXI dataset"""
    PROSTATE = "prostate"
    """The prostate MRI super resolution dataset"""
