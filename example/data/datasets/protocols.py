from enum import Enum


class SupportedMedicalDatasets(Enum):
    """Enum for the paired datasets."""
    BRATS = "brats"
    """The BraTS 2018 dataset."""
    ISEG = "iseg"
    """The iSeg 2017 dataset."""
    IXI = "ixi"
    """The IXI dataset"""
    PROSTATE = "prostate"
    """The prostate MRI super resolution dataset"""
