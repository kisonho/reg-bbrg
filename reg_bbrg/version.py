from torchmanager_core import Version, deprecated as _deprecated

API_VERSION = Version("v1.0")
VERSION = Version("v1.0")
DESCRIPTION = f"Regularized Brownian Bridges for Deterministic Medical Image Translation {VERSION}"

VersionType = str | Version

def deprecated(target_version: VersionType, removing_version: VersionType | None, *, current_version: VersionType = API_VERSION):
    
    '''
    Deprecated decorator function

    - Parameters:
        - target_version: `Any` type of version for the deprecation
        - removing_version: `Any` type of version for removing
        - current_version: `Any` type of version for the current version
    '''
    return _deprecated(target_version, removing_version, current_version=current_version)
