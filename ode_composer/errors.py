#  Copyright (c) 2020. Zoltan A Tuza, Guy-Bart Stan. All Rights Reserved.
#  This code is published under the MIT License.
#  Department of Bioengineering,
#  Centre for Synthetic Biology,
#  Imperial College London, London, UK
#  contact: ztuza@imperial.ac.uk, gstan@imperial.ac.uk


class SBLError(ValueError):
    def __init__(self, str):
        super(SBLError, self).__init__(str)


class ODEError(ValueError):
    def __init__(self, str):
        super(ODEError, self).__init__(str)
