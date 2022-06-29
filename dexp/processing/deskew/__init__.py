from dexp.processing.deskew.classic_deskew import classic_deskew
from dexp.processing.deskew.utils import skew
from dexp.processing.deskew.yang_deskew import yang_deskew

deskew_functions = dict(
    classic=classic_deskew,
    yang=yang_deskew,
)
