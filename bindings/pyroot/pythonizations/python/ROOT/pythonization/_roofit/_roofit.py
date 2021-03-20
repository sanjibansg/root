# Author: Hinnerk C. Schmidt 02/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import cppyy

from ROOT import pythonization


def __getter(k, v):
    # helper function to get CmdArg attribute from `RooFit`
    # Parameters:
    # k: key of the kwarg
    # v: value of the kwarg
    if isinstance(v, (tuple, list)):
        attr = getattr(cppyy.gbl.RooFit, k)(*v)
    elif isinstance(v, (dict, )):
        attr = getattr(cppyy.gbl.RooFit, k)(**v)
    else:
        attr = getattr(cppyy.gbl.RooFit, k)(v)
    return attr


def _pythonization(self, *args, **kwargs):
    """
    Docstring
    """
    # Pythonized functions redefinition for keyword arguments.
    # the keywords must correspond to the CmdArg of the provided function.
    # Parameters:
    # self: instance of class provided
    # *args: arguments passed
    # **kwargs: keyword arguments pased
    if not kwargs:
        return self._OriginalFunction(*args)
    else:
        nargs = args + tuple((__getter(k, v) for k, v in kwargs.items()))
        return self._OriginalFunction(*nargs)


def _pythonizedFunction(klass,function):
    klass._OriginalFunction = getattr(klass,function)
    return _pythonization
