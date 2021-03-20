import cppyy

from ROOT import pythonization
from ._roofit import _pythonizedFunction


@pythonization()
def pythonize_rooabsreal(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'RooAbsReal':
        # Add pythonization of `plotOn` function
        functions=['plotOn']

        for i in functions:
            setattr(klass,i,_pythonizedFunction(klass,i))

    return True
