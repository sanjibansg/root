from ._roofit import _pythonizedFunction


def pythonize_rooabspdf(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'RooAbsPdf':
        # Add pythonization of `fitTo` function
        functions=['fitTo']

        for i in functions:
            setattr(klass,i,_pythonizedFunction(klass,i))

    return True
