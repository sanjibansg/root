import cppyy
from ROOT import pythonization


@pythonization()
def roofitPythonization(klass,name):
    if name[:3] =='Roo':
        from ._rooabspdf import pythonize_rooabspdf
        from ._rooabsreal import pythonize_rooabsreal

        pythonize_rooabspdf(klass,name)
        pythonize_rooabsreal(klass,name)

