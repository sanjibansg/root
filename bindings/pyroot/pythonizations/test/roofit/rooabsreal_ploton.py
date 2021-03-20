import unittest

import ROOT

class RooAbsRealPlotOn(unittest.TestCase):
    """
    Test for the plotOn callable
    """
    x = ROOT.RooRealVar("x", "x", -10, 10)
    mean = ROOT.RooRealVar("mean", "mean of gaussian", 1, -10, 10)
    sigma = ROOT.RooRealVar("sigma", "width of gaussian", 1, 0.1, 10)
    gauss = ROOT.RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

    dgdx=gauss.derivative(x,1)
    xframe = x.frame(ROOT.RooFit.Title("d(Gauss)/dx"))

    def test_wrong_kwargs(self):
        self.assertRaises(AttributeError, self.dgdx.plotOn, self.xframe, ThisIsNotACmdArg=True)

    def test_identical_result(self):
        x = ROOT.RooRealVar("x", "x", -10, 10)
        mean = ROOT.RooRealVar("mean", "mean of gaussian", 1, -10, 10)
        sigma = ROOT.RooRealVar("sigma", "width of gaussian", 1, 0.1, 10)
        gauss = ROOT.RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

        dgdx=gauss.derivative(x,1)
        xframe = x.frame(ROOT.RooFit.Title("d(Gauss)/dx"))
        gauss.plotOn(xframe)

        res=dgdx.plotOn(xframe, LineColor=ROOT.kMagenta)
        self.assertEqual(type(res), ROOT.RooPlot)

    def test_mixed_styles(self):
        x = ROOT.RooRealVar("x", "x", -10, 10)
        mean = ROOT.RooRealVar("mean", "mean of gaussian", 1, -10, 10)
        sigma = ROOT.RooRealVar("sigma", "width of gaussian", 1, 0.1, 10)
        gauss = ROOT.RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

        dgdx=gauss.derivative(x,1)
        xframe = x.frame(ROOT.RooFit.Title("d(Gauss)/dx"))
        gauss.plotOn(xframe)

        res_1=dgdx.plotOn(xframe, ROOT.RooFit.LineColor(ROOT.kMagenta), LineStyle=ROOT.kDashed)
        res_2=dgdx.plotOn(xframe, ROOT.RooFit.LineStyle(ROOT.kDashed),LineColor=ROOT.kMagenta)
        self.assertEqual(type(res_1), ROOT.RooPlot)
        self.assertEqual(type(res_2), ROOT.RooPlot)



if __name__ == '__main__':
    unittest.main()
