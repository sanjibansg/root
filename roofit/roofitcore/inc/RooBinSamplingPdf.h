// Authors: Stephan Hageboeck, CERN; Andrea Sciandra, SCIPP-UCSC/Atlas; Nov 2020

/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2018, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_BIN_SAMPLING__PDF
#define ROO_BIN_SAMPLING__PDF

#include "RooAbsReal.h"
#include "RooTemplateProxy.h"
#include "RooAbsPdf.h"

#include "Math/Integrator.h"

#include <memory>

class RooBinSamplingPdf : public RooAbsPdf {
public:

  RooBinSamplingPdf() { };
  RooBinSamplingPdf(const char *name, const char *title, RooAbsRealLValue& observable, RooAbsPdf& inputPdf,
      double epsilon = 1.E-4);

  RooBinSamplingPdf(const RooBinSamplingPdf& other, const char* name = nullptr);

  TObject* clone(const char* newname=nullptr) const override {
    return new RooBinSamplingPdf(*this, newname);
  }

  // Analytical Integration handling
  bool forceAnalyticalInt(const RooAbsArg& dep) const override {
    return _pdf->forceAnalyticalInt(dep);
  }
  /// Forwards to the PDF's implementation.
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet,
      const char* rangeName=nullptr) const override {
    return _pdf->getAnalyticalIntegralWN(allVars, analVars, normSet, rangeName);
  }
  /// Forwards to the PDF's implementation.
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars,
      const char* rangeName=nullptr) const override {
    return _pdf->getAnalyticalIntegral(allVars, numVars, rangeName);
  }
  /// Forwards to the PDF's implementation.
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const override {
    return _pdf->analyticalIntegralWN(code, normSet, rangeName);
  }
  /// Forwards to the PDF's implementation.
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override {
    return _pdf->analyticalIntegral(code, rangeName);
  }
  /// Forwards to the PDF's implementation.
  std::unique_ptr<RooAbsReal> createExpectedEventsFunc(const RooArgSet *nset) const override {
    return _pdf->createExpectedEventsFunc(nset);
  }

  /// Forwards to the PDF's implementation.
  bool selfNormalized() const override { return true; }

  /// Forwards to the PDF's implementation.
  RooFit::OwningPtr<RooAbsReal> createIntegral(const RooArgSet& iset,
                             const RooArgSet* nset=nullptr,
                             const RooNumIntConfig* cfg=nullptr,
                             const char* rangeName=nullptr) const override {
    return _pdf->createIntegral(iset, nset, cfg, rangeName);
  }

  ExtendMode extendMode() const override { return _pdf->extendMode(); }
  double expectedEvents(const RooArgSet* nset) const override { return _pdf->expectedEvents(nset); }

  /// Forwards to the PDF's implementation.
  Int_t getGenerator(const RooArgSet& directVars, RooArgSet& generateVars, bool staticInitOK = true) const override {
    return _pdf->getGenerator(directVars, generateVars, staticInitOK);
  }
  /// Forwards to the PDF's implementation.
  void initGenerator(Int_t code) override { _pdf->initGenerator(code); }
  /// Forwards to the PDF's implementation.
  void generateEvent(Int_t code) override { _pdf->generateEvent(code); }
  /// Forwards to the PDF's implementation.
  bool isDirectGenSafe(const RooAbsArg& arg) const override { return _pdf->isDirectGenSafe(arg); }


  // Hints for optimized brute-force sampling
  Int_t getMaxVal(const RooArgSet& vars) const override { return _pdf->getMaxVal(vars); }
  double maxVal(Int_t code) const override { return _pdf->maxVal(code); }
  Int_t minTrialSamples(const RooArgSet& arGenObs) const override { return _pdf->minTrialSamples(arGenObs); }

  // Plotting and binning hints
  /// Returns true, since this PDF is meant to be binned.
  bool isBinnedDistribution(const RooArgSet& /*obs*/) const override { return true; }
  std::list<double>* binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const override;
  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override;

  std::unique_ptr<ROOT::Math::IntegratorOneDim>& integrator() const;

  static std::unique_ptr<RooAbsPdf> create(RooAbsPdf& pdf, RooAbsData const &data, double precision);

  double epsilon() const { return _relEpsilon; }
  const RooAbsPdf& pdf() const { return _pdf.arg(); }
  const RooAbsReal& observable() const { return _observable.arg(); }

  void doEval(RooFit::EvalContext &) const override;

protected:
  double evaluate() const override;
  std::span<const double> binBoundaries() const;

private:
  template<typename Func>
  friend class ROOT::Math::WrappedFunction;
  // Call operator for our internal integrator.
  double operator()(double x) const;
  double integrate(const RooArgSet* normSet, double low, double high) const;


  RooTemplateProxy<RooAbsPdf> _pdf;
  RooTemplateProxy<RooAbsRealLValue> _observable;
  double _relEpsilon{1.E-4}; ///< Default integrator precision.

  mutable std::unique_ptr<ROOT::Math::IntegratorOneDim> _integrator{nullptr}; ///<! Integrator used to sample bins.
  mutable std::vector<double> _binBoundaries; ///<! Workspace to store data for bin sampling

  ClassDefOverride(RooBinSamplingPdf,1)
};

#endif
