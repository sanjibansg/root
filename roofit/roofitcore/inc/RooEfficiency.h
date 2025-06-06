/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooEfficiency.h,v 1.6 2007/05/11 10:14:56 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_EFFICIENCY
#define ROO_EFFICIENCY

#include "RooAbsPdf.h"
#include "RooCategoryProxy.h"
#include "RooRealProxy.h"
#include "TString.h"

class RooEfficiency : public RooAbsPdf {
public:
  // Constructors, assignment etc
  /// Default constructor
  inline RooEfficiency() {
  }
  RooEfficiency(const char *name, const char *title, const RooAbsReal& effFunc, const RooAbsCategory& cat, const char* sigCatName);
  RooEfficiency(const RooEfficiency& other, const char* name=nullptr);
  TObject* clone(const char* newname=nullptr) const override { return new RooEfficiency(*this,newname); }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

  RooAbsCategory const& cat() const { return *_cat; }
  RooAbsReal const& effFunc() const { return *_effFunc; }
  std::string sigCatName() const { return _sigCatName.Data(); }

protected:

  // Function evaluation
  double evaluate() const override ;

  RooCategoryProxy _cat ; ///< Accept/reject categort
  RooRealProxy _effFunc ; ///< Efficiency modeling function
  TString _sigCatName ;   ///< Name of accept state of accept/reject category

  ClassDefOverride(RooEfficiency,1) // Generic PDF defined by string expression and list of variables
};

#endif
