/// \file
/// \ingroup tutorial_spectrum
/// Example to illustrate the Gold deconvolution (class TSpectrum2).
///
/// \macro_image
/// \macro_code
///
/// \authors Miroslav Morhac, Olivier Couet

#include <TSpectrum2.h>

void Deconvolution2_1()
{
   Int_t i, j;
   const Int_t nbinsx = 256;
   const Int_t nbinsy = 256;
   Double_t xmin = 0;
   Double_t xmax = (Double_t)nbinsx;
   Double_t ymin = 0;
   Double_t ymax = (Double_t)nbinsy;
   Double_t **source = new Double_t *[nbinsx];
   for (i = 0; i < nbinsx; i++)
      source[i] = new Double_t[nbinsy];
   TString dir = gROOT->GetTutorialDir();
   TString file = dir + "/legacy/spectrum/TSpectrum2.root";
   TFile *f = new TFile(file.Data());
   auto decon = (TH2F *)f->Get("decon1");
   Double_t **response = new Double_t *[nbinsx];
   for (i = 0; i < nbinsx; i++)
      response[i] = new Double_t[nbinsy];
   auto resp = (TH2F *)f->Get("resp1");
   gStyle->SetOptStat(0);
   auto *s = new TSpectrum2();
   for (i = 0; i < nbinsx; i++) {
      for (j = 0; j < nbinsy; j++) {
         source[i][j] = decon->GetBinContent(i + 1, j + 1);
      }
   }
   for (i = 0; i < nbinsx; i++) {
      for (j = 0; j < nbinsy; j++) {
         response[i][j] = resp->GetBinContent(i + 1, j + 1);
      }
   }
   s->Deconvolution(source, response, nbinsx, nbinsy, 1000, 1, 1);
   for (i = 0; i < nbinsx; i++) {
      for (j = 0; j < nbinsy; j++)
         decon->SetBinContent(i + 1, j + 1, source[i][j]);
   }
   decon->Draw("SURF2");
}
