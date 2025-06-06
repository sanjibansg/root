// @(#)root/test:$Id$
// Author: Rene Brun   19/01/97

////////////////////////////////////////////////////////////////////////
//
//             A simple example with a ROOT tree
//             =================================
//
//  This program creates :
//    - a ROOT file
//    - a tree
//  Additional arguments can be passed to the program to control the flow
//  of execution. (see comments describing the arguments in the code).
//      Event  nevent comp split fill tracks IMT compression
//  All arguments are optional. Default is:
//      Event  400      1    1     1     400   0           1
//
//  In this example, the tree consists of one single "super branch"
//  The statement ***tree->Branch("event", &event, 64000,split);*** below
//  will parse the structure described in Event.h and will make
//  a new branch for each data member of the class if split is set to 1.
//    - 9 branches corresponding to the basic types fType, fNtrack,fNseg,
//           fNvertex,fFlag,fTemperature,fMeasures,fMatrix,fClosesDistance.
//    - 3 branches corresponding to the members of the subobject EventHeader.
//    - one branch for each data member of the class Track of TClonesArray.
//    - one branch for the TRefArray of high Pt tracks
//    - one branch for the TRefArray of muon tracks
//    - one branch for the reference pointer to the last track
//    - one branch for the object fH (histogram of class TH1F).
//
//  if split = 0 only one single branch is created and the complete event
//  is serialized in one single buffer.
//  if split = -2 the event is split using the old TBranchObject mechanism
//  if split = -1 the event is streamed using the old TBranchObject mechanism
//  if split > 0  the event is split using the new TBranchElement mechanism.
//
//  if comp = 0 no compression at all.
//  if comp = 1 event is compressed.
//  if comp = 2 same as 1. In addition branches with floats in the TClonesArray
//                         are also compressed.
//  The 4th argument fill can be set to 0 if one wants to time
//     the percentage of time spent in creating the event structure and
//     not write the event in the file.
//  The 5th argument will enable IMT mode (Implicit Multi-Threading), allowing
//  ROOT to use multiple threads internally, if enabled.
//  The 6th argument allows the user to specify the compression algorithm:
//  - 1 - zlib.
//  - 2 - LZMA.
//  - 3 - "old ROOT algorithm"  A variant of zlib; do not use, kept for
//        backwards compatability.
//  - 4 - LZ4.
//  In this example, one loops over nevent events.
//  The branch "event" is created at the first event.
//  The branch address is set for all other events.
//  For each event, the event header is filled and ntrack tracks
//  are generated and added to the TClonesArray list.
//  For each event the event histogram is saved as well as the list
//  of all tracks.
//
//  The two TRefArray contain only references to the original tracks owned by
//  the TClonesArray fTracks.
//
//  The number of events can be given as the first argument to the program.
//  By default 400 events are generated.
//  The compression option can be activated/deactivated via the second argument.
//
//  Additionally, if the environment ENABLE_TTREEPERFSTATS is set, then detailed
//  statistics about IO performance will be reported.
//
//   ---Running/Linking instructions----
//  This program consists of the following files and procedures.
//    - Event.h event class description
//    - Event.C event class implementation
//    - MainEvent.C the main program to demo this class might be used (this file)
//    - EventCint.C  the CINT dictionary for the event and Track classes
//        this file is automatically generated by rootcint (see Makefile),
//        when the class definition in Event.h is modified.
//
//   ---Analyzing the Event.root file with the interactive root
//        example of a simple session
//   Root > TFile f("Event.root")
//   Root > T.Draw("fNtrack")   //histogram the number of tracks per event
//   Root > T.Draw("fPx")       //histogram fPx for all tracks in all events
//   Root > T.Draw("fXfirst:fYfirst","fNtrack>600")
//                              //scatter-plot for x versus y of first point of each track
//   Root > T.Draw("fH.GetRMS()")  //histogram of the RMS of the event histogram
//
//   Look also in the same directory at the following macros:
//     - eventa.C  an example how to read the tree
//     - eventb.C  how to read events conditionally
//
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TFile.h"
#include "TNetFile.h"
#include "TRandom.h"
#include "TTree.h"
#include "TTreePerfStats.h"
#include "TBranch.h"
#include "TClonesArray.h"
#include "TStopwatch.h"

#include "Event.h"

#ifdef R__HAS_DEFAULT_LZ4
constexpr int defaultComp = 4;
#else
constexpr int defaultComp = 1;
#endif

////////////////////////////////////////////////////////////////////////////////

int MainEvent(int nevent, int comp, int split, int arg4, int arg5, int enable_imt = 0, int compAlg = defaultComp,
              std::vector<std::string> outFiles = {})
{
   while (outFiles.size() > 1) {
      MainEvent(nevent, comp, split, arg4, arg5, enable_imt, compAlg, {outFiles.back()});
      outFiles.pop_back();
   }

   Int_t write  = 1;       // by default the tree is filled
   Int_t hfill  = 0;       // by default histograms are not filled
   Int_t read   = 0;
   Int_t netf   = 0;
   Int_t punzip = 0;

   if (arg4 ==  0) { write = 0; hfill = 0; read = 1;}
   if (arg4 ==  1) { write = 1; hfill = 0;}
   if (arg4 ==  2) { write = 0; hfill = 0;}
   if (arg4 == 10) { write = 0; hfill = 1;}
   if (arg4 == 11) { write = 1; hfill = 1;}
   if (arg4 == 20) { write = 0; read  = 1;}  //read sequential
   if (arg4 == 21) { write = 0; read  = 1;  punzip = 1;}  //read sequential + parallel unzipping
   if (arg4 == 25) { write = 0; read  = 2;}  //read random
   if (arg4 >= 30) { netf  = 1; }            //use TNetFile
   if (arg4 == 30) { write = 0; read  = 1;}  //netfile + read sequential
   if (arg4 == 35) { write = 0; read  = 2;}  //netfile + read random
   if (arg4 == 36) { write = 1; }            //netfile + write sequential
   Int_t branchStyle = 1; //new style by default
   if (split < 0) {branchStyle = 0; split = -1-split;}

#ifdef R__USE_IMT
   if (enable_imt) {
     ROOT::EnableImplicitMT();
   }
#else
   if (enable_imt) {
     std::cerr << "IMT mode requested, but this version of ROOT "
                  "is built without IMT support." << std::endl;
     return 1;
   }
#endif

   TFile *hfile;
   TTree *tree;
   TTreePerfStats *ioperf = nullptr;
   Event *event = 0;

   // Fill event, header and tracks with some random numbers
   //   Create a timer object to benchmark this loop
   TStopwatch timer;
   timer.Start();
   Long64_t nb = 0;
   Int_t ev;
   Int_t bufsize;
   Double_t told = 0;
   Double_t tnew = 0;
   Int_t printev = 100;
   if (arg5 < 100) printev = 1000;
   if (arg5 < 10)  printev = 10000;

//         Read case
   if (read) {
      if (netf) {
         hfile = new TNetFile("root://localhost/root/test/EventNet.root");
      } else
         hfile = new TFile(outFiles.back().c_str());
      tree = (TTree*)hfile->Get("T");
      TBranch *branch = tree->GetBranch("event");
      branch->SetAddress(&event);
      Int_t nentries = (Int_t)tree->GetEntries();
      nevent = TMath::Min(nevent,nentries);
      if (read == 1) {  //read sequential
         ioperf = getenv("ENABLE_TTREEPERFSTATS") ? new TTreePerfStats("Perf Stats", tree) : nullptr;
         //by setting the read cache to -1 we set it to the AutoFlush value when writing
         Int_t cachesize = -1;
         if (punzip) tree->SetParallelUnzip();
         tree->SetCacheSize(cachesize);
         tree->SetCacheLearnEntries(1); //one entry is sufficient to learn
         tree->SetCacheEntryRange(0,nevent);
         for (ev = 0; ev < nevent; ev++) {
            tree->LoadTree(ev);  //this call is required when using the cache
            if (ev%printev == 0) {
               tnew = timer.RealTime();
               printf("event:%d, rtime=%f s\n",ev,tnew-told);
               told=tnew;
               timer.Continue();
            }
            nb += tree->GetEntry(ev);        //read complete event in memory
         }
         if (ioperf) {
            ioperf->Finish();
         }
      } else {    //read random
         Int_t evrandom;
         for (ev = 0; ev < nevent; ev++) {
            if (ev%printev == 0) std::cout<<"event="<<ev<<std::endl;
            evrandom = Int_t(nevent*gRandom->Rndm());
            nb += tree->GetEntry(evrandom);  //read complete event in memory
         }
      }
   } else {
//         Write case
      // Create a new ROOT binary machine independent file.
      // Note that this file may contain any kind of ROOT objects, histograms,
      // pictures, graphics objects, detector geometries, tracks, events, etc..
      // This file is now becoming the current directory.
      if (netf) {
         hfile = new TNetFile("root://localhost/root/test/EventNet.root","RECREATE","TTree benchmark ROOT file");
      } else
         hfile = new TFile(outFiles.back().c_str(),"RECREATE","TTree benchmark ROOT file");
      hfile->SetCompressionLevel(comp);
      hfile->SetCompressionAlgorithm(compAlg);

      // Create histogram to show write_time in function of time
      Float_t curtime = -0.5;
      Int_t ntime = nevent / printev;
      TH1F *htime = new TH1F("htime", "Real-Time to write versus time", ntime, 0, ntime);
      HistogramManager *hm = 0;
      if (hfill) {
         TDirectory *hdir = new TDirectory("histograms", "all histograms");
         hm = new HistogramManager(hdir);
      }

      // Create a ROOT Tree and one superbranch
      tree = new TTree("T","An example of a ROOT tree");
      tree->SetAutoSave(1000000000); // autosave when 1 Gbyte written
      tree->SetCacheSize(10000000);  // set a 10 MBytes cache (useless when writing local files)
      bufsize = 64000;
      if (split)  bufsize /= 4;
      event = new Event();           // By setting the value, we own the pointer and must delete it.
      TTree::SetBranchStyle(branchStyle);
      TBranch *branch = tree->Branch("event", &event, bufsize,split);
      branch->SetAutoDelete(kFALSE);
      if(split >= 0 && branchStyle) tree->BranchRef();
      Float_t ptmin = 1;

      for (ev = 0; ev < nevent; ev++) {
         if (ev%printev == 0) {
            tnew = timer.RealTime();
            printf("event:%d, rtime=%f s\n",ev,tnew-told);
            htime->Fill(curtime,tnew-told);
            curtime += 1;
            told=tnew;
            timer.Continue();
         }

         event->Build(ev, arg5, ptmin);

         if (write) nb += tree->Fill();  //fill the tree

         if (hm) hm->Hfill(event);      //fill histograms
      }
      if (write) {
         hfile = tree->GetCurrentFile(); //just in case we switched to a new file
         hfile->Write();
         tree->Print();
      }
   }
   // We own the event (since we set the branch address explicitly), we need to delete it.
   delete event;  event = 0;

   //  Stop timer and print results
   timer.Stop();
   Float_t mbytes = 0.000001*nb;
   Double_t rtime = timer.RealTime();
   Double_t ctime = timer.CpuTime();


   printf("\n%d events and %lld bytes processed.\n",nevent,nb);
   printf("RealTime=%f seconds, CpuTime=%f seconds\n",rtime,ctime);
   if (read) {
      tree->PrintCacheStats();
      if (ioperf) {
         ioperf->Print();
      }
      printf("You read %f Mbytes/Realtime seconds\n", mbytes / rtime);
      printf("You read %f Mbytes/Cputime seconds\n", mbytes / ctime);
   } else {
      printf("compression level=%d, split=%d, arg4=%d, IMT=%d, compression algorithm=%d\n", comp, split, arg4,
             enable_imt, compAlg);
      printf("You write %f Mbytes/Realtime seconds\n",mbytes/rtime);
      printf("You write %f Mbytes/Cputime seconds\n",mbytes/ctime);
      //printf("file compression factor = %f\n",hfile.GetCompressionFactor());
   }
   hfile->Close();
   return 0;
}

int main(int argc, char **argv)
{
   Int_t nevent = 400;     // by default create 400 events
   Int_t comp   = 1;       // by default file is compressed
   Int_t split  = 1;       // by default, split Event in sub branches
   Int_t arg4   = 1;
   Int_t arg5   = 600;     //default number of tracks per event
   Int_t enable_imt = 0;   // Whether to enable IMT mode.
   Int_t compAlg = defaultComp; // Allow user to specify underlying compression algorithm.

   if (argc > 1)  nevent = atoi(argv[1]);
   if (argc > 2)  comp   = atoi(argv[2]);
   if (argc > 3)  split  = atoi(argv[3]);
   if (argc > 4)  arg4   = atoi(argv[4]);
   if (argc > 5)  arg5   = atoi(argv[5]);
   if (argc > 6)  enable_imt = atoi(argv[6]);
   if (argc > 7) compAlg = atoi(argv[7]);

   // all futher arguments are interpreted as additional output file names
   std::vector<std::string> outFiles;
   outFiles.push_back("Event.root");
   for (int i = 8; i < argc; ++i) {
      outFiles.push_back(argv[i]);
   }

   return MainEvent(nevent, comp, split, arg4, arg5, enable_imt, compAlg, outFiles);
}
