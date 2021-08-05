// Author: Sanjiban Sengupta, 2021
// Description: 
//           This is to test the Serialisation property of RModel
//           object defined in SOFIE. The program is run when the 
//           target 'TestCustomModelsFromROOT' is built. The program
//           generates the required .hxx file after reading a written
//           ROOT file which stores the object of the RModel class.

#include <iostream>

#include "RModel.hxx"
#include "RModelParser_ONNX.hxx"

using namespace TMVA::Experimental::SOFIE;

int main(int argc, char *argv[]){
   if (argc < 2) {
      std::cerr << "[ERROR]: Missing ONNX input file\n";
      return -1;
   }
   std::string outname=argv[2];
   RModelParser_ONNX parser;
   RModel model = parser.Parse(argv[1]);
   TFile fileWrite(outname+"FromROOT.root","RECREATE");
   model.Write("model");
   fileWrite.Close();
   TFile fileRead(outname+".root","READ");
   SOFIE::RModel *modelPtr;
   fileRead.GetObject("model",modelPtr);
   fileRead.Close();
   modelPtr->Generate();
   modelPtr->OutputGenerated(outname+"_FromROOT.hxx");   
   return 0;
} 