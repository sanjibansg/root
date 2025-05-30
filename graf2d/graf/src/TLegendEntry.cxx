// @(#)root/graf:$Id$
// Author: Matthew.Adam.Dobbs   06/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cstdio>

#include "TLegendEntry.h"
#include "TVirtualPad.h"
#include "TROOT.h"
#include <iostream>

ClassImp(TLegendEntry);

/** \class TLegendEntry
\ingroup BasicGraphics

Storage class for one entry of a TLegend.
*/

////////////////////////////////////////////////////////////////////////////////
/// TLegendEntry do-nothing default constructor

TLegendEntry::TLegendEntry(): TAttText(), TAttLine(), TAttFill(), TAttMarker()
{
   fObject = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// TLegendEntry normal constructor for one entry in a TLegend.
///
/// obj is the object this entry will represent. If obj has
/// line/fill/marker attributes, then the TLegendEntry will display
/// these attributes.
///
/// label is the text that will describe the entry, it is displayed using
/// TLatex, so may have a complex format.
///
/// option may have values
///  - L draw line associated w/ TAttLine if obj inherits from TAttLine
///  - P draw polymarker assoc. w/ TAttMarker if obj inherits from TAttMarker
///  - F draw a box with fill associated w/ TAttFill if obj inherits TAttFill
///    default is object = "LPF"

TLegendEntry::TLegendEntry(const TObject* obj, const char* label, Option_t* option )
             :TAttText(0,0,0,0,0), TAttLine(1,1,1), TAttFill(0,0), TAttMarker(1,21,1)
{
   fObject = nullptr;
   if ( !label && obj ) fLabel = obj->GetTitle();
   else                 fLabel = label;
   fOption = option;
   if (obj) SetObject((TObject*)obj);
}

////////////////////////////////////////////////////////////////////////////////
/// TLegendEntry copy constructor

TLegendEntry::TLegendEntry(const TLegendEntry &entry) : TObject(entry), TAttText(entry), TAttLine(entry), TAttFill(entry), TAttMarker(entry)
{
   entry.TLegendEntry::Copy(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// TLegendEntry default destructor

TLegendEntry::~TLegendEntry()
{
   fObject = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// copy this TLegendEntry into obj

void TLegendEntry::Copy( TObject &obj ) const
{
   TObject::Copy(obj);
   TAttText::Copy((TLegendEntry&)obj);
   TAttLine::Copy((TLegendEntry&)obj);
   TAttFill::Copy((TLegendEntry&)obj);
   TAttMarker::Copy((TLegendEntry&)obj);
   ((TLegendEntry&)obj).fObject = fObject;
   ((TLegendEntry&)obj).fLabel = fLabel;
   ((TLegendEntry&)obj).fOption = fOption;
}

////////////////////////////////////////////////////////////////////////////////
/// dump this TLegendEntry to std::cout

void TLegendEntry::Print( Option_t *) const
{
   TString output;
   std::cout << "TLegendEntry: Object ";
   if ( fObject ) output = fObject->GetName();
   else output = "NULL";
   std::cout << output << " Label ";
   if ( fLabel ) output = fLabel.Data();
   else output = "NULL";
   std::cout << output << " Option ";
   if (fOption ) output = fOption.Data();
   else output = "NULL";
   std::cout << output << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Save this TLegendEntry as C++ statements on output stream out
///  to be used with the SaveAs .C option

void TLegendEntry::SaveEntry(std::ostream &out, const char* name)
{
   if (gROOT->ClassSaved(TLegendEntry::Class()))
      out << "   legentry = ";
   else
      out << "   TLegendEntry *legentry = ";

   TString objname = fObject ? fObject->GetName() : "NULL";
   out << name << "->AddEntry(\"" << objname << "\",\"" << TString(fLabel).ReplaceSpecialCppChars() << "\",\""
       << TString(fOption).ReplaceSpecialCppChars() << "\");\n";
   // if default style is detected - copy attributes from object
   // it can happen when legend was not paint before writing into the file
   if (fObject && GetFillStyle() == 0 && GetFillColor() == 0 && GetLineStyle() == 1 && GetLineColor() == 1 && GetLineWidth() == 1) {
      TString opt = fOption;
      opt.ToLower();
      if (opt.Contains("f") && fObject->InheritsFrom(TAttFill::Class()))
         dynamic_cast<TAttFill*>(fObject)->Copy(*this);
      if (opt.Contains("p") && fObject->InheritsFrom(TAttMarker::Class()))
         dynamic_cast<TAttMarker*>(fObject)->Copy(*this);
      if ((opt.Contains("l") || opt.Contains("f")) && fObject->InheritsFrom(TAttLine::Class()))
         dynamic_cast<TAttLine*>(fObject)->Copy(*this);
   }

   SaveFillAttributes(out, "legentry", 0, 0);
   SaveLineAttributes(out, "legentry", 1, 1, 1);
   SaveMarkerAttributes(out, "legentry", 1, 21, 1);
   SaveTextAttributes(out, "legentry", 0, 0, 0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// (re)set the obj pointed to by this entry

void TLegendEntry::SetObject(TObject* obj )
{
   if ( ( fObject && fLabel == fObject->GetTitle() ) || !fLabel ) {
      if (obj) fLabel = obj->GetTitle();
   }
   fObject = obj;
}

////////////////////////////////////////////////////////////////////////////////
/// (re)set the obj pointed to by this entry

void TLegendEntry::SetObject(const char* objectName)
{
   TList *padprimitives = gPad ? gPad->GetListOfPrimitives() : nullptr;
   TObject *obj = padprimitives ? padprimitives->FindObject(objectName) : nullptr;
   if (obj) SetObject(obj);
}
