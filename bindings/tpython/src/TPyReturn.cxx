// Author: Enric Tejedor CERN  08/2019
// Original PyROOT code by Wim Lavrijsen, LBL
//
// /*************************************************************************
//  * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
//  * All rights reserved.                                                  *
//  *                                                                       *
//  * For the licensing terms see $ROOTSYS/LICENSE.                         *
//  * For the list of contributors see $ROOTSYS/README/CREDITS.             *
//  *************************************************************************/

// Bindings
#include "CPyCppyy/API.h"
#include "TPyReturn.h"

// ROOT
#include "TObject.h"
#include "TInterpreter.h"

// Standard
#include <stdexcept>

//______________________________________________________________________________
//                        Python expression eval result
//                        =============================
//
// Transport class for bringing objects from python (dynamically typed) to Cling
// (statically typed). It is best to immediately cast a TPyReturn to the real
// type, either implicitly (for builtin types) or explicitly (through a void*
// cast for pointers to ROOT objects).
//
// Examples:
//
//  root [0] TBrowser* b = (void*)TPython::Eval( "ROOT.TBrowser()" );
//  root [1] int i = TPython::Eval( "1+1" );
//  root [2] i
//  (int)2
//  root [3] double d = TPython::Eval( "1+3.1415" );
//  root [4] d
//  (double)4.14150000000000063e+00

//- data ---------------------------------------------------------------------
ClassImp(TPyReturn);

namespace {
   class PyGILRAII {
      PyGILState_STATE m_GILState;
   public:
      PyGILRAII() : m_GILState(PyGILState_Ensure()) { }
      ~PyGILRAII() { PyGILState_Release(m_GILState); }
   };
}

//- constructors/destructor --------------------------------------------------
TPyReturn::TPyReturn()
{
   PyGILRAII gilRaii;

   // Construct a TPyReturn object from Py_None.
   Py_IncRef(Py_None);
   fPyObject = Py_None;
}

////////////////////////////////////////////////////////////////////////////////
/// Construct a TPyReturn from a python object. The python object may represent
/// a ROOT object. Steals reference to given python object.

TPyReturn::TPyReturn(PyObject *pyobject)
{
   PyGILRAII gilRaii;

   if (!pyobject) {
      Py_IncRef(Py_None);
      fPyObject = Py_None;
   } else
      fPyObject = pyobject; // steals reference
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor. Applies python object reference counting.

TPyReturn::TPyReturn(const TPyReturn &other)
{
   PyGILRAII gilRaii;

   Py_IncRef(other.fPyObject);
   fPyObject = other.fPyObject;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator. Applies python object reference counting.

TPyReturn &TPyReturn::operator=(const TPyReturn &other)
{
   PyGILRAII gilRaii;

   if (this != &other) {
      Py_IncRef(other.fPyObject);
      Py_DecRef(fPyObject);
      fPyObject = other.fPyObject;
   }

   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. Reference counting for the held python object is in effect.

TPyReturn::~TPyReturn()
{
   PyGILRAII gilRaii;

   Py_DecRef(fPyObject);
}

//- public members -----------------------------------------------------------
TPyReturn::operator char *() const
{
   PyGILRAII gilRaii;

   // Cast python return value to C-style string (may fail).
   return (char *)((const char *)*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C-style string (may fail).

TPyReturn::operator const char *() const
{
   PyGILRAII gilRaii;

   if (fPyObject == Py_None) // for void returns
      return 0;

   const char *s = PyUnicode_AsUTF8(fPyObject);
   if (PyErr_Occurred()) {
      PyErr_Print();
      return 0;
   }

   return s;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C++ char (may fail).

TPyReturn::operator Char_t() const
{
   PyGILRAII gilRaii;

   std::string s = operator const char *();
   if (s.size())
      return s[0];

   return '\0';
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C++ long (may fail).

TPyReturn::operator Long_t() const
{
   PyGILRAII gilRaii;

   Long_t l = PyLong_AsLong(fPyObject);

   if (PyErr_Occurred())
      PyErr_Print();

   return l;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C++ unsigned long (may fail).

TPyReturn::operator ULong_t() const
{
   PyGILRAII gilRaii;

   ULong_t ul = PyLong_AsUnsignedLong(fPyObject);

   if (PyErr_Occurred())
      PyErr_Print();

   return ul;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C++ double (may fail).

TPyReturn::operator Double_t() const
{
   PyGILRAII gilRaii;

   Double_t d = PyFloat_AsDouble(fPyObject);

   if (PyErr_Occurred())
      PyErr_Print();

   return d;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to ROOT object with dictionary (may fail; note that
/// you have to use the void* converter, as CINT will not call any other).

TPyReturn::operator void *() const
{
   PyGILRAII gilRaii;

   if (fPyObject == Py_None)
      return 0;

   return static_cast<void *>(CPyCppyy::PyResult{fPyObject});
}

////////////////////////////////////////////////////////////////////////////////
/// Direct return of the held PyObject; note the new reference.

TPyReturn::operator PyObject *() const
{
   PyGILRAII gilRaii;

   if (fPyObject == Py_None)
      return 0;

   Py_IncRef(fPyObject);
   return fPyObject;
}
