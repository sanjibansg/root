#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
ROOTTEST_ADD_TEST(chain
                  MACRO runchain.C
                  OUTREF chain.ref)

ROOTTEST_ADD_TEST(listarray
                  MACRO execTEntryListArray.cxx
                  OUTREF execTEntryListArray.ref)

# Regression test for https://github.com/root-project/root/issues/11026
ROOT_ADD_GTEST(test_tentrylist_regression_intoverflow regression_intoverflow.cxx LIBRARIES ROOT::Core ROOT::Tree)
