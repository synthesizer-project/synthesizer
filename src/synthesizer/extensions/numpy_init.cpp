#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#undef NO_IMPORT_ARRAY // Allow import_array() to be defined
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy_init.h"

int numpy_import() {
  if (_import_array() < 0)
    return -1;
  return 0;
}
