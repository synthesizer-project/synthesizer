/******************************************************************************
 * A C module containing macros for use in C extensions.
 *****************************************************************************/

/* Define a macro to handle that bzero is non-standard. */
#ifndef bzero
#define bzero(b, len) (memset((b), '\0', (len)), (void)0)
#endif
