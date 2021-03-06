#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([mtbs], [0.1.0], [cezanne@codemayo.com])
AC_CONFIG_SRCDIR([src/mtbs/mtbs.c])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([foreign])

AM_SILENT_RULES([yes])

AM_MAINTAINER_MODE([disable])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_MAKE_SET
AC_PROG_RANLIB
AX_CHECK_CUDA

# FIXME: Replace `main' with a function in `-lpthread':
AC_CHECK_LIB([pthread], [main])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h strings.h sys/time.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([gettimeofday sqrt strchr])

AC_CONFIG_FILES([Makefile
                 src/Makefile
		 src/mtbs/Makefile
                 benchmarks/Makefile
                 benchmarks/inhouse/Makefile
                 benchmarks/inhouse/loopcalc/Makefile
                 benchmarks/inhouse/mklc/Makefile
                 benchmarks/inhouse/gma/Makefile
                 benchmarks/inhouse/lma/Makefile
                 benchmarks/inhouse/irregular/Makefile
                 benchmarks/inhouse/syncsum/Makefile
                 benchmarks/mandelbrot/Makefile
                 benchmarks/mm/Makefile
                 benchmarks/kmeans/Makefile])

AC_OUTPUT
