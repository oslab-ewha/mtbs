CLEANFILES = *~
DISTCLEANFILES = *~

LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

bin_PROGRAMS = mtbs

mtbs_SOURCES = mtbs.c tbs_sd.cu tbs_native.cu benchmgr.cu benchrun.cu skrun.cu \
	fedkern.cu report.cu misc.cu mtbs.h mtbs_cu.h benchapi.cu sched.cu \
	sched_dyn.h sched_dyn_rr.cu sched_dyn_rrf.cu \
	sched_dyn.cu sched_dyn.cuh \
	sched_static.cu sched_static.cuh \
	sched_hw.cu \
	sched_pagoda.cu sched_pagoda.h sched_pagoda.cuh \
	sched_gemtc.cu sched_gemtc.cuh sched_gemtcP.cu sched_gemtcP.cuh \
	stream.cu stream.h mem.cu cudainit.cu

sched_dyn.o: sched_dyn.cuh
sched_static.o: sched_static.cuh
sched_pagoda.o: sched_pagoda.cuh
sched_gemtc.o:  sched_gemtc.cuh
sched_gemtcP.o:  sched_gemtcP.cuh

mtbs_LDADD = ../../benchmarks/inhouse/loopcalc/libloopcalc.a \
		../../benchmarks/inhouse/mklc/libmklc.a \
		../../benchmarks/inhouse/gma/libgma.a \
		../../benchmarks/inhouse/lma/liblma.a \
		../../benchmarks/kmeans/libkmeans.a \
		../../benchmarks/mandelbrot/libmandelbrot.a \
		../../benchmarks/mm/libmm.a \
		../../benchmarks/inhouse/irregular/libirregular.a \
		-lcuda -lcudart

include $(top_srcdir)/makefile.cu
