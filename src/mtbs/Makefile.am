CLEANFILES = *~
DISTCLEANFILES = *~

LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

bin_PROGRAMS = mtbs

mtbs_SOURCES = mtbs.c tbs_sd.cu tbs_native.cu benchmgr.cu benchrun.cu skrun.cu \
	fedkern.cu report.cu misc.cu mtbs.h mtbs_cu.h benchapi.cu sched.cu \
	sched_dyn.cu sched_dyn.h sched_dyn_rr.cu sched_dyn_rrf.cu sched_dyn_fca.cu sched_dyn_rrm.cu \
	stream.cu stream.h

mtbs_LDADD = ../../benchmarks/inhouse/loopcalc/libloopcalc.a \
		../../benchmarks/inhouse/mklc/libmklc.a \
		../../benchmarks/inhouse/gma/libgma.a \
		../../benchmarks/inhouse/lma/liblma.a \
		../../benchmarks/kmeans/libkmeans.a \
		-lcuda -lcudart

include $(top_srcdir)/makefile.cu
