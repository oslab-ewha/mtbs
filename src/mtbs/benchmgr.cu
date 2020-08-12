#include "mtbs_cu.h"

#define BENCH_PROTO(name)	int name(dim3 dimGrid, dim3 dimBlock, void *args[])
#define BENCHMARK(base)		BENCH_PROTO(bench_##base);

BENCHMARK(loopcalc)
BENCHMARK(mklc)
BENCHMARK(gma)
BENCHMARK(lma)
BENCHMARK(kmeans)
BENCHMARK(mandelbrot)
BENCHMARK(irregular)
BENCHMARK(mm)
BENCHMARK(syncsum)

benchrun_t	*benchruns;
int	n_benches;
int	n_benches_alloc;
int	n_tbs_submitted;
int	n_mtbs_submitted;

static benchinfo_t	benchinfos[] = {
	{ "lc", LOOPCALC, bench_loopcalc },
	{ "mklc", MKLC, bench_mklc },
	{ "gma", GMA, bench_gma },
	{ "lma", LMA, bench_lma },
	{ "kmeans", KMEANS, bench_kmeans },
	{ "mb", MANDELBROT, bench_mandelbrot },
	{ "irr", IRREGULAR, bench_irregular },
	{ "mm", MM, bench_mm },
	{ "syncsum", SYNCSUM, bench_syncsum },
	{ NULL, 0, NULL }
};

static benchinfo_t *
find_benchinfo(const char *code)
{
	int	i;

	for (i = 0; benchinfos[i].code != NULL; i++) {
		if (strcmp(benchinfos[i].code, code) == 0)
			return benchinfos + i;
	}
	return NULL;
}

static BOOL
parse_int(const char **pc_args, int *pval)
{
	const char	*p;
	int	val = 0;

	if (**pc_args == '\0')
		return FALSE;
	for (p = *pc_args; *p && *p != ','; p++) {
		if (p - *pc_args > 31)
			return FALSE;
		if (*p < '0' || *p > '9')
			return FALSE;
		val *= 10;
		val += (*p - '0');
	}
	if (*p == ',')
		p++;
	*pc_args = p;
	*pval = val;
	return TRUE;
}

static BOOL
parse_args(const char *c_args, benchrun_t *brun)
{
	int	i;

	if (c_args == NULL)
		return FALSE;
	if (!parse_int(&c_args, (int *)&brun->dimGrid.x))
		return FALSE;
	if (!parse_int(&c_args, (int *)&brun->dimGrid.y))
		return FALSE;
	if (!parse_int(&c_args, (int *)&brun->dimBlock.x))
		return FALSE;
	if (!parse_int(&c_args, (int *)&brun->dimBlock.y))
		return FALSE;
	brun->dimGrid.z = 1;
	brun->dimBlock.z = 1;
	for (i = 0; i < MAX_ARGS; i++) {
		int	arg;

		if (*c_args == '\0')
			return TRUE;
		if (!parse_int(&c_args, &arg))
			return FALSE;
		brun->args[i] = (void *)(long long)arg;
	}
	return TRUE;
}

static benchrun_t *
alloc_brun(void)
{
	benchrun_t	*brun;

	if (n_benches == n_benches_alloc) {
		benchruns = (benchrun_t *)realloc(benchruns, (n_benches + 100) * sizeof(benchrun_t));
		n_benches_alloc += 100;
	}

	brun = benchruns + n_benches;
	n_benches++;

	return brun;
}

extern "C" BOOL
add_bench(unsigned runcount, const char *code, const char *args)
{
	benchinfo_t	*info;
	unsigned	i;

	info = find_benchinfo(code);
	if (info == NULL)
		return FALSE;
	for (i = 0; i < runcount; i++) {
		benchrun_t	*brun = alloc_brun();
		int	n_tbs;

		brun->info = info;
		if (!parse_args(args, brun))
			return FALSE;
		n_tbs = brun->dimGrid.x * brun->dimGrid.y;
		n_tbs_submitted += n_tbs;
		n_mtbs_submitted += (n_tbs * (brun->dimBlock.x * brun->dimBlock.y / N_THREADS_PER_mTB));
	}

	return TRUE;
}
