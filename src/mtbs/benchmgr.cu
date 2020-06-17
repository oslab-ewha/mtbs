#include "mtbs_cu.h"

#define BENCH_PROTO(name)	int name(dim3 dimGrid, dim3 dimBlock, void *args[])
#define BENCHMARK(base)		BENCH_PROTO(bench_##base);
#define BENCH_COOKARG(name)	int cookarg_##name(dim3 dimGrid, dim3 dimBlock, void *args[]);

BENCHMARK(loopcalc)
BENCHMARK(mklc)
BENCHMARK(gma)
BENCHMARK(lma)
BENCHMARK(kmeans)
BENCHMARK(mandelbrot)
BENCH_COOKARG(gma)
BENCH_COOKARG(lma)
BENCH_COOKARG(kmeans)

benchrun_t	benchruns[MAX_BENCHES];
int	n_benches;
int	n_tbs_submitted;
int	n_mtbs_submitted;

static benchinfo_t	benchinfos[] = {
	{ "lc", LOOPCALC, NULL, bench_loopcalc },
	{ "mklc", MKLC, NULL, bench_mklc },
	{ "gma", GMA, cookarg_gma, bench_gma },
	{ "lma", LMA, cookarg_lma, bench_lma },
	{ "kmeans", KMEANS, cookarg_kmeans, bench_kmeans },
	{ "mb", MANDELBROT, NULL, bench_mandelbrot },
	{ NULL, 0, NULL, NULL }
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

extern "C" BOOL
add_bench(unsigned runcount, const char *code, const char *args)
{
	benchinfo_t	*info;
	unsigned	i;

	info = find_benchinfo(code);
	if (info == NULL)
		return FALSE;
	for (i = 0; i < runcount; i++) {
		benchrun_t	*brun = benchruns + n_benches;
		int	n_tbs;

		brun->info = info;
		if (!parse_args(args, brun))
			return FALSE;
		if (info->cookarg_func != NULL) {
			if (info->cookarg_func(brun->dimGrid, brun->dimBlock, brun->args) < 0) {
				error("failed to cook arguments");
				return FALSE;
			}
		}
		n_tbs = brun->dimGrid.x * brun->dimGrid.y;
		n_tbs_submitted += n_tbs;
		n_mtbs_submitted += (n_tbs * (brun->dimBlock.x * brun->dimBlock.y / N_THREADS_PER_mTB));
		
		n_benches++;
	}

	return TRUE;
}
