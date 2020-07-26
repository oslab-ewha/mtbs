#include "../../config.h"

#include "mtbs_cu.h"

#include <sys/times.h>

unsigned	n_sm_count;
unsigned	n_threads_per_MTB;	/* per macro TB */
unsigned	n_MTBs_per_sm;

static struct timespec  started_ts;

__device__ uint
get_smid(void)
{
	uint	ret;
	asm("mov.u32 %0, %smid;" : "=r"(ret));
	return ret;
}

__device__ uint
get_laneid(void)
{
	uint	ret;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ void
sleep_in_kernel(void)
{
#if CUDA_COMPUTE >= 70
	asm("nanosleep.u32 1;");
#endif
}

unsigned long long
get_ticks(void)
{
	struct timespec	ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

extern "C" void
error(const char *fmt, ...)
{
	char	*msg;
	va_list	ap;
	int	n;

	va_start(ap, fmt);
	n = vasprintf(&msg, fmt, ap);
	va_end(ap);
	if (n >= 0) {
		fprintf(stderr, "error: %s\n", msg);
		free(msg);
	}
}

const char *
get_cuda_error_msg(CUresult err)
{
	const char	*msg;
	CUresult	res;

	res = cuGetErrorString(err, &msg);
	if (res != CUDA_SUCCESS)
		return "";
	return msg;
}

void
init_tickcount(void)
{
        clock_gettime(CLOCK_MONOTONIC, &started_ts);
}

/* microsecs */
unsigned
get_tickcount(void)
{
	struct timespec	ts;
	unsigned	ticks;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	if (ts.tv_nsec < started_ts.tv_nsec) {
		ticks = ((unsigned)(ts.tv_sec - started_ts.tv_sec - 1)) * 1000000;
		ticks += (1000000000 + ts.tv_nsec - started_ts.tv_nsec) / 1000;
	}
	else {
		ticks = ((unsigned)(ts.tv_sec - started_ts.tv_sec)) * 1000000;
		ticks += (ts.tv_nsec - started_ts.tv_nsec) / 1000;
        }

	return ticks;
}
