#include "mtbs.h"

static void
usage(void)
{
	printf(
"mtbs <options> [<run prefix>]<benchmark spec>...\n"
"<options>:\n"
"  -d <device no>: select GPU device\n"
"  -s <sched type>: scheduling type\n"
"     supported scheduling: hw(hardware scheduling, default)\n"
"                           dynamic(dynamic software-defined scheduling)\n"
"                           static(static software-defined scheduling)\n"
"  -w <# of workers>: # of sumission worker, default: 128\n"
"  -S <# of pre-allocated streams>: # of streams, default: 128\n"
"  -M <MTB count per sm>\n"
"  -T <thread count per MTB>\n"
"  -h: help\n"
"<run prefix>: NN*\n"
"  following benchmark is submitted NN times\n"
"<benchmark spec>: <code>:<arg string>\n"
" <code>:\n"
"  lc: repetitive loop calculation(in-house)\n"
"  mklc: multi kernel loop calculation(in-house)\n"
"  gma: global memory access(in-house)\n"
"  lma: localized memory access(in-house)\n"
"  kmeans: kmeans\n"
"  mb: mandelbrot\n"
" <arg string>:\n"
"   NOTE: First 4 arguments are <grid width>,<grid height>,<tb width>,<tb height>\n"
"   lc: <calculation type>,<iterations>,<# iterations for calculation type greater than 3>\n"
"        calculation type: 1(int),2(float),3(double),default:empty\n"
"                          4(float/double),5(int/float),6(int/double)\n"
"                          7(float/double tb),8(int/float tb),9(int/double tb)\n"
"   mklc: <# of kernels>,<iterations>\n"
"   gma: <global mem in KB>,<iterations>\n"
"   lma: <chunk size in byte>,<reference span>,<iterations>\n"
"   kmeans: <# of points per thread>,<# of clusters>,<# of features>,<iterations>\n"
"   mb: <image width>,<image height>\n"
		);
}

unsigned	devno;
unsigned	arg_n_MTBs_per_sm;
unsigned	arg_n_threads_per_MTB;
unsigned	n_submission_workers = 128;
unsigned	n_streams = 128;

static int
parse_benchargs(int argc, char *argv[])
{
	int	i;

	if (argc == 0) {
		error("no benchmark provided");
		return -1;
	}
	for (i = 0; i < argc; i++) {
		char	*bencharg, *colon, *args = NULL;
		unsigned	runcount = 1;

		if (sscanf(argv[i], "%u*", &runcount) == 1) {
			char	*star;
			star = strchr(argv[i], '*');
			bencharg = strdup(star + 1);
		}
		else
			bencharg = strdup(argv[i]);
		colon = strchr(bencharg, ':');
		if (colon != NULL) {
			*colon = '\0';
			args = strdup(colon + 1);
		}
		if (!add_bench(runcount, bencharg, args)) {
			free(bencharg);
			error("invalid benchmark code or arguments: %s", argv[i]);
			return -1;
		}
		free(bencharg);
	}
	return 0;
}

static void
select_device(const char *str_devno)
{
	if (sscanf(str_devno, "%u", &devno) != 1) {
		error("%s: invalid GPU device number", str_devno);
		return;
	}
	if (!select_gpu_device(devno)) {
		error("failed to set GPU device: %s", str_devno);
	}
}

static void
setup_n_MTBs(const char *str_n_MTBs)
{
	if (sscanf(str_n_MTBs, "%u", &arg_n_MTBs_per_sm) != 1 || arg_n_MTBs_per_sm == 0) {
		error("%s: invalid number of MTBs per SM", str_n_MTBs);
	}
}

static void
setup_n_threads(const char *str_n_threads)
{
	if (sscanf(str_n_threads, "%u", &arg_n_threads_per_MTB) != 1 || arg_n_threads_per_MTB == 0) {
		error("%s: invalid number of threads per MTB", str_n_threads);
	}
}

static int
parse_options(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "d:s:w:S:M:T:h")) != -1) {
		switch (c) {
		case 'd':
			select_device(optarg);
			break;
		case 's':
			setup_sched(optarg);
			break;
		case 'w':
			sscanf(optarg, "%u", &n_submission_workers);
			break;
		case 'S':
			sscanf(optarg, "%u", &n_streams);
			break;
		case 'M':
			setup_n_MTBs(optarg);
			break;
		case 'T':
			setup_n_threads(optarg);
			break;
		case 'h':
			usage();
			return -100;
		default:
			usage();
			return -1;
		}
	}
	return 0;
}

int
main(int argc, char *argv[])
{
	unsigned	elapsed;

	if (parse_options(argc, argv) < 0) {
		return 1;
	}

	if (parse_benchargs(argc - optind, argv + optind) < 0) {
		return 2;
	}

	if (run_tbs(&elapsed)) {
		report(elapsed);
		return 0;
	}
	return 4;
}
