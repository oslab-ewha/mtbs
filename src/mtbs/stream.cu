#include "mtbs_cu.h"

#include <pthread.h>

#include "stream.h"

extern unsigned	n_streams;

static vStrm_t	*vStrms;
static unsigned	idx_allocator;
static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;

vstream_t
create_vstream(void)
{
	vstrm_t	vstrm;

	pthread_mutex_lock(&mutex);
	vstrm = &vStrms[idx_allocator];
	idx_allocator = (idx_allocator + 1) % n_streams;
	vstrm->refcnt++;
	pthread_mutex_unlock(&mutex);

	return vstrm;
}

void
destroy_vstream(vstream_t strm)
{
	vstrm_t	vstrm = (vstrm_t)strm;

	pthread_mutex_lock(&mutex);
	assert(vstrm->refcnt > 0);
	vstrm->refcnt--;
	pthread_mutex_unlock(&mutex);
}

void
init_streams(void)
{
	unsigned	i;

	vStrms = (vStrm_t *)malloc(sizeof(vStrm_t) * n_streams);
	for (i = 0; i < n_streams; i++) {
		cudaStreamCreate(&vStrms[i].cudaStrm);
		vStrms[i].refcnt = 0;
	}
}
