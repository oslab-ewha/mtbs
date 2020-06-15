#include "mtbs_cu.h"

#include "stream.h"

#define MAX_STREAMS	120

static vStrm_t	vStrms[MAX_STREAMS];
static unsigned	idx_allocator;

vstream_t
create_vstream(void)
{
	vstrm_t	vstrm;
	
	vstrm = &vStrms[idx_allocator];
	idx_allocator = (idx_allocator + 1) % MAX_STREAMS;
	vstrm->refcnt++;
	return vstrm;
}

void
destroy_vstream(vstream_t strm)
{
	vstrm_t	vstrm = (vstrm_t)strm;

	assert(vstrm->refcnt > 0);
	vstrm->refcnt--;
}

void
init_streams(void)
{
	unsigned	i;

	for (i = 0; i < MAX_STREAMS; i++) {
		cudaStreamCreate(&vStrms[i].cudaStrm);
		vStrms[i].refcnt = 0;
	}
}
