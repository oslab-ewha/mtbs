#include "mtbs_cu.h"
#include "ecm_list.h"

#include <pthread.h>

#define N_MAX_UNITS	25
#define MIN_UNIT	64
#define N_HASH		4096
#define MEMCHUNK	(MIN_UNIT << (N_MAX_UNITS - 1))

static pthread_spinlock_t      lock;

typedef struct _mem {
	unsigned	size;
	void	*ptr, *base;
	struct list_head	list_mem;
	struct list_head	list_free;
	struct list_head	list_hash;
} mem_t;

unsigned long long	max_memsize = 500 * 1024 * 1024;
static unsigned long long	cur_memsize;
static unsigned		memchunk = MEMCHUNK;

static struct list_head	hash_head[N_HASH];

/* 1G, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 512k, 256k, 128k, 64k, 32k, 16k, 8k, 4k, 2k, 1k, 512, 256, 128, 64 */
static struct list_head	free_head[N_MAX_UNITS];

static unsigned
get_mem_unit_idx(unsigned size)
{
	unsigned	idx = 0;

	size >>= 7;
	while (size > 0) {
		size >>= 1;
		idx++;
	}
	return idx;
}

static unsigned
get_mem_unit_idx_ge(unsigned size)
{
	unsigned	idx;

	idx = get_mem_unit_idx(size);
	if (size > (MIN_UNIT << idx))
		idx++;
	return idx;
}

static unsigned
get_mem_unit_idx_le(unsigned size)
{
	if (size < MIN_UNIT)
		return N_MAX_UNITS;
	return get_mem_unit_idx(size);
}

static unsigned
get_mem_unit_idx_lt(unsigned size)
{
	unsigned	idx;

	if (size <= MIN_UNIT)
		return N_MAX_UNITS;
	idx = get_mem_unit_idx(size);
	if (size == (MIN_UNIT << idx))
		return N_MAX_UNITS;
	return idx;
}

static mem_t *
create_mem(void *base, void *ptr, unsigned size)
{
	mem_t	*mem;

	mem = (mem_t *)malloc(sizeof(mem_t));
	mem->base = base;
	mem->ptr = ptr;
	mem->size = size;
	INIT_LIST_HEAD(&mem->list_mem);
	INIT_LIST_HEAD(&mem->list_free);
	INIT_LIST_HEAD(&mem->list_hash);

	return mem;
}

static void
add_freemem(mem_t *mem)
{
	unsigned	idx;

	idx = get_mem_unit_idx(mem->size);
	list_add_tail(&mem->list_free, &free_head[idx]);
}

static unsigned crc32_table[256] =
{
	0x00000000,0x04c11db7,0x09823b6e,0x0d4326d9,0x130476dc,0x17c56b6b,0x1a864db2,0x1e475005,
	0x2608edb8,0x22c9f00f,0x2f8ad6d6,0x2b4bcb61,0x350c9b64,0x31cd86d3,0x3c8ea00a,0x384fbdbd,
	0x4c11db70,0x48d0c6c7,0x4593e01e,0x4152fda9,0x5f15adac,0x5bd4b01b,0x569796c2,0x52568b75,
	0x6a1936c8,0x6ed82b7f,0x639b0da6,0x675a1011,0x791d4014,0x7ddc5da3,0x709f7b7a,0x745e66cd,
	0x9823b6e0,0x9ce2ab57,0x91a18d8e,0x95609039,0x8b27c03c,0x8fe6dd8b,0x82a5fb52,0x8664e6e5,
	0xbe2b5b58,0xbaea46ef,0xb7a96036,0xb3687d81,0xad2f2d84,0xa9ee3033,0xa4ad16ea,0xa06c0b5d,
	0xd4326d90,0xd0f37027,0xddb056fe,0xd9714b49,0xc7361b4c,0xc3f706fb,0xceb42022,0xca753d95,
	0xf23a8028,0xf6fb9d9f,0xfbb8bb46,0xff79a6f1,0xe13ef6f4,0xe5ffeb43,0xe8bccd9a,0xec7dd02d,
	0x34867077,0x30476dc0,0x3d044b19,0x39c556ae,0x278206ab,0x23431b1c,0x2e003dc5,0x2ac12072,
	0x128e9dcf,0x164f8078,0x1b0ca6a1,0x1fcdbb16,0x018aeb13,0x054bf6a4,0x0808d07d,0x0cc9cdca,
	0x7897ab07,0x7c56b6b0,0x71159069,0x75d48dde,0x6b93dddb,0x6f52c06c,0x6211e6b5,0x66d0fb02,
	0x5e9f46bf,0x5a5e5b08,0x571d7dd1,0x53dc6066,0x4d9b3063,0x495a2dd4,0x44190b0d,0x40d816ba,
	0xaca5c697,0xa864db20,0xa527fdf9,0xa1e6e04e,0xbfa1b04b,0xbb60adfc,0xb6238b25,0xb2e29692,
	0x8aad2b2f,0x8e6c3698,0x832f1041,0x87ee0df6,0x99a95df3,0x9d684044,0x902b669d,0x94ea7b2a,
	0xe0b41de7,0xe4750050,0xe9362689,0xedf73b3e,0xf3b06b3b,0xf771768c,0xfa325055,0xfef34de2,
	0xc6bcf05f,0xc27dede8,0xcf3ecb31,0xcbffd686,0xd5b88683,0xd1799b34,0xdc3abded,0xd8fba05a,
	0x690ce0ee,0x6dcdfd59,0x608edb80,0x644fc637,0x7a089632,0x7ec98b85,0x738aad5c,0x774bb0eb,
	0x4f040d56,0x4bc510e1,0x46863638,0x42472b8f,0x5c007b8a,0x58c1663d,0x558240e4,0x51435d53,
	0x251d3b9e,0x21dc2629,0x2c9f00f0,0x285e1d47,0x36194d42,0x32d850f5,0x3f9b762c,0x3b5a6b9b,
	0x0315d626,0x07d4cb91,0x0a97ed48,0x0e56f0ff,0x1011a0fa,0x14d0bd4d,0x19939b94,0x1d528623,
	0xf12f560e,0xf5ee4bb9,0xf8ad6d60,0xfc6c70d7,0xe22b20d2,0xe6ea3d65,0xeba91bbc,0xef68060b,
	0xd727bbb6,0xd3e6a601,0xdea580d8,0xda649d6f,0xc423cd6a,0xc0e2d0dd,0xcda1f604,0xc960ebb3,
	0xbd3e8d7e,0xb9ff90c9,0xb4bcb610,0xb07daba7,0xae3afba2,0xaafbe615,0xa7b8c0cc,0xa379dd7b,
	0x9b3660c6,0x9ff77d71,0x92b45ba8,0x9675461f,0x8832161a,0x8cf30bad,0x81b02d74,0x857130c3,
	0x5d8a9099,0x594b8d2e,0x5408abf7,0x50c9b640,0x4e8ee645,0x4a4ffbf2,0x470cdd2b,0x43cdc09c,
	0x7b827d21,0x7f436096,0x7200464f,0x76c15bf8,0x68860bfd,0x6c47164a,0x61043093,0x65c52d24,
	0x119b4be9,0x155a565e,0x18197087,0x1cd86d30,0x029f3d35,0x065e2082,0x0b1d065b,0x0fdc1bec,
	0x3793a651,0x3352bbe6,0x3e119d3f,0x3ad08088,0x2497d08d,0x2056cd3a,0x2d15ebe3,0x29d4f654,
	0xc5a92679,0xc1683bce,0xcc2b1d17,0xc8ea00a0,0xd6ad50a5,0xd26c4d12,0xdf2f6bcb,0xdbee767c,
	0xe3a1cbc1,0xe760d676,0xea23f0af,0xeee2ed18,0xf0a5bd1d,0xf464a0aa,0xf9278673,0xfde69bc4,
	0x89b8fd09,0x8d79e0be,0x803ac667,0x84fbdbd0,0x9abc8bd5,0x9e7d9662,0x933eb0bb,0x97ffad0c,
	0xafb010b1,0xab710d06,0xa6322bdf,0xa2f33668,0xbcb4666d,0xb8757bda,0xb5365d03,0xb1f740b4
};

static unsigned
get_hashcode(void *ptr)
{
	unsigned	crc;
	unsigned	i;

	crc = 0xffffffff;
	for (i = 0; i < 8; i++) {
		unsigned char	val = ((unsigned char *)&ptr)[i];
		crc = (crc << 8) ^ crc32_table[(crc >> 24) ^ val];
	}
	return ~crc % N_HASH;
}

static void
insert_hash(mem_t *mem)
{
	unsigned        hashcode;
	hashcode = get_hashcode(mem->ptr);
	list_add_tail(&mem->list_hash, &hash_head[hashcode]);
}

static void
destroy_mem(mem_t *mem)
{
	list_del_init(&mem->list_free);
	list_del_init(&mem->list_hash);
	list_del_init(&mem->list_mem);
	free(mem);
}

static BOOL
is_free_mem(mem_t *mem)
{
	if (!list_empty(&mem->list_free))
		return TRUE;
	return FALSE;
}

static void
add_mem(void *ptr, unsigned size)
{
	mem_t	*mem;

	mem = create_mem(ptr, ptr, size);
	insert_hash(mem);
	add_freemem(mem);
}

static mem_t *
get_first_freemem(unsigned idx)
{
	unsigned	i;

	for (i = idx; i < N_MAX_UNITS; i++) {
		mem_t	*mem;

		if (list_empty(&free_head[i]))
			continue;

		mem = list_entry(free_head[i].next, mem_t, list_free);
		list_del_init(&mem->list_free);
		return mem;
	}
	return NULL;
}

static mem_t *
get_prev_mem(mem_t *mem)
{
	mem_t	*mem_prev = list_entry(mem->list_mem.prev, mem_t, list_mem);

	if (mem_prev->ptr < mem->ptr)
		return mem_prev;
	return NULL;
}

static mem_t *
get_next_mem(mem_t *mem)
{
	mem_t	*mem_next = list_entry(mem->list_mem.next, mem_t, list_mem);

	if (mem->ptr < mem_next->ptr)
		return mem_next;
	return NULL;
}

static mem_t *
get_free_prev_mem(mem_t *mem)
{
	mem_t	*mem_prev = get_prev_mem(mem);

	if (mem_prev != NULL && is_free_mem(mem_prev))
		return mem_prev;
	return NULL;
}

static mem_t *
get_free_next_mem(mem_t *mem)
{
	mem_t	*mem_next = get_next_mem(mem);

	if (mem_next != NULL && is_free_mem(mem_next))
		return mem_next;
	return NULL;
}

static void coalesce_free_mem(mem_t *mem);

static void
merge_free_mem(mem_t *mem1, mem_t *mem2)
{
	assert((char *)mem1->ptr + mem1->size == mem2->ptr);
	assert(mem1->size == mem2->size);

	mem1->size *= 2;
	destroy_mem(mem2);

	list_del(&mem1->list_free);
	coalesce_free_mem(mem1);
}

static BOOL
is_same_unit_idx(mem_t *mem1, mem_t *mem2)
{
	unsigned	idx1, idx2;

	idx1 = get_mem_unit_idx(mem1->size);
	idx2 = get_mem_unit_idx(mem2->size);

	return (idx1 == idx2);
}

static BOOL
is_left(mem_t *mem)
{
	if (((char *)mem->base - (char *)mem->ptr) / mem->size % 2)
		return FALSE;
	return TRUE;
}

static void
coalesce_free_mem(mem_t *mem)
{
	if (!list_empty(&mem->list_mem)) {
		if (is_left(mem)) {
			mem_t	*mem_next = get_free_next_mem(mem);
			if (mem_next != NULL && is_same_unit_idx(mem, mem_next)) {
				merge_free_mem(mem, mem_next);
				return;
			}
		}
		else {
			mem_t	*mem_prev = get_free_prev_mem(mem);
			if (mem_prev != NULL && is_same_unit_idx(mem_prev, mem)) {
				merge_free_mem(mem_prev, mem);
				return;
			}
		}
	}

	add_freemem(mem);
}

static mem_t *
create_mem_buddy(mem_t *mem, unsigned size_buddy, BOOL is_left)
{
	mem_t	*mem_buddy;

	mem_buddy = create_mem(mem->base, (char *)mem->ptr + mem->size - size_buddy, size_buddy);
	mem->size -= size_buddy;
	list_add(&mem_buddy->list_mem, &mem->list_mem);
	insert_hash(mem_buddy);

	return mem_buddy;
}

static void
split_mem_buddy(mem_t *mem, unsigned size_buddy)
{
	mem_t	*mem_buddy;

	mem_buddy = create_mem_buddy(mem, size_buddy, FALSE);
	coalesce_free_mem(mem_buddy);
}

static void
split_over_mem(mem_t *mem, unsigned size)
{
	while (TRUE) {
		unsigned	idx;

		idx = get_mem_unit_idx_le(mem->size - size);
		if (idx == N_MAX_UNITS)
			return;
		split_mem_buddy(mem, MIN_UNIT << idx);
	}
}

static void
split_under_mem(mem_t *mem)
{
	mem_t		*mem_buddy;
	unsigned	idx;

	idx = get_mem_unit_idx_lt(mem->size);
	if (idx == N_MAX_UNITS) {
		coalesce_free_mem(mem);
		return;
	}

	mem_buddy = create_mem_buddy(mem, mem->size - (MIN_UNIT << idx), TRUE);
	split_under_mem(mem_buddy);
	coalesce_free_mem(mem);
}

void *
mtbs_cudaMalloc(unsigned size)
{
	mem_t	*mem;
	unsigned	idx;

	if (size == 0)
		return NULL;
	if (size > memchunk) {
		error("too large memory size: %u", size);
		exit(1);
	}

	idx = get_mem_unit_idx_ge(size);

	pthread_spin_lock(&lock);
	mem = get_first_freemem(idx);
	if (mem == NULL) {
		pthread_spin_unlock(&lock);
		return NULL;
	}
	split_over_mem(mem, size);
	pthread_spin_unlock(&lock);

	return mem->ptr;
}

void
mtbs_cudaFree(void *ptr)
{
	unsigned	hashcode;
	struct list_head	*lp;

	if (ptr == NULL)
		return;

	hashcode = get_hashcode(ptr);

	pthread_spin_lock(&lock);
	list_for_each (lp, &hash_head[hashcode]) {
		mem_t	*mem = list_entry(lp, mem_t, list_hash);
		if (mem->ptr == ptr) {
			split_under_mem(mem);
			pthread_spin_unlock(&lock);
			return;
		}
	}
	pthread_spin_unlock(&lock);
	error("invalid cudaFree");
}

static void
check_mem_cleaned_up(void)
{
	unsigned	free_memsize = 0;
	unsigned        i;

	for (i = 0; i < N_MAX_UNITS - 1; i++) {
		struct list_head	*lp;
		unsigned	count = 0;

		list_for_each (lp, &free_head[i]) {
			mem_t	*mem;

			if (count == 1) {
				error("not coalesced free memory");
				exit(1);
			}
			mem = list_entry(lp, mem_t, list_free);
			free_memsize += mem->size;
			count++;
		}
	}
	if (free_memsize != cur_memsize) {
		error("free mem != total mem: %d", free_memsize);
		exit(1);
	}

}

#ifdef SHOW_MEM

static void
show_free_mems(void)
{
	struct list_head	*lp;
	unsigned	i;

	for (i = 0; i < N_MAX_UNITS; i++) {
		printf("[%08x] ", MIN_UNIT << i);
		list_for_each (lp, &free_head[i]) {
			mem_t *m = list_entry(lp, mem_t, list_free);
			printf("%p ", m);
		}
		printf("\n");
	}
}

#define N_LAYOUT_MEMS	256

static void
show_mem_layout(void)
{
	mem_t	*mems[N_LAYOUT_MEMS] = { NULL };
	unsigned	n_mems = 0;
	unsigned	i;

	for (i = 0; i < N_HASH; i++) {
		struct list_head	*lp;

		if (list_empty(&hash_head[i]))
			continue;
		list_for_each (lp, &hash_head[i]) {
			mem_t	*mem = list_entry(lp, mem_t, list_hash);
			unsigned	j;

			if (n_mems == N_LAYOUT_MEMS)
				goto show;

			for (j = 0; j < n_mems; j++) {
				if (mem->ptr < mems[j]->ptr) {
					memcpy(mems + j + 1, mems, (n_mems - j) * sizeof(mem_t *));
					mems[j] = mem;
					n_mems++;
					break;
				}
			}
			if (j == n_mems) {
				mems[n_mems] = mem;
				n_mems++;
			}
		}
	}
show:
	for (i = 0; i < n_mems; i++) {
		printf("[%p:%x:%p] ", mems[i], mems[i]->size, mems[i]->ptr);
	}
	printf("\n");
}

#endif

#ifdef TEST_MEM

static void
test_mem(void)
{
	void *ptr3 = mtbs_cudaMalloc(192);
	void *ptr2 = mtbs_cudaMalloc(128);
	void *ptr1 = mtbs_cudaMalloc(64);

	mtbs_cudaFree(ptr1);
	mtbs_cudaFree(ptr2);

	void *ptr4 = mtbs_cudaMalloc(1);

	mtbs_cudaFree(ptr3);

	void *ptr5 = mtbs_cudaMalloc(768);

	mtbs_cudaFree(ptr4);
	mtbs_cudaFree(ptr5);

	check_mem_cleaned_up();

	exit(0);
}

#endif

void
init_mem(void)
{
	int	i;

	for (i = 0; i < N_HASH; i++)
		INIT_LIST_HEAD(&hash_head[i]);
	for (i = 0; i < N_MAX_UNITS; i++)
		INIT_LIST_HEAD(&free_head[i]);

	while (cur_memsize < max_memsize) {
		void	*ptr;
		unsigned	size = max_memsize - cur_memsize;
		cudaError_t	err;

		if (size > memchunk)
			size = memchunk;
		else {
			unsigned	idx = get_mem_unit_idx_le(size);
			if (idx == N_MAX_UNITS)
				break;
			size = MIN_UNIT << idx;
		}
		err = cudaMalloc(&ptr, size);
		if (err != cudaSuccess) {
			if (memchunk > MIN_UNIT && cur_memsize == 0) {
				memchunk /= 2;
				continue;
			}
			break;
		}

		add_mem(ptr, size);
		cur_memsize += size;
	}

	if (cur_memsize == 0) {
		error("empty memory assigned");
		exit(1);
	}

	pthread_spin_init(&lock ,0);

	
#ifdef TEST_MEM
	test_mem();
#endif
}

void
fini_mem(void)
{
	unsigned        i;

	check_mem_cleaned_up();

	for (i = 0; i < N_HASH; i++) {
		struct list_head        *lp, *next;

		list_for_each_n (lp, &hash_head[i], next) {
			mem_t   *mem = list_entry(lp, mem_t, list_hash);

			if (!list_empty(&mem->list_mem)) {
				error("weird head mem exists");
				exit(1);
			}
			cudaFree(mem->ptr);
			destroy_mem(mem);
		}
	}
}
