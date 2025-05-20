__kernel void bitonic_fast(__global TYPE* g_mem, int j, int stage) {
    int id    = get_global_id(0);
    int lid   = get_local_id(0);

    __local TYPE l_mem[LCL_SZ];
    l_mem[lid] = g_mem[id];

    barrier(CLK_LOCAL_MEM_FENCE);

    int i;
    int to_cmp;
    int direction = (id & stage) ? 1 : 0;
    for (i = j; i > 0; i >>= 1) {
        to_cmp = i ^ lid;

        if (lid < to_cmp) {
            if (direction == (l_mem[to_cmp] > l_mem[lid])) {
                TYPE tmp = l_mem[to_cmp];
                l_mem[to_cmp] = l_mem[lid];
                l_mem[lid] = tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    g_mem[id] = l_mem[lid];
}


__kernel void bitonic_slow(__global TYPE* g_mem, int j, int stage) {
    int to_cmp, direction;

    TYPE tmp;
    int id = get_global_id(0);
    to_cmp = j ^ id;
    direction = (id & stage) ? 1 : 0;

    if (id < to_cmp) {
        if (direction == (g_mem[to_cmp] > g_mem[id])) {
            tmp = g_mem[to_cmp];
            g_mem[to_cmp] = g_mem[id];
            g_mem[id] = tmp;
        }
    }
}
