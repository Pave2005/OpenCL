__kernel void bitonicSort (__global int* data, int block_size, int step)
{
    int id = get_global_id(0);

    int pairIdx = id ^ step;

    if (pairIdx > id)
    {
        bool is_ascending = ((id & block_size) == 0);

        if ((data[id] > data[pairIdx]) == is_ascending)
        {
            int temp = data[id];
            data[id] = data[pairIdx];
            data[pairIdx] = temp;
        }
    }
}
