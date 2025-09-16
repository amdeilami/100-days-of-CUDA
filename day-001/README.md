## Summary:

This kernel performs vector addition for two floating-point vectors of numbers.

## What I learned:

I read the **_PMPP_** till the end of chapter 2, and learned (and also reviewd!) plenty of good points, including but not limited to:

- Different design philosophies for CPU and GPU
- Where parallelsim can actually be effective, how to calculate the overall performance improvement
- Memory allocation, deallocation and data transfers in CUDA C
- Writing a kernel function and using reserved (or predefined) variables for distiguishing different threads
- Concept of grid, thread-block (or block), dimensions and thread indexing in CUDA C
- Calling kernel functions and configure the block size using `<<<` and `>>>` operators
