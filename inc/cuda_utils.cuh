#ifndef COVID_MODEL_CUDA_UTILS_CUH
#define COVID_MODEL_CUDA_UTILS_CUH


#include "base_header.hpp"


/* Return the ith younger bit of a number (starting with 1 up to 32) */
__device__ int get_required_bit_cuda(int window, size_t position) {
    return (window >> (WINDOW_SIZE - position)) & 1;
}


/* Return the ith (taking into consideration offset (col - row)) symmetric bit corresponding to ith row */
__device__ int get_col_required_bit_cuda(int window, size_t col, size_t row) {
    size_t bit_idx = (col - row) % WINDOW_SIZE + 1; // we add 1 at the end as we start bit indexing from 1
    return get_required_bit_cuda(window, bit_idx);
}


/* Return a sequence of uniformly distributed random values (0.0, 1.0] */
__device__ double* rand_sequence(int *thread_range) {
    // unique thread identificator
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize random number generator for a specific thread
    curandState localState;
    //curand_init(SEED + idx, idx, 0, &localState);

    // generate random sequence
    double rands[THREAD_RANGE];
    for (int i = 0; i < *thread_range; i++) {
        rands[i] = curand_uniform_double(&localState);
    }
    return rands;
}


/* Return a random number in range (0.0, 1.0]. Extremely inefficient function! */
__device__ double rand_float(int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize random number generator for a specific thread
    curandState localState;
    curand_init(seed + idx, idx, 0, &localState);

    // generate random sequence
    return  curand_uniform_double(&localState);
}


/*
 * Return the person's next state while being in state S.
 * This function represents a transition state table and is used to
 * determine the next state via bool functions. 
 */
__device__ char s_next_state(struct gpu_vars_t* gpu_vars, char* s_trans_states, double rand_f, double beta, int person) {
    int tr_state = 0;

    bool a = rand_f < beta;
    bool b = rand_f >= beta && rand_f <= beta + gpu_vars->omega && gpu_vars->day >= gpu_vars->start_vaccine;
    bool c = (gpu_vars->who_vaccinated && person < gpu_vars->max_contacts&& gpu_vars->day >= gpu_vars->start_vaccine) || \
        (!gpu_vars->who_vaccinated && gpu_vars->max_contacts < person&& person < (gpu_vars->max_contacts * 2) && gpu_vars->day >= gpu_vars->start_vaccine);

    tr_state += c;
    tr_state = (tr_state << 1) + b;
    tr_state = (tr_state << 1) + a;

    return s_trans_states[(int)ceilf(log2f(tr_state + 1))]; // add 1 to balance out binary numbers
}

#endif //COVID_MODEL_CUDA_UTILS_CUH