#ifndef COVID_MODEL_CUDA_KERNELS_CUH
#define COVID_MODEL_CUDA_KERNELS_CUH


#include "cuda_utils.cuh"


/* Encode bits in parallel using CUDA API */
__global__ void encode_bits(int* adj_matrix, int* window_indices, double* prob_connect, int* people_num, int* thread_range, int* seed) {
    // get required window index for certain thread
    int window_idx = window_indices[blockIdx.x] + threadIdx.x;
    
    // check whether we are not out of range
    int last_window_idx = window_indices[*people_num - 1];
    if (window_idx > last_window_idx)
        return;
    
    // initialize random generator with some local state
    double* rands = rand_sequence(thread_range);
    int rand_counter = 0;

    // determine thread range boundaries
    int start = blockIdx.x + threadIdx.x * *thread_range;
    int end = start + *thread_range;
    
    // encode bits
    for (size_t m = start; m < end; m += WINDOW_SIZE) {
        if (m > last_window_idx)
            break;
        // let the spare bits of the last window be encoded
        // as we don't care about what values these bits have
        int encoded_bits = 0;
        for (size_t entry = 1; entry < WINDOW_SIZE; entry++) {
            encoded_bits = (encoded_bits << 1) + (rands[rand_counter++] < prob_connect[blockIdx.x]);
        }
        adj_matrix[window_idx] = encoded_bits;
        window_idx += 1;
    }
}


__global__ void s_state_simulation(struct gpu_vars_t* gpu_vars, char* s_trans_states, int* s_state_people, char* temp_states, char* people_states,
    int* window_indices, int* adj_matrix) {
    // check if we are not out of range
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > gpu_vars->last_person)
        return;

    // generate random float (consider generating a sequence)
    double rand_f = rand_float(gpu_vars->seed);

    // check if healthy man is in contact with the infected ones
    int person = s_state_people[idx];
    int inf = 0, n_inf = 0;
    bool is_infected;
    // USE FORMULA
    for (int upper_neighbour = 0; upper_neighbour < person; upper_neighbour++) {
        // determine window index and required bit of a neighbour
        int window_idx = window_indices[upper_neighbour] + (person - upper_neighbour) / WINDOW_SIZE;
        int connection = get_col_required_bit_cuda(adj_matrix[window_idx], person, upper_neighbour);

        is_infected = (connection && (people_states[upper_neighbour] == E_STATE || people_states[upper_neighbour] == I_STATE));

        inf += is_infected;
        n_inf += (!is_infected) && connection;
    }

    int window_idx = window_indices[person];
    for (int lower_neighbour = person; lower_neighbour < gpu_vars->people_num; lower_neighbour += WINDOW_SIZE) {
        // check if those encoded bits are on the edge of a matrix,
        // thus define the endpoint of a window
        bool is_window_edge = lower_neighbour + WINDOW_SIZE >= gpu_vars->people_num;
        int window_end = WINDOW_SIZE * (!is_window_edge) + (int)(gpu_vars->people_num - lower_neighbour) * is_window_edge;
        // decode bits
        for (int bit = 1; bit <= window_end; bit++) {
            int connection = get_required_bit_cuda(adj_matrix[window_idx], bit);
            is_infected = (connection && (people_states[lower_neighbour] == E_STATE || people_states[lower_neighbour] == I_STATE));

            inf += is_infected;
            n_inf += (!is_infected) && connection;
        }
        window_idx++;
    }

    // examine the beta coefficicent
    double beta = (inf + n_inf != 0) * (gpu_vars->beta * ((double)inf / (double)(n_inf + inf)));
    // switch state
    temp_states[person] = s_next_state(gpu_vars, s_trans_states, rand_f, beta, person);
}


#endif // COVID_MODEL_CUDA_KERNELS_CUH
