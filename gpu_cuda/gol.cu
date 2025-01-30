#include "gpu_bitmap.h"
#include <ctime>
#include <unistd.h>

#define WIDTH 1920
#define HEIGHT 1080
#define DIM 16

#define K 3
#define M 200
#define G 28


__global__ void color(int *ts, uchar4 *buf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < WIDTH && y < HEIGHT) {
        int offset = y * WIDTH + x;

        float t = ts[offset];
        float r, g, b;

        if (t <= 0) {
            r = 255;
            g = 255;
            b = 255;
        } else if ( t> 0 && t < M) {
            r = 255;
            g = 0;
            b = 0;
        } else {
            r = 0;
            g = 0;
            b = 0;
        }

        buf[offset].x = r;
        buf[offset].y = g;
        buf[offset].z = b;
        buf[offset].w = 255;
    }
}


/*
__global__ void simulate_next_state(int *t_current, int*t_next) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < WIDTH && y < HEIGHT) {
        int offset = y * WIDTH + x;

        int top = y == HEIGHT - 1 ? offset : offset + WIDTH;
        int bottom = y == 0 ? offset : offset - WIDTH;
        int left = x == 0 ? offset : offset - 1;
        int right = x == WIDTH - 1 ? offset : offset + 1;

        if(t_current[offset] >= M) {
            t_next[offset] = 0;

        } else if(t_current[offset] <= 0){
            int nb_sick_neighbors = 0;
            if(t_current[top] == M) nb_sick_neighbors++;
            if(t_current[bottom] == M) nb_sick_neighbors++;
            if(t_current[left] == M) nb_sick_neighbors++;
            if(t_current[right] == M) nb_sick_neighbors++;

            t_next[offset] = nb_sick_neighbors / K;

        } else {
            int neighbors_states = 0;
            neighbors_states+=t_current[top];
            neighbors_states+=t_current[bottom];
            neighbors_states+=t_current[left];
            neighbors_states+=t_current[right];

            t_next[offset] = neighbors_states/4 + G;
        }
    }
}
*/


//Kernell qui implémente un automate cellulaire
__global__ void simulate_next_state(int *t_current, int *t_next) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < WIDTH && y < HEIGHT) {
        int offset = y * WIDTH + x;
        int current_cell_state = t_current[offset];

        int num_infected_neighbors = 0;
        int sum_infected_neighbors_states = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int neighbor_x = x + i;
                int neighbor_y = y + j;

                // Vérifier si le voisin est dans les limites de la grille
                if (neighbor_x >= 0 && neighbor_x < WIDTH && neighbor_y >= 0 && neighbor_y < HEIGHT) {
                    int neighbor_offset = neighbor_y * WIDTH + neighbor_x;
                    int neighbor_state = t_current[neighbor_offset];

                    if (neighbor_state != M && neighbor_state != 0) {
                        num_infected_neighbors++;
                        sum_infected_neighbors_states+=neighbor_state;
                    }
                }
            }
        }

        // Partie vérifie si la cellule est malade, saine ou infectée
        int next_cell_state;
        if (current_cell_state == M) {
            next_cell_state = 0;
        }else if (current_cell_state == 0) {
            if (num_infected_neighbors > 0) {
                next_cell_state = num_infected_neighbors/K;
            }
        }else {
            float average_state = (current_cell_state + sum_infected_neighbors_states) / num_infected_neighbors;
            next_cell_state = average_state + G;
        }

        if (next_cell_state > M)
            next_cell_state = M;

        t_next[offset] = next_cell_state;
    }
}



struct Data {
    int *t1;
    int *t2;
    dim3 blocks;
    dim3 threads;
};

void render_callback(uchar4 *buf, Data *data, int ticks) {
    simulate_next_state<<<data->blocks, data->threads>>>(data->t1, data->t2);
    usleep(1000);
    simulate_next_state<<<data->blocks, data->threads>>>(data->t2, data->t1);

    color<<<data->blocks, data->threads>>>(data->t1, buf);
}

void clean_callback(Data *data) {
    HANDLE_CUDA_ERR(cudaFree(data->t1));
    HANDLE_CUDA_ERR(cudaFree(data->t2));
}

int main() {
    Data data;
    GPUBitmap bitmap(WIDTH, HEIGHT, &data, "GoL");

    int *t_initial = (int *)calloc(WIDTH * HEIGHT, sizeof(int));

    //srand(time(NULL));
    for(int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; x++) {
            //t_initial[y * WIDTH + x] = rand() % (M + 1 - 0) + 0;    // random number between 0 and M.

             if((rand()*1.0)/RAND_MAX < 0.009)
                t_initial[y * WIDTH + x] = rand() % (M + 1 - 0) + 0;    // random number between 0 and M.
             else
                t_initial[y * WIDTH + x] = M;

        }
    }

    data.blocks = dim3((WIDTH + DIM - 1) / DIM, (HEIGHT + DIM - 1) / DIM);
    data.threads = dim3(DIM, DIM);

    size_t size = WIDTH * HEIGHT * sizeof(int);
    HANDLE_CUDA_ERR(cudaMalloc(&data.t1, size));
    HANDLE_CUDA_ERR(cudaMalloc(&data.t2, size));
    HANDLE_CUDA_ERR(cudaMemcpy(data.t1, t_initial, size, cudaMemcpyHostToDevice));

    bitmap.animate((void (*)(uchar4*, void*, int))render_callback, (void (*)(void*))clean_callback);
    return 0;
}

