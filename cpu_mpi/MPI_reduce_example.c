#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define N 20

int main(int argc, char** argv) {
    int rank, nbproc, packetsize;



    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbproc);

    packetsize = N/nbproc;

    int*w;
    int u[packetsize];
    int v;
    int result;

    if(rank == 0) {
        // init our data
        w = (int *) malloc(N * sizeof(int));
        for (int i = 0; i < N; ++i) {
            w[i] = 0;
        }
    }

    // scatter our data
    MPI_Scatter(w, packetsize, MPI_INT, u, packetsize, MPI_INT, 0, MPI_COMM_WORLD);

    // parallel computing
    for (int i = 0; i < packetsize; ++i) {
        u[i]++;
    }

    // gather the data
    MPI_Gather(u, packetsize, MPI_INT, w, packetsize, MPI_INT, 0, MPI_COMM_WORLD);


    // rescatter the data using diffrent packetsize (packetsize=1) since we need to reduce afterward
    MPI_Scatter(w, 1, MPI_INT, &v, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // reduce the data, this will go to the diffrent processes and retreive the blocks of 1*sizeof(int) and sum them.
    // that is why we need a second scatter. This is not efficient, it is just to showcase how reduce works.

    // TL;DR: MPI_Reduce needs to go and fetch all the data that it will reduce from the processes. So the processes need
    // to be the ones hosting the chunks to reduce.
    MPI_Reduce(&v, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);



    if(rank == 0) {
        free(w); w = NULL;
        printf("result: %d\n", result);
    }
    MPI_Finalize();
    return 0;
}

