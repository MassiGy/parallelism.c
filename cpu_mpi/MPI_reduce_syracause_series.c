#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define N 10

int main(int argc, char** argv) {
    int rank, nbproc;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbproc);

    MPI_Status status;
    int packetsize = N/nbproc;
    int w[] = {3, 10, 5, 16, 8,4,2,1,4};
    int u[packetsize+1];
    int v = 1;
    int V= 1;

    // scatter our data
    MPI_Scatter(w, packetsize, MPI_INT, &(u[1]), packetsize, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank ==0) {
        MPI_Send(&(u[packetsize]), 1, MPI_INT, 1, 7, MPI_COMM_WORLD);
    }else {
        MPI_Recv(&(u[0]), 1, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(rank < nbproc-1)
            MPI_Send(&(u[packetsize]), 1, MPI_INT, rank+1, 7, MPI_COMM_WORLD);
    }

    for (int i = 0; i < packetsize; ++i) {
        if(u[i] %2 == 0){
            v *= ( u[i+1] == u[i] / 2 )  ? 1:0;
        }else {
            v *= ( u[i+1] == 3* u[i] +1 )  ? 1:0;
        }
    }


    MPI_Reduce(&v, &V, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);

    if(rank ==0) {
        printf("%d\n", V);
    }

    MPI_Finalize();
    return 0;
}
