#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define N 10000

int main(int argc, char** argv) {
    int rank, nbproc;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbproc);

    double h = 1.0/N;
    double PI = 0.0; // this will be our result;
    int R = N/nbproc;
    double u=0.0;

    for (int i = rank*R; i < (rank+1)*R; ++i) {
        double xi = (h*(i+0.5));
        u += 4 * h * 1 / (1+ (xi*xi) );
    }


    MPI_Reduce(&u, &PI, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("%.50f\n", PI);
    }

    MPI_Finalize();
    return 0;
}
