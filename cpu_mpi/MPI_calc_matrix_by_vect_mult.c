#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define N 4

int main(int argc, char** argv) {
    int rank, nbproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbproc);

    /*
     * The idea behind this program is to make matrix by vector multiplication
     * parallel. Our matrix will be named A, our multiplication vector will be
     * named b, and the result vector will be named c.
     *
     * Be sure that N%nbproc == 0, otherwise the matrix layout can not be uniformly
     * scattered.
    */
    if(N%nbproc!=0){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*
     * To simplify things up, we will assume that we have exactly N processes
     * such us each process will take charge of the multiplication of one line
     * of our matrix and b. (since our matrix has N lines )

        if(nbproc != N) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // workload_size will be equal to 1;
   */


    int workload_size= N / nbproc;

    // declare a utility type
    MPI_Datatype row;
    MPI_Type_contiguous (N, MPI_INT, &row);
    MPI_Type_commit(&row);


    // declare our matrix and our b & c vector
    int *A, *c;
    int b[N], Ai[workload_size*N];    // will store a row of our matrix
    int clocal[workload_size];



    if(rank == 0) {
        A = (int*) malloc(N * N * sizeof(int));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i*N+j] = 1;
            }
        }
        for (int i = 0; i < N; ++i) {
            b[i] = 1;
        }
        c = (int*) malloc(N * sizeof(int));
        memset(c, 0, N * sizeof(int));
    }


    // broadcast our b vector (from process 0)
    MPI_Bcast(b, N, MPI_INT, 0, MPI_COMM_WORLD);
    // scatter our matrix row by row (from process 0)
    MPI_Scatter(A, workload_size, row, Ai, workload_size, row, 0, MPI_COMM_WORLD);


    // do the computation (in parallel)
    memset(clocal, 0, sizeof(int) * workload_size);
    for (int k = 0; k < workload_size; ++k) {
        for (int i = 0; i < N; ++i) {
            clocal[k]+=Ai[i+k*workload_size]*b[k];
        }
    }

    // gather back the data
    MPI_Gather(clocal, workload_size, MPI_INT, c, workload_size, MPI_INT, 0, MPI_COMM_WORLD);




    if(rank == 0) {
        // print A * b;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%d ", A[i*N+j]);
            }
            if(i == N/2){
                printf("* %d ", b[i]);
                printf("= %d\n", c[i]);
            }else {
                printf("  %d ", b[i]);
                printf("  %d\n", c[i]);
            }
        }

        free(A); A=NULL;
        free(c); c=NULL;
    }

    MPI_Finalize();
    return 0;
}
