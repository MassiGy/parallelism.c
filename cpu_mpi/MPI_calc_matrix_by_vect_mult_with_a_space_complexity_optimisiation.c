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
    MPI_Status status;

    /*
     * ./MPI_calc_matrix_by_vect_mult.c:
     * The idea behind this program is to make matrix by vector multiplication
     * parallel. Our matrix will be named A, our multiplication vector will be
     * named b, and the result vector will be named c.
     *
     * %:
     * in this program, the idea is still the same, but we won't broadcast our
     * b vector, this will be a more optimized version since we won't take nbproc * dim(b)
     * just for b. ( it is a space complexity optimisation ).
     *
     * Warning: if b is small, copying it maybe better in terms of performance (time complexity)
     * since all the MPI_Send and MPI_Recv calls that we will use to not copy b do have
     * performance overhead.
    */

    /*
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
    int *A, *c, *b;
    int Ai[workload_size*N];    // will store a row of our matrix
    int clocal[workload_size];
    int blocal[workload_size];



    if(rank == 0) {
        printf("workload_size=%d\n", workload_size);

        A = (int*) malloc(N * N * sizeof(int));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i*N+j] = 1;
            }
        }

        b = (int *) malloc(N * sizeof(int));
        for (int i = 0; i < N; ++i) {
            b[i] = 1;
        }

        c = (int*) malloc(N * sizeof(int));
        memset(c, 0, N * sizeof(int));
    }


    // broadcast our b vector (from process 0)
    // MPI_Bcast(b, N, MPI_INT, 0, MPI_COMM_WORLD);     // this is the copy operation that we are willing to avoid
    // we will rather scatter it
    memset(blocal, 0, sizeof(int) * workload_size);
    MPI_Scatter(b, workload_size, MPI_INT, blocal, workload_size, MPI_INT, 0, MPI_COMM_WORLD);

    // scatter our matrix row by row (from process 0)
    MPI_Scatter(A, workload_size, row, Ai, workload_size, row, 0, MPI_COMM_WORLD);


    // do the computation (in parallel)
    memset(clocal, 0, sizeof(int) * workload_size);


    int index;
    int next_rank = (rank+1)%nbproc;
    int prev_rank = (rank-1+nbproc)%nbproc;

    for (int i = 0; i < nbproc; ++i) {
        index = (rank-i+nbproc)%nbproc;

        for (int j = 0; j < workload_size; ++j) {
            for (int k = 0; k < workload_size; ++k) {
                clocal[j]+=Ai[ j*nbproc + index*workload_size + k] * blocal[k];
            }
        }

        if(rank==0){
            MPI_Send(blocal, workload_size, MPI_INT, next_rank, 99, MPI_COMM_WORLD);
            MPI_Recv(blocal, workload_size, MPI_INT, prev_rank, 99, MPI_COMM_WORLD, &status);
        }else{
            int temp[workload_size];

            MPI_Recv(temp, workload_size, MPI_INT, prev_rank, 99, MPI_COMM_WORLD, &status);
            MPI_Send(blocal, workload_size, MPI_INT, next_rank, 99, MPI_COMM_WORLD);

            memcpy(blocal, temp, workload_size * sizeof(int));
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
        free(b); b=NULL;
        free(c); c=NULL;
    }

    MPI_Finalize();
    return 0;
}
