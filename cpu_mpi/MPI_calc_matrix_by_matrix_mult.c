#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
     * @author: Massiles GHERNAOUT.
     * @login: gm213204
     *
     * @description:
     *
     * The idea behind this program it to compute in a parallel way a matrix by matrix
     * multiplication.
     *
     * i.e: calc MatA * MatB in a parallel way.
     *
     * For the sake of simplicity we will only use square matricies.
     * In this example, we will have A and B as 4by4 matricies.
*/

#define N 4
int main(int argc, char** argv) {
    int rank, nbproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbproc);
    MPI_Status status;

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
     * of our matrix A by one column of our matrix B.
   */
    if(nbproc != N) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // workload_size will be equal to 1;
    int workload_size= N / nbproc;

    // declare a row type (this will be used to scatter our A matrix)
    MPI_Datatype row;
    MPI_Type_contiguous (N, MPI_INT, &row);
    MPI_Type_commit(&row);

    // declare a column & col type ( these will be used to scatter our B matrix)
    /*
     * column is a vector with a block2block offset of N * MPI_INT. This type
     * will allow us to jump through different sectors of our B matrix memory layout.
     * ( do not forget that our B matrix is just a vector after all ).
     *
     * Then col is a resized version of column. Col is basically the same as column,
     * but this time the next_read_offset is stripped down from N*MPI_INT to 1*MPI_INT.
     *
     * This is crucial to properly scatter the columns to our processes. Since in memory
     * B is laid out like this:
     * -------------------------------------------------------------------------
     * | b00 b01 b02 b03 | b10 b11 b12 b13 | b20 b21 b22 b23 | b30 b31 b32 b33 |
     * -------------------------------------------------------------------------
     *
     * The first column is [b00 b10 b20 b30]
     * And the second column should be fetched with the same block2block offset
     * but with a next_read_offset of 1*MPI_INT so as it reads starting from b01.
     *
     * That is where the combination of column and col will come in to play.
     * column gives us the right block2block offset, and col gives us the right
     * next_read_offset.
     *
    */
    MPI_Datatype column;
    MPI_Type_vector(N, 1, N, MPI_INT, &column);
    MPI_Type_commit(&column);

    MPI_Datatype col;
    MPI_Type_create_resized(column,0, sizeof(int), &col);
    MPI_Type_commit(&col);


    // declare our matricies
    int *A, *B, *C;
    int Ai[workload_size*N];
    int Bj[workload_size*N];
    int Ci[workload_size*N];



    if(rank == 0) {
        printf("workload_size=%d\n", workload_size);        //workload_size=1

        A = (int*) malloc(N * N * sizeof(int));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i*N+j] = 1;
            }
        }


        B = (int*) malloc(N * N * sizeof(int));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                B[i*N+j] = 1;
            }
        }

        C = (int*) malloc(N * N * sizeof(int));
        memset(C, 0, N*N*sizeof(int));
    }

    // scatter our matrix row by row (from process 0)
    MPI_Scatter(A, workload_size, row, Ai, workload_size, row, 0, MPI_COMM_WORLD);

    // scatter our B matrix col by col ( from process 0 )
    MPI_Scatter(B, workload_size, col, Bj, 1, row, 0, MPI_COMM_WORLD);


    // setup our variables for the parallel computation
    int index;
    int next_rank = (rank+1)%nbproc;
    int prev_rank = (rank-1+nbproc)%nbproc;
    memset(Ci, 0, workload_size* N* sizeof(int));

    // do the computation (in parallel)
    for (int i = 0; i < nbproc; ++i) {
        index = (rank-i+nbproc)%nbproc;

        for (int j = 0; j < N; ++j) {
            Ci[index]+= Ai[j] * Bj[j];
        }

        if(rank==0){
            MPI_Send(Bj, workload_size, col, next_rank, 99, MPI_COMM_WORLD);
            MPI_Recv(Bj, 1, row, prev_rank, 99, MPI_COMM_WORLD, &status);
        }else{
            int temp[workload_size];

            MPI_Recv(temp, 1, row, prev_rank, 99, MPI_COMM_WORLD, &status);
            MPI_Send(Bj, workload_size, col, next_rank, 99, MPI_COMM_WORLD);

            memcpy(Bj, temp, workload_size * sizeof(int));
        }
    }

    // gather back the data
    MPI_Gather(Ci, 1, row, C, 1, row, 0, MPI_COMM_WORLD);


    if(rank == 0) {
        // print A * b;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%d ", A[i*N+j]);
            }
            if(i == N/2){
                printf("* ");
            }else {
                printf("  ");
            }
            for (int j = 0; j < N; ++j) {
                printf("%d ", B[i*N+j]);
            }
            if(i == N/2){
                printf("= ");
            }else {
                printf("  ");
            }
            for (int j = 0; j < N; ++j) {
                printf(" %d ", C[i*N+j]);
            }
            printf("\n");
        }

        free(A); A=NULL;
        free(B); B=NULL;
        free(C); C=NULL;
    }

    MPI_Finalize();
    return 0;
}
