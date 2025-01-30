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
     *
     * We are going to leverage the power of MPI_Communicators to split our processes into
     * rows and columns of processes.
     *
     * For instance, a process can be in row 1 and in column 2. We differentiate these using colors.
     *
     * Then, once the grid and the layout-ing is done, we will calculate the C matrix (output of A*B),
     * Each process will calculate one coefficient of the C matrix and then all of these will be gathered
     * to reconstitute C.
*/


#define N 4
int main(int argc, char** argv) {
    int rank, nbproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbproc);
    MPI_Status status;

    if(nbproc%N!=0){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int row_color = rank%N;
    int column_color = rank/N;
    int order_key = rank;

    // declare our row and column communicators
    MPI_Comm row_comm, column_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, order_key, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, column_color, order_key, &column_comm);

    int in_row_comm_rank, in_column_comm_rank;
    int row_comm_size, column_comm_size;

    MPI_Comm_rank(row_comm, &in_row_comm_rank);
    MPI_Comm_rank(column_comm, &in_column_comm_rank);

    MPI_Comm_size(row_comm, &row_comm_size);
    MPI_Comm_size(column_comm, &column_comm_size);


    // declare a row type (this will be used to scatter our A matrix)
    MPI_Datatype row;
    MPI_Type_contiguous (N, MPI_INT, &row);
    MPI_Type_commit(&row);

    // declare a column & col type ( these will be used to scatter our B matrix)
    MPI_Datatype column;
    MPI_Type_vector(N, 1, N, MPI_INT, &column);
    MPI_Type_commit(&column);

    MPI_Datatype col;
    MPI_Type_create_resized(column,0, sizeof(int), &col);
    MPI_Type_commit(&col);


    // declare our matricies
    int *A, *B, *C;
    int Ai[N], Bj[N], clocal;


    if(rank == 0) {
        C = (int*) malloc(N * N * sizeof(int));
    }

    if(in_row_comm_rank == 0){
        // init A
        A = (int*) malloc(N * N * sizeof(int));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i*N +j] = 1;
            }
        }
    }
    if(in_column_comm_rank == 0){
        // init B
        B = (int*) malloc(N * N * sizeof(int));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                B[i*N +j] = 1;
            }
        }
    }

    // scatter our matrix row by row (from process 0)
    MPI_Scatter(A, 1, row, Ai, 1, row, 0, row_comm);

    // scatter our B matrix col by col ( from process 0 )
    MPI_Scatter(B, 1, col, Bj, 1, row, 0, column_comm);


    // do the computation (in parallel)
    clocal = 0;
    for (int j = 0; j < N; ++j) {
        clocal+= Ai[j] * Bj[j];
    }


    // gather back the data
    MPI_Gather(&clocal, 1, MPI_INT, C, 1, MPI_INT, 0, MPI_COMM_WORLD);



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
