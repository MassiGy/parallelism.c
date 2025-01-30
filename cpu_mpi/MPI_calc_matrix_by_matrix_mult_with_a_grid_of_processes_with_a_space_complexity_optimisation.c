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
    int Alocal, Blocal, Clocals;

    if(rank == 0){
        C = (int*) malloc(N*N*sizeof(int));
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
    MPI_Scatter(A, 1, row, &Alocal, N, MPI_INT, 0, row_comm);

    // scatter our B matrix col by col ( from process 0 )
    MPI_Scatter(B, 1, col, &Blocal, N, MPI_INT, 0, column_comm);


    // setup our variables for the parallel computation
    int in_row_comm_next_rank = (in_row_comm_rank+1)%N;
    int in_col_comm_next_rank = (in_column_comm_rank+1)%N;

    int in_row_comm_prev_rank = (in_row_comm_rank-1+N)%N;
    int in_col_comm_prev_rank = (in_column_comm_rank-1+N)%N;


    // do the computation (in parallel)
    Clocals = 0;
    for (int i = 0; i < N; ++i) {

        for (int j = 0; j < N; ++j) {
            Clocals += Alocal * Blocal;
        }

        // in the row_comm exchange Ai's
        if(in_row_comm_rank == 0){
            MPI_Send(&Alocal, 1, MPI_INT, in_row_comm_next_rank, 99, row_comm);
            MPI_Recv(&Alocal, 1, MPI_INT, in_row_comm_prev_rank, 99, row_comm, &status);
        }else if(in_row_comm_rank != 0) {
            int temp;
            MPI_Recv(&temp, 1, MPI_INT, in_row_comm_prev_rank, 99, row_comm, &status);
            MPI_Send(&Alocal, 1, MPI_INT, in_row_comm_next_rank, 99, row_comm);

            memcpy(&Alocal, &temp,   sizeof(int));
        }

        // in the col_comm exchange Bj's
        if(in_column_comm_rank == 0){
            MPI_Send(&Blocal, 1, MPI_INT, in_col_comm_next_rank, 99, column_comm);
            MPI_Recv(&Blocal, 1, MPI_INT, in_col_comm_prev_rank, 99, column_comm, &status);

        } else if(in_column_comm_rank != 0) {
            int temp;
            MPI_Recv(&temp, 1, MPI_INT, in_col_comm_prev_rank, 99, column_comm, &status);
            MPI_Send(&Blocal, 1, MPI_INT, in_col_comm_next_rank, 99, column_comm);

            memcpy(&Blocal, &temp,   sizeof(int));
        }
    }



    // gather back the data
    MPI_Gather(&Clocals, 1, MPI_INT, C, 1, MPI_INT, 0, MPI_COMM_WORLD);



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


// print the coordinates for every process
// printf("in_world_comm_rank: %d\t    in_row_comm_rank: %d\t   in_column_comm_rank: %d    row_color: %d\t    column_color: %d\n", rank, in_row_comm_rank, in_column_comm_rank, row_color, column_color);
