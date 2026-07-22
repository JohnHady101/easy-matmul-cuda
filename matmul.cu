#include <cstdlib>


__global__ void matmul(int *a, int *b, int *c, int N){
    // a * b = c
    // (N x N) * (N x N) = (N x N)


    // calculate the global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // as long as you are increasing in y coordinate you are reaching higher row indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // vice versa for columns

    // boundry check for the matrix
    if(row < N && col < N){
        int tmp = 0;
        // dot product of row of a and a col of b
        for(int i = 0; i < N; i++){
            tmp += a[row * N + i] * b[i * N + col];
        }
        // write the matrix c
        c[row * N + col] = tmp;

    }


}

// initializes a square matrix with random numbers between 0-100
void init_matrix(int *m, int N){
    for(int i = 0; i < N * N; i++){
        m[i] = rand() % 100;
    }
} 

int main(){
    // set the matrix dimensions (2^10 * 2^10)
    int N = 1 << 10; // the one is shifted by 10 positions in the binary representation
    size_t bytes = N * N * sizeof(int); /// sizeof(int) = 4 bytes

    // allocating memory for matrices
    int *a, *b, *c;
    // cudamallocmanaged simply moves matrices from host to gpu memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // initialize the matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // set the CTA (Cooperative Thread Array) and Grid dimensions
    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    // setup the kernel launch parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // launch the kernel
    matmul<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();
    



}