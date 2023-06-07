#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
using namespace std;
typedef vector<vector<float> > matrix;

__global__ void cul_b(float *b, float *u, float *v, 
                      float rho, float dt, int nx, int ny, float dx, float dy){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= (ny-2)*(nx-2)) return;
  int j = tid/(nx-2)+1;
  int i = tid%(nx-2)+1;
  b[j*nx+i] = rho * (1 / dt *
             ((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx) + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)) -
             ((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx))*((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx))
             - 2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2 * dy) *
             (v[j*nx+i+1] - v[j*nx+i-1]) / (2 * dx)) - 
             ((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy))*((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)));
}

__global__ void cul_pn(float *p, float *pn, int nx, int ny){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= ny*nx) return;
  int j = tid/nx;
  int i = tid%nx;
  pn[j*nx+i] = p[j*nx+i];
}

__global__ void cul_p(float *b, float *p, float *pn, 
                       int nx, int ny, float dx, float dy){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= (ny-2)*(nx-2)) return;
  int j = tid/(nx-2)+1;
  int i = tid%(nx-2)+1;
  p[j*nx+i] = (dy*dy * (pn[j*nx+i+1] + pn[j*nx+i-1]) +
                     dx*dx * (pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) -
                     b[j*nx+i] * dx*dx * dy*dy) / (2 * (dx*dx + dy*dy));
}

__global__ void bound_p(float *p, int nx, int ny){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nx-2){
    int i = tid + 1;
    p[0*nx+i] = p[1*nx+i];
    p[(ny-1)*nx+i] = p[(ny-2)*nx+i];
  }
  if(tid < ny-2){
    int j = tid + 1;
    p[j*nx+nx-1] = p[j*nx+nx-2];
    p[j*nx+0] = p[j*nx+1];
  }
}

__global__ void cul_unvn(float *u, float *v, float *un, float *vn, 
                         int nx, int ny){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= ny*nx) return;
  int j = tid/nx;
  int i = tid%nx;
  un[j*nx+i] = u[j*nx+i];
  vn[j*nx+i] = v[j*nx+i];
}

__global__ void cul_uv(float *u, float *v, float *un, float *vn, float *p, 
                       float nu, float rho, float dt, int nx, int ny, float dx, float dy){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= (ny-2)*(nx-2)) return;
  int j = tid/(nx-2)+1;
  int i = tid%(nx-2)+1;
  u[j*nx+i] = un[j*nx+i] - un[j*nx+i] * dt / dx * (un[j*nx+i] - un[j*nx+i-1])
                           - un[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j-1)*nx+i])
                           - dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])
                           + nu * dt / dx*dx * (un[j*nx+i+1] - 2 * un[j*nx+i] + un[j*nx+i-1])
                           + nu * dt / dy*dy * (un[(j+1)*nx+i] - 2 * un[j*nx+i] + un[(j-1)*nx+i]);
  v[j*nx+i] = vn[j*nx+i] - vn[j*nx+i] * dt / dx * (vn[j*nx+i] - vn[j*nx+i-1])
                           - vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j-1)*nx+i])
                           - dt / (2 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])
                           + nu * dt / dx*dx * (vn[j*nx+i+1] - 2 * vn[j*nx+i] + vn[j*nx+i-1])
                           + nu * dt / dy*dy * (vn[(j+1)*nx+i] - 2 * vn[j*nx+i] + vn[(j-1)*nx+i]);
}

__global__ void bound_uv(float *u, float *v, int nx, int ny){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < nx-2){
    int i = tid + 1;
    u[i] = 0;
    u[(nx-1)*nx+i] = 1;
    v[i] = 0;
    v[(nx-1)*nx+i] = 0;
  }
  if(tid < ny-2){
    int j = tid + 1;
    u[j*nx+0] = 0;
    u[j*nx+nx-1] = 0;
    v[j*nx+0] = 0;
    v[j*nx+nx-1] = 0;
  }
}

int main(){
  int nx = 41;
  int ny =41;
  int nt = 500;
  int nit = 50;
  float dx = 2. / (nx - 1);
  float dy = 2. / (ny - 1);
  float dt = .01;
  float rho = 1;
  float nu = 0.02;
  vector<float> x(nx);
  vector<float> y(ny);
  for(int i=0; i<nx; i++)
    x[i] = i * dx;
  for(int i=0; i<ny; i++)
    y[i] = i * dy;

  float *u, *v, *b, *p, *un, *vn, *pn;
  cudaMallocManaged(&u, ny*nx*sizeof(float));
  cudaMallocManaged(&v, ny*nx*sizeof(float));
  cudaMallocManaged(&b, ny*nx*sizeof(float));
  cudaMallocManaged(&p, ny*nx*sizeof(float));
  cudaMallocManaged(&un, ny*nx*sizeof(float));
  cudaMallocManaged(&vn, ny*nx*sizeof(float));
  cudaMallocManaged(&pn, ny*nx*sizeof(float));
  for(int i=0; i<ny*nx; i++){
    u[i] = 0;
    v[i] = 0;
    b[i] = 0;
    p[i] = 0;
    un[i] = 0;
    vn[i] = 0;
    pn[i] = 0;
  }
  int N = 6400;
  int M = 128;
  for(int n=0; n<nt; n++){
    auto tic = chrono::steady_clock::now();
    cul_b<<<N/M,M>>>(b, u, v, rho, dt, nx, ny, dx, dy);
    cudaDeviceSynchronize();
    for(int it=0;  it<nit; it++){
      cul_pn<<<N/M,M>>>(p, pn, nx, ny);
      cudaDeviceSynchronize();
      cul_p<<<N/M,M>>>(b, p, pn, nx, ny, dx, dy);
      cudaDeviceSynchronize();
      bound_p<<<N/M,M>>>(p, nx, ny);
      cudaDeviceSynchronize();
    }
    cul_unvn<<<N/M,M>>>(u, v, un, vn, nx, ny);
    cudaDeviceSynchronize();
    cul_uv<<<N/M,M>>>(u, v, un, vn, p, nu, rho, dt, nx, ny, dx, dy);
    cudaDeviceSynchronize();
    bound_uv<<<N/M,M>>>(u, v,  nx, ny);
    cudaDeviceSynchronize();
    auto toc = chrono::steady_clock::now();
    double time = chrono::duration<double>(toc - tic).count();
    printf("step=%d: %lf s\n", n, time);
  }
  for(int i=0; i<ny*nx; i++) printf("%lf\n", b[i]);
  cudaFree(u);
  cudaFree(v);
  cudaFree(b);
  cudaFree(p);
  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);
  return 0;
}

