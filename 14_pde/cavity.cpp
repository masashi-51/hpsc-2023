#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
using namespace std;
typedef vector<vector<float> > matrix;

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
  matrix u(ny, vector<float>(nx,0));
  matrix v(ny, vector<float>(nx,0));
  matrix b(ny, vector<float>(nx,0));
  matrix p(ny, vector<float>(nx,0));
  matrix un(ny, vector<float>(nx,0));
  matrix vn(ny, vector<float>(nx,0));
  matrix pn(ny, vector<float>(nx,0));
  vector<float> x(nx);
  vector<float> y(ny);
  for(int i=0; i<nx; i++)
    x[i] = i * dx;
  for(int i=0; i<ny; i++)
    y[i] = i * dy;


  for(int n=0; n<nt; n++){
    auto tic = chrono::steady_clock::now();
    for(int j=1; j<ny-1; j++){
      for(int i=1; i<nx-1; i++){
        b[j][i] = rho * (1 / dt *
         ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
         ((u[j][i+1] - u[j][i-1]) / (2 * dx))*((u[j][i+1] - u[j][i-1]) / (2 * dx))
          - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
         (v[j][i+1] - v[j][i-1]) / (2 * dx)) - 
         ((v[j+1][i] - v[j-1][i]) / (2 * dy))*((v[j+1][i] - v[j-1][i]) / (2 * dy)));
      }
    }
    for(int it=0;  it<nit; it++){
      for(int j=0; j<ny; j++){
        for(int i=0; i<nx; i++){
          pn[j][i] = p[j][i];
        }
      }
      for(int j=1; j<ny-1; j++){
        for(int i=1; i<nx-1; i++){
          p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                     dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                     b[j][i] * dx*dx * dy*dy) / (2 * (dx*dx + dy*dy));
        }
      }
      for(int j=1; j<ny-1; j++){
        p[j][nx-1] = p[j][nx-2];
        p[j][0] = p[j][1];
      }
      for(int i=1; i<nx-1; i++){
        p[0][i] = p[1][i];
        p[ny-1][i] = p[ny-2][i];
      }
    }
    for(int j=0; j<ny; j++){
      for(int i=0; i<nx; i++){
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }
    for(int j=1; j<ny-1; j++){
      for(int i=1; i<nx-1; i++){
        u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                           - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                           - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                           + nu * dt / dx*dx * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                           + nu * dt / dy*dy * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
        v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                           - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                           - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                           + nu * dt / dx*dx * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                           + nu * dt / dy*dy * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
      }
    }
    for(int j=1; j<ny-1; j++){
      u[j][0] = 0;
      u[j][nx-1] = 0;
      v[j][0] = 0;
      v[j][nx-1] = 0;
    }
    for(int i=1; i<nx-1; i++){
      u[0][i] = 0;
      u[nx-1][i] = 1;
      v[0][i] = 0;
      v[nx-1][i] = 0;
    }
    auto toc = chrono::steady_clock::now();
    double time = chrono::duration<double>(toc - tic).count();
    printf("step=%d: %lf s\n", n, time);
  }
  for(int j=0; j<ny; j++){
    for(int i=0; i<nx; i++){
      printf("%lf\n", b[j][i]);
    }
  }
}

