#include <cstdio>
#include <omp.h>

int main() {
#pragma omp parallel for schedule(dynamic)
  for(int i=0; i<8; i++) {
    printf("%d: %d\n",omp_get_thread_num(),i);
  }
}
