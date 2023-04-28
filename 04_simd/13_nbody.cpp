#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  float tmp[N];
  for(int i=0; i<N; i++) {
/*
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
*/
    //mask(i番目は1、それ以外は0)
    __m256 ivec = _mm256_set1_ps(i);
    __m256 jvec = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
    __m256 mask = _mm256_cmp_ps(ivec, jvec, _CMP_EQ_OQ);
    __m256 one = _mm256_set1_ps(1);
    __m256 zero = _mm256_set1_ps(0);
    mask = _mm256_blendv_ps(zero, one, mask);
    //float rx = x[i] - x[j];
    //float ry = y[i] - y[j];
    __m256 xvec = _mm256_load_ps(x);
    __m256 yvec = _mm256_load_ps(y);
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);
    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);
    //float r = std::sqrt(rx * rx + ry * ry);
    xvec = _mm256_mul_ps(rxvec, rxvec);
    yvec = _mm256_mul_ps(ryvec, ryvec);
    __m256 rvec = _mm256_add_ps(xvec, yvec);
    rvec = _mm256_add_ps(mask, rvec);//i=jの時r=1となるようにmaskを足す
    rvec = _mm256_rsqrt_ps(rvec);
    __m256 rrvec = _mm256_mul_ps(rvec, rvec);
    rrvec = _mm256_mul_ps(rrvec, rvec);
    //fx[i] -= rx * m[j] / (r * r * r);
    //fy[i] -= ry * m[j] / (r * r * r);
    __m256 mvec = _mm256_load_ps(m);
    rxvec = _mm256_mul_ps(rxvec, mvec);
    ryvec = _mm256_mul_ps(ryvec, mvec);
    rxvec = _mm256_mul_ps(rxvec, rrvec);
    ryvec = _mm256_mul_ps(ryvec, rrvec);
    _mm256_store_ps(tmp, rxvec);
    fx[i] -= tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, ryvec);
    fy[i] -= tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
