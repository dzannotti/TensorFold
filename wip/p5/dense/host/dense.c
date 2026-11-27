// usage: dense LIBHIP HSACO KERNEL ROWS COLS M N K [threads]: random bf16 x [M][K], w [N][K]; times KERNEL(x, w, out, m, n, k),
// prints TFLOPS, power and an FNV hash of out (fp32 [M][N]).
#include <dlfcn.h>
#include <glob.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef int E;
static long power_uw(void) {
  glob_t g;
  long uw = 0;
  if (glob("/sys/class/drm/card*/device/hwmon/hwmon*/power1_average", 0, 0, &g) == 0 && g.gl_pathc) {
    FILE *p = fopen(g.gl_pathv[0], "r");
    if (p) { if (fscanf(p, "%ld", &uw) != 1) uw = 0; fclose(p); }
  }
  globfree(&g);
  return uw;
}
int main(int argc, char **argv) {
  if (argc < 9) return 2;
  void *lib = dlopen(argv[1], RTLD_NOW);
  if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
  E (*Init)(unsigned) = dlsym(lib, "hipInit");
  E (*Malloc)(void **, size_t) = dlsym(lib, "hipMalloc");
  E (*Memcpy)(void *, const void *, size_t, int) = dlsym(lib, "hipMemcpy");
  E (*Sync)(void) = dlsym(lib, "hipDeviceSynchronize");
  E (*Load)(void **, const void *) = dlsym(lib, "hipModuleLoadData");
  E (*Get)(void **, void *, const char *) = dlsym(lib, "hipModuleGetFunction");
  E (*Launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) =
      dlsym(lib, "hipModuleLaunchKernel");
  E (*EvC)(void **, unsigned) = dlsym(lib, "hipEventCreateWithFlags");
  E (*EvR)(void *, void *) = dlsym(lib, "hipEventRecord");
  E (*EvS)(void *) = dlsym(lib, "hipEventSynchronize");
  E (*EvT)(float *, void *, void *) = dlsym(lib, "hipEventElapsedTime");
  FILE *f = fopen(argv[2], "rb");
  if (!f) return 1;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *img = malloc(n);
  if (fread(img, 1, n, f) != (size_t)n) return 1;
  fclose(f);
  int br = atoi(argv[4]), bc = atoi(argv[5]), m = atoi(argv[6]), nn = atoi(argv[7]), k = atoi(argv[8]);
  int thr = argc > 9 ? atoi(argv[9]) : 256;
  Init(0);
  void *mod, *fn, *dx, *dw, *dout, *e0, *e1;
  if (Load(&mod, img) || Get(&fn, mod, argv[3])) { fprintf(stderr, "load failed\n"); return 1; }
  size_t xe = (size_t)m * k, we = (size_t)nn * k, ob = (size_t)m * nn * 4;
  uint16_t *hx = malloc(xe * 2), *hw = malloc(we * 2);
  uint32_t s = 12345;
  for (size_t i = 0; i < xe + we; i++) {
    s = s * 1664525u + 1013904223u;
    uint16_t v = (uint16_t)(0x3c00 | ((s >> 9) & 0x7f) | ((s >> 31) << 15));
    if (i < xe) hx[i] = v; else hw[i - xe] = v;
  }
  Malloc(&dx, xe * 2); Malloc(&dw, we * 2); Malloc(&dout, ob);
  Memcpy(dx, hx, xe * 2, 1); Memcpy(dw, hw, we * 2, 1);
  void *args[] = {&dx, &dw, &dout, &m, &nn, &k};
  unsigned gx = (nn + bc - 1) / bc, gy = (m + br - 1) / br;
  EvC(&e0, 0); EvC(&e1, 0);
  E le = Launch(fn, gx, gy, 1, thr, 1, 1, 0, 0, args, 0), se = Sync();
  if (le || se) { fprintf(stderr, "launch %d sync %d\n", le, se); return 1; }
  double flops = 2.0 * m * nn * k, best = 1e30;
  for (int r = 0; r < 5; r++) {
    float ms;
    EvR(e0, 0);
    for (int i = 0; i < 10; i++) Launch(fn, gx, gy, 1, thr, 1, 1, 0, 0, args, 0);
    EvR(e1, 0); EvS(e1); EvT(&ms, e0, e1);
    ms /= 10;
    if (ms < best) best = ms;
    printf("run %d: %.3f ms  %.2f TFLOPS  %.1f W\n", r, ms, flops / ms / 1e9, power_uw() / 1e6);
  }
  float *h = malloc(ob);
  Memcpy(h, dout, ob, 2);
  uint64_t fnv = 1469598103934665603ull;
  const unsigned char *b = (const unsigned char *)h;
  for (size_t i = 0; i < ob; i++) fnv = (fnv ^ b[i]) * 1099511628211ull;
  printf("BEST %s %s m%d n%d k%d: %.3f ms %.2f TFLOPS hash %016llx\n", argv[3], argv[2], m, nn, k, best,
         flops / best / 1e9, (unsigned long long)fnv);
  return 0;
}
