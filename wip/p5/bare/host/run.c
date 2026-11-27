// usage: run LIBHIP HSACO KERNEL GROUPS ITERS WMMA_PER_GROUP_ITER [fill]: times 5 x 10 launches of KERNEL(x, out, iters),
// prints TFLOPS, power and an FNV hash of out (groups x 256 floats). x = 8 KiB bf16: fill 0 = 1.0, 1 = pseudo-random.
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
    if (p) {
      if (fscanf(p, "%ld", &uw) != 1) uw = 0;
      fclose(p);
    }
  }
  globfree(&g);
  return uw;
}
int main(int argc, char **argv) {
  if (argc < 7) return 2;
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
  int groups = atoi(argv[4]), iters = atoi(argv[5]), wpi = atoi(argv[6]), fill = argc > 7 ? atoi(argv[7]) : 0;
  Init(0);
  void *mod, *fn, *a, *o, *e0, *e1;
  if (Load(&mod, img) || Get(&fn, mod, argv[3])) { fprintf(stderr, "load failed\n"); return 1; }
  uint16_t hx[4096];
  uint32_t s = 12345;
  for (int i = 0; i < 4096; i++) {
    s = s * 1664525u + 1013904223u;
    hx[i] = fill ? (uint16_t)(0x3c00 | ((s >> 9) & 0x7f) | ((s >> 31) << 15)) : 0x3f80;
  }
  size_t ob = (size_t)groups * 256 * 4;
  Malloc(&a, sizeof hx);
  Malloc(&o, ob);
  Memcpy(a, hx, sizeof hx, 1);
  void *args[] = {&a, &o, &iters};
  EvC(&e0, 0);
  EvC(&e1, 0);
  E le = Launch(fn, groups, 1, 1, 256, 1, 1, 0, 0, args, 0);
  E se = Sync();
  if (le || se) { fprintf(stderr, "launch %d sync %d\n", le, se); return 1; }
  double best = 1e30;
  for (int r = 0; r < 5; r++) {
    float ms;
    EvR(e0, 0);
    for (int i = 0; i < 10; i++) Launch(fn, groups, 1, 1, 256, 1, 1, 0, 0, args, 0);
    EvR(e1, 0);
    EvS(e1);
    EvT(&ms, e0, e1);
    ms /= 10;
    if (ms < best) best = ms;
    double flops = (double)groups * 8 * iters * wpi * 8192.0;
    printf("run %d: %.3f ms  %.2f TFLOPS  %.1f W\n", r, ms, flops / ms / 1e9, power_uw() / 1e6);
  }
  float *h = malloc(ob);
  Memcpy(h, o, ob, 2);
  uint64_t fnv = 1469598103934665603ull;
  const unsigned char *b = (const unsigned char *)h;
  for (size_t i = 0; i < ob; i++) fnv = (fnv ^ b[i]) * 1099511628211ull;
  double flops = (double)groups * 8 * iters * wpi * 8192.0;
  printf("BEST %s %s: %.3f ms %.2f TFLOPS hash %016llx out0 %g\n", argv[3], argv[2], best, flops / best / 1e9,
         (unsigned long long)fnv, h[0]);
  return 0;
}
