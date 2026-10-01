#include "libs.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

bool check_matmul() {
    struct { int m, n, k; float eps; } cases[] = {
        {1, 1, 1, 1e-4f}, {2, 2, 2, 1e-4f}, {4, 4, 4, 1e-4f},
        {8, 8, 8, 1e-4f}, {16, 16, 16, 1e-4f}, {32, 32, 32, 1e-4f},
        {64, 64, 64, 1e-4f}, {1, 100, 50, 1e-4f}, {100, 1, 50, 1e-4f},
        {100, 50, 1, 1e-4f}, {3, 5, 7, 1e-4f}, {7, 13, 17, 1e-4f},
        {17, 31, 63, 1e-4f}, {33, 65, 129, 1e-4f}, {128, 128, 128, 1e-3f},
        {256, 256, 256, 1e-3f}, {512, 512, 512, 1e-3f}, {1, 4096, 1, 1e-4f},
        {4096, 1, 1, 1e-4f}, {1, 1, 4096, 1e-4f},
    };
    int total = sizeof(cases) / sizeof(cases[0]);
    int passed = 0;

    for (int ci = 0; ci < total; ci++) {
        int m = cases[ci].m, n = cases[ci].n, k = cases[ci].k;
        float eps = cases[ci].eps;
        size_t sA = (size_t)m * k, sB = (size_t)k * n, sC = (size_t)m * n;

        float* A = (float*)malloc(sA * sizeof(float));
        float* B = (float*)malloc(sB * sizeof(float));
        float* C_ref = (float*)malloc(sC * sizeof(float));
        float* C_got = (float*)malloc(sC * sizeof(float));

        if (!A || !B || !C_ref || !C_got) {
            free(A); free(B); free(C_ref); free(C_got);
            continue;
        }

        srand(42 + ci);
        for (size_t i = 0; i < sA; i++) A[i] = (rand() % 2000 - 1000) / 1000.0f;
        for (size_t i = 0; i < sB; i++) B[i] = (rand() % 2000 - 1000) / 1000.0f;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int t = 0; t < k; t++)
                    s += (double)A[i*k+t] * (double)B[t*n+j];
                C_ref[i*n+j] = (float)s;
            }

        matmul(m, n, k, A, B, C_got);

        bool ok = true;
        for (size_t i = 0; i < sC; i++) {
            if (std::fabs(C_ref[i] - C_got[i]) > eps) {
                printf("FAIL: m=%d n=%d k=%d idx=%zu ref=%.6f got=%.6f\n",
                       m, n, k, i, C_ref[i], C_got[i]);
                ok = false;
                break;
            }
        }
        if (ok) passed++;

        free(A); free(B); free(C_ref); free(C_got);
    }

    printf("check_matmul: %d/%d passed\n", passed, total);
    return passed == total;
}