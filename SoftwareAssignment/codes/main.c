#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static long get_file_size(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    if (fseek(f, 0, SEEK_END) != 0)
    {
        fclose(f);
        return -1;
    }
    long sz = ftell(f);
    fclose(f);
    return sz;
}

static void skip_comments(FILE *f)
{
    int c;
    while ((c = fgetc(f)) == '#')
    {
        while ((c = fgetc(f)) != '\n' && c != EOF)
            ;
    }
    if (c != EOF)
        ungetc(c, f);
}

static unsigned char *read_p5_pgm(const char *path, int *width, int *height, int *maxval)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    char magic[3] = {0};
    if (!fgets(magic, sizeof(magic), f))
    {
        fclose(f);
        return NULL;
    }
    if (strncmp(magic, "P5", 2) != 0)
    {
        fprintf(stderr, "Only binary P5 PGM supported (found %s)\n", magic);
        fclose(f);
        return NULL;
    }
    skip_comments(f);
    if (fscanf(f, "%d", width) != 1)
    {
        fclose(f);
        return NULL;
    }
    skip_comments(f);
    if (fscanf(f, "%d", height) != 1)
    {
        fclose(f);
        return NULL;
    }
    skip_comments(f);
    if (fscanf(f, "%d", maxval) != 1)
    {
        fclose(f);
        return NULL;
    }
    fgetc(f);
    if (*maxval > 255)
    {
        fprintf(stderr, "Only maxval <= 255 supported in this program.\n");
        fclose(f);
        return NULL;
    }
    long npix = (long)(*width) * (*height);
    unsigned char *buf = malloc(npix);
    if (!buf)
    {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, npix, f);
    if (got != (size_t)npix)
    {
        fprintf(stderr, "Unexpected pixel data size: got %zu expected %ld\n", got, npix);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return buf;
}

static int write_p5_pgm(const char *path, double *mat_col_major, int m, int n, int maxval)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    fprintf(f, "P5\n%d %d\n%d\n", n, m, maxval);
    long npix = (long)m * n;
    unsigned char *buf = malloc(npix);
    if (!buf)
    {
        fclose(f);
        return -1;
    }

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double v = mat_col_major[j * (long)m + i];
            if (v < 0)
                v = 0;
            if (v > maxval)
                v = maxval;
            buf[i * (long)n + j] = (unsigned char)(v + 0.5);
        }
    }
    fwrite(buf, 1, npix, f);
    free(buf);
    fclose(f);
    return 0;
}

static double vec_norm(int len, const double *x)
{
    double s = 0.0;
    for (int i = 0; i < len; ++i)
        s += x[i] * x[i];
    return sqrt(s);
}

static void mat_vec_mult(int m, int n, const double *M, const double *v, double *out)
{
    for (int i = 0; i < m; ++i)
        out[i] = 0.0;
    for (int j = 0; j < n; ++j)
    {
        double vj = v[j];
        const double *col = M + (long)j * m;
        for (int i = 0; i < m; ++i)
            out[i] += col[i] * vj;
    }
}

static void mat_T_vec_mult(int m, int n, const double *M, const double *u, double *out)
{
    for (int j = 0; j < n; ++j)
    {
        const double *col = M + (long)j * m;
        double s = 0.0;
        for (int i = 0; i < m; ++i)
            s += col[i] * u[i];
        out[j] = s;
    }
}

static void compute_topk_svd(const double *Aorig, int m, int n, int K,
                             double *S, double *U, double *VT, int max_iter, double tol)
{
    long MN = (long)m * n;
    double *R = malloc(sizeof(double) * MN);
    if (!R)
        return;
    memcpy(R, Aorig, sizeof(double) * MN);

    double *u = malloc(sizeof(double) * m);
    double *v = malloc(sizeof(double) * n);
    double *tmpm = malloc(sizeof(double) * ((m > n) ? m : n));
    if (!u || !v || !tmpm)
    {
        free(R);
        free(u);
        free(v);
        free(tmpm);
        return;
    }

    for (int comp = 0; comp < K; ++comp)
    {
        unsigned int seed = 12345 + comp;
        for (int j = 0; j < n; ++j)
        {
            seed = seed * 1103515245u + 12345u;
            v[j] = ((double)(seed & 0x7fffffff)) / 2147483647.0;
        }

        double prev_sigma = 0.0;
        int iter_used = 0;
        for (int iter = 0; iter < max_iter; ++iter)
        {
            mat_vec_mult(m, n, R, v, u);
            double norm_u = vec_norm(m, u);
            if (norm_u == 0.0)
                break;
            for (int i = 0; i < m; ++i)
                u[i] /= norm_u;

            mat_T_vec_mult(m, n, R, u, v);
            double norm_v = vec_norm(n, v);
            if (norm_v == 0.0)
                break;
            for (int j = 0; j < n; ++j)
                v[j] /= norm_v;

            mat_vec_mult(m, n, R, v, tmpm);
            double sigma = 0.0;
            for (int i = 0; i < m; ++i)
                sigma += u[i] * tmpm[i];

            if (fabs(sigma - prev_sigma) < tol * sigma)
            {
                S[comp] = sigma;
                iter_used = iter + 1;
                break;
            }
            prev_sigma = sigma;

            if (iter == max_iter - 1)
            {
                S[comp] = prev_sigma;
                iter_used = max_iter;
            }
        }

        mat_vec_mult(m, n, R, v, u);
        double sigma = vec_norm(m, u);
        if (sigma == 0.0)
        {
            S[comp] = 0.0;
            for (int i = 0; i < m; ++i)
                U[comp * (long)m + i] = 0.0;
            for (int j = 0; j < n; ++j)
                VT[comp + (long)K * j] = 0.0;
            continue;
        }
        for (int i = 0; i < m; ++i)
            U[comp * (long)m + i] = u[i] / sigma;
        S[comp] = sigma;

        mat_T_vec_mult(m, n, R, u, v);
        double nv = vec_norm(n, v);
        if (nv == 0.0)
            nv = 1.0;
        for (int j = 0; j < n; ++j)
            VT[comp + (long)K * j] = v[j] / nv;

        for (int j = 0; j < n; ++j)
        {
            double vj = VT[comp + (long)K * j];
            double *col = R + (long)j * m;
            for (int i = 0; i < m; ++i)
                col[i] -= S[comp] * U[comp * (long)m + i] * vj;
        }
    }

    free(R);
    free(u);
    free(v);
    free(tmpm);
}

int main(int argc, char **argv)
{
    int write_pgm = 1;

    if (argc > 1 && strcmp(argv[1], "-p") == 0)
    {
        write_pgm = 1;
        argv++;
        argc--;
    }

    const char *infile = argv[1];
    int width = 0, height = 0, maxval = 0;

    unsigned char *pix = read_p5_pgm(infile, &width, &height, &maxval);
    if (!pix)
    {
        fprintf(stderr, "Failed to read %s\n", infile);
        return 1;
    }

    int m = height, n = width;
    int minmn = (m < n) ? m : n;

    double *A = malloc(sizeof(double) * (long)m * n);
    if (!A)
    {
        free(pix);
        return 2;
    }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[j * m + i] = (double)pix[i * width + j];

    double *Aorig = malloc(sizeof(double) * (long)m * n);
    if (!Aorig)
    {
        free(pix);
        free(A);
        return 3;
    }
    memcpy(Aorig, A, sizeof(double) * (long)m * n);
    free(pix);

    int max_k = 0;
    for (int ai = 2; ai < argc; ++ai)
    {
        int kv = atoi(argv[ai]);
        if (kv > max_k)
            max_k = kv;
    }
    if (max_k <= 0)
    {
        fprintf(stderr, "No positive k requested\n");
        return 1;
    }
    if (max_k > minmn)
        max_k = minmn;

    double *S = calloc(max_k, sizeof(double));
    double *U = calloc((long)m * max_k, sizeof(double));
    double *VT = calloc((long)max_k * n, sizeof(double));
    if (!S || !U || !VT)
    {
        fprintf(stderr, "Allocation failure\n");
        free(A);
        free(S);
        free(U);
        free(VT);
        return 4;
    }

    compute_topk_svd(A, m, n, max_k, S, U, VT, 100, 1e-6);

    for (int ai = 2; ai < argc; ++ai)
    {
        int k = atoi(argv[ai]);
        if (k <= 0)
        {
            fprintf(stderr, "Skipping invalid k=%s (must be positive)\n", argv[ai]);
            continue;
        }
        if (k > max_k)
        {
            fprintf(stdout, "Note: reducing k=%d to %d (max computed)\n", k, max_k);
            k = max_k;
        }
        fprintf(stdout, "\nReconstructing with k=%d\n", k);

        double *Ak = calloc((long)m * n, sizeof(double));
        if (!Ak)
        {
            fprintf(stderr, "Allocation Ak failed\n");
            continue;
        }

        for (int comp = 0; comp < k; ++comp)
        {
            double sigma = S[comp];
            double *u_col = &U[(long)comp * m];
            for (int j = 0; j < n; ++j)
            {
                double vj = VT[comp + (long)max_k * j];
                double mult = sigma * vj;
                double *colAk = Ak + (long)j * m;
                for (int i = 0; i < m; ++i)
                    colAk[i] += u_col[i] * mult;
            }
        }

        double err2 = 0.0;
        long total = (long)m * n;
        for (long idx = 0; idx < total; ++idx)
        {
            double d = Aorig[idx] - Ak[idx];
            err2 += d * d;
        }
        double frob = sqrt(err2);
        fprintf(stdout, "Frobenius error: %g\n", frob);

        if (write_pgm)
        {
            const char *base = strrchr(infile, '/');
            if (base)
                base++;
            else
                base = infile;
            char namebuf[1024];
            snprintf(namebuf, sizeof(namebuf), "%s", base);
            char *dot = strrchr(namebuf, '.');
            if (dot)
                *dot = '\0';
            char outpath[1200];
            snprintf(outpath, sizeof(outpath), "%s_k%d.pgm", namebuf, k);
            if (write_p5_pgm(outpath, Ak, m, n, maxval) != 0)
            {
                fprintf(stderr, "Failed to write %s\n", outpath);
            }
            else
            {
                fprintf(stdout, "%s created\n", outpath);
            }
        }

        free(Ak);
    }

    free(Aorig);
    free(A);
    free(S);
    free(U);
    free(VT);

    return 0;
}
