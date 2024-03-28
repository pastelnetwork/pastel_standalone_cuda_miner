// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.

#include <src/cuda/memutils.h>
#include <blake2b.h>

__device__ __constant__ uint8_t blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

__constant__ uint64_t blake2b_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

__device__ void secure_zero_memory_device(void *v, size_t n)
{
    volatile uint8_t *p = (volatile uint8_t *)v;
    while (n--)
        *p++ = 0;
}

__device__ uint64_t load64(const void *src)
{
    const uint8_t *p = (const uint8_t *)src;
    return ((uint64_t)(p[0]) << 0) |
           ((uint64_t)(p[1]) << 8) |
           ((uint64_t)(p[2]) << 16) |
           ((uint64_t)(p[3]) << 24) |
           ((uint64_t)(p[4]) << 32) |
           ((uint64_t)(p[5]) << 40) |
           ((uint64_t)(p[6]) << 48) |
           ((uint64_t)(p[7]) << 56);
}

__device__ void store64(void *dst, uint64_t w)
{
    uint8_t *p = (uint8_t *)dst;
    p[0] = (uint8_t)(w >> 0);
    p[1] = (uint8_t)(w >> 8);
    p[2] = (uint8_t)(w >> 16);
    p[3] = (uint8_t)(w >> 24);
    p[4] = (uint8_t)(w >> 32);
    p[5] = (uint8_t)(w >> 40);
    p[6] = (uint8_t)(w >> 48);
    p[7] = (uint8_t)(w >> 56);
}

__device__ uint64_t rotr64(uint64_t x, uint64_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ void G(int r, int i, uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, const uint64_t m[16])
{
    a = a + b + m[blake2b_sigma[r][2 * i + 0]];
    d = rotr64(d ^ a, 32);
    c = c + d;
    b = rotr64(b ^ c, 24);
    a = a + b + m[blake2b_sigma[r][2 * i + 1]];
    d = rotr64(d ^ a, 16);
    c = c + d;
    b = rotr64(b ^ c, 63);
}

__device__ void blake2b_increment_counter(blake2b_state *S, uint64_t inc)
{
    S->t[0] += inc;
    S->t[1] += (S->t[0] < inc);
}

__device__ void blake2b_init_device(blake2b_state *S, size_t outlen)
{
    S->h[0] = blake2b_IV[0] ^ (0x01010000 | (outlen << 8));

    for (int i = 1; i < 8; ++i)
        S->h[i] = blake2b_IV[i];

    S->t[0] = 0;
    S->t[1] = 0;
    S->f[0] = 0;
    S->f[1] = 0;
    S->buflen = 0;
    S->outlen = outlen;
    S->last_node = 0;
}

__device__ void blake2b_compress_device(blake2b_state *S, const uint8_t *block)
{
    uint64_t m[16];
    uint64_t v[16];

    for (int i = 0; i < 16; ++i)
        m[i] = load64(block + i * sizeof(m[i]));

    for (int i = 0; i < 8; ++i)
        v[i] = S->h[i];

    v[8] = blake2b_IV[0];
    v[9] = blake2b_IV[1];
    v[10] = blake2b_IV[2];
    v[11] = blake2b_IV[3];
    v[12] = blake2b_IV[4] ^ S->t[0];
    v[13] = blake2b_IV[5] ^ S->t[1];
    v[14] = blake2b_IV[6] ^ S->f[0];
    v[15] = blake2b_IV[7] ^ S->f[1];

    // Mixing rounds
    for (int r = 0; r < 12; ++r)
    {
        G(r, 0, v[0], v[4], v[8], v[12], m);
        G(r, 1, v[1], v[5], v[9], v[13], m);
        G(r, 2, v[2], v[6], v[10], v[14], m);
        G(r, 3, v[3], v[7], v[11], v[15], m);
        G(r, 4, v[0], v[5], v[10], v[15], m);
        G(r, 5, v[1], v[6], v[11], v[12], m);
        G(r, 6, v[2], v[7], v[8], v[13], m);
        G(r, 7, v[3], v[4], v[9], v[14], m);
    }

    for (int i = 0; i < 8; ++i)
        S->h[i] ^= v[i] ^ v[i + 8];
}

__device__ void blake2b_update_device(blake2b_state *S, const void *pin, size_t inlen)
{
    const uint8_t *in = (const uint8_t *)pin;

    if (S->buflen + inlen > BLAKE2B_BLOCKBYTES)
    {
        size_t left = S->buflen;
        size_t fill = BLAKE2B_BLOCKBYTES - left;
        memcpy(S->buf + left, in, fill);
        S->buflen += fill;
        blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
        blake2b_compress_device(S, S->buf);
        in += fill;
        inlen -= fill;
        S->buflen = 0;
    }
    memcpy(S->buf + S->buflen, in, inlen);
    S->buflen += inlen;
}

__device__ void blake2b_final_device(blake2b_state *S, void *out, size_t outlen)
{
    uint8_t buffer[BLAKE2B_OUTBYTES] = {0};

    if (outlen > BLAKE2B_OUTBYTES)
        outlen = BLAKE2B_OUTBYTES;

    if (S->buflen > BLAKE2B_BLOCKBYTES)
    {
        blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
        blake2b_compress_device(S, S->buf);
        S->buflen = 0;
    }
    blake2b_increment_counter(S, S->buflen);
    S->f[0] = (uint64_t)-1;
    memset(S->buf + S->buflen, 0, BLAKE2B_BLOCKBYTES - S->buflen);
    blake2b_compress_device(S, S->buf);

    for (int i = 0; i < 8; ++i)
        store64(buffer + sizeof(S->h[i]) * i, S->h[i]);
    memcpy(out, buffer, outlen);

    // Clear state
    secure_zero_memory_device(buffer, sizeof(buffer));
    secure_zero_memory_device(S->buf, sizeof(S->buf));
    secure_zero_memory_device(S->h, sizeof(S->h));
    secure_zero_memory_device(S->t, sizeof(S->t));
    secure_zero_memory_device(S->f, sizeof(S->f));
    S->buflen = 0;
    S->last_node = 0;
}

