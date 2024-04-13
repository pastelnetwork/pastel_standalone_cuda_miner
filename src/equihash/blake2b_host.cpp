// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <memory>
#include <cstring>
#include <cstdint>

#include <blake2b.h>

static constexpr uint8_t blake2b_sigma[12][16] =
{
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

static constexpr uint64_t blake2b_IV[8] =
{
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

void secure_zero_memory(void *v, size_t n)
{
    volatile uint8_t *p = (volatile uint8_t *)v;
    while (n--)
        *p++ = 0;
}

#define rotr64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

#define G(r, i, a, b, c, d)                                     \
    do {                                                        \
        a = a + b + m[blake2b_sigma[r][2 * i + 0]];             \
        d = rotr64(d ^ a, 32);                                  \
        c = c + d;                                              \
        b = rotr64(b ^ c, 24);                                  \
        a = a + b + m[blake2b_sigma[r][2 * i + 1]];             \
        d = rotr64(d ^ a, 16);                                  \
        c = c + d;                                              \
        b = rotr64(b ^ c, 63);                                  \
    } while (0)

#define ROUND(r)                                                \
    do {                                                        \
        G(r, 0, v[0], v[4], v[8], v[12]);                       \
        G(r, 1, v[1], v[5], v[9], v[13]);                       \
        G(r, 2, v[2], v[6], v[10], v[14]);                      \
        G(r, 3, v[3], v[7], v[11], v[15]);                      \
        G(r, 4, v[0], v[5], v[10], v[15]);                      \
        G(r, 5, v[1], v[6], v[11], v[12]);                      \
        G(r, 6, v[2], v[7], v[8], v[13]);                       \
        G(r, 7, v[3], v[4], v[9], v[14]);                       \
    } while (0)

static inline uint64_t load64(const void *src)
{
    uint64_t w;
    memcpy(&w, src, sizeof w);
    return w;
}

static inline void store64(void *dst, uint64_t w)
{
    memcpy(dst, &w, sizeof w);
}


bool blake2b_init_host(blake2b_state *state, size_t outlen)
{
    if (!state || !outlen || outlen > BLAKE2B_OUTBYTES)
        return false;

    state->h[0] = blake2b_IV[0] ^ (0x01010000 | static_cast<uint8_t>(outlen));
    for (size_t i = 1; i < 8; ++i)
        state->h[i] = blake2b_IV[i]; 

    state->t[0] = 0;
    state->t[1] = 0;
    state->f[0] = 0;
    state->f[1] = 0;
    state->buflen = 0;
    state->outlen = outlen;
    state->last_node = 0;
    return true;
}

static void blake2b_compress(blake2b_state *state, const uint8_t *block)
{
    uint64_t m[16];
    uint64_t v[16];

    for (size_t i = 0; i < 16; ++i)
        m[i] = load64(block + i * sizeof(m[i]));

    for (size_t i = 0; i < 8; ++i)
        v[i] = state->h[i];

    v[8] = blake2b_IV[0];
    v[9] = blake2b_IV[1];
    v[10] = blake2b_IV[2];
    v[11] = blake2b_IV[3];
    v[12] = state->t[0] ^ blake2b_IV[4];
    v[13] = state->t[1] ^ blake2b_IV[5];
    v[14] = state->f[0] ^ blake2b_IV[6];
    v[15] = state->f[1] ^ blake2b_IV[7];

    ROUND(0); ROUND(1);  ROUND(2);
    ROUND(3); ROUND(4);  ROUND(5);
    ROUND(6); ROUND(7);  ROUND(8);
    ROUND(9); ROUND(10); ROUND(11);

    for (size_t i = 0; i < 8; ++i)
        state->h[i] = state->h[i] ^ v[i] ^ v[i + 8];
}

static inline void blake2b_increment_counter(blake2b_state *state, const uint64_t inc)
{
    state->t[0] += inc;
    state->t[1] += (state->t[0] < inc);
}

void blake2b_update_host(blake2b_state *state, const void *pin, size_t inlen)
{
    const uint8_t *in = static_cast<const uint8_t *>(pin);

    if (inlen == 0)
        return; // No input, nothing to do

    // If there's already data in the buffer, and the new data fills it
    size_t left = state->buflen;
    size_t fill = BLAKE2B_BLOCKBYTES - left;
    if (inlen > fill)
    {
        state->buflen = 0;
        memcpy(state->buf + left, in, fill); // Fill the buffer
        blake2b_increment_counter(state, BLAKE2B_BLOCKBYTES);
        blake2b_compress(state, state->buf); // Compress the full buffer
        in += fill;
        inlen -= fill;

        // Process remaining input data in BLAKE2B_BLOCKBYTES chunks
        while (inlen > BLAKE2B_BLOCKBYTES)
        {
            blake2b_increment_counter(state, BLAKE2B_BLOCKBYTES);
            blake2b_compress(state, in);
            in += BLAKE2B_BLOCKBYTES;
            inlen -= BLAKE2B_BLOCKBYTES;
        }
    }

    // Copy remaining input data into the buffer
    memcpy(state->buf + state->buflen, in, inlen);
    state->buflen += inlen;
}

static inline int blake2b_is_lastblock( const blake2b_state *S )
{
  return S->f[0] != 0;
}

static inline void blake2b_set_lastnode( blake2b_state *S )
{
  S->f[1] = (uint64_t)(-1);
}

static inline void blake2b_set_lastblock( blake2b_state *S )
{
  if( S->last_node )
    blake2b_set_lastnode( S );

  S->f[0] = (uint64_t)(-1);
}

bool blake2b_final_host(blake2b_state *S, void *out, size_t outlen)
{
    if (!out || outlen < S->outlen)
        return false;

    if (blake2b_is_lastblock(S))
        return false;

    blake2b_increment_counter(S, S->buflen);
    blake2b_set_lastblock(S);
    memset(S->buf + S->buflen, 0, BLAKE2B_BLOCKBYTES - S->buflen); // Padding
    blake2b_compress(S, S->buf);

    uint8_t buffer[BLAKE2B_OUTBYTES] = {0};
    for (size_t i = 0; i < 8; ++i)
        store64(buffer + sizeof(S->h[i]) * i, S->h[i]);

    memcpy(out, buffer, S->outlen);
    secure_zero_memory(buffer, sizeof(buffer));
    return true;
}

bool blake2b_init_salt_personal_host(blake2b_state *state, 
    const uint8_t* key, size_t nKeyLength, 
    size_t outlen,
    const uint8_t* salt, const size_t nSaltLength,
    const uint8_t* personal, const size_t nPersonaLength)
{
    if (!state || !outlen || outlen > BLAKE2B_OUTBYTES)
        return false;

    if (key && nKeyLength > BLAKE2B_KEYBYTES)
        return false;

    if (salt && nSaltLength > BLAKE2B_SALTBYTES)
        return false;

    if (personal && nPersonaLength > BLAKE2B_PERSONALBYTES) 
        return false;

    state->h[0] = blake2b_IV[0] ^ (0x01010000 | (nKeyLength << 8) | outlen);
    for (size_t i = 1; i < 8; ++i)
        state->h[i] = blake2b_IV[i]; 

    state->t[0] = 0;
    state->t[1] = 0;
    state->f[0] = 0;
    state->f[1] = 0;
    state->buflen = 0;
    state->outlen = outlen;
    state->last_node = 0;

    memset(state->buf, 0, sizeof(state->buf));
    memset(state->buf, 0, sizeof(state->buf));

    if (nKeyLength > 0)
    {
        uint8_t block[BLAKE2B_BLOCKBYTES];
        memset(block, 0, BLAKE2B_BLOCKBYTES);
        memcpy(block, key, nKeyLength);
        blake2b_update_host(state, block, BLAKE2B_BLOCKBYTES);
        secure_zero_memory(block, BLAKE2B_BLOCKBYTES);
    }

    if (salt && nSaltLength)
        blake2b_update_host(state, salt, nSaltLength);

    if (personal && nPersonaLength)
        blake2b_update_host(state, personal, nPersonaLength);

    return true;
}
