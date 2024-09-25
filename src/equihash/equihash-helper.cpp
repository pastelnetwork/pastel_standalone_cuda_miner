// Copyright (c) 2016 Jack Grigg
// Copyright (c) 2016 The Zcash developers
// Copyright (c) 2021-2024 The Pastel Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <cassert>
#include <vector>
#include <cstring>

#include <blake2b.h>
#include <compat/endian.h>
#include <src/equihash/equihash-types.h>
#include <src/equihash/equihash-helper.h>
#include <src/equihash/blake2b_host.h>

using namespace std;

void GenerateHash(const blake2b_state& base_state, eh_index g,
                  unsigned char* hash, size_t hLen)
{
    blake2b_state state = base_state;
    eh_index lei = htole32(g);
    blake2b_update_host(&state, (const unsigned char*) &lei,
                                      sizeof(eh_index));
    blake2b_final_host(&state, hash, hLen);
}

void ExpandArray(const unsigned char* in, size_t in_len,
                 unsigned char* out, size_t out_len,
                 size_t bit_len, size_t byte_pad)
{
    assert(bit_len >= 8);
    assert(8*sizeof(uint32_t) >= 7+bit_len);

    size_t out_width { (bit_len+7)/8 + byte_pad };
    assert(out_len == 8*out_width*in_len/bit_len);

    uint32_t bit_len_mask { ((uint32_t)1 << bit_len) - 1 };

    // The acc_bits least-significant bits of acc_value represent a bit sequence
    // in big-endian order.
    size_t acc_bits = 0;
    uint32_t acc_value = 0;

    size_t j = 0;
    for (size_t i = 0; i < in_len; i++) {
        acc_value = (acc_value << 8) | in[i];
        acc_bits += 8;

        // When we have bit_len or more bits in the accumulator, write the next
        // output element.
        if (acc_bits >= bit_len) {
            acc_bits -= bit_len;
            for (size_t x = 0; x < byte_pad; x++) {
                out[j+x] = 0;
            }
            for (size_t x = byte_pad; x < out_width; x++) {
                out[j+x] = (
                    // Big-endian
                    acc_value >> (acc_bits+(8*(out_width-x-1)))
                ) & (
                    // Apply bit_len_mask across byte boundaries
                    (bit_len_mask >> (8*(out_width-x-1))) & 0xFF
                );
            }
            j += out_width;
        }
    }
}

void CompressArray(const unsigned char* in, size_t in_len,
                   unsigned char* out, size_t out_len,
                   size_t bit_len, size_t byte_pad)
{
    assert(bit_len >= 8);
    assert(8*sizeof(uint32_t) >= 7+bit_len);

    size_t in_width { (bit_len+7)/8 + byte_pad };
    assert(out_len == bit_len*in_len/(8*in_width));

    uint32_t bit_len_mask { ((uint32_t)1 << bit_len) - 1 };

    // The acc_bits least-significant bits of acc_value represent a bit sequence
    // in big-endian order.
    size_t acc_bits = 0;
    uint32_t acc_value = 0;

    size_t j = 0;
    for (size_t i = 0; i < out_len; i++) {
        // When we have fewer than 8 bits left in the accumulator, read the next
        // input element.
        if (acc_bits < 8) {
            acc_value = acc_value << bit_len;
            for (size_t x = byte_pad; x < in_width; x++) {
                acc_value = acc_value | (
                    (
                        // Apply bit_len_mask across byte boundaries
                        in[j+x] & ((bit_len_mask >> (8*(in_width-x-1))) & 0xFF)
                    ) << (8*(in_width-x-1))); // Big-endian
            }
            j += in_width;
            acc_bits += bit_len;
        }

        acc_bits -= 8;
        out[i] = (acc_value >> acc_bits) & 0xFF;
    }
}

// Big-endian so that lexicographic array comparison is equivalent to integer
// comparison
void EhIndexToArray(const eh_index i, unsigned char* array)
{
    static_assert(sizeof(eh_index) == 4);
    eh_index bei = htobe32(i);
    memcpy(array, &bei, sizeof(eh_index));
}

// Big-endian so that lexicographic array comparison is equivalent to integer
// comparison
eh_index ArrayToEhIndex(const unsigned char* array)
{
    static_assert(sizeof(eh_index) == 4);
    eh_index bei;
    memcpy(&bei, array, sizeof(eh_index));
    return be32toh(bei);
}

eh_trunc TruncateIndex(const eh_index i, const unsigned int ilen)
{
    // Truncate to 8 bits
    static_assert(sizeof(eh_trunc) == 1);
    return (i >> (ilen - 8)) & 0xff;
}

eh_index UntruncateIndex(const eh_trunc t, const eh_index r, const unsigned int ilen)
{
    eh_index i{t};
    return (i << (ilen - 8)) | r;
}

vector<eh_index> GetIndicesFromMinimal(const v_uint8 &minimal, const size_t cBitLen)
{
    assert(((cBitLen+1)+7)/8 <= sizeof(eh_index));
    
    const size_t lenIndices = 8 * sizeof(eh_index) * minimal.size() / (cBitLen + 1);
    const size_t bytePad = sizeof(eh_index) - ((cBitLen + 1) + 7) / 8;
    v_uint8 array(lenIndices);
    ExpandArray(minimal.data(), minimal.size(),
                array.data(), lenIndices, cBitLen+1, bytePad);
    vector<eh_index> ret;
    ret.reserve(lenIndices / sizeof(eh_index));
    for (int i = 0; i < lenIndices; i += sizeof(eh_index))
        ret.push_back(ArrayToEhIndex(array.data()+i));
    return ret;
}

v_uint8 GetMinimalFromIndices(const vector<eh_index> &indices, const size_t cBitLen)
{
    assert(((cBitLen+1)+7)/8 <= sizeof(eh_index));

    const size_t lenIndices = indices.size() * sizeof(eh_index);
    const size_t minLen = (cBitLen + 1) * lenIndices / (8 * sizeof(eh_index));
    const size_t bytePad = sizeof(eh_index) - ((cBitLen + 1) + 7) / 8;
    v_uint8 array(lenIndices);
    for (size_t i = 0; i < indices.size(); ++i)
        EhIndexToArray(indices[i], array.data()+(i*sizeof(eh_index)));
    v_uint8 ret(minLen);
    CompressArray(array.data(), lenIndices, ret.data(), minLen, cBitLen+1, bytePad);
    return ret;
}

template<size_t WIDTH>
StepRow<WIDTH>::StepRow(const unsigned char* hashIn, size_t hInLen,
                        size_t hLen, size_t cBitLen)
{
    assert(hLen <= WIDTH);
    ExpandArray(hashIn, hInLen, hash, hLen, cBitLen, 0);
}

template<size_t WIDTH> template<size_t W>
StepRow<WIDTH>::StepRow(const StepRow<W>& a)
{
    static_assert(W <= WIDTH);
    copy(a.hash, a.hash+W, hash);
}

template<size_t WIDTH>
FullStepRow<WIDTH>::FullStepRow(const unsigned char* hashIn, size_t hInLen,
                                size_t hLen, size_t cBitLen, eh_index i) :
        StepRow<WIDTH> {hashIn, hInLen, hLen, cBitLen}
{
    EhIndexToArray(i, hash+hLen);
}

template<size_t WIDTH> template<size_t W>
FullStepRow<WIDTH>::FullStepRow(const FullStepRow<W>& a, const FullStepRow<W>& b, size_t len, size_t lenIndices, int trim) :
        StepRow<WIDTH> {a}
{
    assert(len+lenIndices <= W);
    assert(len-trim+(2*lenIndices) <= WIDTH);
    for (int i = trim; i < len; i++)
        hash[i-trim] = a.hash[i] ^ b.hash[i];
    if (a.IndicesBefore(b, len, lenIndices)) {
        copy(a.hash+len, a.hash+len+lenIndices, hash+len-trim);
        copy(b.hash+len, b.hash+len+lenIndices, hash+len-trim+lenIndices);
    } else {
        copy(b.hash+len, b.hash+len+lenIndices, hash+len-trim);
        copy(a.hash+len, a.hash+len+lenIndices, hash+len-trim+lenIndices);
    }
}

template<size_t WIDTH>
FullStepRow<WIDTH>& FullStepRow<WIDTH>::operator=(const FullStepRow<WIDTH>& a)
{
    copy(a.hash, a.hash+WIDTH, hash);
    return *this;
}

template<size_t WIDTH>
bool StepRow<WIDTH>::IsZero(size_t len)
{
    // This doesn't need to be constant time.
    for (int i = 0; i < len; i++) {
        if (hash[i] != 0)
            return false;
    }
    return true;
}

template<size_t WIDTH>
v_uint8 FullStepRow<WIDTH>::GetIndices(size_t len, size_t lenIndices, size_t cBitLen) const
{
    assert(((cBitLen + 1) + 7) / 8 <= sizeof(eh_index));
    const size_t minLen = (cBitLen + 1) * lenIndices / (8 * sizeof(eh_index));
    const size_t bytePad = sizeof(eh_index) - ((cBitLen + 1) + 7) / 8;
    v_uint8 ret(minLen);
    CompressArray(hash+len, lenIndices, ret.data(), minLen, cBitLen+1, bytePad);
    return ret;
}

// Checks if the intersection of a.indices and b.indices is empty
template<size_t WIDTH>
bool DistinctIndices(const FullStepRow<WIDTH>& a, const FullStepRow<WIDTH>& b, size_t len, size_t lenIndices)
{
    for(size_t i = 0; i < lenIndices; i += sizeof(eh_index)) {
        for(size_t j = 0; j < lenIndices; j += sizeof(eh_index)) {
            if (memcmp(a.hash+len+i, b.hash+len+j, sizeof(eh_index)) == 0) {
                return false;
            }
        }
    }
    return true;
}

template<size_t WIDTH>
bool HasCollision(StepRow<WIDTH>& a, StepRow<WIDTH>& b, const size_t l)
{
    // This doesn't need to be constant time.
    for (size_t j = 0; j < l; ++j)
    {
        if (a.hash[j] != b.hash[j])
            return false;
    }
    return true;
}

// explicit template instantiation
template class StepRow<Eh200_9::FinalFullWidth>;
template class FullStepRow<Eh200_9::FinalFullWidth>;
template FullStepRow<Eh200_9::FinalFullWidth>::FullStepRow(const FullStepRow<Eh200_9::FinalFullWidth>&, const FullStepRow<Eh200_9::FinalFullWidth>&, size_t, size_t, int);
template bool HasCollision<Eh200_9::FinalFullWidth>(StepRow<Eh200_9::FinalFullWidth>& a, StepRow<Eh200_9::FinalFullWidth>& b, const size_t l);
template bool DistinctIndices<Eh200_9::FinalFullWidth>(const FullStepRow<Eh200_9::FinalFullWidth>& a, const FullStepRow<Eh200_9::FinalFullWidth>& b, size_t len, size_t lenIndices);

