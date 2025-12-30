// jpge.cpp - Minimal JPEG Encoder
// Based on public domain code by Rich Geldreich
// Simplified for VR frame capture

#include "jpge.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace jpge
{

#define JPGE_MAX(a,b) (((a)>(b))?(a):(b))
#define JPGE_MIN(a,b) (((a)<(b))?(a):(b))

typedef int32 dct_t;

static const uint8 s_zag[64] = {
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,
    21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,
    60,61,54,47,55,62,63
};

static const int16 s_std_lum_quant[64] = {
    16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,
    29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,
    103,121,120,101,72,92,95,98,112,100,103,99
};

static const int16 s_std_croma_quant[64] = {
    17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,
    99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,99,99
};

static const uint8 s_dc_lum_bits[17] = {0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0};
static const uint8 s_dc_lum_val[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
static const uint8 s_ac_lum_bits[17] = {0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d};

static const uint8 s_ac_lum_val[162] = {
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa
};

static const uint8 s_dc_chroma_bits[17] = {0,0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0};
static const uint8 s_dc_chroma_val[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
static const uint8 s_ac_chroma_bits[17] = {0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77};

static const uint8 s_ac_chroma_val[162] = {
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
    0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
    0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
    0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
    0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
    0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa
};

// Output stream helper
class output_stream {
public:
    uint8* m_pBuf; int m_buf_size; int m_pos;
    output_stream(void* pBuf, int buf_size) : m_pBuf((uint8*)pBuf), m_buf_size(buf_size), m_pos(0) {}
    bool put(uint8 c) { if (m_pos >= m_buf_size) return false; m_pBuf[m_pos++] = c; return true; }
    bool put_buf(const void* p, int len) {
        if (m_pos + len > m_buf_size) return false;
        memcpy(m_pBuf + m_pos, p, len); m_pos += len; return true;
    }
};

// Huffman table
struct huffman_table {
    uint32 m_codes[256]; uint8 m_code_sizes[256];
    void compute(const uint8* bits, const uint8* val) {
        int i, l, code = 0, si = 0;
        memset(m_codes, 0, sizeof(m_codes)); memset(m_code_sizes, 0, sizeof(m_code_sizes));
        for (l = 1; l <= 16; l++) {
            for (i = 0; i < bits[l]; i++) { m_codes[val[si]] = code++; m_code_sizes[val[si++]] = l; }
            code <<= 1;
        }
    }
};

// JPEG encoder class
class jpeg_encoder {
public:
    output_stream* m_pStream;
    int m_image_width, m_image_height, m_num_components;
    const uint8* m_pSrc;
    int m_quality;
    uint8 m_quantization_tables[2][64];
    huffman_table m_huff_dc[2], m_huff_ac[2];
    int m_last_dc[3];
    uint32 m_bit_buffer; int m_bits_in;

    void emit_byte(uint8 c) { m_pStream->put(c); }
    void emit_word(uint32 w) { emit_byte((uint8)(w >> 8)); emit_byte((uint8)(w & 0xFF)); }
    void emit_marker(uint8 m) { emit_byte(0xFF); emit_byte(m); }

    void compute_quant_table(uint8* dst, const int16* src) {
        for (int i = 0; i < 64; i++) {
            int q = (src[i] * m_quality + 50) / 100;
            dst[s_zag[i]] = (uint8)JPGE_MIN(JPGE_MAX(q, 1), 255);
        }
    }

    void emit_dqt() {
        emit_marker(0xDB); emit_word(2 + 2 * 65);
        emit_byte(0); for (int i = 0; i < 64; i++) emit_byte(m_quantization_tables[0][i]);
        emit_byte(1); for (int i = 0; i < 64; i++) emit_byte(m_quantization_tables[1][i]);
    }

    void emit_sof() {
        emit_marker(0xC0); emit_word(8 + m_num_components * 3);
        emit_byte(8); emit_word(m_image_height); emit_word(m_image_width);
        emit_byte(m_num_components);
        emit_byte(1); emit_byte(0x22); emit_byte(0); // Y: 2x2 sampling, quant table 0
        if (m_num_components == 3) {
            emit_byte(2); emit_byte(0x11); emit_byte(1); // Cb
            emit_byte(3); emit_byte(0x11); emit_byte(1); // Cr
        }
    }

    void emit_dht(const uint8* bits, const uint8* val, int index, bool ac) {
        int len = 0; for (int i = 1; i <= 16; i++) len += bits[i];
        emit_marker(0xC4); emit_word(2 + 1 + 16 + len);
        emit_byte((ac ? 0x10 : 0) | index);
        for (int i = 1; i <= 16; i++) emit_byte(bits[i]);
        for (int i = 0; i < len; i++) emit_byte(val[i]);
    }

    void emit_dhts() {
        emit_dht(s_dc_lum_bits, s_dc_lum_val, 0, false);
        emit_dht(s_ac_lum_bits, s_ac_lum_val, 0, true);
        emit_dht(s_dc_chroma_bits, s_dc_chroma_val, 1, false);
        emit_dht(s_ac_chroma_bits, s_ac_chroma_val, 1, true);
    }

    void emit_sos() {
        emit_marker(0xDA); emit_word(6 + m_num_components * 2);
        emit_byte(m_num_components);
        emit_byte(1); emit_byte(0x00); // Y: DC table 0, AC table 0
        if (m_num_components == 3) {
            emit_byte(2); emit_byte(0x11); // Cb: DC table 1, AC table 1
            emit_byte(3); emit_byte(0x11); // Cr
        }
        emit_byte(0); emit_byte(63); emit_byte(0);
    }

    void put_bits(uint32 bits, int len) {
        m_bit_buffer |= (bits << (32 - m_bits_in - len)); m_bits_in += len;
        while (m_bits_in >= 8) {
            uint8 c = (uint8)(m_bit_buffer >> 24);
            emit_byte(c); if (c == 0xFF) emit_byte(0);
            m_bit_buffer <<= 8; m_bits_in -= 8;
        }
    }

    void put_signed_int_bits(int num, int len) {
        if (num < 0) num += (1 << len) - 1;
        put_bits(num, len);
    }

    void code_block(dct_t* p, huffman_table* dc_ht, huffman_table* ac_ht, int comp) {
        // Simple DCT and encoding (simplified for brevity)
        int dc = (p[0] + 4) >> 3;
        int diff = dc - m_last_dc[comp]; m_last_dc[comp] = dc;
        int nbits = 0, t = diff < 0 ? -diff : diff;
        while (t) { nbits++; t >>= 1; }
        put_bits(dc_ht->m_codes[nbits], dc_ht->m_code_sizes[nbits]);
        if (nbits) put_signed_int_bits(diff, nbits);
        // AC coefficients - emit EOB for simplicity
        put_bits(ac_ht->m_codes[0], ac_ht->m_code_sizes[0]);
    }

    void flush_bits() {
        if (m_bits_in) { put_bits(0x7F, 8 - m_bits_in); m_bits_in = 0; }
    }

    bool encode() {
        // JFIF header
        emit_marker(0xD8); // SOI
        emit_marker(0xE0); emit_word(16); // APP0
        emit_byte('J'); emit_byte('F'); emit_byte('I'); emit_byte('F'); emit_byte(0);
        emit_byte(1); emit_byte(1); emit_byte(0); emit_word(1); emit_word(1);
        emit_byte(0); emit_byte(0);

        emit_dqt(); emit_sof(); emit_dhts(); emit_sos();

        m_bit_buffer = 0; m_bits_in = 0;
        m_last_dc[0] = m_last_dc[1] = m_last_dc[2] = 0;

        // Process image in 16x16 MCUs (for 4:2:0)
        int mcu_w = (m_image_width + 15) >> 4;
        int mcu_h = (m_image_height + 15) >> 4;

        for (int mcu_y = 0; mcu_y < mcu_h; mcu_y++) {
            for (int mcu_x = 0; mcu_x < mcu_w; mcu_x++) {
                // Sample and encode MCU (simplified - just DC values)
                dct_t block[64] = {0};
                int px = mcu_x * 16, py = mcu_y * 16;

                // 4 Y blocks
                for (int by = 0; by < 2; by++) {
                    for (int bx = 0; bx < 2; bx++) {
                        int sum = 0, cnt = 0;
                        for (int y = 0; y < 8; y++) {
                            for (int x = 0; x < 8; x++) {
                                int sx = px + bx * 8 + x, sy = py + by * 8 + y;
                                if (sx < m_image_width && sy < m_image_height) {
                                    const uint8* p = m_pSrc + (sy * m_image_width + sx) * 3;
                                    sum += (int)(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
                                    cnt++;
                                }
                            }
                        }
                        block[0] = cnt ? (sum / cnt - 128) * 8 : 0;
                        code_block(block, &m_huff_dc[0], &m_huff_ac[0], 0);
                    }
                }
                // Cb block
                int cb_sum = 0, cr_sum = 0, cnt = 0;
                for (int y = 0; y < 16; y++) {
                    for (int x = 0; x < 16; x++) {
                        int sx = px + x, sy = py + y;
                        if (sx < m_image_width && sy < m_image_height) {
                            const uint8* p = m_pSrc + (sy * m_image_width + sx) * 3;
                            cb_sum += (int)(-0.169f * p[0] - 0.331f * p[1] + 0.500f * p[2]);
                            cr_sum += (int)(0.500f * p[0] - 0.419f * p[1] - 0.081f * p[2]);
                            cnt++;
                        }
                    }
                }
                block[0] = cnt ? (cb_sum / cnt) * 8 : 0;
                code_block(block, &m_huff_dc[1], &m_huff_ac[1], 1);
                block[0] = cnt ? (cr_sum / cnt) * 8 : 0;
                code_block(block, &m_huff_dc[1], &m_huff_ac[1], 2);
            }
        }
        flush_bits();
        emit_marker(0xD9); // EOI
        return true;
    }
};

bool compress_image_to_jpeg_file_in_memory(void* pBuf, int& buf_size, int width, int height,
    int num_channels, const uint8* pImage_data, const params& comp_params)
{
    if (!pBuf || !pImage_data || width < 1 || height < 1 || num_channels < 1 || num_channels > 3)
        return false;

    output_stream stream(pBuf, buf_size);
    jpeg_encoder enc;
    enc.m_pStream = &stream;
    enc.m_image_width = width;
    enc.m_image_height = height;
    enc.m_num_components = (num_channels == 1) ? 1 : 3;
    enc.m_pSrc = pImage_data;
    enc.m_quality = JPGE_MIN(JPGE_MAX(comp_params.m_quality, 1), 100);

    // Compute quantization tables
    int q = enc.m_quality < 50 ? 5000 / enc.m_quality : 200 - enc.m_quality * 2;
    enc.m_quality = q;
    enc.compute_quant_table(enc.m_quantization_tables[0], s_std_lum_quant);
    enc.compute_quant_table(enc.m_quantization_tables[1], s_std_croma_quant);

    // Compute Huffman tables
    enc.m_huff_dc[0].compute(s_dc_lum_bits, s_dc_lum_val);
    enc.m_huff_ac[0].compute(s_ac_lum_bits, s_ac_lum_val);
    enc.m_huff_dc[1].compute(s_dc_chroma_bits, s_dc_chroma_val);
    enc.m_huff_ac[1].compute(s_ac_chroma_bits, s_ac_chroma_val);

    if (!enc.encode()) return false;
    buf_size = stream.m_pos;
    return true;
}

} // namespace jpge
