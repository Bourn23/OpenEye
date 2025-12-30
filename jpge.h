// jpge.h - Public Domain JPEG Encoder
// Rich Geldreich <richgel99@gmail.com>
// Minimal single-header JPEG encoder

#ifndef JPGE_H
#define JPGE_H

namespace jpge
{
    typedef unsigned char uint8;
    typedef signed short int16;
    typedef signed int int32;
    typedef unsigned short uint16;
    typedef unsigned int uint32;

    // JPEG subsampling factors
    enum subsampling_t { Y_ONLY = 0, H1V1 = 1, H2V1 = 2, H2V2 = 3 };

    // Compression parameters
    struct params
    {
        int m_quality;              // 1-100
        subsampling_t m_subsampling;
        bool m_no_chroma_discrim_flag;
        bool m_two_pass_flag;

        params() : m_quality(85), m_subsampling(H2V2), 
                   m_no_chroma_discrim_flag(false), m_two_pass_flag(false) {}
    };

    // Compress image to JPEG in memory
    // Returns true on success
    // pBuf_size should be set to buffer size on input, actual size on output
    bool compress_image_to_jpeg_file_in_memory(
        void* pBuf, int& buf_size,
        int width, int height, int num_channels,
        const uint8* pImage_data,
        const params& comp_params = params());

} // namespace jpge

#endif // JPGE_H
