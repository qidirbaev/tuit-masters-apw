/*
 * Bilateral image filter — OpenCL C 1.2
 *
 * Execution model:
 *   one OpenCL work-item computes one output pixel;
 *   get_global_id(0/1) gives the pixel coordinate;
 *   work-items are grouped into 2-D work-groups by the NDRange local size.
 *
 * The host pads the global range to a multiple of the local range, so this
 * kernel must guard both dimensions against the true image width/height.
 */
__kernel void bilateral_filter(
    __global const uchar *input_rgba,
    __global uchar *output_rgba,
    const uint width,
    const uint height,
    const uint radius,
    const float sigma_space_squared,
    const float sigma_color_squared)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);

    if (x >= (int)width || y >= (int)height) {
        return;
    }

    const int center_index = (y * (int)width + x) * 4;
    const float center_r = (float)input_rgba[center_index + 0];
    const float center_g = (float)input_rgba[center_index + 1];
    const float center_b = (float)input_rgba[center_index + 2];

    float weighted_r = 0.0f;
    float weighted_g = 0.0f;
    float weighted_b = 0.0f;
    float total_weight = 0.0f;

    for (int dy = -(int)radius; dy <= (int)radius; ++dy) {
        const int sample_y = clamp(y + dy, 0, (int)height - 1);

        for (int dx = -(int)radius; dx <= (int)radius; ++dx) {
            const int sample_x = clamp(x + dx, 0, (int)width - 1);
            const int sample_index = (sample_y * (int)width + sample_x) * 4;

            const float sample_r = (float)input_rgba[sample_index + 0];
            const float sample_g = (float)input_rgba[sample_index + 1];
            const float sample_b = (float)input_rgba[sample_index + 2];

            const float spatial_distance_squared = (float)(dx * dx + dy * dy);
            const float dr = sample_r - center_r;
            const float dg = sample_g - center_g;
            const float db = sample_b - center_b;
            const float color_distance_squared = dr * dr + dg * dg + db * db;

            const float exponent =
                -(spatial_distance_squared / sigma_space_squared)
                -(color_distance_squared / sigma_color_squared);
            const float weight = exp(exponent);

            weighted_r += sample_r * weight;
            weighted_g += sample_g * weight;
            weighted_b += sample_b * weight;
            total_weight += weight;
        }
    }

    /* Positive float -> uchar truncation intentionally mirrors Uint8Array. */
    output_rgba[center_index + 0] = (uchar)clamp(weighted_r / total_weight, 0.0f, 255.0f);
    output_rgba[center_index + 1] = (uchar)clamp(weighted_g / total_weight, 0.0f, 255.0f);
    output_rgba[center_index + 2] = (uchar)clamp(weighted_b / total_weight, 0.0f, 255.0f);
    output_rgba[center_index + 3] = (uchar)255;
}
