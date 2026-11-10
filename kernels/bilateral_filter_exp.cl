/*
 * ================================================================
 * BILATERAL IMAGE FILTER — OpenCL C 1.2
 * ================================================================
 *
 * PURPOSE:
 *   Smooth an image while preserving important edges.
 *
 * INPUT:
 *   An RGBA image represented as a one-dimensional array of bytes.
 *
 * OUTPUT:
 *   A filtered RGBA image, also represented as a one-dimensional
 *   array of bytes.
 *
 * EXECUTION MODEL:
 *   One OpenCL work-item calculates exactly one output pixel.
 *
 *   For example:
 *
 *       Work-item (0, 0) -> output pixel (0, 0)
 *       Work-item (1, 0) -> output pixel (1, 0)
 *       Work-item (2, 0) -> output pixel (2, 0)
 *       ...
 *
 *   The GPU executes many work-items in parallel.
 *
 *   Work-items are organized into work-groups.
 */


/*
 * __kernel
 *
 * Indicates that this function is an OpenCL kernel.
 *
 * A kernel is a special function that the host program (CPU)
 * can launch on an OpenCL device, such as a GPU.
 *
 * Unlike an ordinary C function, the host does not directly call
 * this function. It sets its arguments and launches an NDRange.
 *
 * bilateral_filter
 *
 * The name of our kernel. The host program uses this name to
 * retrieve and execute the compiled kernel.
 */
__kernel void bilateral_filter(

    /*
     * __global
     *
     * Specifies the OpenCL global memory address space.
     *
     * Global memory is accessible by all work-items executing
     * this kernel.
     *
     * const
     *
     * Means this kernel must not modify the input data.
     *
     * uchar
     *
     * An unsigned 8-bit integer (0 to 255).
     *
     * Each image channel occupies one byte.
     *
     * For an RGBA image:
     *
     *   R = Red
     *   G = Green
     *   B = Blue
     *   A = Alpha (transparency)
     *
     * One pixel therefore occupies 4 bytes.
     *
     * The pointer points to the first byte of the image.
     */
    __global const uchar *input_rgba,


    /*
     * Output image buffer.
     *
     * Unlike the input pointer, this pointer is not const,
     * because the kernel must write its calculated pixels.
     *
     * Each work-item writes its own output pixel.
     */
    __global uchar *output_rgba,


    /*
     * Image width in pixels.
     *
     * uint = unsigned 32-bit integer.
     *
     * Example:
     *   width = 1920
     */
    const uint width,


    /*
     * Image height in pixels.
     *
     * Example:
     *   height = 1080
     */
    const uint height,


    /*
     * Neighborhood radius.
     *
     * Determines how many surrounding pixels are examined.
     *
     * radius = 1 -> 3 x 3 neighborhood
     * radius = 2 -> 5 x 5 neighborhood
     * radius = 3 -> 7 x 7 neighborhood
     *
     * General formula:
     *
     *   neighborhood_size = (2 * radius + 1)^2
     */
    const uint radius,


    /*
     * Spatial Gaussian parameter.
     *
     * Controls how strongly the distance between pixels
     * affects their contribution to the result.
     *
     * A larger value means distant pixels can contribute
     * more strongly.
     *
     * It is called sigma_space_squared because it represents
     * the denominator used with spatial distance squared.
     */
    const float sigma_space_squared,


    /*
     * Color Gaussian parameter.
     *
     * Controls how strongly color differences affect
     * the contribution of neighboring pixels.
     *
     * A larger value means pixels with different colors
     * can contribute more strongly.
     *
     * It is the denominator used with color distance squared.
     */
    const float sigma_color_squared
)
{
    /*
     * ============================================================
     * STEP 1: IDENTIFY THE PIXEL THIS WORK-ITEM MUST PROCESS
     * ============================================================
     */

    /*
     * get_global_id(0)
     *
     * Returns the global ID of the current work-item
     * along dimension 0.
     *
     * We use dimension 0 as the horizontal image coordinate (x).
     *
     * Example:
     *   First column  -> x = 0
     *   Second column -> x = 1
     *   Third column  -> x = 2
     *
     * get_global_id returns size_t, which is an unsigned
     * integer type capable of representing global IDs.
     *
     * We convert it to int because we perform signed
     * coordinate calculations later.
     */
    const int x = (int)get_global_id(0);


    /*
     * get_global_id(1)
     *
     * Returns the global ID along dimension 1.
     *
     * We use dimension 1 as the vertical image coordinate (y).
     *
     * Example:
     *   First row  -> y = 0
     *   Second row -> y = 1
     *   Third row  -> y = 2
     */
    const int y = (int)get_global_id(1);


    /*
     * ============================================================
     * STEP 2: CHECK WHETHER THIS WORK-ITEM IS INSIDE THE IMAGE
     * ============================================================
     */

    /*
     * The host program may round the global work size up
     * to a multiple of the local work size.
     *
     * Therefore, some work-items may have coordinates
     * outside the actual image.
     *
     * Example:
     *
     *   Actual image width = 1000
     *   Global work size   = 1024
     *
     * Work-items with x = 1000, 1001, ... 1023 are unnecessary.
     *
     * The same situation can occur vertically.
     *
     * If the current work-item is outside the image,
     * terminate its execution immediately.
     *
     * return;
     *
     * Ends execution of the current work-item only.
     * It does NOT stop other work-items or the entire kernel.
     */
    if (x >= (int)width || y >= (int)height) {
        return;
    }


    /*
     * ============================================================
     * STEP 3: CALCULATE THE INPUT BUFFER INDEX OF THE PIXEL
     * ============================================================
     */

    /*
     * Images are logically two-dimensional, but our RGBA
     * image is stored as a one-dimensional byte array.
     *
     * To access pixel (x, y), we first calculate its pixel index:
     *
     *   pixel_index = y * width + x
     *
     * Each pixel contains 4 channels:
     *
     *   R -> offset 0
     *   G -> offset 1
     *   B -> offset 2
     *   A -> offset 3
     *
     * Therefore, the byte index is:
     *
     *   center_index = (y * width + x) * 4
     *
     * Example:
     *
     *   width = 100
     *   x = 5
     *   y = 3
     *
     *   pixel_index  = 3 * 100 + 5 = 305
     *   center_index = 305 * 4 = 1220
     *
     * So:
     *
     *   input_rgba[1220] -> Red
     *   input_rgba[1221] -> Green
     *   input_rgba[1222] -> Blue
     *   input_rgba[1223] -> Alpha
     */
    const int center_index = (y * (int)width + x) * 4;


    /*
     * ============================================================
     * STEP 4: READ THE CENTER PIXEL'S COLOR
     * ============================================================
     */

    /*
     * The center pixel is the pixel currently being processed.
     *
     * We read its RGB components from global memory.
     *
     * uchar -> float conversion is necessary because:
     *
     *   1. Bilateral filtering requires floating-point calculations.
     *   2. Gaussian weights are floating-point numbers.
     *   3. Weighted color sums need floating-point precision.
     *
     * These values will be used to compare neighboring
     * pixels with the center pixel.
     */

    const float center_r = (float)input_rgba[center_index + 0];

    const float center_g = (float)input_rgba[center_index + 1];

    const float center_b = (float)input_rgba[center_index + 2];


    /*
     * ============================================================
     * STEP 5: INITIALIZE WEIGHTED ACCUMULATORS
     * ============================================================
     */

    /*
     * These variables accumulate the weighted color values
     * of all pixels in the neighborhood.
     *
     * Initially, they are zero because no neighboring pixels
     * have been processed yet.
     *
     * weighted_r = sum(sample_r * weight)
     * weighted_g = sum(sample_g * weight)
     * weighted_b = sum(sample_b * weight)
     */
    float weighted_r = 0.0f;
    float weighted_g = 0.0f;
    float weighted_b = 0.0f;


    /*
     * Stores the sum of all calculated weights.
     *
     * total_weight = sum(weight)
     *
     * We need this value to normalize the weighted color sums.
     */
    float total_weight = 0.0f;


    /*
     * ============================================================
     * STEP 6: ITERATE THROUGH THE PIXEL NEIGHBORHOOD
     * ============================================================
     */

    /*
     * Outer loop: vertical neighborhood coordinate.
     *
     * We start at -radius and finish at +radius.
     *
     * Example, radius = 2:
     *
     *   dy = -2, -1, 0, 1, 2
     *
     * This means we examine two rows above the current pixel,
     * the current row, and two rows below it.
     *
     * Notice that dy is a signed integer.
     * Negative values are necessary to access pixels above
     * or to the left of the center pixel.
     */
    for (int dy = -(int)radius; dy <= (int)radius; ++dy) {


        /*
         * Calculate the actual vertical coordinate of the
         * neighboring pixel.
         *
         * sample_y = y + dy
         *
         * However, this coordinate may go outside the image.
         *
         * Example:
         *
         *   y = 0
         *   dy = -1
         *
         *   sample_y = -1
         *
         * There is no row -1 in the image.
         *
         * clamp(value, minimum, maximum)
         *
         * Restricts a value to a specified range.
         *
         * Here:
         *
         *   Minimum = 0
         *   Maximum = height - 1
         *
         * This implements border replication.
         *
         * For example:
         *
         *   sample_y = -1 -> 0
         *   sample_y = -2 -> 0
         *   sample_y = height -> height - 1
         *
         * In other words, coordinates outside the image
         * are replaced by the closest valid border coordinate.
         */
        const int sample_y = clamp(y + dy, 0, (int)height - 1);


        /*
         * Inner loop: horizontal neighborhood coordinate.
         *
         * We examine pixels from:
         *
         *   x - radius
         *
         * to:
         *
         *   x + radius
         *
         * Together, the two loops create a square neighborhood.
         *
         * For radius = 1:
         *
         *   3 x 3 = 9 pixels
         *
         * For radius = 2:
         *
         *   5 x 5 = 25 pixels
         *
         * For radius = 3:
         *
         *   7 x 7 = 49 pixels
         */
        for (int dx = -(int)radius; dx <= (int)radius; ++dx) {


            /*
             * Calculate the horizontal coordinate of
             * the neighboring pixel.
             *
             * sample_x = x + dx
             *
             * clamp() guarantees that the coordinate
             * remains inside the image.
             *
             * This uses the same border-replication strategy
             * as sample_y.
             */
            const int sample_x = clamp(x + dx, 0, (int)width - 1);


            /*
             * Convert the neighboring pixel's 2D position
             * into a 1D RGBA buffer index.
             *
             * Formula:
             *
             *   sample_index = (sample_y * width + sample_x) * 4
             *
             * Now we can read the neighboring pixel's
             * color channels from the input buffer.
             */
            const int sample_index = (sample_y * (int)width + sample_x) * 4;


            /*
             * ====================================================
             * STEP 7: READ THE NEIGHBORING PIXEL'S COLOR
             * ====================================================
             *
             * Read the RGB values of the neighboring pixel.
             *
             * Convert each 8-bit unsigned integer into float.
             *
             * We need these values to calculate the color
             * difference between this pixel and the center pixel.
             */

            const float sample_r = (float)input_rgba[sample_index + 0];

            const float sample_g = (float)input_rgba[sample_index + 1];

            const float sample_b = (float)input_rgba[sample_index + 2];


            /*
             * ====================================================
             * STEP 8: CALCULATE SPATIAL DISTANCE
             * ====================================================
             */

            /*
             * Spatial distance measures how far the neighboring
             * pixel is from the center pixel in image coordinates.
             *
             * We calculate squared Euclidean distance:
             *
             *   d_space^2 = dx^2 + dy^2
             *
             * Examples:
             *
             *   Center pixel:
             *       dx = 0, dy = 0
             *       distance squared = 0
             *
             *   Direct horizontal neighbor:
             *       dx = 1, dy = 0
             *       distance squared = 1
             *
             *   Diagonal neighbor:
             *       dx = 1, dy = 1
             *       distance squared = 2
             *
             * The square root is unnecessary because
             * the Gaussian function uses squared distance.
             */
            const float spatial_distance_squared =
                (float)(dx * dx + dy * dy);


            /*
             * ====================================================
             * STEP 9: CALCULATE COLOR DISTANCE
             * ====================================================
             */

            /*
             * Calculate the difference between the center pixel
             * and the neighboring pixel for each RGB channel.
             *
             * A positive value means the neighboring channel
             * is brighter than the center channel.
             *
             * A negative value means it is darker.
             *
             * The sign does not matter in the final squared
             * Euclidean distance.
             */

            const float dr = sample_r - center_r;
            const float dg = sample_g - center_g;
            const float db = sample_b - center_b;


            /*
             * Calculate squared RGB color distance:
             *
             *   d_color^2 = dr^2 + dg^2 + db^2
             *
             * Example:
             *
             *   Center pixel = (100, 100, 100)
             *   Sample pixel = (110, 100, 100)
             *
             *   dr = 10
             *   dg = 0
             *   db = 0
             *
             *   Color distance squared = 100
             *
             * A small color distance means the two pixels
             * have similar colors.
             *
             * A large color distance means their colors
             * are very different.
             *
             * Notice that this implementation uses ordinary
             * Euclidean distance in RGB space, not a perceptually
             * uniform color space such as CIELAB.
             */
            const float color_distance_squared =
                dr * dr + dg * dg + db * db;


            /*
             * ====================================================
             * STEP 10: CALCULATE THE BILATERAL FILTER WEIGHT
             * ====================================================
             */

            /*
             * The bilateral filter combines two Gaussian influences:
             *
             *   1. Spatial influence
             *   2. Color influence
             *
             * The exponent is:
             *
             *   exponent =
             *       -(d_space^2 / sigma_space^2)
             *       -(d_color^2 / sigma_color^2)
             *
             * A more explicit mathematical form:
             *
             *   exponent = -(
             *       spatial_distance_squared / sigma_space_squared
             *       +
             *       color_distance_squared / sigma_color_squared
             *   )
             *
             * Why the negative sign?
             *
             * Because the Gaussian function decreases as
             * the distance increases.
             *
             * Small distance:
             *   exponent is close to zero.
             *   Weight is close to 1.
             *
             * Large distance:
             *   exponent becomes more negative.
             *   Weight approaches zero.
             *
             * This means:
             *
             *   - Nearby pixels receive higher spatial influence.
             *   - Similar-colored pixels receive higher color influence.
             *
             * IMPORTANT:
             *
             * This implementation does not include the conventional
             * factor 2 in the Gaussian denominator:
             *
             *   exp(-d^2 / (2 * sigma^2))
             *
             * Instead, the host must provide denominators matching:
             *
             *   exp(-d^2 / sigma_squared_parameter)
             *
             * Both conventions are valid if the host and kernel
             * agree on the parameter definition.
             */
            const float exponent =
                -(spatial_distance_squared / sigma_space_squared)
                -(color_distance_squared / sigma_color_squared);


            /*
             * exp(exponent)
             *
             * Calculates the exponential function:
             *
             *   e ^ exponent
             *
             * where e is approximately 2.71828.
             *
             * This produces the final bilateral weight.
             *
             * The weight is normally between 0 and 1.
             *
             * Example:
             *
             *   exponent = 0    -> weight = 1
             *   exponent = -1   -> weight ≈ 0.3679
             *   exponent = -2   -> weight ≈ 0.1353
             *   exponent = -5   -> weight ≈ 0.0067
             *
             * The center pixel always has:
             *
             *   dx = 0
             *   dy = 0
             *   dr = 0
             *   dg = 0
             *   db = 0
             *
             * Therefore:
             *
             *   exponent = 0
             *   weight = 1
             *
             * This also ensures that total_weight is positive,
             * assuming the sigma parameters are valid.
             */
            const float weight = exp(exponent);


            /*
             * ====================================================
             * STEP 11: ACCUMULATE WEIGHTED COLOR VALUES
             * ====================================================
             */

            /*
             * Multiply each neighboring color channel
             * by its calculated weight.
             *
             * Then add the result to the corresponding accumulator.
             *
             * Mathematically:
             *
             *   weighted_r += sample_r * weight
             *   weighted_g += sample_g * weight
             *   weighted_b += sample_b * weight
             *
             * Pixels with high weights contribute more.
             *
             * Pixels with low weights contribute less.
             */
            weighted_r += sample_r * weight;
            weighted_g += sample_g * weight;
            weighted_b += sample_b * weight;


            /*
             * Add the current weight to the total weight.
             *
             * This is necessary for normalization.
             *
             * After all neighborhood pixels are processed:
             *
             *   total_weight = sum(all neighborhood weights)
             */
            total_weight += weight;
        }
    }


    /*
     * ================================================================
     * STEP 12: NORMALIZE AND WRITE THE OUTPUT PIXEL
     * ================================================================
     */

    /*
     * The weighted color accumulators contain sums, not final
     * pixel values.
     *
     * To calculate the weighted average, divide each sum
     * by the total weight:
     *
     *   output_red = weighted_r / total_weight
     *
     *   output_green = weighted_g / total_weight
     *
     *   output_blue = weighted_b / total_weight
     *
     * Why normalization?
     *
     * Without normalization, the output would become dependent
     * on the number and magnitude of accumulated weights.
     *
     * Dividing by total_weight creates a weighted average.
     *
     *
     * clamp(value, 0.0f, 255.0f)
     *
     * Ensures that the result is a valid 8-bit color value.
     *
     * Floating-point to uchar conversion:
     *
     *   (uchar)123.9f -> 123
     *
     * Positive fractional values are truncated, not rounded.
     *
     * Finally, the result is written into the output image.
     */

    output_rgba[center_index + 0] =
        (uchar)clamp(weighted_r / total_weight, 0.0f, 255.0f);

    output_rgba[center_index + 1] =
        (uchar)clamp(weighted_g / total_weight, 0.0f, 255.0f);

    output_rgba[center_index + 2] =
        (uchar)clamp(weighted_b / total_weight, 0.0f, 255.0f);


    /*
     * Set the output alpha channel to 255.
     *
     * Alpha = 255 means fully opaque.
     *
     * This implementation does not filter or preserve
     * the input alpha channel.
     *
     * If the input image contains transparency, this operation
     * makes every output pixel fully opaque.
     */
    output_rgba[center_index + 3] = (uchar)255;
}