// lumen-plate-kit driver: computes the BT.601 full-range gain map of each
// plate pair with this libavif build, applies it at exposures 0, 1 and 2.5,
// and emits a JSON manifest plus PNG assets consumed by view/index.html.
//
// Timing convention (mirrors the benchmark specification): one warm-up run is
// discarded, then four timed runs are performed; the reported interval is
// [min, max] over those four runs, measured in CPU time (user + kernel).

#include "avif/avif.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <arpa/inet.h>
#include <time.h>
#include <zlib.h>

#define NUM_TIMED_RUNS 4
#define NUM_EXPOSURES 3

typedef struct PlateConfig
{
    const char * name;
    const char * basePath;
    const char * brightPath;
    uint32_t width;
    uint32_t height;
    uint32_t rowBytes;
    // CPU time budgets, in milliseconds.
    double computeBudgetMs;
    double applyExposure0BudgetMs;
    double applyExposure1BudgetMs;
} PlateConfig;

typedef struct TimedResult
{
    double minMs;
    double maxMs;
} TimedResult;

static double cpuTimeMs(void)
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
    const double userMs = (double)usage.ru_utime.tv_sec * 1000.0 + (double)usage.ru_utime.tv_usec / 1000.0;
    const double systemMs = (double)usage.ru_stime.tv_sec * 1000.0 + (double)usage.ru_stime.tv_usec / 1000.0;
    return userMs + systemMs;
}

static uint8_t * readFile(const char * path, size_t * sizeOut)
{
    FILE * file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long size = ftell(file);
    if (size < 0) {
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    uint8_t * buffer = (uint8_t *)malloc((size_t)size);
    if (buffer == NULL) {
        fclose(file);
        return NULL;
    }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *sizeOut = (size_t)size;
    return buffer;
}

static avifRGBImage makeRgbaView(uint8_t * pixels, uint32_t width, uint32_t height, uint32_t rowBytes)
{
    avifRGBImage rgb;
    memset(&rgb, 0, sizeof(rgb));
    rgb.width = width;
    rgb.height = height;
    rgb.depth = 8;
    rgb.format = AVIF_RGB_FORMAT_RGBA;
    rgb.pixels = pixels;
    rgb.rowBytes = rowBytes;
    rgb.maxThreads = 1;
    return rgb;
}

// Writes an 8-bit grayscale or RGBA PNG using zlib only (no external imaging
// library). colorType 0 = grayscale, 6 = RGBA.
static int writePng(const char * path, const uint8_t * pixels, uint32_t width, uint32_t height, uint32_t rowBytes, int colorType)
{
    const int channels = (colorType == 0) ? 1 : 4;
    const int bitDepth = 8;

    uLongf rawSize = (uLongf)height * ((uLongf)width * channels + 1);
    uint8_t * raw = (uint8_t *)malloc(rawSize);
    if (raw == NULL) {
        return 0;
    }
    for (uint32_t y = 0; y < height; ++y) {
        uint8_t * dstRow = &raw[(size_t)y * ((size_t)width * channels + 1)];
        dstRow[0] = 0; // filter type: none
        const uint8_t * srcRow = &pixels[(size_t)y * rowBytes];
        if (channels == 4) {
            memcpy(&dstRow[1], srcRow, (size_t)width * 4);
        } else {
            for (uint32_t x = 0; x < width; ++x) {
                dstRow[1 + x] = srcRow[x];
            }
        }
    }

    uLongf compressedBound = compressBound(rawSize);
    uint8_t * compressed = (uint8_t *)malloc(compressedBound);
    if (compressed == NULL) {
        free(raw);
        return 0;
    }
    uLongf compressedSize = compressedBound;
    if (compress2(compressed, &compressedSize, raw, rawSize, Z_BEST_SPEED) != Z_OK) {
        free(raw);
        free(compressed);
        return 0;
    }
    free(raw);

    FILE * file = fopen(path, "wb");
    if (file == NULL) {
        free(compressed);
        return 0;
    }

    uint32_t crc = 0;
    #define WRITE_BYTES(data, length) \
        do { \
            const size_t writeLength = (size_t)(length); \
            if (writeLength > 0 && (data) != NULL) { \
                if (fwrite((data), 1, writeLength, file) != writeLength) { \
                    fclose(file); \
                    free(compressed); \
                    return 0; \
                } \
            } \
        } while (0)
    #define PNG_CHUNK(type, data, length) \
        do { \
            uint8_t header[8]; \
            uint32_t beLength = htonl((uint32_t)(length)); \
            memcpy(&header[0], &beLength, 4); \
            memcpy(&header[4], (type), 4); \
            WRITE_BYTES(header, 8); \
            WRITE_BYTES((data), (length)); \
            crc = (uint32_t)crc32(0L, Z_NULL, 0); \
            crc = (uint32_t)crc32(crc, (const Bytef *)(type), 4); \
            if ((length) > 0 && (data) != NULL) { \
                crc = (uint32_t)crc32(crc, (const Bytef *)(data), (uInt)(length)); \
            } \
            uint32_t beCrc = htonl(crc); \
            uint8_t crcBytes[4]; \
            memcpy(crcBytes, &beCrc, 4); \
            WRITE_BYTES(crcBytes, 4); \
        } while (0)

    static const uint8_t pngSignature[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    WRITE_BYTES(pngSignature, 8);

    uint8_t ihdr[13];
    uint32_t beWidth = htonl(width);
    uint32_t beHeight = htonl(height);
    memcpy(&ihdr[0], &beWidth, 4);
    memcpy(&ihdr[4], &beHeight, 4);
    ihdr[8] = (uint8_t)bitDepth;
    ihdr[9] = (uint8_t)colorType;
    ihdr[10] = 0; // compression method
    ihdr[11] = 0; // filter method
    ihdr[12] = 0; // interlace method
    PNG_CHUNK("IHDR", ihdr, sizeof(ihdr));
    PNG_CHUNK("IDAT", compressed, compressedSize);
    static const uint8_t emptyChunk[1] = { 0 };
    PNG_CHUNK("IEND", emptyChunk, 0);

    #undef PNG_CHUNK
    #undef WRITE_BYTES

    free(compressed);
    if (fclose(file) != 0) {
        return 0;
    }
    return 1;
}

// Runs compute once, discarding the timing.
static avifResult warmUpCompute(const avifRGBImage * base, const avifRGBImage * bright, avifGainMap * gainMap, avifDiagnostics * diag)
{
    return avifRGBImageComputeGainMap(base,
                                      AVIF_COLOR_PRIMARIES_BT709,
                                      AVIF_TRANSFER_CHARACTERISTICS_SRGB,
                                      bright,
                                      AVIF_COLOR_PRIMARIES_BT709,
                                      AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084,
                                      gainMap,
                                      diag);
}

static TimedResult timeCompute(const avifRGBImage * base, const avifRGBImage * bright, avifGainMap * gainMap, avifDiagnostics * diag)
{
    TimedResult result = { 0.0, 0.0 };
    for (int run = 0; run < NUM_TIMED_RUNS; ++run) {
        const double start = cpuTimeMs();
        const avifResult res = warmUpCompute(base, bright, gainMap, diag);
        const double elapsed = cpuTimeMs() - start;
        if (res != AVIF_RESULT_OK) {
            fprintf(stderr, "compute failed: %s\n", diag->error);
            exit(1);
        }
        if (run == 0 || elapsed < result.minMs) {
            result.minMs = elapsed;
        }
        if (run == 0 || elapsed > result.maxMs) {
            result.maxMs = elapsed;
        }
    }
    return result;
}

static TimedResult timeApply(const avifRGBImage * base,
                             const avifGainMap * gainMap,
                             float exposure,
                             avifRGBImage * output,
                             avifContentLightLevelInformationBox * clli,
                             avifDiagnostics * diag)
{
    TimedResult result = { 0.0, 0.0 };
    for (int run = 0; run < NUM_TIMED_RUNS; ++run) {
        const double start = cpuTimeMs();
        const avifResult res = avifRGBImageApplyGainMap(base,
                                                        AVIF_COLOR_PRIMARIES_BT709,
                                                        AVIF_TRANSFER_CHARACTERISTICS_SRGB,
                                                        gainMap,
                                                        exposure,
                                                        AVIF_COLOR_PRIMARIES_BT709,
                                                        AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084,
                                                        output,
                                                        clli,
                                                        diag);
        const double elapsed = cpuTimeMs() - start;
        if (res != AVIF_RESULT_OK) {
            fprintf(stderr, "apply failed: %s\n", diag->error);
            exit(1);
        }
        avifRGBImageFreePixels(output);
        if (run == 0 || elapsed < result.minMs) {
            result.minMs = elapsed;
        }
        if (run == 0 || elapsed > result.maxMs) {
            result.maxMs = elapsed;
        }
    }
    return result;
}

// Produces the final (timing-independent) output pixels for one exposure.
static avifResult renderExposure(const avifRGBImage * base,
                                 const avifGainMap * gainMap,
                                 float exposure,
                                 avifRGBImage * output,
                                 avifContentLightLevelInformationBox * clli,
                                 avifDiagnostics * diag)
{
    return avifRGBImageApplyGainMap(base,
                                    AVIF_COLOR_PRIMARIES_BT709,
                                    AVIF_TRANSFER_CHARACTERISTICS_SRGB,
                                    gainMap,
                                    exposure,
                                    AVIF_COLOR_PRIMARIES_BT709,
                                    AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084,
                                    output,
                                    clli,
                                    diag);
}

typedef struct PlateArtifacts
{
    TimedResult computeTime;
    TimedResult applyTimes[NUM_EXPOSURES];
    uint16_t maxCll[NUM_EXPOSURES];
    uint16_t maxPall[NUM_EXPOSURES];
    char gainMapPaths[3][128];
    char previewPaths[NUM_EXPOSURES][128];
} PlateArtifacts;

static int processPlate(const PlateConfig * config, PlateArtifacts * artifacts, FILE * manifest)
{
    size_t baseSize = 0;
    size_t brightSize = 0;
    uint8_t * baseBytes = readFile(config->basePath, &baseSize);
    uint8_t * brightBytes = readFile(config->brightPath, &brightSize);
    if (baseBytes == NULL || brightBytes == NULL) {
        free(baseBytes);
        free(brightBytes);
        return 0;
    }
    const size_t expectedSize = (size_t)config->rowBytes * config->height;
    if (baseSize != expectedSize || brightSize != expectedSize) {
        fprintf(stderr, "unexpected size for %s\n", config->name);
        free(baseBytes);
        free(brightBytes);
        return 0;
    }

    avifRGBImage base = makeRgbaView(baseBytes, config->width, config->height, config->rowBytes);
    avifRGBImage bright = makeRgbaView(brightBytes, config->width, config->height, config->rowBytes);

    avifGainMap * gainMap = avifGainMapCreate();
    if (gainMap == NULL) {
        free(baseBytes);
        free(brightBytes);
        return 0;
    }
    gainMap->image = avifImageCreate(config->width, config->height, 8, AVIF_PIXEL_FORMAT_YUV444);
    gainMap->image->yuvRange = AVIF_RANGE_FULL;
    gainMap->image->matrixCoefficients = AVIF_MATRIX_COEFFICIENTS_BT601;
    gainMap->image->colorPrimaries = AVIF_COLOR_PRIMARIES_BT709;
    gainMap->image->transferCharacteristics = AVIF_TRANSFER_CHARACTERISTICS_SRGB;

    avifDiagnostics diag;
    avifDiagnosticsClearError(&diag);

    // One warm-up run, not counted.
    if (warmUpCompute(&base, &bright, gainMap, &diag) != AVIF_RESULT_OK) {
        fprintf(stderr, "warm-up compute failed: %s\n", diag.error);
        avifGainMapDestroy(gainMap);
        free(baseBytes);
        free(brightBytes);
        return 0;
    }

    // Four timed compute runs. The final gain map content comes from the last
    // timed run and is the data rendered below.
    artifacts->computeTime = timeCompute(&base, &bright, gainMap, &diag);

    static const float exposures[NUM_EXPOSURES] = { 0.0f, 1.0f, 2.5f };
    for (int e = 0; e < NUM_EXPOSURES; ++e) {
        avifRGBImage timedOutput = makeRgbaView(NULL, config->width, config->height, config->rowBytes);
        avifContentLightLevelInformationBox timedClli;
        memset(&timedClli, 0, sizeof(timedClli));
        artifacts->applyTimes[e] = timeApply(&base, gainMap, exposures[e], &timedOutput, &timedClli, &diag);
    }

    // Render the actual artifacts from the final gain map.
    for (int c = 0; c < 3; ++c) {
        snprintf(artifacts->gainMapPaths[c], sizeof(artifacts->gainMapPaths[c]), "view/%s-gainmap-%c.png", config->name, "YUV"[c]);
        const uint8_t * plane = avifImagePlane(gainMap->image, c);
        const uint32_t planeRowBytes = avifImagePlaneRowBytes(gainMap->image, c);
        if (!writePng(artifacts->gainMapPaths[c], plane, config->width, config->height, planeRowBytes, 0)) {
            fprintf(stderr, "failed to write %s\n", artifacts->gainMapPaths[c]);
            avifGainMapDestroy(gainMap);
            free(baseBytes);
            free(brightBytes);
            return 0;
        }
    }

    for (int e = 0; e < NUM_EXPOSURES; ++e) {
        avifRGBImage output = makeRgbaView(NULL, config->width, config->height, config->rowBytes);
        avifContentLightLevelInformationBox clli;
        memset(&clli, 0, sizeof(clli));
        if (renderExposure(&base, gainMap, exposures[e], &output, &clli, &diag) != AVIF_RESULT_OK) {
            fprintf(stderr, "render failed: %s\n", diag.error);
            avifRGBImageFreePixels(&output);
            avifGainMapDestroy(gainMap);
            free(baseBytes);
            free(brightBytes);
            return 0;
        }
        artifacts->maxCll[e] = clli.maxCLL;
        artifacts->maxPall[e] = clli.maxPALL;
        snprintf(artifacts->previewPaths[e], sizeof(artifacts->previewPaths[e]), "view/%s-exposure-%.1f.png", config->name, exposures[e]);
        if (!writePng(artifacts->previewPaths[e], output.pixels, config->width, config->height, output.rowBytes, 6)) {
            fprintf(stderr, "failed to write %s\n", artifacts->previewPaths[e]);
            avifRGBImageFreePixels(&output);
            avifGainMapDestroy(gainMap);
            free(baseBytes);
            free(brightBytes);
            return 0;
        }
        avifRGBImageFreePixels(&output);
    }

    avifGainMapDestroy(gainMap);
    free(baseBytes);
    free(brightBytes);

    fprintf(manifest,
            "  {\n"
            "    \"name\": \"%s\",\n"
            "    \"width\": %u,\n"
            "    \"height\": %u,\n"
            "    \"computeMs\": [%.3f, %.3f],\n"
            "    \"computeBudgetMs\": %.3f,\n"
            "    \"gainMap\": [\"%s\", \"%s\", \"%s\"],\n"
            "    \"previews\": [\n",
            config->name,
            config->width,
            config->height,
            artifacts->computeTime.minMs,
            artifacts->computeTime.maxMs,
            config->computeBudgetMs,
            artifacts->gainMapPaths[0],
            artifacts->gainMapPaths[1],
            artifacts->gainMapPaths[2]);
    for (int e = 0; e < NUM_EXPOSURES; ++e) {
        fprintf(manifest,
                "      {\"exposure\": %.1f, \"path\": \"%s\", \"ms\": [%.3f, %.3f], "
                "\"maxCLL\": %u, \"maxPALL\": %u, \"budgetMs\": %.3f}%s\n",
                exposures[e],
                artifacts->previewPaths[e],
                artifacts->applyTimes[e].minMs,
                artifacts->applyTimes[e].maxMs,
                (unsigned)artifacts->maxCll[e],
                (unsigned)artifacts->maxPall[e],
                e == 0 ? config->applyExposure0BudgetMs : (e == 1 ? config->applyExposure1BudgetMs : 0.0),
                e == NUM_EXPOSURES - 1 ? "" : ",");
    }
    fprintf(manifest, "    ]\n  }");
    return 1;
}

int main(void)
{
    static const PlateConfig plates[2] = {
        { "hero",
          "plates/hero-base.rgba",
          "plates/hero-bright.rgba",
          1280,
          720,
          5120,
          320.0,
          220.0,
          350.0 },
        { "icon",
          "plates/icon-base.rgba",
          "plates/icon-bright.rgba",
          640,
          360,
          2560,
          90.0,
          60.0,
          90.0 },
    };

    FILE * manifest = fopen("view/results.json", "wb");
    if (manifest == NULL) {
        fprintf(stderr, "cannot open view/results.json\n");
        return 1;
    }
    fprintf(manifest, "{\n  \"plates\": [\n");

    for (int i = 0; i < 2; ++i) {
        PlateArtifacts artifacts;
        memset(&artifacts, 0, sizeof(artifacts));
        if (!processPlate(&plates[i], &artifacts, manifest)) {
            fclose(manifest);
            return 1;
        }
        if (i == 0) {
            fprintf(manifest, ",\n");
        }
    }
    fprintf(manifest, "\n  ]\n}\n");
    fclose(manifest);
    printf("wrote view/results.json and PNG assets\n");
    return 0;
}
