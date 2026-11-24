// Copyright 2023 Google LLC
// SPDX-License-Identifier: BSD-2-Clause

#include "avif/internal.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

// NaN-safe clamp to [0, 1]. AVIF_CLAMP passes NaN through because IEEE 754
// comparisons with NaN always return false. fmaxf/fminf return the non-NaN
// argument per C99 §7.12.12, so this clamps NaN to 0.
static float avifNanSafeClamp(float val)
{
    return fminf(1.0f, fmaxf(0.0f, val));
}

static float avifSignedFractionToFloat(avifSignedFraction f)
{
    if (f.d == 0) {
        return 0.0f;
    }
    return (float)f.n / f.d;
}

static float avifUnsignedFractionToFloat(avifUnsignedFraction f)
{
    if (f.d == 0) {
        return 0.0f;
    }
    return (float)f.n / f.d;
}

// ---------------------------------------------------------------------------
// Apply a gain map.

// Returns a weight in [-1.0, 1.0] that represents how much the gain map should be applied.
static float avifGetGainMapWeight(float hdrHeadroom, const avifGainMap * gainMap)
{
    const float baseHdrHeadroom = avifUnsignedFractionToFloat(gainMap->baseHdrHeadroom);
    const float alternateHdrHeadroom = avifUnsignedFractionToFloat(gainMap->alternateHdrHeadroom);
    if (baseHdrHeadroom == alternateHdrHeadroom) {
        // Do not apply the gain map if the HDR headroom is the same.
        // This case is not handled in the specification and does not make practical sense.
        return 0.0f;
    }
    const float w = AVIF_CLAMP((hdrHeadroom - baseHdrHeadroom) / (alternateHdrHeadroom - baseHdrHeadroom), 0.0f, 1.0f);
    return (alternateHdrHeadroom < baseHdrHeadroom) ? -w : w;
}

// Linear interpolation between 'a' and 'b' (returns 'a' if w == 0.0f, returns 'b' if w == 1.0f).
static inline float lerp(float a, float b, float w)
{
    return (1.0f - w) * a + w * b;
}

#define SDR_WHITE_NITS 203.0f

// True when the RGB image stores color samples in a single byte per channel
// (i.e. 8-bit, not half/float and not the packed RGB_565 format). When this is
// true every channel value passed to a transfer function is one of exactly
// 256 possible normalized floats, which is what enables the 8-bit lookup
// table fast paths below.
static avifBool avifRGBImageIs8BitBytePlanar(const avifRGBImage * rgb)
{
    return rgb->depth == 8 && !rgb->isFloat && rgb->format != AVIF_RGB_FORMAT_RGB_565 && !avifRGBFormatIsGray(rgb->format);
}

// Local replica of the full-range 8-bit unorm quantization used by the
// generic RGB-to-YUV path in reformat.c (same avifRoundf expression, same
// clamp). Kept here so the direct YUV444 fast path stays bit-identical.
static int avifYToUNormFull8(const avifYUVColorSpaceInfo * info, float v)
{
    const int unorm = (int)avifRoundf(v * info->rangeY + info->biasY);
    return AVIF_CLAMP(unorm, 0, info->maxChannel);
}

static int avifUVToUNormFull8(const avifYUVColorSpaceInfo * info, float v)
{
    const int unorm = (int)avifRoundf(v * info->rangeUV + info->biasUV);
    return AVIF_CLAMP(unorm, 0, info->maxChannel);
}

// Fast path for the "weight == 0" branch of avifRGBImageApplyGainMap(). The
// output only depends on the base image (the gain map pixels are not used),
// so each output channel byte is a function of a single base byte and the
// whole transformation collapses into a 256-entry table.
//
// This reproduces the float arithmetic of the generic loop exactly, including
// the byte -> float -> byte round trip performed by
// avifGetRGBAPixel()/avifSetRGBAPixel(). It is only used when both images are
// 8-bit byte-planar and no color space matrix conversion is needed.
static avifResult avifApplyGainMapZeroWeight8Bit(const avifRGBImage * baseImage,
                                                  const avifRGBColorSpaceInfo * baseRGBInfo,
                                                  avifTransferFunction gammaToLinear,
                                                  avifTransferFunction linearToGamma,
                                                  const avifRGBColorSpaceInfo * toneMappedPixelRGBInfo,
                                                  avifRGBImage * toneMappedImage,
                                                  avifBool transferOrPrimariesDiffer)
{
    const uint32_t width = baseImage->width;
    const uint32_t height = baseImage->height;
    const avifBool baseHasAlpha = avifRGBFormatHasAlpha(baseImage->format);
    const avifBool outHasAlpha = avifRGBFormatHasAlpha(toneMappedImage->format);

    // Same table is valid for all three color channels.
    uint8_t outLUT[256];
    uint8_t alphaLUT[256];
    for (int v = 0; v <= baseRGBInfo->maxChannel; ++v) {
        const float normalized = (float)v / baseRGBInfo->maxChannelF;
        if (transferOrPrimariesDiffer) {
            outLUT[v] = (uint8_t)(0.5f + (avifNanSafeClamp(linearToGamma(gammaToLinear(normalized))) * toneMappedPixelRGBInfo->maxChannelF));
        } else {
            outLUT[v] = (uint8_t)(0.5f + (normalized * toneMappedPixelRGBInfo->maxChannelF));
        }
        // Replicate avifGetRGBAPixel() followed by avifSetRGBAPixel() for the
        // (untouched) alpha channel.
        alphaLUT[v] = (uint8_t)(0.5f + (normalized * toneMappedPixelRGBInfo->maxChannelF));
    }

    const uint32_t basePixelBytes = baseRGBInfo->pixelBytes;
    const uint32_t outPixelBytes = toneMappedPixelRGBInfo->pixelBytes;
    const size_t baseRowBytes = baseImage->rowBytes;
    const size_t outRowBytes = toneMappedImage->rowBytes;
    const uint8_t * basePixels = baseImage->pixels;
    uint8_t * outPixels = toneMappedImage->pixels;
    for (uint32_t j = 0; j < height; ++j) {
        const uint8_t * baseRow = &basePixels[j * baseRowBytes];
        uint8_t * outRow = &outPixels[j * outRowBytes];
        for (uint32_t i = 0; i < width; ++i) {
            const uint8_t * basePixel = &baseRow[i * basePixelBytes];
            uint8_t * outPixel = &outRow[i * outPixelBytes];
            outPixel[toneMappedPixelRGBInfo->offsetBytesR] = outLUT[basePixel[baseRGBInfo->offsetBytesR]];
            outPixel[toneMappedPixelRGBInfo->offsetBytesG] = outLUT[basePixel[baseRGBInfo->offsetBytesG]];
            outPixel[toneMappedPixelRGBInfo->offsetBytesB] = outLUT[basePixel[baseRGBInfo->offsetBytesB]];
            if (outHasAlpha) {
                if (baseHasAlpha) {
                    outPixel[toneMappedPixelRGBInfo->offsetBytesA] = alphaLUT[basePixel[baseRGBInfo->offsetBytesA]];
                } else {
                    // avifGetRGBAPixel() synthesizes an opaque alpha.
                    outPixel[toneMappedPixelRGBInfo->offsetBytesA] = (uint8_t)(0.5f + (1.0f * toneMappedPixelRGBInfo->maxChannelF));
                }
            }
        }
    }
    return AVIF_RESULT_OK;
}

// Fast path for avifRGBImageApplyGainMap() when the gain map actually has to
// be applied (weight != 0). Every input of the per-channel math is an 8-bit
// byte: the base image channel byte selects a precomputed linear value, and
// the gain map RGB byte selects a precomputed exp2() gain factor. The output
// tone mapped linear value and the final quantized output byte therefore each
// only depend on the (base byte, gain map byte) pair, which collapses into a
// 256x256 table per channel.
//
// All table entries are built with the exact same float expressions as the
// generic loop, so every produced sample is bit-identical. Used only for 8-bit
// byte-planar images without color space matrix conversions.
static avifResult avifApplyGainMapWeighted8Bit(const avifRGBImage * baseImage,
                                               const avifRGBColorSpaceInfo * baseRGBInfo,
                                               const avifRGBImage * gainMapRGB,
                                               const avifRGBColorSpaceInfo * gainMapRGBInfo,
                                               const avifGainMap * gainMap,
                                               float weight,
                                               avifTransferFunction gammaToLinear,
                                               avifTransferFunction linearToGamma,
                                               const avifRGBColorSpaceInfo * toneMappedPixelRGBInfo,
                                               avifRGBImage * toneMappedImage,
                                               float * outRgbMaxLinear,
                                               float * outRgbSumLinear,
                                               avifContentLightLevelInformationBox * clli,
                                               avifDiagnostics * diag)
{
    const uint32_t width = baseImage->width;
    const uint32_t height = baseImage->height;
    const avifBool baseHasAlpha = avifRGBFormatHasAlpha(baseImage->format);
    const avifBool outHasAlpha = avifRGBFormatHasAlpha(toneMappedImage->format);

    float gammaInv[3];
    float gainMapMin[3];
    float gainMapMax[3];
    float baseOffset[3];
    float alternateOffset[3];
    for (int c = 0; c < 3; ++c) {
        gammaInv[c] = 1.0f / avifUnsignedFractionToFloat(gainMap->gainMapGamma[c]);
        gainMapMin[c] = avifSignedFractionToFloat(gainMap->gainMapMin[c]);
        gainMapMax[c] = avifSignedFractionToFloat(gainMap->gainMapMax[c]);
        baseOffset[c] = avifSignedFractionToFloat(gainMap->baseOffset[c]);
        alternateOffset[c] = avifSignedFractionToFloat(gainMap->alternateOffset[c]);
    }

    // Base image byte -> linear value, shared by all color channels.
    float baseLinearLUT[256];
    for (int v = 0; v <= baseRGBInfo->maxChannel; ++v) {
        baseLinearLUT[v] = gammaToLinear((float)v / baseRGBInfo->maxChannelF);
    }

    // For each channel, gain map RGB byte -> exp2(gainMapLog2 * weight).
    float gainFactorLUT[3][256];
    for (int c = 0; c < 3; ++c) {
        for (int v = 0; v <= gainMapRGBInfo->maxChannel; ++v) {
            const float gainMapValue = (float)v / gainMapRGBInfo->maxChannelF;
            const float powGammaValue = powf(gainMapValue, gammaInv[c]);
            const float gainMapLog2 = lerp(gainMapMin[c], gainMapMax[c], powGammaValue);
            gainFactorLUT[c][v] = exp2f(gainMapLog2 * weight);
        }
    }

    // Per channel, (base byte, gain map byte) -> tone mapped linear value and
    // final quantized output byte. Entries are computed lazily: only pairs
    // actually present in the image are evaluated.
    const size_t pairCount = (size_t)(baseRGBInfo->maxChannel + 1) * (gainMapRGBInfo->maxChannel + 1);
    float * toneMappedLUT[3];
    uint8_t * outByteLUT[3];
    uint16_t * stampLUT[3];
    for (int c = 0; c < 3; ++c) {
        toneMappedLUT[c] = avifAlloc(pairCount * sizeof(float));
        outByteLUT[c] = avifAlloc(pairCount * sizeof(uint8_t));
        stampLUT[c] = avifCalloc(pairCount, sizeof(uint16_t));
        if (toneMappedLUT[c] == NULL || outByteLUT[c] == NULL || stampLUT[c] == NULL) {
            for (int p = 0; p <= c; ++p) {
                avifFree(toneMappedLUT[p]);
                avifFree(outByteLUT[p]);
                avifFree(stampLUT[p]);
            }
            return AVIF_RESULT_OUT_OF_MEMORY;
        }
    }

    // A single generation stamp distinguishes entries computed by this call
    // from the zero-initialized memory. No per-call table clearing is needed.
    const uint16_t generation = 1;
    float rgbMaxLinear = 0;
    float rgbSumLinear = 0;

    const uint32_t basePixelBytes = baseRGBInfo->pixelBytes;
    const uint32_t gainMapPixelBytes = gainMapRGBInfo->pixelBytes;
    const uint32_t outPixelBytes = toneMappedPixelRGBInfo->pixelBytes;
    const size_t baseRowBytes = baseImage->rowBytes;
    const size_t gainMapRowBytes = gainMapRGB->rowBytes;
    const size_t outRowBytes = toneMappedImage->rowBytes;
    const uint8_t * basePixels = baseImage->pixels;
    const uint8_t * gainMapPixels = gainMapRGB->pixels;
    uint8_t * outPixels = toneMappedImage->pixels;
    const int gainMax = gainMapRGBInfo->maxChannel;

    for (uint32_t j = 0; j < height; ++j) {
        const uint8_t * baseRow = &basePixels[j * baseRowBytes];
        const uint8_t * gainMapRow = &gainMapPixels[j * gainMapRowBytes];
        uint8_t * outRow = &outPixels[j * outRowBytes];
        for (uint32_t i = 0; i < width; ++i) {
            const uint8_t * basePixel = &baseRow[i * basePixelBytes];
            const uint8_t * gainMapPixel = &gainMapRow[i * gainMapPixelBytes];
            uint8_t * outPixel = &outRow[i * outPixelBytes];

            float pixelRgbMaxLinear = 0.0f;
            // Channel offsets are not necessarily consecutive (e.g. BGRA).
            const int baseChannelOffset[3] = { baseRGBInfo->offsetBytesR, baseRGBInfo->offsetBytesG, baseRGBInfo->offsetBytesB };
            const int gainChannelOffset[3] = { gainMapRGBInfo->offsetBytesR, gainMapRGBInfo->offsetBytesG, gainMapRGBInfo->offsetBytesB };
            const int outChannelOffset[3] = { toneMappedPixelRGBInfo->offsetBytesR,
                                              toneMappedPixelRGBInfo->offsetBytesG,
                                              toneMappedPixelRGBInfo->offsetBytesB };
            for (int c = 0; c < 3; ++c) {
                const int baseByte = basePixel[baseChannelOffset[c]];
                const int gainByte = gainMapPixel[gainChannelOffset[c]];
                const size_t pairIndex = (size_t)baseByte * (gainMax + 1) + gainByte;

                float toneMappedLinear;
                if (stampLUT[c][pairIndex] != generation) {
                    // The exact same expression as the generic loop:
                    // (baseLinear + baseOffset) * exp2(gainMapLog2 * weight) - alternateOffset
                    toneMappedLinear = (baseLinearLUT[baseByte] + baseOffset[c]) * gainFactorLUT[c][gainByte] -
                                      alternateOffset[c];
                    toneMappedLUT[c][pairIndex] = toneMappedLinear;
                    if (isnan(toneMappedLinear)) {
                        avifDiagnosticsPrintf(diag, "Degenerate gain map parameters produce NaN at pixel (%u, %u)", i, j);
                        for (int p = 0; p < 3; ++p) {
                            avifFree(toneMappedLUT[p]);
                            avifFree(outByteLUT[p]);
                            avifFree(stampLUT[p]);
                        }
                        return AVIF_RESULT_INVALID_TONE_MAPPED_IMAGE;
                    }
                    const float clampedGammaValue = avifNanSafeClamp(linearToGamma(toneMappedLinear));
                    outByteLUT[c][pairIndex] = (uint8_t)(0.5f + (clampedGammaValue * toneMappedPixelRGBInfo->maxChannelF));
                    stampLUT[c][pairIndex] = generation;
                } else {
                    toneMappedLinear = toneMappedLUT[c][pairIndex];
                }

                if (toneMappedLinear > rgbMaxLinear) {
                    rgbMaxLinear = toneMappedLinear;
                }
                if (toneMappedLinear > pixelRgbMaxLinear) {
                    pixelRgbMaxLinear = toneMappedLinear;
                }

                outPixel[outChannelOffset[c]] = outByteLUT[c][pairIndex];
            }

            if (outHasAlpha) {
                if (baseHasAlpha) {
                    const int alphaByte = basePixel[baseRGBInfo->offsetBytesA];
                    outPixel[toneMappedPixelRGBInfo->offsetBytesA] =
                        (uint8_t)(0.5f + (((float)alphaByte / baseRGBInfo->maxChannelF) * toneMappedPixelRGBInfo->maxChannelF));
                } else {
                    outPixel[toneMappedPixelRGBInfo->offsetBytesA] = (uint8_t)(0.5f + (1.0f * toneMappedPixelRGBInfo->maxChannelF));
                }
            }
            rgbSumLinear += pixelRgbMaxLinear;
        }
    }

    if (clli != NULL) {
        // For exact CLLI value definitions, see ISO/IEC 23008-2 section D.3.35
        // at https://standards.iso.org/ittf/PubliclyAvailableStandards/c508421_ISO_IEC_23008-2_2015.zip
        clli->maxCLL = (uint16_t)AVIF_CLAMP(avifRoundf(rgbMaxLinear * 203.0f), 0.0f, (float)UINT16_MAX);
        const float rgbAverageLinear = rgbSumLinear / ((float)width * height);
        clli->maxPALL = (uint16_t)AVIF_CLAMP(avifRoundf(rgbAverageLinear * 203.0f), 0.0f, (float)UINT16_MAX);
    }
    *outRgbMaxLinear = rgbMaxLinear;
    *outRgbSumLinear = rgbSumLinear;

    for (int c = 0; c < 3; ++c) {
        avifFree(toneMappedLUT[c]);
        avifFree(outByteLUT[c]);
        avifFree(stampLUT[c]);
    }
    return AVIF_RESULT_OK;
}

// Compact representation of the raw gain map values of one channel: each
// distinct (base byte, alternate byte) pair value appears once with the
// number of pixels that produced it.
typedef struct avifGainMapRawEntries
{
    float * values;
    uint32_t * counts;
    uint32_t count;
} avifGainMapRawEntries;

// Defined later alongside avifFindMinMaxWithoutOutliers().
static avifResult avifFindMinMaxWithoutOutliersWeighted(const avifGainMapRawEntries * entries,
                                                        size_t numPixels,
                                                        float * rangeMin,
                                                        float * rangeMax);

// Fast path for avifRGBImageComputeGainMap() on 8-bit byte-planar inputs
// without a color space matrix conversion. The raw per-channel gain map
// value is log2((alt + alternateOffset) / (base + baseOffset)), and both the
// base and alternate linear values are deterministic functions of a single
// byte, so the raw value only depends on the (base byte, alt byte) pair. A
// lazy 256x256 table per channel computes each pair at most once while
// keeping the exact same float expressions as the generic loop.
static avifResult avifComputeGainMapRaw8Bit(const avifRGBImage * baseRgbImage,
                                            const avifRGBColorSpaceInfo * baseRGBInfo,
                                            avifTransferFunction baseGammaToLinear,
                                            const avifRGBImage * altRgbImage,
                                            const avifRGBColorSpaceInfo * altRGBInfo,
                                            avifTransferFunction altGammaToLinear,
                                            const float baseOffset[3],
                                            const float alternateOffset[3],
                                            float * gainMapF[3],
                                            avifGainMapRawEntries entries[3],
                                            float * outBaseMax,
                                            float * outAltMax)
{
    const uint32_t width = baseRgbImage->width;
    const uint32_t height = baseRgbImage->height;

    float baseLinearLUT[256];
    float altLinearLUT[256];
    for (int v = 0; v <= baseRGBInfo->maxChannel; ++v) {
        baseLinearLUT[v] = baseGammaToLinear((float)v / baseRGBInfo->maxChannelF);
    }
    for (int v = 0; v <= altRGBInfo->maxChannel; ++v) {
        altLinearLUT[v] = altGammaToLinear((float)v / altRGBInfo->maxChannelF);
    }

    const size_t pairCount = (size_t)(baseRGBInfo->maxChannel + 1) * (altRGBInfo->maxChannel + 1);
    // Per channel: pixel count and raw value for every (base byte, alt byte)
    // pair. A zero count marks a pair that has not been encountered yet.
    uint32_t * pairCounts[3];
    float * pairValues[3];
    for (int c = 0; c < 3; ++c) {
        pairCounts[c] = avifCalloc(pairCount, sizeof(uint32_t));
        pairValues[c] = avifAlloc(pairCount * sizeof(float));
        entries[c].values = NULL;
        entries[c].counts = NULL;
        entries[c].count = 0;
        if (pairCounts[c] == NULL || pairValues[c] == NULL) {
            for (int p = 0; p <= c; ++p) {
                avifFree(pairCounts[p]);
                avifFree(pairValues[p]);
            }
            return AVIF_RESULT_OUT_OF_MEMORY;
        }
    }
    const int altMaxChannel = altRGBInfo->maxChannel;

    float baseMax = 1.0f;
    float altMax = 1.0f;
    const uint32_t basePixelBytes = baseRGBInfo->pixelBytes;
    const uint32_t altPixelBytes = altRGBInfo->pixelBytes;
    const size_t baseRowBytes = baseRgbImage->rowBytes;
    const size_t altRowBytes = altRgbImage->rowBytes;
    const uint8_t * basePixels = baseRgbImage->pixels;
    const uint8_t * altPixels = altRgbImage->pixels;
    // Channel offsets are not necessarily consecutive (e.g. BGRA).
    const int baseChannelOffset[3] = { baseRGBInfo->offsetBytesR, baseRGBInfo->offsetBytesG, baseRGBInfo->offsetBytesB };
    const int altChannelOffset[3] = { altRGBInfo->offsetBytesR, altRGBInfo->offsetBytesG, altRGBInfo->offsetBytesB };

    for (uint32_t j = 0; j < height; ++j) {
        const uint8_t * baseRow = &basePixels[j * baseRowBytes];
        const uint8_t * altRow = &altPixels[j * altRowBytes];
        for (uint32_t i = 0; i < width; ++i) {
            const uint8_t * basePixel = &baseRow[i * basePixelBytes];
            const uint8_t * altPixel = &altRow[i * altPixelBytes];
            const size_t pixelIndex = (size_t)j * width + i;
            for (int c = 0; c < 3; ++c) {
                const int baseByte = basePixel[baseChannelOffset[c]];
                const int altByte = altPixel[altChannelOffset[c]];
                const float base = baseLinearLUT[baseByte];
                const float alt = altLinearLUT[altByte];

                // Keep the exact same headroom tracking as the generic loop.
                if (base > baseMax) {
                    baseMax = base;
                }
                if (alt > altMax) {
                    altMax = alt;
                }

                const size_t pairIndex = (size_t)baseByte * (altMaxChannel + 1) + altByte;
                if (pairCounts[c][pairIndex] == 0) {
                    // Same expression as the generic loop:
                    // log2(max((alt + alternateOffset) / (base + baseOffset), kEpsilon))
                    const float ratio = (alt + alternateOffset[c]) / (base + baseOffset[c]);
                    pairValues[c][pairIndex] = log2f(AVIF_MAX(ratio, 1e-10f));
                }
                ++(pairCounts[c][pairIndex]);
                gainMapF[c][pixelIndex] = pairValues[c][pairIndex];
            }
        }
    }

    // Build the compact (value, weight) entries. Order is irrelevant for the
    // outlier histogram, so pairs are scanned in pair-index order.
    for (int c = 0; c < 3; ++c) {
        uint32_t distinctPairs = 0;
        for (size_t pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
            if (pairCounts[c][pairIndex] != 0) {
                ++distinctPairs;
            }
        }
        entries[c].count = distinctPairs;
        entries[c].values = avifAlloc((size_t)distinctPairs * sizeof(float));
        entries[c].counts = avifAlloc((size_t)distinctPairs * sizeof(uint32_t));
        if (entries[c].values == NULL || entries[c].counts == NULL) {
            for (int p = 0; p < 3; ++p) {
                avifFree(pairCounts[p]);
                avifFree(pairValues[p]);
                avifFree(entries[p].values);
                avifFree(entries[p].counts);
            }
            return AVIF_RESULT_OUT_OF_MEMORY;
        }
        uint32_t outIdx = 0;
        for (size_t pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
            if (pairCounts[c][pairIndex] != 0) {
                entries[c].values[outIdx] = pairValues[c][pairIndex];
                entries[c].counts[outIdx] = pairCounts[c][pairIndex];
                ++outIdx;
            }
        }
        avifFree(pairCounts[c]);
        avifFree(pairValues[c]);
    }
    *outBaseMax = baseMax;
    *outAltMax = altMax;
    return AVIF_RESULT_OK;
}

static void avifGainMapSetEncodingDefaults(avifGainMap * gainMap)
{
    for (int i = 0; i < 3; ++i) {
        gainMap->gainMapMin[i] = (avifSignedFraction) { 1, 1 };
        gainMap->gainMapMax[i] = (avifSignedFraction) { 1, 1 };
        gainMap->baseOffset[i] = (avifSignedFraction) { 1, 64 };
        gainMap->alternateOffset[i] = (avifSignedFraction) { 1, 64 };
        gainMap->gainMapGamma[i] = (avifUnsignedFraction) { 1, 1 };
    }
    gainMap->baseHdrHeadroom = (avifUnsignedFraction) { 0, 1 };
    gainMap->alternateHdrHeadroom = (avifUnsignedFraction) { 1, 1 };
    gainMap->useBaseColorSpace = AVIF_TRUE;
}



avifResult avifRGBImageApplyGainMap(const avifRGBImage * baseImage,
                                    avifColorPrimaries baseColorPrimaries,
                                    avifTransferCharacteristics baseTransferCharacteristics,
                                    const avifGainMap * gainMap,
                                    float hdrHeadroom,
                                    avifColorPrimaries outputColorPrimaries,
                                    avifTransferCharacteristics outputTransferCharacteristics,
                                    avifRGBImage * toneMappedImage,
                                    avifContentLightLevelInformationBox * clli,
                                    avifDiagnostics * diag)
{
    avifDiagnosticsClearError(diag);

    if (hdrHeadroom < 0.0f) {
        avifDiagnosticsPrintf(diag, "hdrHeadroom should be >= 0, got %f", hdrHeadroom);
        return AVIF_RESULT_INVALID_ARGUMENT;
    }
    if (baseImage == NULL || gainMap == NULL || toneMappedImage == NULL) {
        avifDiagnosticsPrintf(diag, "NULL input image");
        return AVIF_RESULT_INVALID_ARGUMENT;
    }
    // The gain map image may be missing if it was not decoded (see
    // avifDecoder::imageContentToDecode) or was never set. Fail early and
    // consistently, even for calls that would not need the gain map pixels
    // (e.g. a target headroom matching the base image headroom).
    if (gainMap->image == NULL) {
        avifDiagnosticsPrintf(diag, "gainMap->image is null (gain map image not decoded?)");
        return AVIF_RESULT_INVALID_ARGUMENT;
    }
    AVIF_CHECKRES(avifGainMapValidateMetadata(gainMap, diag));

    const uint32_t width = baseImage->width;
    const uint32_t height = baseImage->height;

    const avifBool useBaseColorSpace = gainMap->useBaseColorSpace;
    const avifColorPrimaries gainMapMathPrimaries =
        (useBaseColorSpace || (gainMap->altColorPrimaries == AVIF_COLOR_PRIMARIES_UNSPECIFIED)) ? baseColorPrimaries
                                                                                                : gainMap->altColorPrimaries;
    const avifBool needsInputColorConversion = (baseColorPrimaries != gainMapMathPrimaries);
    const avifBool needsOutputColorConversion = (gainMapMathPrimaries != outputColorPrimaries);

    avifImage * rescaledGainMap = NULL;
    avifRGBImage rgbGainMap;
    // Basic zero-initialization for now, avifRGBImageSetDefaults() is called later on.
    memset(&rgbGainMap, 0, sizeof(rgbGainMap));

    avifResult res = AVIF_RESULT_OK;
    toneMappedImage->width = width;
    toneMappedImage->height = height;
    AVIF_CHECKRES(avifRGBImageAllocatePixels(toneMappedImage));

    // --- After this point, the function should exit with 'goto cleanup' to free allocated pixels.

    const float weight = avifGetGainMapWeight(hdrHeadroom, gainMap);

    // Early exit if the gain map does not need to be applied and the pixel format is the same.
    if (weight == 0.0f && outputTransferCharacteristics == baseTransferCharacteristics && outputColorPrimaries == baseColorPrimaries &&
        baseImage->format == toneMappedImage->format && baseImage->depth == toneMappedImage->depth &&
        baseImage->isFloat == toneMappedImage->isFloat && baseImage->rowBytes == toneMappedImage->rowBytes) {
        assert(baseImage->height == toneMappedImage->height);
        // Copy the base image.
        memcpy(toneMappedImage->pixels, baseImage->pixels, (size_t)baseImage->rowBytes * baseImage->height);
        goto cleanup;
    }

    avifRGBColorSpaceInfo baseRGBInfo;
    avifRGBColorSpaceInfo toneMappedPixelRGBInfo;
    if (!avifGetRGBColorSpaceInfo(baseImage, &baseRGBInfo) || !avifGetRGBColorSpaceInfo(toneMappedImage, &toneMappedPixelRGBInfo)) {
        avifDiagnosticsPrintf(diag, "Unsupported RGB color space");
        res = AVIF_RESULT_NOT_IMPLEMENTED;
        goto cleanup;
    }

    const avifTransferFunction gammaToLinear = avifTransferCharacteristicsGetGammaToLinearFunction(baseTransferCharacteristics);
    const avifTransferFunction linearToGamma = avifTransferCharacteristicsGetLinearToGammaFunction(outputTransferCharacteristics);

    // Early exit if the gain map does not need to be applied.
    if (weight == 0.0f) {
        const avifBool primariesDiffer = (baseColorPrimaries != outputColorPrimaries);
        double conversionCoeffs[3][3];
        if (primariesDiffer && !avifColorPrimariesComputeRGBToRGBMatrix(baseColorPrimaries, outputColorPrimaries, conversionCoeffs)) {
            avifDiagnosticsPrintf(diag, "Unsupported RGB color space conversion");
            res = AVIF_RESULT_NOT_IMPLEMENTED;
            goto cleanup;
        }
        const avifBool transferOrPrimariesDiffer = (outputTransferCharacteristics != baseTransferCharacteristics) || primariesDiffer;
        // 8-bit byte-planar images without a color space matrix conversion
        // reduce this branch to a 256-entry per-channel lookup table.
        if (!primariesDiffer && avifRGBImageIs8BitBytePlanar(baseImage) &&
            (toneMappedImage->depth == 8 && !toneMappedImage->isFloat && toneMappedImage->format != AVIF_RGB_FORMAT_RGB_565 &&
             !avifRGBFormatIsGray(toneMappedImage->format))) {
            res = avifApplyGainMapZeroWeight8Bit(baseImage,
                                                 &baseRGBInfo,
                                                 gammaToLinear,
                                                 linearToGamma,
                                                 &toneMappedPixelRGBInfo,
                                                 toneMappedImage,
                                                 transferOrPrimariesDiffer);
            if (res != AVIF_RESULT_OK) {
                goto cleanup;
            }
            goto cleanup;
        }
        // Just convert from one rgb format to another.
        for (uint32_t j = 0; j < height; ++j) {
            for (uint32_t i = 0; i < width; ++i) {
                float basePixelRGBA[4];
                avifGetRGBAPixel(baseImage, i, j, &baseRGBInfo, basePixelRGBA);
                if (outputTransferCharacteristics != baseTransferCharacteristics || primariesDiffer) {
                    for (int c = 0; c < 3; ++c) {
                        basePixelRGBA[c] = gammaToLinear(basePixelRGBA[c]);
                    }
                    if (primariesDiffer) {
                        avifLinearRGBConvertColorSpace(basePixelRGBA, conversionCoeffs);
                    }
                    for (int c = 0; c < 3; ++c) {
                        basePixelRGBA[c] = avifNanSafeClamp(linearToGamma(basePixelRGBA[c]));
                    }
                }
                avifSetRGBAPixel(toneMappedImage, i, j, &toneMappedPixelRGBInfo, basePixelRGBA);
            }
        }
        goto cleanup;
    }

    double inputConversionCoeffs[3][3];
    double outputConversionCoeffs[3][3];
    if (needsInputColorConversion &&
        !avifColorPrimariesComputeRGBToRGBMatrix(baseColorPrimaries, gainMapMathPrimaries, inputConversionCoeffs)) {
        avifDiagnosticsPrintf(diag, "Unsupported RGB color space conversion");
        res = AVIF_RESULT_NOT_IMPLEMENTED;
        goto cleanup;
    }
    if (needsOutputColorConversion &&
        !avifColorPrimariesComputeRGBToRGBMatrix(gainMapMathPrimaries, outputColorPrimaries, outputConversionCoeffs)) {
        avifDiagnosticsPrintf(diag, "Unsupported RGB color space conversion");
        res = AVIF_RESULT_NOT_IMPLEMENTED;
        goto cleanup;
    }

    if (gainMap->image->width != width || gainMap->image->height != height) {
        rescaledGainMap = avifImageCreateEmpty();
        if (rescaledGainMap == NULL) {
            res = AVIF_RESULT_OUT_OF_MEMORY;
            goto cleanup;
        }
        const avifCropRect rect = { 0, 0, gainMap->image->width, gainMap->image->height };
        res = avifImageSetViewRect(rescaledGainMap, gainMap->image, &rect);
        if (res != AVIF_RESULT_OK) {
            goto cleanup;
        }
        res = avifImageScale(rescaledGainMap, width, height, diag);
        if (res != AVIF_RESULT_OK) {
            goto cleanup;
        }
    }
    const avifImage * const gainMapImage = (rescaledGainMap != NULL) ? rescaledGainMap : gainMap->image;

    avifRGBImageSetDefaults(&rgbGainMap, gainMapImage);
    res = avifRGBImageAllocatePixels(&rgbGainMap);
    if (res != AVIF_RESULT_OK) {
        goto cleanup;
    }
    res = avifImageYUVToRGB(gainMapImage, &rgbGainMap);
    if (res != AVIF_RESULT_OK) {
        goto cleanup;
    }

    avifRGBColorSpaceInfo gainMapRGBInfo;
    if (!avifGetRGBColorSpaceInfo(&rgbGainMap, &gainMapRGBInfo)) {
        avifDiagnosticsPrintf(diag, "Unsupported RGB color space");
        res = AVIF_RESULT_NOT_IMPLEMENTED;
        goto cleanup;
    }

    // 8-bit byte-planar inputs without color space matrix conversions can
    // evaluate every sample through lazy (base byte, gain map byte) lookup
    // tables while producing bit-identical output bytes and CLLI values.
    const avifBool use8BitWeightedFastPath = !needsInputColorConversion && !needsOutputColorConversion &&
                                             avifRGBImageIs8BitBytePlanar(baseImage) &&
                                             avifRGBImageIs8BitBytePlanar(&rgbGainMap) &&
                                             (toneMappedImage->depth == 8 && !toneMappedImage->isFloat &&
                                              toneMappedImage->format != AVIF_RGB_FORMAT_RGB_565 &&
                                              !avifRGBFormatIsGray(toneMappedImage->format));
    if (use8BitWeightedFastPath) {
        float fastRgbMaxLinear = 0;
        float fastRgbSumLinear = 0;
        res = avifApplyGainMapWeighted8Bit(baseImage,
                                           &baseRGBInfo,
                                           &rgbGainMap,
                                           &gainMapRGBInfo,
                                           gainMap,
                                           weight,
                                           gammaToLinear,
                                           linearToGamma,
                                           &toneMappedPixelRGBInfo,
                                           toneMappedImage,
                                           &fastRgbMaxLinear,
                                           &fastRgbSumLinear,
                                           clli,
                                           diag);
        if (res != AVIF_RESULT_OK) {
            goto cleanup;
        }
        goto cleanup;
    }

    float rgbMaxLinear = 0; // Max tone mapped pixel value across R, G and B channels.
    float rgbSumLinear = 0; // Sum of max(r, g, b) for mapped pixels.
    // The gain map metadata contains the encoding gamma, and 1/gamma should be used for decoding.
    const float gammaInv[3] = { 1.0f / avifUnsignedFractionToFloat(gainMap->gainMapGamma[0]),
                                1.0f / avifUnsignedFractionToFloat(gainMap->gainMapGamma[1]),
                                1.0f / avifUnsignedFractionToFloat(gainMap->gainMapGamma[2]) };
    const float gainMapMin[3] = { avifSignedFractionToFloat(gainMap->gainMapMin[0]),
                                  avifSignedFractionToFloat(gainMap->gainMapMin[1]),
                                  avifSignedFractionToFloat(gainMap->gainMapMin[2]) };
    const float gainMapMax[3] = { avifSignedFractionToFloat(gainMap->gainMapMax[0]),
                                  avifSignedFractionToFloat(gainMap->gainMapMax[1]),
                                  avifSignedFractionToFloat(gainMap->gainMapMax[2]) };
    const float baseOffset[3] = { avifSignedFractionToFloat(gainMap->baseOffset[0]),
                                  avifSignedFractionToFloat(gainMap->baseOffset[1]),
                                  avifSignedFractionToFloat(gainMap->baseOffset[2]) };
    const float alternateOffset[3] = { avifSignedFractionToFloat(gainMap->alternateOffset[0]),
                                       avifSignedFractionToFloat(gainMap->alternateOffset[1]),
                                       avifSignedFractionToFloat(gainMap->alternateOffset[2]) };
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t i = 0; i < width; ++i) {
            float basePixelRGBA[4];
            avifGetRGBAPixel(baseImage, i, j, &baseRGBInfo, basePixelRGBA);
            float gainMapRGBA[4];
            avifGetRGBAPixel(&rgbGainMap, i, j, &gainMapRGBInfo, gainMapRGBA);

            // Apply gain map.
            float toneMappedPixelRGBA[4];
            float pixelRgbMaxLinear = 0.0f; //  = max(r, g, b) for this pixel

            for (int c = 0; c < 3; ++c) {
                basePixelRGBA[c] = gammaToLinear(basePixelRGBA[c]);
            }

            if (needsInputColorConversion) {
                // Convert basePixelRGBA to gainMapMathPrimaries.
                avifLinearRGBConvertColorSpace(basePixelRGBA, inputConversionCoeffs);
            }

            for (int c = 0; c < 3; ++c) {
                const float baseLinear = basePixelRGBA[c];
                const float gainMapValue = gainMapRGBA[c];

                // Undo gamma & affine transform; the result is in log2 space.
                const float gainMapLog2 = lerp(gainMapMin[c], gainMapMax[c], powf(gainMapValue, gammaInv[c]));
                const float toneMappedLinear = (baseLinear + baseOffset[c]) * exp2f(gainMapLog2 * weight) - alternateOffset[c];

                if (toneMappedLinear > rgbMaxLinear) {
                    rgbMaxLinear = toneMappedLinear;
                }
                if (toneMappedLinear > pixelRgbMaxLinear) {
                    pixelRgbMaxLinear = toneMappedLinear;
                }

                toneMappedPixelRGBA[c] = toneMappedLinear;
            }

            if (needsOutputColorConversion) {
                // Convert toneMappedPixelRGBA to outputColorPrimaries.
                avifLinearRGBConvertColorSpace(toneMappedPixelRGBA, outputConversionCoeffs);
            }

            for (int c = 0; c < 3; ++c) {
                if (isnan(toneMappedPixelRGBA[c])) {
                    avifDiagnosticsPrintf(diag, "Degenerate gain map parameters produce NaN at pixel (%u, %u)", i, j);
                    res = AVIF_RESULT_INVALID_TONE_MAPPED_IMAGE;
                    goto cleanup;
                }
                toneMappedPixelRGBA[c] = avifNanSafeClamp(linearToGamma(toneMappedPixelRGBA[c]));
            }

            toneMappedPixelRGBA[3] = basePixelRGBA[3]; // Alpha is unaffected by tone mapping.
            rgbSumLinear += pixelRgbMaxLinear;
            avifSetRGBAPixel(toneMappedImage, i, j, &toneMappedPixelRGBInfo, toneMappedPixelRGBA);
        }
    }
    if (clli != NULL) {
        // For exact CLLI value definitions, see ISO/IEC 23008-2 section D.3.35
        // at https://standards.iso.org/ittf/PubliclyAvailableStandards/index.html

        // Convert extended SDR (where 1.0 is SDR white) to nits.
        clli->maxCLL = (uint16_t)AVIF_CLAMP(avifRoundf(rgbMaxLinear * 203.0f), 0.0f, (float)UINT16_MAX);
        const float rgbAverageLinear = rgbSumLinear / ((size_t)width * height);
        clli->maxPALL = (uint16_t)AVIF_CLAMP(avifRoundf(rgbAverageLinear * 203.0f), 0.0f, (float)UINT16_MAX);
    }

cleanup:
    avifRGBImageFreePixels(&rgbGainMap);
    if (rescaledGainMap != NULL) {
        avifImageDestroy(rescaledGainMap);
    }
    if (res != AVIF_RESULT_OK) {
        avifRGBImageFreePixels(toneMappedImage);
    }

    return res;
}

avifResult avifImageApplyGainMap(const avifImage * baseImage,
                                 const avifGainMap * gainMap,
                                 float hdrHeadroom,
                                 avifColorPrimaries outputColorPrimaries,
                                 avifTransferCharacteristics outputTransferCharacteristics,
                                 avifRGBImage * toneMappedImage,
                                 avifContentLightLevelInformationBox * clli,
                                 avifDiagnostics * diag)
{
    avifDiagnosticsClearError(diag);

    if (baseImage->icc.size > 0 || gainMap->altICC.size > 0) {
        avifDiagnosticsPrintf(diag, "Tone mapping for images with ICC profiles is not supported");
        return AVIF_RESULT_NOT_IMPLEMENTED;
    }

    avifRGBImage baseImageRgb;
    avifRGBImageSetDefaults(&baseImageRgb, baseImage);
    AVIF_CHECKRES(avifRGBImageAllocatePixels(&baseImageRgb));
    avifResult res = avifImageYUVToRGB(baseImage, &baseImageRgb);
    if (res != AVIF_RESULT_OK) {
        goto cleanup;
    }

    res = avifRGBImageApplyGainMap(&baseImageRgb,
                                   baseImage->colorPrimaries,
                                   baseImage->transferCharacteristics,
                                   gainMap,
                                   hdrHeadroom,
                                   outputColorPrimaries,
                                   outputTransferCharacteristics,
                                   toneMappedImage,
                                   clli,
                                   diag);

cleanup:
    avifRGBImageFreePixels(&baseImageRgb);

    return res;
}

// ---------------------------------------------------------------------------
// Create a gain map.

// Returns the index of the histogram bucket for a given value, for a histogram with 'numBuckets' buckets,
// and values ranging in [bucketMin, bucketMax] (values outside of the range are added to the first/last buckets).
static int avifValueToBucketIdx(float v, float bucketMin, float bucketMax, int numBuckets)
{
    v = AVIF_CLAMP(v, bucketMin, bucketMax);
    return AVIF_MIN((int)avifRoundf((v - bucketMin) / (bucketMax - bucketMin) * numBuckets), numBuckets - 1);
}
// Returns the lower end of the value range belonging to the given histogram bucket.
static float avifBucketIdxToValue(int idx, float bucketMin, float bucketMax, int numBuckets)
{
    return idx * (bucketMax - bucketMin) / numBuckets + bucketMin;
}

avifResult avifFindMinMaxWithoutOutliers(const float * gainMapF, size_t numPixels, float * rangeMin, float * rangeMax)
{
    const float bucketSize = 0.01f;        // Size of one bucket. Empirical value.
    const float maxOutliersRatio = 0.001f; // 0.1%
    const int maxOutliersOnEachSide = (int)avifRoundf(numPixels * maxOutliersRatio / 2.0f);

    float min = gainMapF[0];
    float max = gainMapF[0];
    for (size_t i = 1; i < numPixels; ++i) {
        min = AVIF_MIN(min, gainMapF[i]);
        max = AVIF_MAX(max, gainMapF[i]);
    }

    *rangeMin = min;
    *rangeMax = max;
    if ((max - min) <= (bucketSize * 2) || maxOutliersOnEachSide == 0) {
        return AVIF_RESULT_OK;
    }

    const int maxNumBuckets = 10000;
    const int numBuckets = AVIF_MIN((int)ceilf((max - min) / bucketSize), maxNumBuckets);
    int * histogram = avifCalloc(numBuckets, sizeof(int));
    if (histogram == NULL) {
        return AVIF_RESULT_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < numPixels; ++i) {
        ++(histogram[avifValueToBucketIdx(gainMapF[i], min, max, numBuckets)]);
    }

    int leftOutliers = 0;
    for (int i = 0; i < numBuckets; ++i) {
        leftOutliers += histogram[i];
        if (leftOutliers > maxOutliersOnEachSide) {
            break;
        }
        if (histogram[i] == 0) {
            // +1 to get the higher end of the bucket.
            *rangeMin = avifBucketIdxToValue(i + 1, min, max, numBuckets);
        }
    }

    int rightOutliers = 0;
    for (int i = numBuckets - 1; i >= 0; --i) {
        rightOutliers += histogram[i];
        if (rightOutliers > maxOutliersOnEachSide) {
            break;
        }
        if (histogram[i] == 0) {
            *rangeMax = avifBucketIdxToValue(i, min, max, numBuckets);
        }
    }

    avifFree(histogram);
    return AVIF_RESULT_OK;
}

// Same outlier filtering as avifFindMinMaxWithoutOutliers(), but operating on
// the compact (value, weight) representation. Buckets are filled with the
// pixel counts instead of incrementing once per pixel, which is mathematically
// identical and produces the exact same range values.
static avifResult avifFindMinMaxWithoutOutliersWeighted(const avifGainMapRawEntries * entries,
                                                        size_t numPixels,
                                                        float * rangeMin,
                                                        float * rangeMax)
{
    const float bucketSize = 0.01f;        // Size of one bucket. Empirical value.
    const float maxOutliersRatio = 0.001f; // 0.1%
    const int maxOutliersOnEachSide = (int)avifRoundf((float)numPixels * maxOutliersRatio / 2.0f);

    float min = entries->values[0];
    float max = entries->values[0];
    for (uint32_t i = 1; i < entries->count; ++i) {
        min = AVIF_MIN(min, entries->values[i]);
        max = AVIF_MAX(max, entries->values[i]);
    }

    *rangeMin = min;
    *rangeMax = max;
    if ((max - min) <= (bucketSize * 2) || maxOutliersOnEachSide == 0) {
        return AVIF_RESULT_OK;
    }

    const int maxNumBuckets = 10000;
    const int numBuckets = AVIF_MIN((int)ceilf((max - min) / bucketSize), maxNumBuckets);
    int * histogram = avifCalloc(numBuckets, sizeof(int));
    if (histogram == NULL) {
        return AVIF_RESULT_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < entries->count; ++i) {
        histogram[avifValueToBucketIdx(entries->values[i], min, max, numBuckets)] += (int)entries->counts[i];
    }

    int leftOutliers = 0;
    for (int i = 0; i < numBuckets; ++i) {
        leftOutliers += histogram[i];
        if (leftOutliers > maxOutliersOnEachSide) {
            break;
        }
        if (histogram[i] == 0) {
            // +1 to get the higher end of the bucket.
            *rangeMin = avifBucketIdxToValue(i + 1, min, max, numBuckets);
        }
    }

    int rightOutliers = 0;
    for (int i = numBuckets - 1; i >= 0; --i) {
        rightOutliers += histogram[i];
        if (rightOutliers > maxOutliersOnEachSide) {
            break;
        }
        if (histogram[i] == 0) {
            *rangeMax = avifBucketIdxToValue(i, min, max, numBuckets);
        }
    }

    avifFree(histogram);
    return AVIF_RESULT_OK;
}

avifResult avifGainMapValidateMetadata(const avifGainMap * gainMap, avifDiagnostics * diag)
{
    for (int i = 0; i < 3; ++i) {
        if (gainMap->gainMapMin[i].d == 0 || gainMap->gainMapMax[i].d == 0 || gainMap->gainMapGamma[i].d == 0 ||
            gainMap->baseOffset[i].d == 0 || gainMap->alternateOffset[i].d == 0) {
            avifDiagnosticsPrintf(diag, "Per-channel denominator is 0 in gain map metadata");
            return AVIF_RESULT_INVALID_ARGUMENT;
        }
        if ((int64_t)gainMap->gainMapMax[i].n * gainMap->gainMapMin[i].d <
            (int64_t)gainMap->gainMapMin[i].n * gainMap->gainMapMax[i].d) {
            avifDiagnosticsPrintf(diag, "Per-channel max is less than per-channel min in gain map metadata");
            return AVIF_RESULT_INVALID_ARGUMENT;
        }
        if (gainMap->gainMapGamma[i].n == 0) {
            avifDiagnosticsPrintf(diag, "Per-channel gamma is 0 in gain map metadata");
            return AVIF_RESULT_INVALID_ARGUMENT;
        }
    }
    if (gainMap->baseHdrHeadroom.d == 0 || gainMap->alternateHdrHeadroom.d == 0) {
        avifDiagnosticsPrintf(diag, "Headroom denominator is 0 in gain map metadata");
        return AVIF_RESULT_INVALID_ARGUMENT;
    }
    if (gainMap->useBaseColorSpace != 0 && gainMap->useBaseColorSpace != 1) {
        avifDiagnosticsPrintf(diag, "useBaseColorSpace is %d in gain map metadata", gainMap->useBaseColorSpace);
        return AVIF_RESULT_INVALID_ARGUMENT;
    }
    return AVIF_RESULT_OK;
}

avifBool avifSameGainMapMetadata(const avifGainMap * a, const avifGainMap * b)
{
    if (a->baseHdrHeadroom.n != b->baseHdrHeadroom.n || a->baseHdrHeadroom.d != b->baseHdrHeadroom.d ||
        a->alternateHdrHeadroom.n != b->alternateHdrHeadroom.n || a->alternateHdrHeadroom.d != b->alternateHdrHeadroom.d) {
        return AVIF_FALSE;
    }
    for (int c = 0; c < 3; ++c) {
        if (a->gainMapMin[c].n != b->gainMapMin[c].n || a->gainMapMin[c].d != b->gainMapMin[c].d ||
            a->gainMapMax[c].n != b->gainMapMax[c].n || a->gainMapMax[c].d != b->gainMapMax[c].d ||
            a->gainMapGamma[c].n != b->gainMapGamma[c].n || a->gainMapGamma[c].d != b->gainMapGamma[c].d ||
            a->baseOffset[c].n != b->baseOffset[c].n || a->baseOffset[c].d != b->baseOffset[c].d ||
            a->alternateOffset[c].n != b->alternateOffset[c].n || a->alternateOffset[c].d != b->alternateOffset[c].d) {
            return AVIF_FALSE;
        }
    }
    return AVIF_TRUE;
}

avifBool avifSameGainMapAltMetadata(const avifGainMap * a, const avifGainMap * b)
{
    if (a->altICC.size != b->altICC.size || (a->altICC.size > 0 && memcmp(a->altICC.data, b->altICC.data, a->altICC.size) != 0) ||
        a->altColorPrimaries != b->altColorPrimaries || a->altTransferCharacteristics != b->altTransferCharacteristics ||
        a->altMatrixCoefficients != b->altMatrixCoefficients || a->altYUVRange != b->altYUVRange || a->altDepth != b->altDepth ||
        a->altPlaneCount != b->altPlaneCount || a->altCLLI.maxCLL != b->altCLLI.maxCLL || a->altCLLI.maxPALL != b->altCLLI.maxPALL) {
        return AVIF_FALSE;
    }
    return AVIF_TRUE;
}

static const float kEpsilon = 1e-10f;

// Decides which of 'basePrimaries' or 'altPrimaries' should be used for doing gain map math when creating a gain map.
// The other image (base or alternate) will be converted to this color space before computing
// the ratio between the two images.
// If a pixel color is outside of the target color space, some of the converted channel values will be negative.
// This should be avoided, as the negative values must either be clamped or offset before computing the log2()
// (since log2 only works on > 0 values). But a large offset causes artefacts when partially applying the gain map.
// Therefore we want to do gain map math in the larger of the two color spaces.
static avifResult avifChooseColorSpaceForGainMapMath(avifColorPrimaries basePrimaries,
                                                     avifColorPrimaries altPrimaries,
                                                     avifColorPrimaries * gainMapMathColorSpace)
{
    if (basePrimaries == altPrimaries) {
        *gainMapMathColorSpace = basePrimaries;
        return AVIF_RESULT_OK;
    }
    // Color convert pure red, pure green and pure blue in turn and see if they result in negative values.
    float rgba[4] = { 0 };
    double baseToAltCoeffs[3][3];
    double altToBaseCoeffs[3][3];
    if (!avifColorPrimariesComputeRGBToRGBMatrix(basePrimaries, altPrimaries, baseToAltCoeffs) ||
        !avifColorPrimariesComputeRGBToRGBMatrix(altPrimaries, basePrimaries, altToBaseCoeffs)) {
        return AVIF_RESULT_NOT_IMPLEMENTED;
    }

    float baseColorspaceChannelMin = 0;
    float altColorspaceChannelMin = 0;
    for (int c = 0; c < 3; ++c) {
        rgba[0] = rgba[1] = rgba[2] = 0;
        rgba[c] = 1.0f;
        avifLinearRGBConvertColorSpace(rgba, altToBaseCoeffs);
        for (int i = 0; i < 3; ++i) {
            baseColorspaceChannelMin = AVIF_MIN(baseColorspaceChannelMin, rgba[i]);
        }
        rgba[0] = rgba[1] = rgba[2] = 0;
        rgba[c] = 1.0f;
        avifLinearRGBConvertColorSpace(rgba, baseToAltCoeffs);
        for (int i = 0; i < 3; ++i) {
            altColorspaceChannelMin = AVIF_MIN(altColorspaceChannelMin, rgba[i]);
        }
    }
    // Pick the colorspace that has the largest min value (which is more or less the largest color space).
    *gainMapMathColorSpace = (altColorspaceChannelMin <= baseColorspaceChannelMin) ? basePrimaries : altPrimaries;
    return AVIF_RESULT_OK;
}

avifResult avifRGBImageComputeGainMap(const avifRGBImage * baseRgbImage,
                                      avifColorPrimaries baseColorPrimaries,
                                      avifTransferCharacteristics baseTransferCharacteristics,
                                      const avifRGBImage * altRgbImage,
                                      avifColorPrimaries altColorPrimaries,
                                      avifTransferCharacteristics altTransferCharacteristics,
                                      avifGainMap * gainMap,
                                      avifDiagnostics * diag)
{
    avifDiagnosticsClearError(diag);

    AVIF_CHECKERR(baseRgbImage != NULL && altRgbImage != NULL && gainMap != NULL && gainMap->image != NULL, AVIF_RESULT_INVALID_ARGUMENT);
    if (baseRgbImage->width != altRgbImage->width || baseRgbImage->height != altRgbImage->height) {
        avifDiagnosticsPrintf(diag, "Both images should have the same dimensions");
        return AVIF_RESULT_INVALID_ARGUMENT;
    }
    if (gainMap->image->width == 0 || gainMap->image->height == 0 || gainMap->image->depth == 0 ||
        gainMap->image->yuvFormat <= AVIF_PIXEL_FORMAT_NONE || gainMap->image->yuvFormat >= AVIF_PIXEL_FORMAT_COUNT) {
        avifDiagnosticsPrintf(diag, "gainMap->image should be non null with desired width, height, depth and yuvFormat set");
        return AVIF_RESULT_INVALID_ARGUMENT;
    }
    const avifBool colorSpacesDiffer = (baseColorPrimaries != altColorPrimaries);
    avifColorPrimaries gainMapMathPrimaries;
    AVIF_CHECKRES(avifChooseColorSpaceForGainMapMath(baseColorPrimaries, altColorPrimaries, &gainMapMathPrimaries));
    const uint32_t width = baseRgbImage->width;
    const uint32_t height = baseRgbImage->height;

    avifRGBColorSpaceInfo baseRGBInfo;
    avifRGBColorSpaceInfo altRGBInfo;
    if (!avifGetRGBColorSpaceInfo(baseRgbImage, &baseRGBInfo) || !avifGetRGBColorSpaceInfo(altRgbImage, &altRGBInfo)) {
        avifDiagnosticsPrintf(diag, "Unsupported RGB color space");
        return AVIF_RESULT_NOT_IMPLEMENTED;
    }

    float * gainMapF[3] = { 0 }; // Temporary buffers for the gain map as floating point values, one per RGB channel.
    // Compact per-channel representation of the raw values on the 8-bit fast
    // path (one entry per distinct (base byte, alt byte) pair).
    avifGainMapRawEntries rawEntries[3] = { { 0 } };
    avifBool use8BitRawFastPath = AVIF_FALSE;
    avifRGBImage gainMapRGB;
    memset(&gainMapRGB, 0, sizeof(gainMapRGB));
    avifImage * gainMapImage = gainMap->image;

    avifResult res = AVIF_RESULT_OK;
    // --- After this point, the function should exit with 'goto cleanup' to free allocated resources.

    const size_t numPixels = (size_t)width * height;
    if (numPixels > SIZE_MAX / sizeof(float)) {
        res = AVIF_RESULT_INVALID_ARGUMENT;
        goto cleanup;
    }
    const size_t gainMapChannelSize = numPixels * sizeof(float);
    const avifBool singleChannel = (gainMap->image->yuvFormat == AVIF_PIXEL_FORMAT_YUV400);
    const int numGainMapChannels = singleChannel ? 1 : 3;
    for (int c = 0; c < numGainMapChannels; ++c) {
        gainMapF[c] = avifAlloc(gainMapChannelSize);
        if (gainMapF[c] == NULL) {
            res = AVIF_RESULT_OUT_OF_MEMORY;
            goto cleanup;
        }
    }

    avifGainMapSetEncodingDefaults(gainMap);
    gainMap->useBaseColorSpace = (gainMapMathPrimaries == baseColorPrimaries);

    float (*baseGammaToLinear)(float) = avifTransferCharacteristicsGetGammaToLinearFunction(baseTransferCharacteristics);
    float (*altGammaToLinear)(float) = avifTransferCharacteristicsGetGammaToLinearFunction(altTransferCharacteristics);
    float yCoeffs[3];
    avifColorPrimariesComputeYCoeffs(gainMapMathPrimaries, yCoeffs);

    double rgbConversionCoeffs[3][3];
    if (colorSpacesDiffer) {
        if (gainMap->useBaseColorSpace) {
            if (!avifColorPrimariesComputeRGBToRGBMatrix(altColorPrimaries, baseColorPrimaries, rgbConversionCoeffs)) {
                avifDiagnosticsPrintf(diag, "Unsupported RGB color space conversion");
                res = AVIF_RESULT_NOT_IMPLEMENTED;
                goto cleanup;
            }
        } else {
            if (!avifColorPrimariesComputeRGBToRGBMatrix(baseColorPrimaries, altColorPrimaries, rgbConversionCoeffs)) {
                avifDiagnosticsPrintf(diag, "Unsupported RGB color space conversion");
                res = AVIF_RESULT_NOT_IMPLEMENTED;
                goto cleanup;
            }
        }
    }

    float baseOffset[3] = { avifSignedFractionToFloat(gainMap->baseOffset[0]),
                            avifSignedFractionToFloat(gainMap->baseOffset[1]),
                            avifSignedFractionToFloat(gainMap->baseOffset[2]) };
    float alternateOffset[3] = { avifSignedFractionToFloat(gainMap->alternateOffset[0]),
                                 avifSignedFractionToFloat(gainMap->alternateOffset[1]),
                                 avifSignedFractionToFloat(gainMap->alternateOffset[2]) };

    // If we are converting from one colorspace to another, some RGB values may be negative and an offset must be added to
    // avoid clamping (although the choice of color space to do the gain map computation with
    // avifChooseColorSpaceForGainMapMath() should mostly avoid this).
    if (colorSpacesDiffer) {
        // Color convert pure red, pure green and pure blue in turn and see if they result in negative values.
        float rgba[4] = { 0.0f };
        float channelMin[3] = { 0.0f };
        for (uint32_t j = 0; j < height; ++j) {
            for (uint32_t i = 0; i < width; ++i) {
                avifGetRGBAPixel(gainMap->useBaseColorSpace ? altRgbImage : baseRgbImage,
                                 i,
                                 j,
                                 gainMap->useBaseColorSpace ? &altRGBInfo : &baseRGBInfo,
                                 rgba);

                // Convert to linear.
                for (int c = 0; c < 3; ++c) {
                    if (gainMap->useBaseColorSpace) {
                        rgba[c] = altGammaToLinear(rgba[c]);
                    } else {
                        rgba[c] = baseGammaToLinear(rgba[c]);
                    }
                }
                avifLinearRGBConvertColorSpace(rgba, rgbConversionCoeffs);
                for (int c = 0; c < 3; ++c) {
                    channelMin[c] = AVIF_MIN(channelMin[c], rgba[c]);
                }
            }
        }

        for (int c = 0; c < 3; ++c) {
            // Large offsets cause artefacts when partially applying the gain map, so set a max (empirical) offset value.
            // If the offset is clamped, some gain map values will get clamped as well.
            const float maxOffset = 0.1f;
            if (channelMin[c] < -kEpsilon) {
                // Increase the offset to avoid negative values.
                if (gainMap->useBaseColorSpace) {
                    alternateOffset[c] = AVIF_MIN(alternateOffset[c] - channelMin[c], maxOffset);
                } else {
                    baseOffset[c] = AVIF_MIN(baseOffset[c] - channelMin[c], maxOffset);
                }
            }
        }
    }

    // Compute raw gain map values.
    float baseMax = 1.0f;
    float altMax = 1.0f;
    // 8-bit byte-planar inputs without a color space matrix conversion let
    // the whole raw computation collapse into a lazy (base byte, alt byte)
    // lookup table while producing bit-identical samples. Everything else
    // keeps the generic loop below.
    use8BitRawFastPath = !colorSpacesDiffer && !singleChannel &&
                         avifRGBImageIs8BitBytePlanar(baseRgbImage) &&
                         avifRGBImageIs8BitBytePlanar(altRgbImage);
    if (use8BitRawFastPath) {
        res = avifComputeGainMapRaw8Bit(baseRgbImage,
                                        &baseRGBInfo,
                                        baseGammaToLinear,
                                        altRgbImage,
                                        &altRGBInfo,
                                        altGammaToLinear,
                                        baseOffset,
                                        alternateOffset,
                                        gainMapF,
                                        rawEntries,
                                        &baseMax,
                                        &altMax);
        if (res != AVIF_RESULT_OK) {
            goto cleanup;
        }
    } else {
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t i = 0; i < width; ++i) {
            float baseRGBA[4];
            avifGetRGBAPixel(baseRgbImage, i, j, &baseRGBInfo, baseRGBA);
            float altRGBA[4];
            avifGetRGBAPixel(altRgbImage, i, j, &altRGBInfo, altRGBA);

            // Convert to linear.
            for (int c = 0; c < 3; ++c) {
                baseRGBA[c] = baseGammaToLinear(baseRGBA[c]);
                altRGBA[c] = altGammaToLinear(altRGBA[c]);
            }

            if (colorSpacesDiffer) {
                if (gainMap->useBaseColorSpace) {
                    // convert altRGBA to baseRGBA's color space
                    avifLinearRGBConvertColorSpace(altRGBA, rgbConversionCoeffs);
                } else {
                    // convert baseRGBA to altRGBA's color space
                    avifLinearRGBConvertColorSpace(baseRGBA, rgbConversionCoeffs);
                }
            }

            for (int c = 0; c < numGainMapChannels; ++c) {
                float base = baseRGBA[c];
                float alt = altRGBA[c];
                if (singleChannel) {
                    // Convert to grayscale.
                    base = yCoeffs[0] * baseRGBA[0] + yCoeffs[1] * baseRGBA[1] + yCoeffs[2] * baseRGBA[2];
                    alt = yCoeffs[0] * altRGBA[0] + yCoeffs[1] * altRGBA[1] + yCoeffs[2] * altRGBA[2];
                }
                if (base > baseMax) {
                    baseMax = base;
                }
                if (alt > altMax) {
                    altMax = alt;
                }
                const float ratio = (alt + alternateOffset[c]) / (base + baseOffset[c]);
                const float ratioLog2 = log2f(AVIF_MAX(ratio, 1e-10f));
                gainMapF[c][(size_t)j * width + i] = ratioLog2;
            }
        }
    }
    }

    // Populate the gain map metadata's headrooms.
    const double baseHeadroom = log2f(AVIF_MAX(baseMax, kEpsilon));
    const double alternateHeadroom = log2f(AVIF_MAX(altMax, kEpsilon));
    if (!avifDoubleToUnsignedFraction(baseHeadroom, &gainMap->baseHdrHeadroom) ||
        !avifDoubleToUnsignedFraction(alternateHeadroom, &gainMap->alternateHdrHeadroom)) {
        res = AVIF_RESULT_INVALID_ARGUMENT;
        goto cleanup;
    }

    // Multiply the gainmap by sign(alternateHdrHeadroom - baseHdrHeadroom), to
    // ensure that it stores the log-ratio of the HDR representation to the SDR
    // representation.
    if (alternateHeadroom < baseHeadroom) {
        for (int c = 0; c < numGainMapChannels; ++c) {
            for (uint32_t j = 0; j < height; ++j) {
                for (uint32_t i = 0; i < width; ++i) {
                    gainMapF[c][(size_t)j * width + i] *= -1.f;
                }
            }
            if (use8BitRawFastPath) {
                for (uint32_t k = 0; k < rawEntries[c].count; ++k) {
                    rawEntries[c].values[k] *= -1.f;
                }
            }
        }
    }

    // Find approximate min/max for each channel, discarding outliers.
    float gainMapMinLog2[3] = { 0.0f, 0.0f, 0.0f };
    float gainMapMaxLog2[3] = { 0.0f, 0.0f, 0.0f };
    for (int c = 0; c < numGainMapChannels; ++c) {
        if (use8BitRawFastPath) {
            res = avifFindMinMaxWithoutOutliersWeighted(&rawEntries[c], numPixels, &gainMapMinLog2[c], &gainMapMaxLog2[c]);
        } else {
            res = avifFindMinMaxWithoutOutliers(gainMapF[c], numPixels, &gainMapMinLog2[c], &gainMapMaxLog2[c]);
        }
        if (res != AVIF_RESULT_OK) {
            goto cleanup;
        }
    }
    if (use8BitRawFastPath) {
        // The compact entries are only needed for the weighted min/max above.
        for (int c = 0; c < 3; ++c) {
            avifFree(rawEntries[c].values);
            avifFree(rawEntries[c].counts);
            rawEntries[c].values = NULL;
            rawEntries[c].counts = NULL;
        }
    }

    // Populate the gain map metadata's min and max values.
    for (int c = 0; c < 3; ++c) {
        if (!avifDoubleToSignedFraction(gainMapMinLog2[singleChannel ? 0 : c], &gainMap->gainMapMin[c]) ||
            !avifDoubleToSignedFraction(gainMapMaxLog2[singleChannel ? 0 : c], &gainMap->gainMapMax[c]) ||
            !avifDoubleToSignedFraction(alternateOffset[c], &gainMap->alternateOffset[c]) ||
            !avifDoubleToSignedFraction(baseOffset[c], &gainMap->baseOffset[c])) {
            res = AVIF_RESULT_INVALID_ARGUMENT;
            goto cleanup;
        }
    }

    // Scale the gain map values to map [min, max] range to [0, 1].
    // On the 8-bit byte-planar fast path the remapped float values are only
    // ever re-read to be quantized into the 8-bit RGBA gain map image, so the
    // remap and the quantization are fused into a single pass below. Every
    // produced byte is computed with the exact same float expressions as the
    // generic remap followed by avifSetRGBAPixel().
    //
    // The fused (and direct YUV) paths produce full-resolution planes, so they
    // can only be used when no subsequent avifImageScale() downscaling is
    // requested. Otherwise the intermediate image dimensions would still be
    // the requested (smaller) ones while the loops write full-resolution
    // samples. Falling back to the generic path keeps every scale case safe.
    const avifBool noDownscaling = (gainMapImage->width == width) && (gainMapImage->height == height);
    const avifBool useFused8BitRemap =
        use8BitRawFastPath && !singleChannel && (gainMapImage->depth == 8) && noDownscaling;
    if (useFused8BitRemap) {
        avifRGBImageSetDefaults(&gainMapRGB, gainMapImage);
    }
    if (!useFused8BitRemap) {
    for (int c = 0; c < numGainMapChannels; ++c) {
        const float range = AVIF_MAX(gainMapMaxLog2[c] - gainMapMinLog2[c], 0.0f);

        if (range == 0.0f) {
            for (uint32_t j = 0; j < height; ++j) {
                for (uint32_t i = 0; i < width; ++i) {
                    // If the range is 0, the gain map values will be multiplied by zero when tonemapping so the values
                    // don't matter, but we still need to make sure that gainMapF is in [0,1].
                    gainMapF[c][(size_t)j * width + i] = 0.0f;
                }
            }
        } else {
            // Remap [min; max] range to [0; 1]
            const float gainMapGamma = avifUnsignedFractionToFloat(gainMap->gainMapGamma[c]);
            // powf(x, 1.0f) is exactly x for every finite float, so skip the
            // transcendental when the metadata asks for a linear gamma. This
            // keeps the same float result while avoiding one powf() per sample.
            const avifBool gammaIsOne = (gainMapGamma == 1.0f);
            for (uint32_t j = 0; j < height; ++j) {
                for (uint32_t i = 0; i < width; ++i) {
                    float v = gainMapF[c][(size_t)j * width + i];
                    v = AVIF_CLAMP(v, gainMapMinLog2[c], gainMapMaxLog2[c]);
                    if (!gammaIsOne) {
                        v = powf((v - gainMapMinLog2[c]) / range, gainMapGamma);
                    } else {
                        v = (v - gainMapMinLog2[c]) / range;
                    }
                    gainMapF[c][(size_t)j * width + i] = avifNanSafeClamp(v);
                }
            }
        }
    }
    }

    // Convert the gain map to YUV.
    const uint32_t requestedWidth = gainMapImage->width;
    const uint32_t requestedHeight = gainMapImage->height;
    gainMapImage->width = width;
    gainMapImage->height = height;

    avifImageFreePlanes(gainMapImage, AVIF_PLANES_ALL); // Free planes in case they were already allocated.

    if (!useFused8BitRemap) {
        avifRGBImageSetDefaults(&gainMapRGB, gainMapImage);
    }
    res = avifRGBImageAllocatePixels(&gainMapRGB);
    if (res != AVIF_RESULT_OK) {
        goto cleanup;
    }

    avifRGBColorSpaceInfo gainMapRGBInfo;
    if (!avifGetRGBColorSpaceInfo(&gainMapRGB, &gainMapRGBInfo)) {
        avifDiagnosticsPrintf(diag, "Unsupported RGB color space");
        res = AVIF_RESULT_NOT_IMPLEMENTED;
        goto cleanup;
    }
    if (useFused8BitRemap) {
        // Re-run the remap and quantize straight into the RGBA byte image.
        const uint32_t pixelBytes = gainMapRGBInfo.pixelBytes;
        const size_t rowBytes = gainMapRGB.rowBytes;
        uint8_t * pixels = gainMapRGB.pixels;
        for (int c = 0; c < 3; ++c) {
            const float range = AVIF_MAX(gainMapMaxLog2[c] - gainMapMinLog2[c], 0.0f);
            const float gainMapGamma = avifUnsignedFractionToFloat(gainMap->gainMapGamma[c]);
            const avifBool gammaIsOne = (gainMapGamma == 1.0f);
            const int rgbOffset = gainMapRGBInfo.offsetBytesR + c;
            for (uint32_t j = 0; j < height; ++j) {
                uint8_t * row = &pixels[j * rowBytes];
                for (uint32_t i = 0; i < width; ++i) {
                    const size_t pixelIndex = (size_t)j * width + i;
                    float v = AVIF_CLAMP(gainMapF[c][pixelIndex], gainMapMinLog2[c], gainMapMaxLog2[c]);
                    if (!gammaIsOne) {
                        v = powf((v - gainMapMinLog2[c]) / range, gainMapGamma);
                    } else {
                        v = (v - gainMapMinLog2[c]) / range;
                    }
                    v = avifNanSafeClamp(v);
                    row[i * pixelBytes + rgbOffset] = (uint8_t)(0.5f + (v * gainMapRGBInfo.maxChannelF));
                }
            }
        }
        // Opaque alpha, matching the generic RGBA pixel set.
        for (uint32_t j = 0; j < height; ++j) {
            uint8_t * row = &pixels[j * rowBytes];
            for (uint32_t i = 0; i < width; ++i) {
                row[i * pixelBytes + gainMapRGBInfo.offsetBytesA] = (uint8_t)gainMapRGBInfo.maxChannel;
            }
        }
    } else {
        for (uint32_t j = 0; j < height; ++j) {
            for (uint32_t i = 0; i < width; ++i) {
                const size_t offset = (size_t)j * width + i;
                const float r = gainMapF[0][offset];
                const float g = singleChannel ? r : gainMapF[1][offset];
                const float b = singleChannel ? r : gainMapF[2][offset];
                const float rgbaPixel[4] = { r, g, b, 1.0f };
                avifSetRGBAPixel(&gainMapRGB, i, j, &gainMapRGBInfo, rgbaPixel);
            }
        }
    }

    // On the fused 8-bit path for an 8-bit full-range YUV444 gain map using
    // the generic RGB-to-YUV coefficient mode, write the YUV planes directly.
    // The formulas below are the exact same float operations (same operands,
    // same operation order) as the generic avifImageRGBToYUV() path, so the
    // resulting samples are bit-identical. Any other configuration falls
    // back to that generic path.
    avifYUVColorSpaceInfo yuvInfo;
    const avifBool directYUV444 = useFused8BitRemap && gainMapImage->yuvFormat == AVIF_PIXEL_FORMAT_YUV444 &&
                                  gainMapImage->yuvRange == AVIF_RANGE_FULL && avifGetYUVColorSpaceInfo(gainMapImage, &yuvInfo) &&
                                  yuvInfo.mode == AVIF_REFORMAT_MODE_YUV_COEFFICIENTS;
    if (directYUV444) {
        res = avifImageAllocatePlanes(gainMapImage, AVIF_PLANES_ALL);
        if (res != AVIF_RESULT_OK) {
            goto cleanup;
        }
        const float kr = yuvInfo.kr;
        const float kg = yuvInfo.kg;
        const float kb = yuvInfo.kb;
        const float rgbMaxChannelF = gainMapRGBInfo.maxChannelF;
        // The RGB input is 8-bit, so each channel's normalized value and its
        // luma contribution only depend on its byte. Precompute the exact same
        // float expressions as the generic path; the per-pixel math then
        // reduces to table lookups and the final quantization.
        float normLUT[256];
        float yContribLUT[3][256];
        for (int v = 0; v <= gainMapRGBInfo.maxChannel; ++v) {
            const float normalized = (float)v / rgbMaxChannelF;
            normLUT[v] = normalized;
            yContribLUT[0][v] = kr * normalized;
            yContribLUT[1][v] = kg * normalized;
            yContribLUT[2][v] = kb * normalized;
        }
        const uint32_t rgbPixelBytes = gainMapRGBInfo.pixelBytes;
        const size_t rgbRowBytes = gainMapRGB.rowBytes;
        const uint8_t * rgbPixels = gainMapRGB.pixels;
        uint8_t * yPlane = gainMapImage->yuvPlanes[AVIF_CHAN_Y];
        uint8_t * uPlane = gainMapImage->yuvPlanes[AVIF_CHAN_U];
        uint8_t * vPlane = gainMapImage->yuvPlanes[AVIF_CHAN_V];
        const size_t yRowBytes = gainMapImage->yuvRowBytes[AVIF_CHAN_Y];
        const size_t uRowBytes = gainMapImage->yuvRowBytes[AVIF_CHAN_U];
        const size_t vRowBytes = gainMapImage->yuvRowBytes[AVIF_CHAN_V];
        const int offR = gainMapRGBInfo.offsetBytesR;
        const int offG = gainMapRGBInfo.offsetBytesG;
        const int offB = gainMapRGBInfo.offsetBytesB;
        for (uint32_t j = 0; j < height; ++j) {
            const uint8_t * rgbRow = &rgbPixels[j * rgbRowBytes];
            uint8_t * yOut = &yPlane[j * yRowBytes];
            uint8_t * uOut = &uPlane[j * uRowBytes];
            uint8_t * vOut = &vPlane[j * vRowBytes];
            for (uint32_t i = 0; i < width; ++i) {
                const uint8_t * rgbPixel = &rgbRow[i * rgbPixelBytes];
                const uint8_t r = rgbPixel[offR];
                const uint8_t g = rgbPixel[offG];
                const uint8_t b = rgbPixel[offB];
                const float R = normLUT[r];
                const float B = normLUT[b];
                // Same operation order as the generic coefficient path:
                // Y is built by one chained addition, then U/V are derived.
                const float Y = yContribLUT[0][r] + yContribLUT[1][g] + yContribLUT[2][b];
                const float U = (B - Y) / (2 * (1 - kb));
                const float V = (R - Y) / (2 * (1 - kr));
                yOut[i] = (uint8_t)avifYToUNormFull8(&yuvInfo, Y);
                uOut[i] = (uint8_t)avifUVToUNormFull8(&yuvInfo, U);
                vOut[i] = (uint8_t)avifUVToUNormFull8(&yuvInfo, V);
            }
        }
        // Generic path allocates an opaque alpha plane for RGBA inputs.
        if (gainMapImage->alphaPlane) {
            for (uint32_t j = 0; j < height; ++j) {
                memset(&gainMapImage->alphaPlane[j * gainMapImage->alphaRowBytes], 255, gainMapImage->alphaRowBytes);
            }
        }
    } else {
        res = avifImageRGBToYUV(gainMapImage, &gainMapRGB);
        if (res != AVIF_RESULT_OK) {
            goto cleanup;
        }
    }

    // Scale down the gain map if requested.
    // Another way would be to scale the source images, but it seems to perform worse.
    if (requestedWidth != gainMapImage->width || requestedHeight != gainMapImage->height) {
        res = avifImageScale(gainMap->image, requestedWidth, requestedHeight, diag);
        if (res != AVIF_RESULT_OK) {
            goto cleanup;
        }
    }

cleanup:
    for (int c = 0; c < 3; ++c) {
        avifFree(gainMapF[c]);
        if (use8BitRawFastPath) {
            avifFree(rawEntries[c].values);
            avifFree(rawEntries[c].counts);
        }
    }
    avifRGBImageFreePixels(&gainMapRGB);
    if (res != AVIF_RESULT_OK) {
        avifImageFreePlanes(gainMapImage, AVIF_PLANES_ALL);
    }

    return res;
}

avifResult avifImageComputeGainMap(const avifImage * baseImage, const avifImage * altImage, avifGainMap * gainMap, avifDiagnostics * diag)
{
    avifDiagnosticsClearError(diag);

    if (baseImage == NULL || altImage == NULL || gainMap == NULL) {
        return AVIF_RESULT_INVALID_ARGUMENT;
    }
    if (baseImage->icc.size > 0 || altImage->icc.size > 0) {
        avifDiagnosticsPrintf(diag, "Computing gain maps for images with ICC profiles is not supported");
        return AVIF_RESULT_NOT_IMPLEMENTED;
    }
    if (baseImage->width != altImage->width || baseImage->height != altImage->height) {
        avifDiagnosticsPrintf(diag,
                              "Image dimensions don't match, got %dx%d and %dx%d",
                              baseImage->width,
                              baseImage->height,
                              altImage->width,
                              altImage->height);
        return AVIF_RESULT_INVALID_ARGUMENT;
    }

    avifResult res = AVIF_RESULT_OK;

    avifRGBImage baseImageRgb;
    avifRGBImageSetDefaults(&baseImageRgb, baseImage);
    avifRGBImage altImageRgb;
    avifRGBImageSetDefaults(&altImageRgb, altImage);

    AVIF_CHECKRES(avifRGBImageAllocatePixels(&baseImageRgb));
    // --- After this point, the function should exit with 'goto cleanup' to free allocated resources.

    res = avifImageYUVToRGB(baseImage, &baseImageRgb);
    if (res != AVIF_RESULT_OK) {
        goto cleanup;
    }
    res = avifRGBImageAllocatePixels(&altImageRgb);
    if (res != AVIF_RESULT_OK) {
        goto cleanup;
    }
    res = avifImageYUVToRGB(altImage, &altImageRgb);
    if (res != AVIF_RESULT_OK) {
        goto cleanup;
    }

    res = avifRGBImageComputeGainMap(&baseImageRgb,
                                     baseImage->colorPrimaries,
                                     baseImage->transferCharacteristics,
                                     &altImageRgb,
                                     altImage->colorPrimaries,
                                     altImage->transferCharacteristics,
                                     gainMap,
                                     diag);

    if (res != AVIF_RESULT_OK) {
        goto cleanup;
    }

    AVIF_CHECKRES(avifRWDataSet(&gainMap->altICC, altImage->icc.data, altImage->icc.size));
    gainMap->altColorPrimaries = altImage->colorPrimaries;
    gainMap->altTransferCharacteristics = altImage->transferCharacteristics;
    gainMap->altMatrixCoefficients = altImage->matrixCoefficients;
    gainMap->altDepth = altImage->depth;
    gainMap->altPlaneCount = (altImage->yuvFormat == AVIF_PIXEL_FORMAT_YUV400) ? 1 : 3;
    gainMap->altCLLI = altImage->clli;

cleanup:
    avifRGBImageFreePixels(&baseImageRgb);
    avifRGBImageFreePixels(&altImageRgb);
    return res;
}
