// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <src/kernel/memutils.h>

class Cuda3dArray : public Cuda3dArrayBase
{
public:
    Cuda3dArray(const size_t width, const size_t height, const size_t depth)
    {
        // Set the channel description for the 3D array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint32_t>();

        // Set the extent of the 3D array
        extent = make_cudaExtent(width, height, depth);

        // Allocate the 3D array
        cudaMalloc3DArray(&d_array, &channelDesc, extent);
        if (!d_array)
            throw std::runtime_error("Failed to allocate 3D array");
    }

    ~Cuda3dArray()
    {
        // Destroy the texture object if it was created
        if (textureObject != 0)
            cudaDestroyTextureObject(textureObject);        

        // Destroy the surface object if it was created
        if (surfaceObject != 0)
            cudaDestroySurfaceObject(surfaceObject);

        // Free the 3D array
        if (d_array)
            cudaFreeArray(d_array);
    }

    void createTextureObject() override
    {
        if (textureObject != 0)
            return;

        // Resource description
        struct cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = d_array;

        // Texture description
        struct cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.addressMode[2] = cudaAddressModeBorder;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;

        // Create texture object
        cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr);
    }

    void createSurfaceObject() override
    {
        if (surfaceObject != 0)
            return;

        // Resource description
        struct cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = d_array;

        // Create surface object
        cudaCreateSurfaceObject(&surfaceObject, &resDesc);
    }

    uint64_t getTextureObject() const override
    {
        return textureObject;
    }

    uint64_t getSurfaceObject() const override
    {
        return surfaceObject;
    }

private:
    cudaArray_t d_array = nullptr;
    cudaExtent extent;
    cudaTextureObject_t textureObject = 0;
    cudaSurfaceObject_t surfaceObject = 0;
};
