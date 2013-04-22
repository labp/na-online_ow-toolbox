#ifndef LFRAWDATA_H
#define LFRAWDATA_H

#include <inttypes.h>
using namespace std;

#include "LFDataBuffer.h"
#include "LFArrayPtr.h"

/**
 * Raw Data block (102)
 */
class LFRawData
{
protected:
    int32_t m_FirstSample;/**< The first sample of an epoch, default == -1 (208) */
    int32_t m_DataSkip;/**< Number of blocks to skip before actual data starts, default == -1 (301) */
    int32_t m_DataSkipSamples;/**< Data skip in samples, default == -1 (303) */
    LFArrayPtr<LFDataBuffer> m_LFDataBuffer;/**< Buffer containing measurement data (300) */
public:
    /**
     * Constructor
     */
    LFRawData();
    /**
     * Destructor
     */
    ~LFRawData();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the first sample of an epoch
     */
    int32_t GetFirstSample();
    /**
     * Returns the number of blocks to skip before actual data starts
     */
    int32_t GetDataSkip();
    /**
     * Returns data skip in samples
     */
    int32_t GetDataSkipSamples();
    /**
     * Returns buffers containing measurement data
     */
    LFArrayPtr<LFDataBuffer>& GetLFDataBuffer();
    /**
     * Sets the first sample of an epoch
     */
    void SetFirstSample( const int32_t src );
    /**
     * Sets the number of blocks to skip before actual data starts
     */
    void SetDataSkip( const int32_t src );
    /**
     * Sets data skip in samples
     */
    void SetDataSkipSamples( const int32_t src );
};

#endif
