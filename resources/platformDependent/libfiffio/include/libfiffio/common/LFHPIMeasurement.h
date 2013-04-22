#ifndef LFHPIMEASUREMENT_H
#define LFHPIMEASUREMENT_H

#include <inttypes.h>

#include "LFHPICoil.h"

/**
 * HPI Measurement block (108)
 */
class LFHPIMeasurement
{
protected:
    float m_SamplingFrequency;/**< Sampling Frequency, Hz, default == -FLT_MAX (201) */
    int32_t m_NumberOfChannels;/**< Number of Channels, default == -1 (200) */
    int32_t m_NumberOfAverages;/**< Number of Averages, default == -1 (207) */
    int32_t m_NumberOfHPICoils;/**< Number of HPI Coils, default == -1 (216) */
    int32_t m_FirstSample;/**< First Sample of Epoch, default == -1 (208) */
    int32_t m_LastSample;/**< Last Sample of Epoch, default == -1 (209) */
    LFArrayPtr<LFHPICoil> m_LFHPICoil;/**< HPI Coil block (110) */
public:
    /**
     * Constructor
     */
    LFHPIMeasurement();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the Sampling Frequency
     */
    float GetSamplingFrequency();
    /**
     * Returns the Number of Channels
     */
    int32_t GetNumberOfChannels();
    /**
     * Returns the Number of Averages
     */
    int32_t GetNumberOfAverages();
    /**
     * Returns the Number of HPI Coils
     */
    int32_t GetNumberOfHPICoils();
    /**
     * Returns the First Sample
     */
    int32_t GetFirstSample();
    /**
     * Returns the Last Sample
     */
    int32_t GetLastSample();
    /**
     * Returns the HPI Coil block
     */
    LFArrayPtr<LFHPICoil>& GetLFHPICoil();
    /**
     * Sets the Sampling Frequency
     */
    void SetSamplingFrequency( const float src );
    /**
     * Sets the Number of Channels
     */
    void SetNumberOfChannels( const int32_t src );
    /**
     * Sets the Number of Averages
     */
    void SetNumberOfAverages( const int32_t src );
    /**
     * Sets the Number of HPI Coils
     */
    void SetNumberOfHPICoils( const int32_t src );
    /**
     * Sets the First Sample
     */
    void SetFirstSample( const int32_t src );
    /**
     * Sets the Last Sample
     */
    void SetLastSample( const int32_t src );
};

#endif
