#ifndef LFMEASUREMENT_H
#define LFMEASUREMENT_H

#include "LFMeasurementInfo.h"
#include "LFRawData.h"
#include "LFBem.h"

/**
 * Measurement block (100)
 */
class LFMeasurement
{
protected:
    LFMeasurementInfo m_LFMeasurementInfo;/**< Measurement-Info block (101) */
    LFRawData m_LFRawData;/**< Raw Data block (102) */
    LFBem m_LFBem;/**< Bem block (310) */
public:
    /**
     * Constructor
     */
    LFMeasurement();
    /**
     * Destructor
     */
    ~LFMeasurement();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the Measurement-Info block (101)
     */
    LFMeasurementInfo& GetLFMeasurementInfo();
    /**
     * Returns the Raw Data block (102)
     */
    LFRawData& GetLFRawData();
    /**
     * Returns the Bem block (310)
     */
    LFBem& GetLFBem();
};

#endif
