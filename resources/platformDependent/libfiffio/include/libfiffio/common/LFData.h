#ifndef LFDATA_H
#define LFDATA_H

#include "LFMeasurement.h"

/**
 * FIFF - Data
 */
class LFData
{
protected:
    LFMeasurement m_LFMeasurement;/**< Measurement block (100) */
public:
    /**
     * Constructor
     */
    LFData();
    /**
     * Destructor
     */
    ~LFData();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the Measurement Block
     */
    LFMeasurement& GetLFMeasurement();
};

#endif
