#ifndef LFHPICOIL_H
#define LFHPICOIL_H

#include <inttypes.h>


#include "LFArrayPtr.h"
#include "LFArrayFloat2d.h"

/**
 * HPI Coil block (110)
 */
class LFHPICoil
{
protected:
    int32_t m_CoilNumber;/**< Coil Number, default == -1 (245) */
    LFArrayFloat2d m_Epoch;/**< Buffer containing one Epoch and channel (302) */
    vector< float > m_Slopes;/**< Slopes, T, T/m (215) */
    vector< float > m_CorrelationCoefficient;/**< Correlation Coefficient (226) */
    float m_CoilFrequency;/**< Hz range, default == -FLT_MAX (236) */
public:
    /**
     * Constructor
     */
    LFHPICoil();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the Coil Number
     */
    int32_t GetCoilNumber();
    /**
     * Returns the Liste der Epochen
     */
    LFArrayFloat2d& GetEpoch();
    /**
     * Returns the Slopes
     */
    vector< float >& GetSlopes();
    /**
     * Returns the Correlation Coefficient
     */
    vector< float >& GetCorrelationCoefficient();
    /**
     * Returns the Hz range
     */
    float GetCoilFrequency();
    /**
     * Sets the Coil Number
     */
    void SetCoilNumber( const int32_t src );
    /**
     * Sets Hz range
     */
    void SetCoilFrequency( const float src );
};
#endif
