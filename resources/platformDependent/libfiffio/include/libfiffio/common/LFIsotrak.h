#ifndef LFISOTRAK_H
#define LFISOTRAK_H

#include "LFDigitisationPoint.h"
#include "LFArrayPtr.h"

/**
 * Isotrak block (107)
 */
class LFIsotrak
{
protected:
    LFArrayPtr<LFDigitisationPoint> m_DigitisationPoint;/**< Digitisation Point (213) */
public:
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns Digitisation Point (213)
     */
    LFArrayPtr<LFDigitisationPoint>& GetLFDigitisationPoint();
};
#endif
