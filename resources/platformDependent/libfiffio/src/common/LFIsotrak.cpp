#include "LFIsotrak.h"

void LFIsotrak::Init()
{
    m_DigitisationPoint.clear();
}

LFArrayPtr<LFDigitisationPoint>& LFIsotrak::GetLFDigitisationPoint()
{
    return m_DigitisationPoint;
}
