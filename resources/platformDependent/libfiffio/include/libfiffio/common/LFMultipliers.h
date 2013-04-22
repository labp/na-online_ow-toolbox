#ifndef LFMULTIPLIERS_H
#define LFMULTIPLIERS_H
/**
 * Value multipliers
 */
enum fiffmultipliers_t
{
    mul_none = 0 /**< 10^0 */
    ,mul_pet = 15 /**< 10^15 */
    ,mul_t = 12 /**< 10^12 */
    ,mul_gig = 9 /**< 10^9 */
    ,mul_meg = 6 /**< 10^6 */
    ,mul_k = 3 /**< 10^3 */
    ,mul_h = 2 /**< 10^2 */
    ,mul_da = 1 /**< 10^1 */
    ,mul_e = 18 /**< 10^18 */
    ,mul_d = -1 /**< 10^-1 */
    ,mul_c = -2 /**< 10^-2 */
    ,mul_m = -3 /**< 10^-3 */
    ,mul_mu = -6 /**< 10^-6 */
    ,mul_n = -9 /**< 10^-9 */
    ,mul_p = -12 /**< 10^-12 */
    ,mul_f = -15 /**< 10^-15 */
    ,mul_a = -18 /**< 10^-18 */
};

#endif
