#ifndef LFUNITS_H
#define LFUNITS_H
/**
 * Value units
 */
enum fiffunits_t
{
    unit_none = -1 /**< No unit */
    ,unit_unitless = 0 /**< unitless */
    ,unit_m = 1 /**< meter */
    ,unit_kg = 2 /**< kilogram */
    ,unit_sec = 3 /**< second */
    ,unit_A = 4 /**< ampere */
    ,unit_K = 5 /**< Kelvin */
    ,unit_mol = 6 /**< mole */
    ,unit_rad = 7 /**< radian */
    ,unit_sr = 8 /**< steradian */
    ,unit_cd = 9 /**< candela */
    ,unit_Hz = 101 /**< herz */
    ,unit_N = 102 /**< Newton */
    ,unit_Pa = 103 /**< pascal */
    ,unit_J = 104 /**< joule */
    ,unit_W = 105 /**< watt */
    ,unit_C = 106 /**< coulomb */
    ,unit_V = 107 /**< volt */
    ,unit_F = 108 /**< farad */
    ,unit_Ohm = 109 /**< ohm */
    ,unit_Mho = 110 /**< one per ohm */
    ,unit_Wb = 111 /**< weber */
    ,unit_T = 112 /**< tesla */
    ,unit_H = 113 /**< Henry */
    ,unit_Cel = 114 /**< celcius */
    ,unit_lm = 115 /**< lumen */
    ,unit_lx = 116 /**< lux */
    ,unit_T_m = 201 /**< T/m */
    ,unit_Am = 202 /**< Am */
};

#endif
