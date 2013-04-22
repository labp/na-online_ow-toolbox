#ifndef RETURNCODES_H
#define RETURNCODES_H

/**
 * Return codes
 */
enum returncode_t
{
    rc_normal=0/**< Normal */
    ,rc_error_file_open/**< Error opening file */
    ,rc_error_file_read/**< File read error */
    ,rc_error_unknown/**< Unknown error */
};

#endif //RETURNCODES_H
