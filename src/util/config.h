/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2023/10/30.
 */

#pragma once

#ifndef BITRUSS_CONFIG_H
#define BITRUSS_CONFIG_H

#define MAX_IDS 99999999
#define THREADS 16
#define BLK_NUMS 168
#define BLK_DIM 1024
#define WARP_SIZE 32
#define WARPS_EACH_BLK (BLK_DIM / WARP_SIZE)
#define N_THREADS (BLK_DIM * BLK_NUMS)

#define GLBUFFER_SIZE 1000000



//#define COUNT_BUTTERFLY
#define LOG_USE_COLOR



#endif //BITRUSS_CONFIG_H
