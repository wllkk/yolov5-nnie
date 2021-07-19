/*
* Copyright (C) Hisilicon Technologies Co., Ltd. 2012-2019. All rights reserved.
* Description:
* Author: Hisilicon multimedia software group
* Create: 2011/06/28
*/


#ifndef __HI_ISP_BIN_H__
#define __HI_ISP_BIN_H__

#include "hi_type.h"
#include "hi_comm_isp.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

/****************************************************************************
 * MACRO DEFINITION                                                         *
 ****************************************************************************/
#define MAX_BIN_REG_NUM 5

/****************************************************************************
 * GENERAL STRUCTURES                                                       *
 ****************************************************************************/
typedef struct hiISP_BIN_REG_ATTR_S {
    HI_U32  u32Addr;    /* register addr */
    HI_U8   u8StartBit; /* start bit of register addr */
    HI_U8   u8EndBit;   /* end bit of register addr */
} ISP_BIN_REG_ATTR_S;

/* -------------------------------------------*
 The base addr of ISP logic register
 The base addr of ISP ext register
 The base addr of Hisi AE ext register
 The base addr of Hisi AWB ext register
* ------------------------------------------- */

ISP_BIN_REG_ATTR_S g_astIspBinRegAttr[ISP_MAX_PIPE_NUM][MAX_BIN_REG_NUM] = {
    [0 ...(ISP_MAX_PIPE_NUM - 1)] = {0}
};


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /* __HI_ISP_BIN_H__ */
