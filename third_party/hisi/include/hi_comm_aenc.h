/*
 * Copyright (C) Hisilicon Technologies Co., Ltd. 2012-2019. All rights reserved.
 * Description: hi_comm_aenc.h
 * Author: Hisilicon multimedia software group
 * Create: 2006/12/15
 * History                 :
 *  1.Date                 :   2006/12/15
 *    Modification         :   Created file
 *  2.Date                 :   2007/5/10
 *    Modification         :   add err code
 */

#ifndef  __HI_COMM_AENC_H__
#define  __HI_COMM_AENC_H__

#include "hi_type.h"
#include "hi_common.h"
#include "hi_comm_aio.h"


#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

typedef struct hiAENC_ATTR_G711_S {
    HI_U32 resv;            /* reserve item */
} AENC_ATTR_G711_S;

typedef struct hiAENC_ATTR_G726_S {
    G726_BPS_E enG726bps;
} AENC_ATTR_G726_S;

typedef struct hiAENC_ATTR_ADPCM_S {
    ADPCM_TYPE_E enADPCMType;
} AENC_ATTR_ADPCM_S;

typedef struct hiAENC_ATTR_LPCM_S {
    HI_U32 resv;            /* reserve item */
} AENC_ATTR_LPCM_S;

typedef struct hiAENC_ENCODER_S {
    PAYLOAD_TYPE_E  enType;
    HI_U32          u32MaxFrmLen;
    HI_CHAR         aszName[17];    /* encoder type,be used to print proc information */
    HI_S32          (*pfnOpenEncoder)(HI_VOID *pEncoderAttr, HI_VOID **ppEncoder); /* pEncoder is the handle to control the encoder */
    HI_S32          (*pfnEncodeFrm)(HI_VOID *pEncoder, const AUDIO_FRAME_S *pstData,
                        HI_U8 *pu8Outbuf, HI_U32 *pu32OutLen);
    HI_S32          (*pfnCloseEncoder)(HI_VOID *pEncoder);
} AENC_ENCODER_S;

typedef struct hiAENC_CHN_ATTR_S {
    PAYLOAD_TYPE_E      enType;
    HI_U32              u32PtNumPerFrm;
    HI_U32              u32BufSize;      /* buf size [2~MAX_AUDIO_FRAME_NUM] */
    HI_VOID ATTRIBUTE   *pValue;  /* point to attribute of definite audio encoder */
} AENC_CHN_ATTR_S;

typedef enum hiEN_AENC_ERR_CODE_E {
    AENC_ERR_ENCODER_ERR     = 64,
    AENC_ERR_VQE_ERR         = 65,
} EN_AENC_ERR_CODE_E;


/* invlalid device ID */
#define HI_ERR_AENC_INVALID_DEVID     HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_INVALID_DEVID)
/* invlalid channel ID */
#define HI_ERR_AENC_INVALID_CHNID     HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_INVALID_CHNID)
/* at lease one parameter is illagal ,eg, an illegal enumeration value  */
#define HI_ERR_AENC_ILLEGAL_PARAM     HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_ILLEGAL_PARAM)
/* channel exists */
#define HI_ERR_AENC_EXIST             HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_EXIST)
/* channel unexists */
#define HI_ERR_AENC_UNEXIST           HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_UNEXIST)
/* using a NULL point */
#define HI_ERR_AENC_NULL_PTR          HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_NULL_PTR)
/* try to enable or initialize system,device or channel, before configing attribute */
#define HI_ERR_AENC_NOT_CONFIG        HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_NOT_CONFIG)
/* operation is not supported by NOW */
#define HI_ERR_AENC_NOT_SUPPORT       HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_NOT_SUPPORT)
/* operation is not permitted ,eg, try to change static attribute */
#define HI_ERR_AENC_NOT_PERM          HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_NOT_PERM)
/* failure caused by malloc memory */
#define HI_ERR_AENC_NOMEM             HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_NOMEM)
/* failure caused by malloc buffer */
#define HI_ERR_AENC_NOBUF             HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_NOBUF)
/* no data in buffer */
#define HI_ERR_AENC_BUF_EMPTY         HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_BUF_EMPTY)
/* no buffer for new data */
#define HI_ERR_AENC_BUF_FULL          HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_BUF_FULL)
/* system is not ready,had not initialed or loaded */
#define HI_ERR_AENC_SYS_NOTREADY      HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, EN_ERR_SYS_NOTREADY)
/* encoder internal err */
#define HI_ERR_AENC_ENCODER_ERR       HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, AENC_ERR_ENCODER_ERR)
/* vqe internal err */
#define HI_ERR_AENC_VQE_ERR       HI_DEF_ERR(HI_ID_AENC, EN_ERR_LEVEL_ERROR, AENC_ERR_VQE_ERR)


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif/* End of #ifndef __HI_COMM_AENC_H__ */

