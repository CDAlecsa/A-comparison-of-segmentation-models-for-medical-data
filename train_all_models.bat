@echo off
echo Starting all model runs...
echo.

set MODELS=^
TU_1 ^
SMP_FPN_1 ^
SMP_LINK_NET_1 ^
SMP_MA_NET_1 ^
SMP_UNET_1 ^
SMP_UNET_PP_1 ^
MTU_1 ^
MTU_2 ^
MTU_3 ^
CBIM_MEDFORMER_1 ^
CBIM_DA_UNET_1 ^
CBIM_ATT_UNET_1 ^
TU_2 ^
TU_3 ^
SU_1 ^
SU_2 ^
SU_3

for %%m in (%MODELS%) do (
    echo =======================================
    echo Running model: %%m
    echo =======================================
    python train.py --config-file %%m
    echo.
)

echo.
echo All model runs finished.
pause
