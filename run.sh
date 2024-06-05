############################################################
# BERT
############################################################
# FP32
export DNNL_DEFAULT_FPMATH_MODE=FP32
python3 bert-clip-vit.py --bert --fp32
# BF16 (Compile, Autocast, Dynamic)
export DNNL_DEFAULT_FPMATH_MODE=BF16
python3 bert-clip-vit.py --bert --fp32 --compile --dynamic --autocast
############################################################
# CLIP-VIT
############################################################
export DNNL_DEFAULT_FPMATH_MODE=FP32
python3 bert-clip-vit.py --clip --fp32
export DNNL_DEFAULT_FPMATH_MODE=BF16
python3 bert-clip-vit.py --clip --fp32 --compile --dynamic --autocast
############################################################
# FB-DETR
############################################################
export DNNL_DEFAULT_FPMATH_MODE=FP32
python3 fb-detr.py --fp32
export DNNL_DEFAULT_FPMATH_MODE=BF16
python3 fb-detr.py --fp32 --compile --dynamic --autocast
############################################################
# T5 - Base
############################################################
export DNNL_DEFAULT_FPMATH_MODE=FP32
python3 t5.py --fp32 --base
export DNNL_DEFAULT_FPMATH_MODE=BF16
python3 t5.py --fp32 --base --compile --dynamic --autocast
############################################################
# T5 - SMALL
############################################################
export DNNL_DEFAULT_FPMATH_MODE=FP32
python3 t5.py --fp32 --small
export DNNL_DEFAULT_FPMATH_MODE=BF16
python3 t5.py --fp32 --small --compile --dynamic --autocast
############################################################
# T5 - SMALL
############################################################
export DNNL_DEFAULT_FPMATH_MODE=FP32
python3 msft_phi2.py --fp32 
export DNNL_DEFAULT_FPMATH_MODE=BF16
python3 msft_phi2.py --fp32 --compile --dynamic --autocast
