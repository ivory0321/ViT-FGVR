# ViT-FGVR
Vehicle classification using ViT and High Pass Filtered Images

**Please visit https://ivory218.com/wordpress/ for full report and data/model download.**

This project uses High Pass Filter to guide attention layer to small intra-class vehicle variations.

By stacking HPF images of 3 different level into 3 channels, rather than standard RGB, the model predicts from reasonable interest areas(grills, doors, wheels, etc.) and have strong noise tolerance such as rain droplets, snow on the surface, and strong lights.

Single channel HPF model is twice faster in training and inference while achieving slightly better performance than the baseline model.

