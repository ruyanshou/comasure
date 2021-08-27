# COMASure - CoCoNet MAML based Super-Resolution Network
This repository stores scripts used to run COMASure and its extensions. The models are studied as part of the requirements for the MSc Data Science and Machine Learning dissertation at UCL.

**Disclaimers:** All the implementations in this repository are made for research exploration and educational purposes only. They are not ready for industrial usages and the effectiveness is not guaranteed. In addition, permission should be obtained from the author before any distribution or reproduction.

## Single Domain Image Super-Resolution
Example: Image 010836 of CelebA\
Bicubic: PSNR 23.2, SSIM 0.780\
Meta Initialization: PSNR 23.4, SSIM 0.785\
![010836](https://user-images.githubusercontent.com/61622080/131056665-0537f629-82c0-4408-858e-4cc5970d439e.png)

## Cross-Domain Image Super-Resolution
Example: Image 0808 of DIV2K\
Bicubic: PSNR 21.0, SSIM 0.479\
Meta Initialization: PSNR 21.2, SSIM 0.494\
![0808](https://user-images.githubusercontent.com/61622080/131056507-ddd12583-8e74-4fc5-b013-d0ee2fd05523.png)

## Code References
CoCoNet: https://github.com/paulbricman/python-fuse-coconet \
CoCoNet+MAML: https://github.com/tancik/learnit \
Fourier Feature Transformation: https://github.com/tancik/fourier-feature-networks \
GAN: https://github.com/eriklindernoren/PyTorch-GAN \
MAML: https://github.com/cbfinn/maml \
SIREN Layer: https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb

## Other References
Please refer to the actual report.
