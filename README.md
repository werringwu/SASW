# A Novel Perspective on Low-Light Image Enhancement: Leveraging Artifact Regularization and Walsh-Hadamard Transform (SASW)

<a href="https://werring.wu.github.io">Weilin Wu*,</a> Shifan Yang, Qizhao Lin, XingHong Chen, Kunping Yang, Jing Wang and Guannan Chen <sup>✉️</sup>

> **Abstract:** *Low-light image enhancement (LLIE) aims to restore low-light images to normal lighting conditions by improving their illumination and fine details, thereby facilitating efficient execution of downstream visual tasks. Traditional LLIE methods improve image quality but often introduce high-frequency artifacts, which are difficult to eliminate, hindering detail recovery and quality enhancement in LLIE.
To solve this problem, we introduce a novel perspective: instead of traditional artifact suppression, sparsification-induced artifacts are repurposed as constructive regularization signals to guide detail recovery. By analyzing the impact of sparsified frequency components and their role in reconstruction artifacts, a detailed mathematical framework is presented. Specifically, we propose a novel loss function SASW-Loss which combining Sparse Artifact Similarity Loss (SAS-Loss) and Walsh-Hadamard Coefficient Loss (WHC-Loss).
SAS-Loss mitigates the over-compensation of missing frequencies, helping the network recover structural details, while WHC-Loss optimizes the frequency-domain representation, restoring luminance, suppressing noise, and enhancing both structure and details. Extensive experiments show that our approach outperforms existing state-of-the-art methods, achieving superior performance in structural detail preservation and noise suppression. These results validate the effectiveness of our new perspective, which leverages sparsification artifacts to guide detail recovery, demonstrating significant improvements and robust performance across multiple models, and opening new avenues for future research.* 
<hr />
The code will coming soon!!!

## Main Results

### Results on UHD-LL and UHDLOL4K
![fig2.png](Figures/Fig2.png)

### Results on LOLv1 and LOLv2-Real
![fig3.png](Figures/Fig3.png)

### Visual Results
![fig4.png](Figures/Fig4.png)
![fig5.png](Figures/Fig5.png)
![fig6.png](Figures/Fig6.png)

## Citation
```
@inproceedings{wu2025sasw,
 title={A Novel Perspective on Low-Light Image Enhancement: Leveraging Artifact Regularization and Walsh-Hadamard Transform},
 author={Weilin Wu and Shifan Yang and Qizhao Lin and XingHong Chen and Kunping Yang and Jing Wang and Guannan Chen},
 booktitle={ACM Multimedia 2025},
 year= {2025},
 url={https://10.1145/3746027.3758142}
}
```
## License

## Acknowledgement
This project is based on [RetinexFormer](), [RetinexMamba](), [HWMNet](), [SNR-Net](), [WaveMamba]() and [FourLLIE]().
