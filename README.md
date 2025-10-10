# A Novel Perspective on Low-Light Image Enhancement: Leveraging Artifact Regularization and Walsh-Hadamard Transform (SASW)

<a href="https://werring.wu.github.io">Weilin Wu*,</a> Shifan Yang, Qizhao Lin, XingHong Chen, Kunping Yang, Jing Wang and Guannan Chen <sup>✉️</sup>

> **Abstract:** *Low-light image enhancement (LLIE) aims to restore low-light images to normal lighting conditions by improving their illumination and fine details, thereby facilitating efficient execution of downstream visual tasks. Traditional LLIE methods improve image quality but often introduce high-frequency artifacts, which are difficult to eliminate, hindering detail recovery and quality enhancement in LLIE.
To solve this problem, we introduce a novel perspective: instead of traditional artifact suppression, sparsification-induced artifacts are repurposed as constructive regularization signals to guide detail recovery. By analyzing the impact of sparsified frequency components and their role in reconstruction artifacts, a detailed mathematical framework is presented. Specifically, we propose a novel loss function SASW-Loss which combining Sparse Artifact Similarity Loss (SAS-Loss) and Walsh-Hadamard Coefficient Loss (WHC-Loss).
SAS-Loss mitigates the over-compensation of missing frequencies, helping the network recover structural details, while WHC-Loss optimizes the frequency-domain representation, restoring luminance, suppressing noise, and enhancing both structure and details. Extensive experiments show that our approach outperforms existing state-of-the-art methods, achieving superior performance in structural detail preservation and noise suppression. These results validate the effectiveness of our new perspective, which leverages sparsification artifacts to guide detail recovery, demonstrating significant improvements and robust performance across multiple models, and opening new avenues for future research.* 
<hr />
The code will coming soon!!!

## Main FlowChart
![flowchart](Imgs/framwork0.jpg)

## Detail of WHT Generate Artifacts
![WHT](Imgs/detail_wht_merge.jpg)

## Main Results

### Visual Results
![fig2.png](Imgs/visual_v1_v2_result_0.jpg)
![fig3.png](Imgs/unpairImg_01.jpg)
![fig4.png](Imgs/unpairImg_02.jpg)
![fig5.png](supplement/supp_compare_0.jpg)
![fig6.png](supplement/supp_compare_1.png)
![fig7.png](supplement/supp_unpair_FourLLIE.jpg)
![fig8.png](supplement/supp_unpair_FourLLIE-0.jpg)
![fig9.png](supplement/supp_unpair_snr.jpg)
![fig10.png](supplement/supp_unpair_waveMamba.jpg)

## Citation
```
@inproceedings{wu2025sasw,
 title={A Novel Perspective on Low-Light Image Enhancement: Leveraging Artifact Regularization and Walsh-Hadamard Transform},
 author={Weilin Wu and Shifan Yang and Qizhao Lin and XingHong Chen and Kunping Yang and Jing Wang and Guannan Chen},
 booktitle={ACM Multimedia 2025},
 year= {2025},
 doi={10.1145/3746027.3758142}
}
```
## License

## Acknowledgement
## Acknowledgement
This project is based on [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer), [RetinexMamba](https://github.com/YhuoyuH/RetinexMamba), [HWMNet](https://github.com/FanChiMao/HWMNet), [SNR-Net](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance), [WaveMamba](https://github.com/AlexZou14/Wave-Mamba) and [FourLLIE](https://github.com/wangchx67/FourLLIE).
