# Ensemble and low-frequency mixing with diffusion models for accelerated MRI reconstruction



[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 

This repo contains the code for our paper  <a href="https://doi.org/10.1016/j.media.2025.103477"> **Ensemble and low-frequency mixing with diffusion models for accelerated MRI reconstruction**  </a>.

![Overview of framework]![Image](https://github.com/user-attachments/assets/08f51efb-7f35-4464-8185-dad3ae43015f)

> Magnetic resonance imaging (MRI) is an important imaging modality in medical diagnosis, providing comprehensive anatomical information with detailed tissue structures. However, the long scan time required to acquire high-quality MR images is a major challenge, especially in urgent clinical scenarios. Although diffusion models have achieved remarkable performance in accelerated MRI, there are several challenges. In particular, they struggle with the long inference time due to the high number of iterations in the reverse process of diffusion models. Additionally, they occasionally create artifacts or ‘hallucinate’ tissues that do not exist in the original anatomy. To address these problems, we propose ensemble and adaptive low-frequency mixing on the diffusion model, namely ELF-Diff for accelerated MRI. The proposed method consists of three key components in the reverse diffusion step: (1) optimization based on unified data consistency; (2) low-frequency mixing; and (3) aggregation of multiple perturbations of the predicted images for the ensemble in each step. We evaluate ELF-Diff on two MRI datasets, FastMRI and SKM-TEA. ELF-Diff surpasses other existing diffusion models for MRI reconstruction. Furthermore, extensive experiments, including a subtask of pathology detection, further demonstrate the superior anatomical precision of our method. ELF-Diff outperforms the existing state-of-the-art MRI reconstruction methods without being limited to specific undersampling patterns.


## Execution Instructions
- Envrionment Setting

```
bash install.sh
```
  
- Build Model
```
from models.segmentation.segment_anything import sam_model_registry
model, img_embedding_size = sam_model_registry[args.vit_type](args, image_size=args.img_size,
                                                num_classes=args.num_classes,
                                                chunk = chunk,
                                                checkpoint=args.resume, pixel_mean=[0., 0., 0.],
                                                pixel_std=[1., 1., 1.])
```

## Pretrained Model Chcekpoints
Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Citation

If you found ELF-Diff useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```bibtex
@article{shin2025ensemble,
  title={Ensemble and low-frequency mixing with diffusion models for accelerated MRI reconstruction},
  author={Shin, Yejee and Son, Geonhui and Hwang, Dosik and Eo, Taejoon},
  journal={Medical Image Analysis},
  pages={103477},
  year={2025},
  publisher={Elsevier}
}
```


## Acknowledgments

* This repo is mainly based on [guided-diffusion](https://github.com/openai/guided-diffusion).
