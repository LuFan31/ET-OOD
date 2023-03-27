# Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection （CVPR 2023）

**SCOOD benchmarks download link:**<br>[![gdrive](https://img.shields.io/badge/SCOOD%20dataset-google%20drive-f39f37)](https://drive.google.com/file/d/1cbLXZ39xnJjxXnDM7g2KODHIjE0Qj4gu/view?usp=sharing)&nbsp;
[![onedrive](https://img.shields.io/badge/SCOOD%20dataset-onedrive-blue)](https://entuedu-my.sharepoint.com/:u:/r/personal/jingkang001_e_ntu_edu_sg/Documents/scood_benchmark.zip?csf=1&web=1&e=vl8nr8)  

The codebase accesses the SCOOD benchmarks from the root directory in a folder named `data/` by default, i.e.
```
├── ...
├── data
│   ├── images
│   └── imglist
├── scood
├── test.py
├── train.py
├── ...
```

## Dependencies
* Python >= 3.8
* Pytorch = 1.8.1
* CUDA >= 11.3
* torchvision=0.9.1
* faiss-gpu=1.7.1


## Experimental Results
You can run the following script (specifying the output and data directories) which perform training & testing for CIFAR10/100 experimental results:
```bash
bash cifar10.sh output_dir data_dir
```
```bash
bash cifar100.sh output_dir data_dir
```
The information during training can be monitored in real-time in `output_dir/log.txt` and the results will be saved in `output_dir/results.csv`.

## Acknowledgments
This paper follows the excellent work from [SCOOD](https://jingkang50.github.io/projects/scood).

## Cite
If our work is useful for your research, please consider citing our [paper](https://arxiv.org/abs/2303.10449) :
```
@inproceedings{lu2022etood,
  title={Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection},
  author={Fan Lu, Kai Zhu, Wei Zhai, Kecheng Zheng and Yang Cao},
  booktitle={CVPR},
  year={2023}
}
```