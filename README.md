# The source code for all experiments in the manuscript


## Requirements
### 1 Linux environment (git, cmake, make and so on)
### 2 opencv 4.0.0 (other version should work as well)

## Installation
Execute the following command in the terminal
### 1 git clone https://github.com/yijieluo/LBPMR_GNP.git
### 2 cd LBPMR_GNP & mkdir Datasets
### 3 Download the datasets: [TC10&TC12](https://drive.google.com/file/d/1-JlAmUXQujDFIzarJZ7w-_BnGNe_eEvU/view?usp=sharing), [CUReT](https://drive.google.com/file/d/1-WU178yAhKVjluUY14rDnKWOEEHiN9ys/view?usp=sharing), [UIUC](https://drive.google.com/file/d/1gMxp502wy5_ll0UJwwRKoZC1OfXYkUPb/view?usp=sharing), [UMD](https://drive.google.com/file/d/1tyw436_fpFb_f15bh72v6xNZeBmfSnId/view?usp=sharing), [KTH-TIPS2-b](https://drive.google.com/file/d/1Q6-qvHShKg6GBItxp3f-XQhfEs5fkVmk/view?usp=sharing), and unzip these into the directory "Datasets"
### 4 mkdir build & cd build
### 5 cmake .. -DCMAKE_BUILD_TYPE=Debug (or cmake .. -DCMAKE_BUILD_TYPE=Release)
### 6 make -j8
### 5 ./a.out