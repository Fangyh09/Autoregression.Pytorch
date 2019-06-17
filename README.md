# Autoregression.Pytorch

AR(2) auto regression model.

Support:
* AR(2)
* FFT decompose


## Installation
`git clone https://github.com/Fangyh09/Autoregression.Pytorch.git`
## Usage
```
python main.py
```

## Screenshot-NoFFT local_autoregression
### input img0, img1, img2
![image](pics/nofft_R0_k=3.png)
![image](pics/nofft_R1_k=3.png)
![image](pics/nofft_R2_k=3.png)

### output img3
![image](pics/nofft_R3_k=3.png)


## Screenshot- FFT local_autoregression
### input img0, img1, img2
![image](pics/R0_k=3.png)
![image](pics/R1_k=3.png)
![image](pics/R2_k=3.png)

### output img3
![image](pics/out_k=3.png)


## Screenshot-FFT
`python main_image_fft.py`
### input img0, img1, img2
<!-- ![image](pics/p0.png)
![image](pics/p1.png)
![image](pics/p2.png) -->
![image](pics/R0.png)
![image](pics/R1.png)
![image](pics/R2.png)


### output img3
<!-- ![image](pics/p3_2.png) -->
![image](pics/R3.png)

## Screenshot-NoFFT
`python main_image.py`
### input img0, img1, img2
![image](pics/nofft_R0.png)
![image](pics/nofft_R1.png)
![image](pics/nofft_R2.png)


### output img3
![image](pics/nofft_R3.png)



## TODOs
* [x] Add fft decompose
* [x] Add local conv
* [ ] Add optical flow

## History

## Credits
Some codes are from @pysteps.
Thanks @ssim-pytorch, @pysteps
