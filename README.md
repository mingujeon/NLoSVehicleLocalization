
# Non-Line-of-Sight Vehicle Localization based on Sound

### T Junction with a wall, Left particles and sound paths
<table align="center">
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/NLoSVehicleLocalization/assets/39543006/a3575085-310d-4198-a701-db318534346f" alt="Site1 left particle path" width="380">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/NLoSVehicleLocalization/assets/39543006/14406aba-1c57-4863-9195-ad526aca25f3" alt="Site1 right particle path" width="380">
      </p>
    </td>
  </tr>
</table>

### T Junction with a wall, Right particles and sound paths
<table align="center">
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/NLoSVehicleLocalization/assets/39543006/b7eb9793-ef76-4f02-bfed-6cb1173ee482" alt="Site1 left particle path" width="380">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/NLoSVehicleLocalization/assets/39543006/c7515452-bb6a-4894-9879-53a0ce474dc0" alt="Site1 right particle path" width="380">
      </p>
    </td>
  </tr>
</table>

### T Junction without a wall, Left particles and sound paths
<table align="center">
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/NLoSVehicleLocalization/assets/39543006/c65dc8cc-4413-4b02-8769-e7e8b078d41e" alt="Site1 left particle path" width="380">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/NLoSVehicleLocalization/assets/39543006/75a2cb32-c4d6-4c44-804b-2d1c9e11e005" alt="Site1 right particle path" width="380">
      </p>
    </td>
  </tr>
</table>

### T Junction without a wall, Right particles and sound paths
<table align="center">
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/NLoSVehicleLocalization/assets/39543006/8ca67457-c720-403a-afa4-842b918578ab" alt="Site1 left particle path" width="380">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/NLoSVehicleLocalization/assets/39543006/8d5b1eca-3091-4ff5-b961-e94b0c08e853" alt="Site1 right particle path" width="380">
      </p>
    </td>
  </tr>
</table>


<!--
## Introduction

This is the repository of the paper “Non-Line-of-Sight Vehicle Localization based on Sound”. The experimental data, code, and the result images are contained.

<p align="center">
  <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/40bbadf6-b774-4ac7-b3cd-557fffd6c6d8" alt="Figure1" width="400" height="320">
</p>


This paper aims to detect NLoS objects based on sound and reduce thinking distance

<p align="center">
  <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/f5f52d6a-2bbf-4b3d-bc08-856eb033e6b6" alt="Figure2" width="650" height="300">
</p>


Comparison of possible paths of sound regarding the position of NLoS vehicle and spatial configuration

---



## Folder Structure (Recommended)

```
${ROOT}
└── cls_features/
    └── Site1
      └── [case name]
        └── .wav # sound file
        └── .csv # result of SRP-PHAT algorithm
        └── ground_truth # ground_truth
    └── Site2
    └── SA2
    └── SB1
└── config
└── utils
└── Graph
```
---
-->
<!--
## OVAD Dataset

The dataset is from the reference study. It is the github repository of the paper "Hearing What You Cannot See: Acoustic Vehicle Detection Around Corners". The details are in the below link.

https://github.com/tudelft-iv/occluded_vehicle_acoustic_detection

- *.csv file is the result of SRP-PHAT algorithm
- *.wav file is a sound recording file
- ground_truth.txt file is a ground truth file in the dataset of “Hearing What You Cannot See: Acoustic Vehicle Detection Around Corners”
-->



<!--
## ARIL Dataset

This dataset contained sound recordings file in the form of .wav and ground truth in the form of .xlsx. 
Our data shows the exact location of the target vehicle through the BEV camera.

A detailed description is provided in the paper

### Site1 Environment

- T junction with wall

<table align="center">
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/fd122903-58f3-4438-b27a-33080eb24bef" alt="Image 1" width="380">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/ef286e6c-6ea3-4ccd-8812-12d93eed689a" alt="Image 2" width="380">
      </p>
    </td>
  </tr>
</table>

### Site2 Environment

- T junction without wall
  
<table align="center">
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/c6ed19c2-5ed6-48f4-aec7-04bcc7e37045" alt="Image 1" width="380">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/7bc240c9-6daa-4227-b048-083956f54ea9" alt="Image 2" width="380">
      </p>
    </td>
  </tr>
</table>
-->
---

<!--
## How to run the code

Install python libraries

```bash
# pip
pip install -r requirments.txt # (tested with python 3.8.18)

# conda 
conda install --yes --file requirements.txt
```

SRP-PHAT algorithm running first

```bash
sh featureExtract.sh
```

For tracking test

```bash
sh tracking_test.sh
```

- tracking experiment results are in **log.txt** file

Get particle filter tracking image

```bash
python plotTrackingImg.py
```

Get particle filter convergence variance image

```bash
python plotVarianceImg.py
```

Get the azimuth map

```bash
python azimuth.py
```

Get the rmse

```bash
python rmse.py
```
-->
## Qualitative result


![image](https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/b52c46c6-2115-455b-bdc2-03f8510f548d)
<!--
## Azimuth
DoA response map
<p align="center">
  <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/2d6d7a80-3ec9-4ee6-964e-4319bf061c4c" alt="azimuth" width="400">
</p>
-->

## Quantitative results in ARIL dataset


|ARIL|Left|Right|
|------|---|---|
|T Junction with wall|<img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/9e40e1a7-4c64-4198-ba5e-45c68cda889b" alt="Site1 Left" width="380">|<img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/29481589-1067-4854-86fc-6fe07ff8215b" alt="Site1 Right" width="380">|
|T Junction without wall|<img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/7ec6ff91-d87a-44c5-8bc3-2030d4969bf2" alt="Site2 Left" width="380">|<img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/afe28ca7-e7bb-452a-a983-e25402e27a87" alt="Site2 Right" width="380">|

<!--
<table align="center">
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/9e40e1a7-4c64-4198-ba5e-45c68cda889b" alt="Site1 Left" width="380">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/29481589-1067-4854-86fc-6fe07ff8215b" alt="Site1 Right" width="380">
      </p>
    </td>
  </tr>
  <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/7ec6ff91-d87a-44c5-8bc3-2030d4969bf2" alt="Site2 Left" width="380">
      </p>
    </td>
  <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/afe28ca7-e7bb-452a-a983-e25402e27a87" alt="Site2 Right" width="380">
      </p>
    </td>
</table>
-->

## Quantitative results in OVAD dataset

|OVAD|Left|Right|
|------|---|---|
|T Junction with wall|<img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/5c6a9598-b4dd-41d0-9029-af2ceb15330d" alt="SA2 left" width="380">|<img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/f58eca7a-11d2-4f59-b32f-f51556c1fbb8" alt="SA2 right" width="380">|
|T Junction without wall|<img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/2fa88987-a506-4355-955e-2645863ab878" alt="SB1 left" width="380">|<img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/e92a519d-44ca-4b3f-aed1-1b235ca10b0a" alt="SB1 right" width="380">|

<!--
<table align="center">
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/5c6a9598-b4dd-41d0-9029-af2ceb15330d" alt="SA2 left" width="380">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/f58eca7a-11d2-4f59-b32f-f51556c1fbb8" alt="SA2 right" width="380">
      </p>
    </td>
  </tr>
  <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/2fa88987-a506-4355-955e-2645863ab878" alt="SB1 left" width="380">
      </p>
    </td>
  <td>
      <p align="center">
        <img src="https://github.com/mingujeon/Acoustic-Recognition-based-Invisible-target-Localization/assets/39543006/e92a519d-44ca-4b3f-aed1-1b235ca10b0a" alt="SB1 right" width="380">
      </p>
    </td>
</table>
-->
---

## Dataset

The dataset is available at [here](https://drive.google.com/file/d/1YlmlHGwIg7H1GrXJ6RFmImYoJjqfe4xG/view?usp=sharing).




## Folder Structure (Recommended)

```
${ROOT}
├── ARILDataset                 # Dataset
|    ├── SA                     # T-Junction with a wall
|    |  ├── left                # Direction of travel for NLoS vehicle
|    |  |  ├── [Scenario name]
|    |  |  | ├── .wav           # sound file
|    |  |  | ├── .csv           # result of SRP-PHAT algorithm
|    |  |  | └── .xlsx          # ground_truth
|    |  └── right  
|    └── SB                     # T-Junction without a wall
├── config
└── utils
```
---

## How to run the code

SRP-PHAT algorithm running first
```bash
sh featureExtract.sh
```

For localization test
```bash
sh localization_test.sh
```

For making azimuth map
```bash
python azimuth.py
```

For calculating RMSE
```bash
python rmse.py
```

## Authors

copyright<br>
Autonomous Robot Intelligence Lab, SNU


Mingu Jeon<br>
Jae-Kyung Cho<br>
Hee-Yeun Kim<br>
Byeonggyu Park<br>
Seung-Woo Seo<br>
Seong-Woo Kim
