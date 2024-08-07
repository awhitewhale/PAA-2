# PAA-2 🌧️🔍
Code for paper *You Can’t Handle the Weather: Progressive Amplified Adverse-Weather-Gradient Projection Adversarial Attack (PAA²)*

![DNNs excel in vision tasks but are vulnerable to adversarial attacks and adverse weather effects, like haze, rain, and snow, posing security risks and impacting applications such as autonomous driving.](third_party/fig1.png)

Deep Neural Networks (DNNs) excel in vision tasks but are vulnerable to adversarial attacks and adverse weather effects, such as haze, rain, and snow. These vulnerabilities pose security risks and impact applications like autonomous driving.

## Installation 🛠️

To install the necessary dependencies, use the following commands:

```bash
git clone https://github.com/awhitewhale/PAA-2.git
cd PAA-2
pip install -r requirements.txt
```
Note: requirements.txt may contain lower versions of Python modules that have been removed from the PyPI library. However, higher versions should also work.

The experiments in this article are based on:

    Operating System: Linux node122 3.10.0-1160.92.1.el7.x86_64
    CPU: Intel(R) Xeon(R) Gold 6230 @ 2.10GHz
    GPU: NVIDIA Corporation GV100GL [Tesla V100 SXM2 32GB]
    Driver Version: NVIDIA-SMI 510.47.03
    CUDA Version: 11.6
    GCC Version: 11.2.1 (Red Hat 11.2.1-9)

## How to Use 📝
Run the following command to execute the script:
```python
python paa2.py -input_csv dataset.csv -input_dir dataset -max_epsilon 16.0 -num_iter_set 10 -batch_size 100 -momentum 1.0 -amplification 10.0 -prob 0.7
```

## Results 📊

ASR (%), PSNR (dB), SSIM, and NIQE values of various gradient-based attacks in a single normally trained model environment. Adversarial examples
are generated using MI-FGSM, TIM, SIM, VT-FGSM, Admix, and PAM, and our PAA^2 attack methods on Inc-v3, Inc-v4, IncRes-v2,
and Res-101 models. * denotes white-box model. Bolded, Underlined, and Italicized represent the first, second, and third ranked performance, respectively.


![](third_party/fig2.png)

Visualization of PAA^2 and PAM for adversarial examples. Here, we randomly selected ten images and generated adversarial examples on Inc-v3. The
variables in parentheses represent the noise types of PAA^2.

![](third_party/fig3.png)

The user study webpage is available at ![User Study](https://github.com/awhitewhale/PAA-2/blob/main/html/userstudy.html).
