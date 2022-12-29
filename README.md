## EMOTIONAL VOICE CONVERSION WITH CYCLE-CONSISTENT ADVERSARIAL NETWORK
We propose a non-parallel voice-conversion (VC) method that can learn a mapping from source to target speech without relying on parallel data.
Emotional Voice Conversion, or emotional VC, is a technique of converting speech from one emotion state into another one, keeping the basic linguistic information and speaker identity. Previous approaches for emotional VC need parallel data and use dynamic time warping (DTW) method to temporally align the source-target speech parameters. These approaches often define a minimum generation loss as the objective function, such as L1 or L2 loss, to learn model parameters. Recently, cycle-consistent generative adversarial networks (CycleGAN) have been used successfully for non-parallel VC. This paper investigates the efficacy of using CycleGAN for emotional VC tasks. Rather than attempting to learn a mapping between parallel training data using a frame-to-frame minimum generation loss, the CycleGAN uses two discriminators and one classifier to guide the learning process, where the discriminators aim to differentiate between the natural and converted speech and the classifier aims to classify the underlying emotion from the natural and converted speech. The training process of the CycleGAN models randomly pairs source-target speech parameters, without any temporal alignment operation. The objective and subjective evaluation results confirm the effectiveness of using CycleGAN models for emotional VC. The non-parallel training for a CycleGAN indicates its potential for non-parallel emotional VC

---

### Model Architectures
<img src=https://github.com/liusongxiang/CycleGAN-EmoVC/blob/master/img/net_fig.png alt="drawing" width="800px"/>
<img src=https://github.com/liusongxiang/CycleGAN-EmoVC/blob/master/img/model_arch.png alt="drawing" width="800px"/>

### Dependencies
* Python 3.6 (or 3.5)
* Pytorch 0.4.0 +
* pyworld
* pycwt
* 
* [تبدیل صدای احساسی یا emovc  یک تکنیک تبدیل گفتار از یک سطح احساسی به سطحی دیگر همراه با حفظ اطلاعات پایه زبانی و هویت گوینده  می باشد 0 سابقا رویکرد های vc  به داده های موازی نیاز داشته و از روش dtw.docx](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322020/emovc.0.vc.dtw.docx)

* numpy 
* scipy
* tqdm
* librosa
* tensorboardX and tensorboard
* 
* 
[مقاله شماره9.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322004/9.pdf)
[مقاله شماره2.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322005/2.pdf)
[مقاله شماره 10.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322007/10.pdf)
[مقاله شماره 8.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322008/8.pdf)
[مقاله شماره 7.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322010/7.pdf)
[مقاله شماره 6.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322011/6.pdf)[

[مقاله شماره 4.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322114/4.pdf)

له شماره 3.pdf…]()

[مقاله شماره 5.pdf](https://github.com/Shabnamlatifian/CycleGAN--
EMOVC/files/10322013/5.pdf)
[مقاله شماره 5.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322116/5.pdf)
[مقاله شماره 3.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322117/3.pdf)

[تبدیل صدای احساسی [مقاله شماره 1.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322118/1.pdf)
یا emovc  یک تکنیک تبدیل گفتار از یک سطح احساسی به سطحی دیگر همراه با حفظ اطلاعات پایه زبانی و هویت گوینده  می باشد 0 سابقا رویکرد های vc  به داده های موازی نیاز داشته و از روش dtw.docx](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322022/emovc.0.vc.dtw.docx)
[ی.docx](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322028/default.docx)
 
 

[مقاله شماره 1.pdf](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322119/1.pdf)
2[ی.docx](https://github.com/Shabnamlatifian/CycleGAN--EMOVC/files/10322122/default.docx)

 shabnam latifian
 40114140111002
