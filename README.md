## EMOTIONAL VOICE CONVERSION WITH CYCLE-CONSISTENT ADVERSARIAL NETWORK

![CycleGAN-EmoVC](https://user-images.githubusercontent.com/115186860/210151557-15eefd00-5773-4b16-9e66-422b3a125d16.png)




We propose a non-parallel voice-conversion (VC) method that can learn a mapping from source to target speech without relying on parallel data.
Emotional Voice Conversion, or emotional VC, is a technique of converting speech from one emotion state into another one, keeping the basic linguistic information and speaker identity. Previous approaches for emotional VC need parallel data and use dynamic time warping (DTW) method to temporally align the source-target speech parameters. These approaches often define a minimum generation loss as the objective function, such as L1 or L2 loss, to learn model parameters. Recently, cycle-consistent generative adversarial networks (CycleGAN) have been used successfully for non-parallel VC. This paper investigates the efficacy of using CycleGAN for emotional VC tasks. Rather than attempting to learn a mapping between parallel training data using a frame-to-frame minimum generation loss, the CycleGAN uses two discriminators and one classifier to guide the learning process, where the discriminators aim to differentiate between the natural and converted speech and the classifier aims to classify the underlying emotion from the natural and converted speech. The training process of the CycleGAN models randomly pairs source-target speech parameters, without any temporal alignment operation. The objective and subjective evaluation results confirm the effectiveness of using CycleGAN models for emotional VC. The non-parallel training for a CycleGAN indicates its potential for non-parallel emotional VC




Emotional Voice Conversion, or emotional VC, is a technique of
converting speech from one emotion state into another one, keeping the basic linguistic information and speaker identity. Previous
approaches for emotional VC need parallel data and use dynamic
time warping (DTW) method to temporally align the source-target
speech parameters. These approaches often define a minimum generation loss as the objective function, such as L1 or L2 loss, to learn
model parameters. Recently, cycle-consistent generative adversarial
networks (CycleGAN) have been used successfully for non-parallel
VC. This paper investigates the efficacy of using CycleGAN for
emotional VC tasks. Rather than attempting to learn a mapping between parallel training data using a frame-to-frame minimum generation loss, the CycleGAN uses two discriminators and one classifier
to guide the learning process, where the discriminators aim to differentiate between the natural and converted speech and the classifier
aims to classify the underlying emotion from the natural and converted speech. The training process of the CycleGAN models randomly pairs source-target speech parameters, without any temporal
alignment operation. The objective and subjective evaluation results
confirm the effectiveness of using CycleGAN models for emotional
VC. The non-parallel training for a CycleGAN indicates its potential
for non-parallel emotional VC.





![images](https://user-images.githubusercontent.com/115186860/210151583-d48f6cdb-ced4-4832-a8d5-1dbf6e8d213b.jpg)



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
 

* numpy 
* scipy
* tqdm
* librosa
* tensorboardX and tensorboard

{the link of my project}

https://github.com/jasonaidm/CycleGAN-EmoVC

* 
*{my main article project }

[dsp .pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327741/dsp.pdf)



[translate.docx](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327744/translate.docx)





{another articles which i use}


[مقاله شماره 1.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327748/1.pdf)



[مقاله شماره2.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327750/2.pdf)



[مقاله شماره 3.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327751/3.pdf)


[مقاله شماره 4.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327753/4.pdf)


[مقاله شماره 5.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327754/5.pdf)



[مقاله شماره 6.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327755/6.pdf)



[مقاله شماره 7.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327756/7.pdf)



[مقاله شماره 8.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327757/8.pdf)




[مقاله شماره9.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327758/9.pdf)



[مقاله شماره 10.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327759/10.pdf)



{comparison articles with each other}

[comparison.docx](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327762/comparison.docx)

[comparison 2.docx](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327764/comparison.2.docx)



{ more articles}

[Nonparallel_Emotional_Speech_Conversion.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327765/Nonparallel_Emotional_Speech_Conversion.pdf)


[Wed-3-10-4.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327767/Wed-3-10-4.pdf)


[Speaker_Independent_Emotional_Voice_Conversion_via_Disentangled_Representations.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327768/Speaker_Independent_Emotional_Voice_Conversion_via_Disentangled_Representations.pdf)


[FAIA-325-FAIA200325.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327771/FAIA-325-FAIA200325.pdf)


[Bao_Neumann_Vu.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327773/Bao_Neumann_Vu.pdf)



 
 


 shabnam latifian
 40114140111002
