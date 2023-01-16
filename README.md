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


{the advantages of cycle Gan for EMOVC}


the advantages offered by the CycleGAN model include
(i)utilizing GAN loss instead of minimum generation loss, (ii)getting
rid of source-target alignment errors and (iii) flexible non-parallel
training, etc. 





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



{suggestions for improving the articles}


article 1 :  improving limitations with GAN
article2:   
Use CWT-based F0 as additional input to the decoder for
Improved experimental spectrum conversion performance

article 3:Mismatch between the listener's evaluation and the speaker's intention
should be examined for feelings


article 4:Margin filling with other features, such as STFT spectroscopy

article 5:Investigate multiple-to-multiple emotion transformations and do further research on data that can be transformed into desired features

article 6:Effectively produce a more effective rhythm and tone in the target voice


article 7 :Increased clarity and emotional power of speech created by
Suggested model


article 8:More reviews and increasing the quality of the joint and separate effect of teaching the characteristics of the spectrum and prosody.



article 9 :Further improvement of conversion quality, and this test on other instruments should be checked


article 10:  more inspection
Parallel input without voice conversion




{ more articles}

[Nonparallel_Emotional_Speech_Conversion.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327765/Nonparallel_Emotional_Speech_Conversion.pdf)


[Wed-3-10-4.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327767/Wed-3-10-4.pdf)


[Speaker_Independent_Emotional_Voice_Conversion_via_Disentangled_Representations.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327768/Speaker_Independent_Emotional_Voice_Conversion_via_Disentangled_Representations.pdf)


[FAIA-325-FAIA200325.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327771/FAIA-325-FAIA200325.pdf)


[Bao_Neumann_Vu.pdf](https://github.com/mahdeslami11/CycleGAN--EMOVC/files/10327773/Bao_Neumann_Vu.pdf)https://my.uupload.ir/dl/mbGEvvXv




{the link of my video}
https://s5.uupload.ir/files/sh1373bme/0A5F195B-7C54-459E-B891-850E368BEA75.MP4


https://s5.uupload.ir/files/sh1373bme/0A5F195B-7C54-459E-B891-850E368BEA75.MP4





 
 


 shabnam latifian
 40114140111002
