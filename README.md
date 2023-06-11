## the effective of another tissue on hepar in pet images
 https://ganj.irandoc.ac.ir/#/articles/48ac31b9d65757154dcd61d4b4bb160f


Many errors occur in PET imaging that reduce the quality of the image. One of these errors is the drop error, which is well resolved by combining CT imaging technique with PET. Another error is the scattering error of annihilation gammas and occurs when one or both annihilation gammas are scattered inside the body or other parts due to the Compton event. By placing an energy window for PET detectors, this error can be reduced in a favorable way. This research tries to investigate the effect of loss and scattering from the adjacent organs of the liver in liver tumor imaging with the help of Monte Carlo simulation method and to optimize the energy window in this imaging.



. For this purpose, first, with the help of MCNPX simulation code, the Pat Siemens detector ring of Biograph 6 model was simulated. Before examining the organs adjacent to the liver in order to further investigate the effect of soft tissue on the scattering and loss of annihilation rays, with the help of a simple geometry of soft tissue and considering two types of spatial distribution for the 511keV gamma source, the effect of soft tissue scattering in PET imaging Checked out. The analysis of the results of the simulations in the third chapter shows that the soft tissue around the point source and the spherical volume with a uniform distribution scatter the gamma rays in such a way that their energy loss is low and almost all of them are within the range of the energy window of PET (-550 350 keV are recorded. The soft tissue around both springs at a distance of 8 cm from the center of the spring has the greatest scattering effect in PET imaging.In the fourth chapter, imaging of a liver tumor was simulated by placing a complete human body phantom with a tumor inside the liver (a spherical gamma source with 511keV energy) and placing it inside the PET ring. To investigate the effect of individual internal organs near the liver, the desired tissue was removed from the full body phantom, and the response function of the PET detector in this case was compared with the response function of the PET detector for the full body phantom.The results showed that the stomach has the most negative effect on liver tumor imaging among the examined organs including heart, kidney, lung and intestine. Examining the results of all the desired tissues on the energy window of the PET detector showed that the effect of scattered rays can be favorably reduced by setting the energy window to 450-550 keV in liver tumor imaging.In the continuation of this investigation, in order to achieve more accuracy, the distribution of radiopharmaceuticals inside the body was also considered in addition to the tumor. Despite the fact that the effect of the loss and dispersion of the organs adjacent to the liver on the destruction gammas is greater in this case compared to the previous case, but the accumulation of the effect of the investigated organs in the detector ring is still in the same range as before. In the fifth chapter, the investigation of braking radiation in PET imaging in soft tissue was discussed. For this purpose, in one step, the 511keV source inside a sphere of soft tissue (in order to investigate the effect of braking radiation of secondary electrons) and in the next step, the source of positron donor 18F inside the same tissue (in order to investigate the effect of braking radiation of positrons) inside the PET detector ring. it placed.The analysis of the results shows that the bremsstrahlung radiation caused by secondary electrons does not have an effect on the imaging due to being outside the energy window range, but on the other hand, the bremsstrahlung radiation caused by positrons causes a slight change in the counts throughout the range of the energy window, which has an adverse effect on the quality. The image can be appropriately reduced by reducing the energy window to 417-550 keV. In the sixth chapter, the scattering effect of annihilation gammas between PET ring detectors was investigated. For this purpose, each PET ring detector was insulated in the simulation so that the gamma ray scattered from aA detector cannot enter another detector. The results showed that the separation of PET ring detectors reduces the number of detectors in the range of the energy window, especially near its lower threshold. This decrease in count was reduced in the presence of a sphere of soft body tissue.

In this project, I have investigated other things that reduce the quality of the image in PET along with the method of correction and application. Such as the distance that a positron travels before annihilation, which I have chosen as the main article.

main article

Positron emission tomography (PET) is a molecular imaging technique that provides a 3D
image of functional processes in the body in vivo. Some of the radionuclides proposed for PET imaging emit high-energy positrons, which travel some distance before they annihilate (positron range),
creating significant blurring in the reconstructed images. Their large positron range compromises the
achievable spatial resolution of the system, which is more significant when using high-resolution
scanners designed for the imaging of small animals. In this work, we trained a deep neural network
named Deep-PRC to correct PET images for positron range effects. Deep-PRC was trained with
modeled cases using a realistic Monte Carlo simulation tool that considers the positron energy
distribution and the materials and tissues it propagates into. Quantification of the reconstructed PET
images corrected with Deep-PRC showed that it was able to restore the images by up to 95% without
any significant noise increase. The proposed method, which is accessible via Github, can provide an
accurate positron range correction in a few seconds for a typical PET acquisition.





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
 
 
 new project 
 
 
 https://ganj.irandoc.ac.ir/#/articles/48ac31b9d65757154dcd61d4b4bb160f


