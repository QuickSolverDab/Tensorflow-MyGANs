# Tensorflow-MyGANs
* Tensorflow implementation related to my research
* All implementation is based on DCGAN arcitecture (https://arxiv.org/abs/1511.06434)
  * RFGAN (Improved Training of Generative Adversarial Networks using Representative Features)
    * http://proceedings.mlr.press/v80/yingzhen18a.html
  * ResembledGAN (Resembled Generative Adversarial Networks: Two Domains with Similar Attributes)
    * http://bmvc2018.org/contents/papers/0737.pdf
  * MGGAN (MGGAN: Solving Mode Collapse using Manifold Guided Training)
    * https://arxiv.org/abs/1804.04391
  * Recycling discriminator (Recycling the Discriminator for Improving the Inference Mapping of GAN)
    * https://openreview.net/pdf?id=HkgnpiR9Y7
    
* Running
  * python main.py --gan_type \<Type\> --dataset \<Dataset\> --epoch \<Epoch\> --batch_size \<Batch_size\>
  * Gan Type: RFGAN, ResembledGAN, MGGAN, Recycling_discriminator
  
* Code Reference 
  * https://github.com/hwalsuklee/tensorflow-generative-model-collections
  * https://github.com/carpedm20/DCGAN-tensorflow
