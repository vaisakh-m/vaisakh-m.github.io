---
title: 'ICLR Blog Post'
date: 2121-01-14
permalink: /posts/2021/01/iclr-blog-post/
tags:
  - cool posts
  - category1
  - category2
---

<!-- ![](https://i.imgur.com/HIGRfyY.jpg) -->
![](https://i.imgur.com/yii8lxs.gif)

# ICLR BlogPost

## Blog Post Sections

## Blog Post Title
Haven't discussed yet.

## Motivation

The amount of data available to us is exploding, however, most of this data is unlabelled. Despite the availabilty of such large amount of data, we are unable to exploit them since most training requires use of labelled source data. The paradigm of learning methods which is required should use source data with fewer or no labels and yet perform well for identifying patterns in unseen data. This learning paradigm, which doesn't assume availabilty of labelled source data for training, is called Self-Supervised Learning (SSL). 

However, traditionally most of the tasks in self supervised learning (SSL) requires domain expertise e.g various data augmentation strategies ------- on data such that it performs well for a particular domain and modality. This creates a bottleneck for the use of such models by non-domain experts as well as usabilty of such models across various domains and modalities. Viewmaker Network attempts to bridge this gap by proposing a network which generates domain invariant data views (i.e. augmentations) that can be used for model training and yet show significant success in the classifier predictions across various datasets. This blogpost seeks to provide a broad, hands-on introduction to the paper on viewmaker network, it's related works and it's implementation detail. The goal is combine both a mathematical presentation and illustrative code examples that highlight some of the key methods and challenges in this setting. With this goal in mind, the blogpost is provided as a static blogpost, but all the sections are also downloadable as Jupyter Notebooks; this lets you try out and build upon the explainations presented here.




## Intro
<!-- About Self-Supervised Visual Representation Learning, Hand-crafted views, recent works on automatizing the process, etc. -->



The idea of representation learning is to learn 'representations' of the data that make it easier to extract useful information when building predictive models. In the case of probabilistic models, a good representation is often one that captures the posterior distribution of the underlying explanatory factors for the observed input. A good representation is also one that is useful as input to a supervised predictor. Most recently, representation learning has seen a lot of success in the deep-learning where the task is defined by the composition of multiple non-linear transformations, with the goal of finding more abstract – and ultimately more useful representations.


For machines to comprehend visual data, learning meaningful representations is necessary. This is what Visual Representation Learning is all about. Visual representation should not just be effective, but also efficient(handling large amounts of data) and robust to changes in lighting or viewpoint. 

--representative image here--

Leaning visual representations in a supervised manner often leads to the learning process relying too much on the task and the labels. These are undesirable as the representations are meant to be ‘general’ i.e. task and label agnostic.

The most intuitive way of mitigating these issues for better Visual Representation Learning is by shifting to a Self-supervised approach. As the training doesn’t involve labels or the knowledge of the downstream task, the representations can be considered more general.

*More to be added


## Basics of Viewmaker Networks
Srishti - About views, adverserial leanrning, lp lp spheres/norms, adverserial + viewmaker

 ## Views

Views are described as various image transformations that a training data undergoes as part of augmetation strategies before being fed to a model for training. 

<figure>
<img src="https://1.bp.blogspot.com/-bO6c2IGpXDY/Xo4cR6ebFUI/AAAAAAAAFpo/CPVNlMP08hUfNPHQb2tKeHju4Y_UsNzegCLcBGAsYHQ/s1600/image3.png" alt="Trulli" style="width:100%">
<figcaption style="text-align:center"> 
    Fig.1: SimCLR data augmentations
</figcaption>
</figure>




It is an accepted notion that bigger datasets result in better Deep Learning models (**any DL reference will fit**). However, manual effort of collecting and leblling this dataset can be a very daunting task, hence the usefulness of views (data transformations).

Different works have proposed many different augmentation stregies for better model training. Historially, random duplication, cropping, flipping and rotations have been used for suhc augmentations. It was, however, AlexNet (**ImageNet Classification with Deep Convolutional Neural Networks**) which popularized the use of augmentations in deep learning which was used to increase the training size of the dataset by a magnitude of 2048. They randomly cropped 224×224 patches from the original images, flipped them horizontally, and changed the intensity of the RGB channels using PCA color augmentation. Since then various augmentation strategies have been proposed for effectiness in various domains. Random Erasing (**Random Erasing Data Augmentation**)(add figures) proposed for classification and detection problems, randomly selects a rectangle region in an image and erases its pixels with random values. A closely related data augmentation approach (**Improved Regularization of Convolutional Neural Networks with Cutout**) is CutOut where authors randomly mask out square regions of input during training, which they call cutout, and is used to improve the robustness and overall performance of convolutional neural networks. Yun et. al. proposed CutMix (**CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features**) strategy where patches are cut and pasted among training images and are mixed proportionally to the area of the patches. This work shares some similarity with Mixup (**mixup: Beyond Empirical Risk Minimization**) in a way that both combine two samples, where the ground truth label of the new sample is given by the linear interpolation of one-hot labels. SamplePairing (**Data Augmentation by Pairing Samples for Images Classification**) creates a new sample from one image by overlaying another image randomly chosen from the training data, and reduces top-1 error rate from 8.22% to 6.93% on the CIFAR-10 dataset. DAGAN (**Data Augmetnation generative eAdverserial network**) used adverserial setup to augment images ans showed how adverserial settings can be used to create data to apply to novel unseen classes of data (e.g. few shot learning). This field of optmizing best augmentation strategies is constantly expanding.

|
|
|code here
|
|
-- code on how some view works or how it's fed to a model --

The intuition behind using different views is that these augmented data simulate realistic samples from the true data distribution (original unaugmented data) and hence provides effective increase in the training data (**good reference?**) (some small experiment or figure may help). This rationale, however, does not explain why unrealistic distortions such as cutout (**Improved Regularization of Convolutional Neural Networks with Cutout**) and mixup (**mixup: Beyond Empirical Risk Minimization**) significantly improve generalization performance. Furthermore, many methods do not always transfer across datasets —Cutout, for example, is useful on CIFAR-10 and not on ImageNet. There is also no standard metrics to quantify if an augmentation strategy is good and how well can it be generalized. Works like Lopez (**Affinity and Diversity: Quantifying Mechanisms of Data Augmentation**) attempt to provide a good metrics to quanity these intutions by providing metrics of Diversity and Affinity and show that more in-distribution and more diverse augmentation policies perform well. Though there metrics are model independent, they do not provide exact augmentations will work on all the models. 


### Concept of $l_p$ norm

Szegedy et al. and Biggio et al. showed that specifically crafted small perturbations of benign inputs can lead machine-learning models to misclassify them. The perturbed inputs are referred to as adversarial examples. Given a machine-learning model,
a sample $\hat{s}$ is considered as an adversarial example if it is similar to a benign sample x (drawn from the data distribution), such that x is correctly classified and $\hat{s}$ is classified differently than x.

For two given images, image X and image Z, we can define L norms as:

**L⁰ distance:** What is the total number of pixels that differ in their value between image X and image Z?
**L¹ distance:** What is the summed absolute value difference between image X and image Z?
(for each pixel, calculate the absolute value difference, and sum it over all pixels)
**L² distance:** What is the squared difference between image X and image Z? (for each pixel, take the distance between image X and Z, square it, and sum that over all pixels)
**L${^\infty}$ distance:** What is the maximum pixel difference between image X and image Z? It is many times referred to as as “Max Norm”. (for each pixel, take the absolute value difference between X and Z, and return the largest such distance found over all pixels).


### Adverserial setup

In this section we will explain the concept of viewmaker network and its view generation using constrained perturbation with the help of psuedo code and related visual representation.


<figure>
<img src="https://i.imgur.com/1kTs3KB.png" alt="Trulli" />
<figcaption style="text-align:center">
    Psuedo Code for Viewmaker Network 
</figcaption>
</figure>


<figure>
<img src="https://i.imgur.com/FIIzNxu.png" alt="Trulli" />
<figcaption style="text-align:center">
    Psuedo Code: Generate perturbations 
</figcaption>
</figure>


<figure>
<img src="https://i.imgur.com/4XZfuJP.png" alt="Trulli" />
<figcaption style="text-align:center">
    Visual Representation: Generate perturbations 
</figcaption>
</figure>


<figure>
<img src="https://i.imgur.com/jmK8Rsm.png" alt="Trulli" />
<figcaption style="text-align:center">
    Psuedo Code: L1 contraint of input image 
</figcaption>
</figure>

<figure>
<img src="https://i.imgur.com/hrkEezs.png" alt="Trulli">
<figcaption style="text-align:center">
    Visual Representation: L1 contraint of input image </figcaption>
</figure>


<figure>
<img src="https://i.imgur.com/8ZGgr0q.png" alt="Trulli" />
<figcaption style="text-align:center">
    Fig.3: Puedo Code: Final image generated by viewmaker network 
</figcaption>
</figure>

<figure>
<img src="https://i.imgur.com/g0UDfFP.png" alt="Trulli" />
<figcaption style="text-align:center">
    Fig.2: Visual Representation: Final image generated by viewmaker network 
</figcaption>
</figure>




<!-- Reword and shorten it: good source: https://adversarial-ml-tutorial.org/introduction/ -->
<!-- Now let’s try to fool this classifier into thinking this image of a dog is something else. To explain this process, we’re going to introduce a bit more notation. Specifically, we’ll define the define the model, or hypothesis function, $h_θ:X→\mathbb{R}^k$ as the mapping from input space (in the above example this would be a three dimensional tensor), to the output space, which is a k-dimensional vector, where k is the number of classes being predicted; note that like in our model above, the output corresponds to the logit space, so these are real-valued numbers that can be positive or negative. The θ vector represents all the parameters defining this model, (i.e., all the convolutional filters, fully-connected layer weight matrices, baises, etc; the $θ$ parameters are what we typically optimize over when we train a neural network. And finally, note that this $h_θ$ corresponds precisely to the model object in the Python code above.


Second, we define a loss function $ℓ:\mathbb{R}_k×Z+→R+$ as a mapping from the model predictions and true labels to a non-negative number. The semantics of this loss function are that the first argument is the model output (logits which can be positive or negative), and the second argument is the index of the true class (that is, a number from 1 to $k$ denoting the index of the true label). Thus, the notation

$$ ℓ(h_θ(x),y) $$

for $x∈X$ the input and $y∈Z$ the true class, denotes the loss that the classifier achieves in its predictions on $x$, assuming the true class is $y$. By far the most common form of loss used in deep learning is the cross entropy loss (also sometimes called the softmax loss), defined as

$$ ℓ(h_θ(x),y)=log(∑_{j=1}^{k}exp(h_θ(x)_j))−h_θ(x)_y $$

where $h_θ(x)_j$ denotes the jth elements of the vector $h_θ(x)$.

Since the convention is that we want to minimize loss (rather than maximizing probability), we use the negation of this quantity as our loss function. We can evaluate this loss in PyTorch as below.

|
|
|code here
|
|

|
to be cont.
|
|
 -->


<!-- conclude with the papers which ppularized adverserial learning -->
<!-- Important paper: Adversarial Self-Supervised Contrastive Learning -->



### Adverserial setup for views


Viewmaker network attempts to bride the gap between continuously evolving augmentation strategies and diffculties in transfering these straegies to different domains because of domain expertise required. The method proposed by the authors is insipired from adverserial setup, in particular with the use of $l_p$ norms used for adverserial robustness. 

To explain the contrained perturbations proposed by the authors, we will refer to the figure below.

<figure>
<img src="https://i.imgur.com/kVc2SdQ.png" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig.1: 3D representation of adding perturbation to input data</b></figcaption>
</figure>



Let us assume a 3D space with centre at point 4. This space defines the region where images can exists. By definition, we can define an L1 norm as $||\textbf{w}||_1 = ∑_{i}^{n}|w_i|$ and hence can be represented as the greenish pyramid structure centred at point 4 and 5 respectively. $L_2$ norm is $||\textbf{w}||_2^2 = ∑_{i}^{n}w_i^2$ and can be represented by the pink and yellow spheres centred around point 4 and 5.


Let Our input data point (here image) be represented by point 6. Adding a new data point or perturbation (say point 1) to input data point (point 6) will, using simple vector addition, give point ........
1 Unconstrained perturbation
2 Perturbation constrained by l2 norm
3 Perturbation constrained by l1 norm
4 origin
5 input data point
6 input data point perturbed by an l1 constrained perturbation
7 input data point perturbed by an l2 constrained perturbation



## Viewmaker Arch
InstDisc vs. SimCLR

Pre-training Architecture:  
- Encoder = ResNet18
- Viewmaker = style transfer network (fully convolutional downsampling using strided/fractionally strided convolutions followed by batch norm/ReLU. Contains 5 residual layers and no pooling). Uniform random noise concatenated to the input and activations before each residual block.

Pre-training Hyperparams:  
- SimCLR temperature = 0.07
- InstDisk negatives = 4096
- InstDisk update rate = 0.5
- SGD optimizer
- Batch size = 256
- Learning rate = 0.03
- Momentum = 0.9
- Weight decay = 1e-4
- Epochs = 200

Linear Eval Architecture:
- todo


Training Procedured:
Two-phase training. 

## Experiments

### Dimensionality Collapse 

### 


<!-- ## Expts with Non-Contrastive SSL
Comparisons, interpretation, etc -->

<!-- ## Expts with 3D data
Adaptability, interpretation, challenges, etc -->

## Conclusion
Putting it all together, future of conditional view generation. 

## References

[1] [Viewmaker Networks: Learning Views for Unsupervised Representation Learning](https://iclr.cc/virtual/2021/poster/2544)
[2] [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
[3] [Understanding Dimensional Collapse in Contrastive Self-supervised Learning](https://arxiv.org/abs/2110.09348)
[4] []()
[5] []()
[6] []()
[7] []()
[8] []()
[9] []()
[10] []()

<!-- 

# My first HackMD note (change me!)

###### tags: `Tag(change me!)`

> This note is yours, feel free to play around.  :video_game: 
> Type on the left :arrow_left: and see the rendered result on the right. :arrow_right: 

## :memo: Where do I start?

### Step 1: Change the title and add a tag

- [x] Create my first HackMD note (this one!)
- [ ] Change its title
- [ ] Add a tag

:rocket: 

### Step 2: Write something in Markdown

Let's try it out!
Apply different styling to this paragraph:
**HackMD gets everyone on the same page with Markdown.** ==Real-time collaborate on any documentation in markdown.== Capture fleeting ideas and formalize tribal knowledge.

- [x] **Bold**
- [ ] *Italic*
- [ ] Super^script^
- [ ] Sub~script~
- [ ] ~~Crossed~~
- [x] ==Highlight==

:::info
:bulb: **Hint:** You can also apply styling from the toolbar at the top :arrow_upper_left: of the editing area.

![](https://i.imgur.com/Cnle9f9.png)
:::

> Drag-n-drop image from your file system to the editor to paste it!

### Step 3: Invite your team to collaborate!

Click on the <i class="fa fa-share-alt"></i> **Sharing** menu :arrow_upper_right: and invite your team to collaborate on this note!

![permalink setting demo](https://i.imgur.com/PjUhQBB.gif)

- [ ] Register and sign-in to HackMD (to use advanced features :tada: ) 
- [ ] Set Permalink for this note
- [ ] Copy and share the link with your team

:::info
:pushpin: Want to learn more? ➜ [HackMD Tutorials](https://hackmd.io/c/tutorials) 
:::

---

## BONUS: More cool ways to HackMD!

- Table

| Features          | Tutorials               |
| ----------------- |:----------------------- |
| GitHub Sync       | [:link:][GitHub-Sync]   |
| Browser Extension | [:link:][HackMD-it]     |
| Book Mode         | [:link:][Book-mode]     |
| Slide Mode        | [:link:][Slide-mode]    | 
| Share & Publish   | [:link:][Share-Publish] |

[GitHub-Sync]: https://hackmd.io/c/tutorials/%2Fs%2Flink-with-github
[HackMD-it]: https://hackmd.io/c/tutorials/%2Fs%2Fhackmd-it
[Book-mode]: https://hackmd.io/c/tutorials/%2Fs%2Fhow-to-create-book
[Slide-mode]: https://hackmd.io/c/tutorials/%2Fs%2Fhow-to-create-slide-deck
[Share-Publish]: https://hackmd.io/c/tutorials/%2Fs%2Fhow-to-publish-note

- LaTeX for formulas

$$
x = {-b \pm \sqrt{b^2-4ac} \over 2a}
$$

- Code block with color and line numbers：
```javascript=16
var s = "JavaScript syntax highlighting";
alert(s);
```


- Auto-generated Table of Content
[ToC]

> Leave in-line comments! [color=#3b75c6]

- Embed YouTube Videos

{%youtube PJuNmlE74BQ %}

> Put your cursor right behind an empty bracket {} :arrow_left: and see all your choices.

- And MORE ➜ [HackMD Tutorials](https://hackmd.io/c/tutorials)
 -->
