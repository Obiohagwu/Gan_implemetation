# Gan_implemetation
A short introduction to GAN's by implementing some GAN papers. Will do more on this in time.

---

## What are GANS?
As you probably know, GANS are generative adverserial networks; key words - *GENERATIVE* and *ADVERSERIAL*.
It is a special class of neural network modelling as operates on the basis of adverserializing outputs between the generator and the discriminator.

## What do we mean by Generator and Discriminator?
One could think of a GAN as two models working in an adverserial manner. The generator, used/trained to generate fake data - and the discriminator, used/trained to differentiatet the fake generated images of the generator from the ground truth dataset given.
The goal of the generator is to create images that are indistinguishable from the ground truth dataset. So in that sense, we could think of the discriminator as some quasi loss function; iteratively shifting the generator to a local minimum over its loss landscpe/ latent space of outputs until an optimal output can be produced.
Both models are trying to "outsmart" each other, so as the generator get's better, the discriminator also improves. 

Depending on the complexity of the product we are trying to implement, the GANs can either be implemented by some fairly simple feed-forward network, or as complexity increases, a convolutional network, or some more complex network like a U-net.

## How do these little guys even work?
Following a simple example that Ian Goodfellow (the inventor of GANs uses to explain the models), we could imagine the generator as some sort  of master forger; think Abagnale from catch me if you can. And we can imagine the discriminator as detective Hanratty, trying to catch the forger in action. The main caveat here is that we would like the detective (discriminator) to scale in detection skill as well as Abagnale (generator) scales in forging skill.

##
