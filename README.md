# Digit recognition

We take a few simple stabs at the digit recognition contest.

In the first, we model with a random forest.  This already
has an accuracy of about `95%`, which isn't so bad.

But by modern standards even that isn't very good.  So we
build a single-layer neural network using the `tensorflow`
library, which makes this a breeze.  This performs a little
bit better, but still has an accuracy in the neighborhood
of `95%` so it's not a substantial improvement even though
it's more complicated and prone to overfitting.

Finally, we implement a CNN (convolutional neural network)
using tensorflow again.  In fact we don't have to do any
heavy work as the tensorflow library has high-level tools
to build and train the CNN.  I'd be pretty lost building
one from scratch so we really just followed the tutorial
on the tensorflow doc pages.  This model gets us to `99%`
accuracy.  This is certainly not state-of-the-art, but
is "respectable".
