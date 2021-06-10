# Stochastic-Activation-Pruning

A tensorflow implementation of https://www.semanticscholar.org/paper/Stochastic-Activation-Pruning-for-Robust-Defense-Dhillon-Azizzadenesheli/2f201c77e7ccdf1f37115e16accac3486a65c03d

Can be added to a Keras model like:

from SAP import SAP

x = layers.Activation(lambda acts: SAP(acts, percent))(x)
