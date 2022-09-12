# Stochastic-Activation-Pruning

A tensorflow implementation of [Stochastic Activation Pruning for Robust Adversarial Defense](https://www.semanticscholar.org/paper/Stochastic-Activation-Pruning-for-Robust-Defense-Dhillon-Azizzadenesheli/2f201c77e7ccdf1f37115e16accac3486a65c03d)

Used as an activation for keras conv2d layers like

```
x = keras.layers.Conv2D(...)
x = keras.layers.Activation(lambda acts: SAP(acts, 0.25))(x)
```
