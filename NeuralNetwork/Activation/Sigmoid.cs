using System;

namespace NeuralNetwork.Activation
{
    public class Sigmoid : IActivation
    {
        public double Function(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double Derivative(double x)
        {
            return x * (1 - x);
        }
    }
}
