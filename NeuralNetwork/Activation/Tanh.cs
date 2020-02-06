using System;

namespace NeuralNetwork.Activation
{
    public class Tanh : IActivation
    {
        public double Function(double x)
        {
            return Math.Tanh(x);
        }

        public double Derivative(double x)
        {
            return 1 - Math.Pow(Function(x), 2);
        }
    }
}
