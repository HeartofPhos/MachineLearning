namespace NeuralNetwork.Activation
{
    public interface IActivation
    {
        double Function(double x);
        double Derivative(double x);
    }
}
