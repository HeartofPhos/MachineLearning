using System;
using NeuralNetwork.Activation;

namespace NeuralNetwork.Networks
{
    public class MultilayerPerceptron<T>
        where T : IActivation, new()
    {
        private static readonly Random s_random = new Random();

        private IActivation Activation { get; set; }

        private Neuron[][] Neurons { get; set; }
        private Synapse[][][] Synapses { get; set; }

        public MultilayerPerceptron(
            int inputCount,
            int[] hiddenLayerCounts,
            int outputCount)
        {
            Activation = new T();

            Neurons = new Neuron[hiddenLayerCounts.Length + 2][];
            Neurons[0] = new Neuron[inputCount];
            for (int i = 0; i < hiddenLayerCounts.Length; i++)
            {
                Neurons[i + 1] = new Neuron[hiddenLayerCounts[i]];
            }
            Neurons[^1] = new Neuron[outputCount];

            for (int i = 1; i < Neurons.Length; i++)
            {
                var layer = Neurons[i];
                for (int j = 0; j < layer.Length; j++)
                {
                    layer[j].Bias = s_random.NextDouble() * 2 - 1;
                }
            }

            Synapses = new Synapse[Neurons.Length - 1][][];
            for (int i = 0; i < Synapses.Length; i++)
            {
                var sourceLayer = Neurons[i];
                var targetLayer = Neurons[i + 1];

                Synapses[i] = new Synapse[sourceLayer.Length][];
                for (int j = 0; j < sourceLayer.Length; j++)
                {
                    Synapses[i][j] = new Synapse[targetLayer.Length];
                    for (int k = 0; k < targetLayer.Length; k++)
                    {
                        Synapses[i][j][k].Weight = s_random.NextDouble() * 2 - 1;
                    }
                }
            }
        }

        public void Predict(double[] inputs, double[] outputs)
        {
            for (int i = 0; i < Neurons[0].Length; i++)
            {
                Neurons[0][i].Value = inputs[i];
            }

            for (int i = 0; i < Neurons.Length - 1; i++)
            {
                var sourceLayer = Neurons[i];
                var targetLayer = Neurons[i + 1];

                for (int j = 0; j < targetLayer.Length; j++)
                {
                    double net = 0;
                    for (int k = 0; k < sourceLayer.Length; k++)
                    {
                        net += Synapses[i][k][j].Weight * sourceLayer[k].Value;
                    }
                    targetLayer[j].Value = Activation.Function(net + targetLayer[j].Bias);
                }
            }

            for (int i = 0; i < Neurons[^1].Length; i++)
            {
                outputs[i] = Neurons[^1][i].Value;
            }
        }

        public void Train(double[] expectedOutput, double learnRate, double momentum)
        {
            for (int i = 0; i < Neurons[^1].Length; i++)
            {
                Neurons[^1][i].Gradient = (expectedOutput[i] - Neurons[^1][i].Value) * Activation.Derivative(Neurons[^1][i].Value);
            }

            for (int i = Neurons.Length - 2; i >= 1; i--)
            {
                var sourceLayer = Neurons[i];
                var targetLayer = Neurons[i + 1];

                for (int j = 0; j < sourceLayer.Length; j++)
                {
                    double net = 0;
                    for (int k = 0; k < targetLayer.Length; k++)
                    {
                        net += targetLayer[k].Gradient * Synapses[i][j][k].Weight;
                    }

                    sourceLayer[j].Gradient = net * Activation.Derivative(sourceLayer[j].Value);
                }
            }

            for (int i = 0; i < Neurons.Length - 1; i++)
            {
                var sourceLayer = Neurons[i];
                var targetLayer = Neurons[i + 1];

                for (int j = 0; j < targetLayer.Length; j++)
                {
                    double prevDelta = targetLayer[j].BiasDelta;
                    targetLayer[j].BiasDelta = learnRate * targetLayer[j].Gradient;
                    targetLayer[j].Bias += targetLayer[j].BiasDelta + momentum * prevDelta;

                    for (int k = 0; k < sourceLayer.Length; k++)
                    {
                        prevDelta = Synapses[i][k][j].WeightDelta;
                        Synapses[i][k][j].WeightDelta = learnRate * targetLayer[j].Gradient * sourceLayer[k].Value;
                        Synapses[i][k][j].Weight += Synapses[i][k][j].WeightDelta + momentum * prevDelta;
                    }
                }
            }
        }

        private struct Neuron
        {
            public double Value { get; set; }
            public double Bias { get; set; }
            public double BiasDelta { get; set; }
            public double Gradient { get; set; }
        }

        private struct Synapse
        {
            public double Weight { get; set; }
            public double WeightDelta { get; set; }
        }
    }
}
