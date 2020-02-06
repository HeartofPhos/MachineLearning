using System;
using NeuralNetwork.Networks;
using NeuralNetwork.Activation;
using NUnit.Framework;
using System.IO;

namespace NeuralNetwork.Tests
{
    [TestFixture]
    public class MultilayerPerceptronTests
    {
        [TestCase(1000, 0.96)]
        [TestCase(10000, 0.98)]
        public void XOR(int iterations, double minimumAccuracy)
        {
            var mlp = new MultilayerPerceptron<Tanh>(2, new int[] { 10, 10 }, 1);

            double[][] inputs = new double[][]{
                new double[]{0,0},
                new double[]{1,0},
                new double[]{0,1},
                new double[]{1,1},
            };
            double[][] outputs = new double[][]{
                new double[]{0},
                new double[]{1},
                new double[]{1},
                new double[]{0},
            };

            double[] output = new double[1];
            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    mlp.Predict(inputs[j], output);
                    mlp.Train(outputs[j], 0.4, 0.9);
                }
            }

            for (int j = 0; j < inputs.Length; j++)
            {
                mlp.Predict(inputs[j], output);
                double diff = 1 - Math.Abs(output[0] - outputs[j][0]);
                Assert.GreaterOrEqual(diff, minimumAccuracy);
            }
        }

        [TestCase(100, 0.95)]
        [TestCase(10000, 1)]
        public void Iris(int iterations, double minimumAccuracy)
        {
            string[] lines = File.ReadAllLines("./Data/iris.csv");

            double[][] inputs = new double[lines.Length][];
            double[][] outputs = new double[lines.Length][];
            for (int i = 0; i < lines.Length; i++)
            {
                inputs[i] = new double[4];
                outputs[i] = new double[1];
                string[] values = lines[i].Split(',');

                for (int j = 0; j < 4; j++)
                {
                    inputs[i][j] = double.Parse(values[j]);
                }

                outputs[i][0] = Map(values[4]);
            }

            for (int i = 0; i < 4; i++)
            {
                double min = double.PositiveInfinity;
                double max = double.NegativeInfinity;
                for (int j = 0; j < inputs.Length; j++)
                {
                    if (inputs[j][i] < min)
                        min = inputs[j][i];
                    if (inputs[j][i] > max)
                        max = inputs[j][i];
                }

                for (int j = 0; j < inputs.Length; j++)
                {
                    inputs[j][i] = Normalize((inputs[j][i] - min) / (max - min));
                }
            }

            var mlp = new MultilayerPerceptron<Tanh>(4, new int[] { 10, 10 }, 1);

            double normalizedCorrect = 0;
            double[] output = new double[1];
            for (int i = 0; i < iterations; i++)
            {
                int correct = 0;
                for (int j = 0; j < inputs.Length; j++)
                {
                    mlp.Predict(inputs[j], output);
                    mlp.Train(outputs[j], 0.1, 0.4);

                    double expectedOutput = outputs[j][0];
                    if (Map(expectedOutput) == Map(output[0]))
                        correct++;
                }

                normalizedCorrect = (double)correct / inputs.Length;
            }

            Assert.GreaterOrEqual(normalizedCorrect, minimumAccuracy);
        }

        static readonly string[] s_classification = new string[] { "setosa", "versicolor", "virginica" };

        static double Normalize(double x) => x * 2 - 1;
        static double Map(string c)
        {
            for (int i = 0; i < s_classification.Length; i++)
            {
                if (s_classification[i] == c)
                    return Normalize(i / (s_classification.Length - 1d));
            }

            throw new ArgumentException($"{c} does not exist in {nameof(s_classification)}");
        }

        static double Denormalize(double x) => (x + 1) / 2;
        static string Map(double x)
        {
            x = Denormalize(x);
            for (int i = 0; i < s_classification.Length; i++)
            {
                double threshold = (i + 1d) / s_classification.Length;
                if (x < threshold)
                    return s_classification[i];
            }

            return s_classification[^1];
        }
    }
}
