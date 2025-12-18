using System;

public class MatrixFactorization
{
    private int numUsers, numItems, numFactors;

    private double[,] userFactors;
    private double[,] itemFactors;

    private double[] userBias;
    private double[] itemBias;

    private double globalMean;

    private double learningRate = 0.01;
    private double regularization = 0.02;

    public MatrixFactorization(int users, int items, int factors = 10)
    {
        numUsers = users;
        numItems = items;
        numFactors = factors;

        userFactors = new double[users, factors];
        itemFactors = new double[items, factors];

        userBias = new double[users];
        itemBias = new double[items];

        Random rnd = new Random();

        for (int u = 0; u < users; u++)
            for (int f = 0; f < factors; f++)
                userFactors[u, f] = rnd.NextDouble() * 0.1;

        for (int i = 0; i < items; i++)
            for (int f = 0; f < factors; f++)
                itemFactors[i, f] = rnd.NextDouble() * 0.1;
    }

    public void Train(int[] users, int[] items, double[] ratings, int epochs = 20)
    {
        // GLOBAL MEAN
        globalMean = 0;
        foreach (var r in ratings)
            globalMean += r;
        globalMean /= ratings.Length;

        for (int e = 0; e < epochs; e++)
        {
            for (int k = 0; k < ratings.Length; k++)
            {
                int u = users[k];
                int i = items[k];

                double prediction = Predict(u, i);
                double error = ratings[k] - prediction;

                // UPDATE BIASES 
                userBias[u] += learningRate * (error - regularization * userBias[u]);
                itemBias[i] += learningRate * (error - regularization * itemBias[i]);

                // UPDATE LATENT FACTORS
                for (int f = 0; f < numFactors; f++)
                {
                    double uf = userFactors[u, f];
                    double ifac = itemFactors[i, f];

                    userFactors[u, f] += learningRate * (error * ifac - regularization * uf);
                    itemFactors[i, f] += learningRate * (error * uf - regularization * ifac);
                }
            }
        }
    }

    public double Predict(int user, int item)
    {
        double dot = 0;
        for (int f = 0; f < numFactors; f++)
            dot += userFactors[user, f] * itemFactors[item, f];

        return globalMean + userBias[user] + itemBias[item] + dot;
    }
}
