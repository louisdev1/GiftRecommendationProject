using System;

namespace GiftApi;

public class MatrixFactorization
{
    private readonly int _nUsers, _nItems, _nFactors;
    private readonly double[,] _userFactors, _itemFactors;
    private readonly double[] _userBias, _itemBias;
    private double _globalMean;
    private const double LR = 0.01, REG = 0.02;

    public MatrixFactorization(int nUsers, int nItems, int numFactors = 20)
    {
        _nUsers = nUsers; _nItems = nItems; _nFactors = numFactors;
        _userFactors = new double[nUsers, numFactors];
        _itemFactors = new double[nItems, numFactors];
        _userBias = new double[nUsers];
        _itemBias = new double[nItems];

        var rnd = new Random(42);
        for (int u = 0; u < nUsers; u++)
            for (int f = 0; f < numFactors; f++)
                _userFactors[u, f] = rnd.NextDouble() * 0.1;
        for (int i = 0; i < nItems; i++)
            for (int f = 0; f < numFactors; f++)
                _itemFactors[i, f] = rnd.NextDouble() * 0.1;
    }

    public void Train(int[] users, int[] items, double[] ratings, int epochs = 20)
    {
        _globalMean = 0;
        for (int k = 0; k < ratings.Length; k++) _globalMean += ratings[k];
        _globalMean /= ratings.Length;

        Console.WriteLine($"Training MF: {_nUsers} users, {_nItems} items, {_nFactors} factors");

        for (int e = 0; e < epochs; e++)
        {
            double err = 0;
            for (int k = 0; k < ratings.Length; k++)
            {
                int u = users[k], i = items[k];
                double pred = Predict(u, i);
                double diff = ratings[k] - pred;
                err += diff * diff;

                _userBias[u] += LR * (diff - REG * _userBias[u]);
                _itemBias[i] += LR * (diff - REG * _itemBias[i]);

                for (int f = 0; f < _nFactors; f++)
                {
                    double uf = _userFactors[u, f], itf = _itemFactors[i, f];
                    _userFactors[u, f] += LR * (diff * itf - REG * uf);
                    _itemFactors[i, f] += LR * (diff * uf - REG * itf);
                }
            }
            Console.WriteLine($"  Epoch {e + 1}/{epochs}: RMSE = {Math.Sqrt(err / ratings.Length):F4}");
        }
    }

    public double Predict(int u, int i)
    {
        double dot = 0;
        for (int f = 0; f < _nFactors; f++) dot += _userFactors[u, f] * _itemFactors[i, f];
        return Math.Clamp(_globalMean + _userBias[u] + _itemBias[i] + dot, 1, 5);
    }

    public double GetItemBias(int i) => _globalMean + _itemBias[i];
}