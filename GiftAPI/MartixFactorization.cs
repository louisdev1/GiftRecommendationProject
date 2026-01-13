using System;

namespace GiftApi;

public class MatrixFactorization
{
    private readonly int _nUsers;      // users
    private readonly int _nItems;      // Psroducts
    private readonly int _nFactors;    // Latent factors 888888888888888888
    
    private readonly double[,] _userFactors; 
    private readonly double[,] _itemFactors;  
    
    // Bias, some users rate uneven, some products are just better
    private readonly double[] _userBias;  // Is this user kind/strict
    private readonly double[] _itemBias;  // Is this product "generally" liked
    
    // Average rating
    private double _globalMean;
    
    // Parameters for training MF model
    private const double LR = 0.01;   // Learning rate
    private const double REG = 0.02;  // Regularization

    // Initialize all the matrices with random values
    public MatrixFactorization(int nUsers, int nItems, int numFactors = 20)
    {
        _nUsers = nUsers;
        _nItems = nItems;
        _nFactors = numFactors;
        
        // Create empty matrices for user and item factors
        _userFactors = new double[nUsers, numFactors];
        _itemFactors = new double[nItems, numFactors];
        
        // Create empty arrays for biases
        _userBias = new double[nUsers];
        _itemBias = new double[nItems];

        // Initialize with small random values
        // Seed = 42
        var rnd = new Random(42);
        
        // Fill user factors with random values between 0 and 0.1
        for (int u = 0; u < nUsers; u++)
            for (int f = 0; f < numFactors; f++)
                _userFactors[u, f] = rnd.NextDouble() * 0.1;
        
        // Fill item factors with random values between 0 and 0.1
        for (int i = 0; i < nItems; i++)
            for (int f = 0; f < numFactors; f++)
                _itemFactors[i, f] = rnd.NextDouble() * 0.1;
    }

    // Train the model using Stochastic Gradient Descent (SGD)
    // This is where the machine learning happens
    public void Train(int[] users, int[] items, double[] ratings, int epochs = 20)
    {
        // Calculate the average rating across all data
        // This serves as our baseline prediction
        _globalMean = 0;
        for (int k = 0; k < ratings.Length; k++) 
            _globalMean += ratings[k];
        _globalMean /= ratings.Length;

        Console.WriteLine($"Training MF: {_nUsers} users, {_nItems} items, {_nFactors} factors");

        // Train for multiple epochs (passes through the data)
        for (int e = 0; e < epochs; e++)
        {
            double err = 0;  // Track total squared error for this epoch
            
            // Go through every rating in the dataset
            for (int k = 0; k < ratings.Length; k++)
            {
                int u = users[k];      // Which user gave this rating
                int i = items[k];      // Which product was rated
                
                // Predict what rating the user would give
                double pred = Predict(u, i);
                
                // Calculate error (actual rating - predicted rating)
                double diff = ratings[k] - pred;
                
                // Add to total error for calculating RMSE later
                err += diff * diff;

                // Update user bias
                // Move it in the direction that reduces error
                // Regularization pulls it back toward zero to prevent overfitting
                _userBias[u] += LR * (diff - REG * _userBias[u]);
                
                // Update item bias (same logic)
                _itemBias[i] += LR * (diff - REG * _itemBias[i]);

                // Update the latent factors using gradient descent
                for (int f = 0; f < _nFactors; f++)
                {
                    // Store current values (we need them for both updates)
                    double uf = _userFactors[u, f];
                    double itf = _itemFactors[i, f];
                    
                    // Update user factor
                    // diff * itf: push toward better prediction
                    // REG * uf: pull toward zero to prevent overfitting
                    _userFactors[u, f] += LR * (diff * itf - REG * uf);
                    
                    // Update item factor (similar logic)
                    _itemFactors[i, f] += LR * (diff * uf - REG * itf);
                }
            }
            
            // Calculate and display RMSE (Root Mean Squared Error) for this epoch
            // This tells us how well the model is performing
            Console.WriteLine($"  Epoch {e + 1}/{epochs}: RMSE = {Math.Sqrt(err / ratings.Length):F4}");
        }
    }

    // Predict what rating a user would give to an item
    // This is the core prediction formula
    public double Predict(int u, int i)
    {
        // Calculate dot product of user and item factors
        // This captures how well user preferences match product characteristics
        double dot = 0;
        for (int f = 0; f < _nFactors; f++) 
            dot += _userFactors[u, f] * _itemFactors[i, f];
        
        // Final prediction = baseline + user bias + item bias + learned interaction
        // Clamp between 1 and 5 since ratings are in this range
        return Math.Clamp(_globalMean + _userBias[u] + _itemBias[i] + dot, 1, 5);
    }

    // Get the general popularity score for an item (without a specific user)
    // Used when we don't have a user ID but still want to recommend products
    public double GetItemBias(int i) => _globalMean + _itemBias[i];
}