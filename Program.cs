using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApp15
{
    class Program
    {

        static void Main(string[] args)
        {

            var mlContext = new MLContext();
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
                                {

                                    Separator = ",",
                                    HasHeader = true,
                                    Column = new[]
                                    {
                                                  new TextLoader.Column("Label", DataKind.BL, 0),
                                                  new TextLoader.Column("CategoricalFeatures", DataKind.TX, 1, 10),
                                                  new TextLoader.Column("NumericalFeatures", DataKind.R4, 11, 17)
                                    }
                                });

            // Read the data.
            var data = reader.Read("../../../normalized.csv");

            // Inspect the first 10 records of the categorical columns to check that they are correctly read.
            var catColumns = data.GetColumn<string[]>(mlContext, "CategoricalFeatures").Take(10).ToArray();

            // Build several alternative featurization pipelines.
            var dynamicPipeline =
                // Convert each categorical feature into one-hot encoding independently.
                mlContext.Transforms.Categorical.OneHotEncoding("CategoricalFeatures", "CategoricalOneHot")
                // Convert all categorical features into indices, and build a 'word bag' of these.
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("CategoricalFeatures", "CategoricalBag", CategoricalTransform.OutputKind.Bag));

            // Of course, if we want to train the model, we will need to compose a single float vector of all the features.
            // Here's how we could do this:

            var fullLearningPipeline = dynamicPipeline
                // Concatenate two of the 3 categorical pipelines, and the numeric features.
                .Append(mlContext.Transforms.Concatenate("Features", "NumericalFeatures", "CategoricalBag"))
                // Now we're ready to train. We chose our FastTree trainer for this classification task.
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numTrees: 50));

            // Train the model.
            var (train, test) = mlContext.BinaryClassification.TrainTestSplit(data, testFraction: 0.2);
            var model = fullLearningPipeline.Fit(train);
            var predictions = model.Transform(test);
            var evaluationResult = mlContext.BinaryClassification.Evaluate(predictions, "Label");
        }


    }
}
