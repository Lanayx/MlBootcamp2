using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApp15
{

    class Input { }
    class Output { }


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
                                                  new TextLoader.Column("CategoricalFeatures", DataKind.TX, 1)
                                    }
                                });

            // Read the data.
            var data = reader.Read("../../../normalized.csv");

            //// Inspect the first 10 records of the categorical columns to check that they are correctly read.
            //var catColumns = data.GetColumn<string[]>(mlContext, "CategoricalFeatures").Take(10).ToArray();

            // Build several alternative featurization pipelines.
            var dynamicPipeline =
                // Convert each categorical feature into one-hot encoding independently.
                mlContext.Transforms.Categorical.OneHotEncoding("CategoricalFeatures", "CategoricalOneHot");

            // Of course, if we want to train the model, we will need to compose a single float vector of all the features.
            // Here's how we could do this:

            var trainer = mlContext.BinaryClassification.Trainers.FastTree("Label", "CategoricalOneHot", numTrees: 20);

            var fullLearningPipeline = dynamicPipeline
                // Concatenate two of the 3 categorical pipelines, and the numeric features.
                .Append(mlContext.Transforms.Concatenate("Features", "CategoricalOneHot"))
                // Now we're ready to train. We chose our FastTree trainer for this classification task.
                .Append(trainer);

            // Train the model.
            var (train, test) = mlContext.BinaryClassification.TrainTestSplit(data, testFraction: 0.5);
            var model = fullLearningPipeline.Fit(train);
            var predictions = model.Transform(test);
            var evaluationResult = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine("Auc = {0}, Trainer = {1}",evaluationResult.Auc, train.GetType().Name);

            var predictionFunction = model.MakePredictionFunction<Input, Output>(mlContext);
        }


    }
}
