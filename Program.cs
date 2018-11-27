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
using Microsoft.ML.Runtime.Api;

namespace ConsoleApp15
{

    public class Input
    {
        [Column(ordinal: "0", name: "Label")]
        public float Label;
        [Column(ordinal: "1")]
        public string CategoricalFeatures;
    }

    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }


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
            }
                                );

            // Read the data.
            var data = reader.Read("../../../normalized.csv");  

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
        }
    }
}
