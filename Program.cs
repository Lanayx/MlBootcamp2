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
using Microsoft.ML.Transforms.Normalizers;

namespace ConsoleApp15
{

    public class ClusteringPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint SelectedClusterId;
        [ColumnName("Score")]
        public float[] Distance;
        [ColumnName("Id")]
        public string Id;
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
                        new TextLoader.Column("Id", DataKind.TX, 0),
                        new TextLoader.Column("NumericalFeatures", DataKind.R4, 1, 75)
                }
            }
                                );

            // Read the data.
            var data = reader.Read("../../../kpi.csv");
            for (var i = 5; i <= 40; i++)
            {

                // Build several alternative featurization pipelines.
                var dynamicPipeline =
                    // Convert each categorical feature into one-hot encoding independently.
                    mlContext.Transforms.Normalize("NumericalFeatures", mode: NormalizingEstimator.NormalizerMode.LogMeanVariance);

                var trainer = mlContext.Clustering.Trainers.KMeans("NumericalFeatures", clustersCount: i);

                var fullLearningPipeline = dynamicPipeline
                    // Now we're ready to train. We chose our FastTree trainer for this classification task.
                    .Append(trainer);

                // Train the model.
                // var (train, test) = mlContext.Clustering.TrainTestSplit(data, testFraction: 0.5);
                var model = fullLearningPipeline.Fit(data);
                var predictions = model.Transform(data);
                var evaluationResult = mlContext.Clustering.Evaluate(predictions);
                Console.WriteLine("AvgMinScore = {0}, Dbi={1}, Nmi={2} i = {3}",
                    evaluationResult.AvgMinScore, evaluationResult.Dbi, evaluationResult.Nmi, i);
               
            }
            //var results =
            //       predictions.AsEnumerable<ClusteringPrediction>(mlContext, false).ToArray();
            //SaveSegmentationCSV(results, "clustering.csv");
        }

        private static void SaveSegmentationCSV(ClusteringPrediction[] predictions, string csvlocation)
        {
            
            using (var w = new System.IO.StreamWriter(csvlocation))
            {
                w.WriteLine($"CELL_LAC_ID,CLUSTER_NUM");
                w.Flush();
                predictions.ToList().ForEach(prediction => {
                    w.WriteLine($"{prediction.Id},{prediction.SelectedClusterId}");
                    w.Flush();
                });
            }

            Console.WriteLine($"CSV location: {csvlocation}");
        }
    }
}
