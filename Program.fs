// Learn more about F# at http://fsharp.org

open System
open FSharp.Data
open System.Linq
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.ML.Runtime.Api
open Microsoft.ML.Runtime.Data
open Microsoft.ML.Transforms.Text
open Microsoft.ML.Trainers
open Microsoft.ML.Transforms
open Microsoft.ML.Runtime
open Microsoft.ML.Legacy.Transforms
open Microsoft.ML.Transforms.Categorical
open Microsoft.ML.StaticPipe


    type Input =   
    
        [<Column(ordinal= "0", name="Label")>]
        val mutable Label: single
        [<Column(ordinal= "1")>]
        val mutable CategoricalFeatures: string
    

    type Output =
    
        [<ColumnName("PredictedLabel")>]
        val mutable Prediction: bool

        [<ColumnName("Probability")>]
        val mutable Probability: single

        [<ColumnName("Score")>]
        val mutable Score: single
    

[<EntryPoint>]
let main argv =

    let mlContext = MLContext()

    let _textLoader = mlContext.Data.TextReader(TextLoader.Arguments
                        (

                            Separator = ",",
                            HasHeader = true,
                            Column =    [|
                                          TextLoader.Column("Label", Nullable(DataKind.BL), 0)
                                          TextLoader.Column("CategoricalFeatures", Nullable(DataKind.TX), 1, 10)
                                          TextLoader.Column("NumericalFeatures", Nullable(DataKind.R4), 11, 17)
                                        |]
                        ))
    // Read the data.
    let data = _textLoader.Read("../../../normalized.csv")
    // Inspect the first 10 records of the categorical columns to check that they are correctly read.
    let catColumns = data.GetColumn<string[]>(mlContext, "CategoricalFeatures").Take(10).ToArray();

    // Build several alternative featurization pipelines.
    let dynamicPipeline =
        // Convert each categorical feature into one-hot encoding independently.
        mlContext.Transforms.Categorical.OneHotEncoding("CategoricalFeatures", "CategoricalOneHot")
        :> IEstimator<CategoricalTransform>
        :?> IEstimator<ITransformer>
    
    let trainer = mlContext.BinaryClassification.Trainers.FastTree("Label", "CategoricalOneHot", numTrees= 20)

    let fullLearningPipeline = 
        dynamicPipeline
            // Concatenate two of the 3 categorical pipelines, and the numeric features.
            .Append(mlContext.Transforms.Concatenate("Features", "CategoricalOneHot"))
            // Now we're ready to train. We chose our FastTree trainer for this classification task.
            .Append(trainer)

    // Train the model.
    let struct(train, test) = mlContext.BinaryClassification.TrainTestSplit(data, testFraction= 0.5)
    let model = fullLearningPipeline.Fit(train)
    let predictions = model.Transform(test)
    let evaluationResult = mlContext.BinaryClassification.Evaluate(predictions, "Label")
    Console.WriteLine("Auc = {0}, Trainer = {1}", evaluationResult.Auc, train.GetType().Name)
    0
