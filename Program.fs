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


//type Normalized = CsvProvider<"normalized.csv",",">

//let userData = new Normalized()

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
    // Convert all categorical features into indices, and build a 'word bag' of these.
    dynamicPipeline.Append(mlContext.Transforms.Categorical.OneHotEncoding("CategoricalFeatures",
        "CategoricalBag", CategoricalTransform.OutputKind.Bag))

    dynamicPipeline
        .Append(mlContext.Transforms.Concatenate("Features", "NumericalFeatures"; "CategoricalBag" ))
        .Append(mlContext.BinaryClassification.Trainers.FastTree(numTrees= 50))

    let struct(train, test) = mlContext.BinaryClassification.TrainTestSplit(data, testFraction = 0.2)
    let model = dynamicPipeline.Fit(train)
    let predictions = model.Transform(test)
    let evaluationResult = mlContext.BinaryClassification.Evaluate(predictions, "Label")
    //classification.CrossValidate(train, est)

    printfn "Hello World from F#!"
    0 // return an integer exit code
