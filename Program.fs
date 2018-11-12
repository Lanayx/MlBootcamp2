// Learn more about F# at http://fsharp.org

open System
open FSharp.Data
open System.Globalization
open Microsoft.ML
open Microsoft.ML.Core.Data
open Microsoft.ML.Runtime.Api
open Microsoft.ML.Runtime.Data
open Microsoft.ML.Transforms.Text
open Microsoft.ML.Trainers
open Microsoft.ML.Transforms


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

    let data = _textLoader.Read("normalized.csv")
    let classification = new BinaryClassificationContext(mlContext)

    let est = ColumnConcatenatingEstimator(mlContext, "Features", "CategoricalFeatures", "NumericalFeatures")
    let struct(train, test) = classification.TrainTestSplit(data, testFraction = 0.2)
    
    est.Append(LinearClassificationTrainer(mlContext, "Features", "Label"))
        
    let model = est.Fit(train)
    let predictions = model.Transform(test);
    let evaluationResult = classification.Evaluate(predictions, "Label")
    //classification.CrossValidate(train, est)

    printfn "Hello World from F#!"
    0 // return an integer exit code
