// Learn more about F# at http://fsharp.org

open System
open FSharp.Data
open System.Globalization
open Microsoft.ML
open Microsoft.ML.Core.Data
open Microsoft.ML.Runtime.Api
open Microsoft.ML.Runtime.Data
open Microsoft.ML.Transforms.Text


type Normalized = CsvProvider<"../data/train/normalized.csv",",">

let userData = new Normalized()

[<EntryPoint>]
let main argv =

    let mlContext = MLContext()

    let _textLoader = mlContext.Data.TextReader(TextLoader.Arguments
                        (

                            Separator = ",",
                            HasHeader = true,
                            Column =    [|
                                          TextLoader.Column("Label", Nullable(DataKind.Bool), 0)
                                          TextLoader.Column("COM_CAT#1", Nullable(DataKind.Num), 1)
                                        |]
                        ))

    printfn "Hello World from F#!"
    0 // return an integer exit code
