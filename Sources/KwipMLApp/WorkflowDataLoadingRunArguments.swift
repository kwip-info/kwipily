import Foundation
import KwipMLCore

/// Selected reader recipes only. Prepared-source authority remains in the captured operation.
enum WorkflowDataLoadingRunArguments {
    static func arguments(dataset:DatasetRecipe?,loader:DataLoaderRecipe?)throws->[String] {
        var result:[String]=[]
        if let dataset {
            try dataset.validate()
            let bytes=try WorkflowArchiveCodec.encode(dataset.configuration.recipeValue)
            guard bytes.count<=32768 else {throw WorkflowPhaseError("Dataset recipe exceeds its launch limit.")}
            result += ["--dataset-recipe-json",String(decoding:bytes,as:UTF8.self)]
        }
        if let loader {
            try loader.validate()
            let bytes=try WorkflowArchiveCodec.encode(loader.configuration.recipeValue)
            guard bytes.count<=32768 else {throw WorkflowPhaseError("DataLoader recipe exceeds its launch limit.")}
            result += ["--loader-recipe-json",String(decoding:bytes,as:UTF8.self)]
        }
        return result
    }
}
