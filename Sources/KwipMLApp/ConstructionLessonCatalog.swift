import Foundation
import KwipMLCore

/// A checked construction is a distinct delivery type, not an ordinary training workflow.
enum ConstructionLessonCatalog {
    struct Choice {let id:String;let title:String}
    static let affineEntries:Set<String> = ["affine","relu-lesson","tanh-lesson","sigmoid-lesson"]
    static let affineIdentities:[ModelConstructionLessonIdentity] = [.composition,.recovery(taskSeed:0),.backwardChain,.frozenStandardization,.relu,.tanh,.sigmoid,.geluExact,.geluTanh]
    static let affineChoices:[Choice] = affineIdentities.map {identity in
        let title:String
        switch identity.assessmentID {
        case ModelConstructionLessonIdentity.recovery(taskSeed:0).assessmentID:title="Recover an affine map"
        case ModelConstructionLessonIdentity.backwardChain.assessmentID:title="Build the backward map"
        case ModelConstructionLessonIdentity.geluTanh.assessmentID:title="Explore approximate GELU"
        default:title=(try? ModelConstructionLessonTasks.publicTask(identity:identity).title) ?? identity.assessmentID
        }
        return .init(id:identity.assessmentID,title:title)
    }
    static let groups:[String:[Choice]] = [
        "affine":affineChoices,
        "tensor-layout":[.init(id:"reshape",title:"Reshape"),.init(id:"transpose",title:"Transpose"),.init(id:"flatten",title:"Flatten")],
        "softmax":[.init(id:"softmax-last-axis",title:"Last sample axis"),.init(id:"softmax-first-sample-axis",title:"First sample axis")],
        "layernorm":LayerNormLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "embedding":EmbeddingLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "embedding-bag":EmbeddingBagLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "program-export":ProgramExportLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "module-composition":ModuleCompositionLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "checkpoint-lifecycle":CheckpointLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "attention-connections":AttentionLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "recurrent-memory":RecurrentLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "channel-normalization":StatefulNormalizationLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "spatial-patches":SpatialPatchLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "tensor-routing":TensorRoutingLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "tensor-composition":TensorCompositionLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "adaptive-pooling":AdaptivePoolingLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "pooling":PoolingLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "configurable-activations":ConfigurableActivationLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "remaining-activations":RemainingActivationLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "convolution":ConvolutionLearningTask.allCases.filter {!$0.isTransposed}.map {.init(id:$0.rawValue,title:$0.title)},
        "transposed-convolution":ConvolutionLearningTask.allCases.filter {$0.isTransposed}.map {.init(id:$0.rawValue,title:$0.title)},
        "matrix":MatrixLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "bilinear":BilinearLearningTask.allCases.map {.init(id:$0.rawValue,title:$0.title)},
        "lazy-linear-lesson":[.init(id:LazyLinearAssessment.operation,title:"Inferred feature width")]
    ]
    static func contains(_ id:String)->Bool {affineEntries.contains(id) || groups[id] != nil}
    static func entryID(operation:String)->String? {groups.keys.sorted().first {groups[$0]?.contains(where:{$0.id==operation}) == true}}
}
