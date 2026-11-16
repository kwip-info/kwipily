import Foundation
import KwipMLCore

/// Exact catalog destinations, never a fuzzy title search or an execution request.
@MainActor enum ComponentLessonDestination {
    private static let catalogIDs = Set(LearnCatalogContent.items.map(\.id))
        .union(LearnCatalogContent.componentExampleEntries.map { $0.item.id })
        .union(LearnCatalogContent.historicalEntryIDs)
    private static let bundledIDs = Set(LearnCatalogContent.componentExampleEntries.map(\.componentID))
    private static let dataOperators: Set<String> = [
        "kwip.data.view", "kwip.data.sourceCSV", "kwip.data.selectRename", "kwip.data.cast", "kwip.data.filter",
        "kwip.data.missing", "kwip.data.split", "kwip.data.standardScalerFit", "kwip.data.transform",
        "kwip.data.imputerFit", "kwip.data.oneHotEncoderFit", "kwip.data.sampleRows",
        "kwip.data.groupSplit", "kwip.data.temporalSplit"
    ]
    private static let destinations: [String: String] = [
        "estimator":"sklearn", "mixture_experts":"mixture-experts",
        "kwip.data.outputDelimited":"table-output",
        "kwip.data.sort":"sort", "kwip.data.unique":"unique", "kwip.data.reshapeTable":"reshapeTable", "kwip.data.splitText":"splitText",
        "kwip.data.preprocess":"preprocess", "kwip.data.missing":"preprocess", "kwip.data.standardScalerFit":"preprocess", "kwip.data.imputerFit":"preprocess", "kwip.data.oneHotEncoderFit":"preprocess", "kwip.data.transform":"preprocess",
        "kwip.data.partition": PartitionPracticeExample.lessonID,
        "kwip.data.formula": FormulaPracticeExample.lessonID,
        "kwip.data.calculate":"formula", "kwip.data.regexExtract":"formula", "kwip.data.regexReplace":"formula",
        "kwip.data.filterRows": FilterPracticeExample.lessonID,
        "kwip.data.textEncode": TextEncodingPracticeExample.lessonID,
        "kwip.data.sequenceWindows":SequenceWindowsPracticeExample.lessonID,
        "kwip.data.columns": ColumnsPracticeExample.lessonID,
        "gameSession":GameEpisodePractice.lessonID,"workflow.gameSession":GameEpisodePractice.lessonID,
        GameRuleConfiguration.operatorID:GameRuleConfiguration.lessonID,
        "kwip.data.decide":TypedDecisionPracticeExample.exampleID,
        "kwip.data.classifyAudio":AudioClassificationPracticeExample.exampleID,
        "kwip.data.transcribeAudio":SpeechTranscriptionPracticeExample.exampleID,
        "kwip.data.extractText":OCRPracticeExample.exampleID,
        "kwip.data.prepareMedia":PrepareMediaPracticeExample.exampleID,
        "kwip.data.sourceFiles": FileInputViewPracticeExample.lessonID,
        "input": "model-input", "output": "model-output", "depthwise_conv2d": "depthwise",
        "kwip.recurrentCell":"recurrent-cell-overview", "kwip.recurrent":"recurrent-overview", "kwip.tensor.axes":"tensor-layout", "kwip.tensor.parts":"tensor-routing", "kwip.reshape":"tensor-layout", "kwip.linear":"linear-overview", "kwip.normalization":"normalization-overview", "kwip.convolution":"convolution-overview", "kwip.pooling":"pooling-overview", "kwip.activation":"activation", "activation":"activation", "linear": "affine", "relu": "relu-lesson", "sigmoid": "sigmoid-lesson", "tanh": "tanh-lesson",
        "bilinear":"bilinear", "lazy_linear": "lazy-linear-lesson", "reshape": "tensor-layout", "transpose": "tensor-layout",
        "flatten": "tensor-layout", "layer_norm": "layernorm", "rms_norm": "rms_norm", "matmul": "matrix",
        "index_input": "model-input", "bag_input": "embedding-bag", "embedding_bag": "embedding-bag",
        "conv_transpose1d": "transposed-convolution", "conv_transpose2d": "transposed-convolution",
        "conv_transpose3d": "transposed-convolution", "attention_mask": "attention-connections", "rotary_position":"rotary_position",
        "dataset_source": "train-data-access", "data_loader": "train-data-access",
        "initialization": "train-data-access", "gradient_policy": "train-data-access",
        "trainability": "train-data-access", "kwip.block": "module-composition",
        "modelDefinition": "module-composition", "trainingData": "train-data-access", "runSetup": "train-data-access",
        "generateText":"text-generation", "workflow.generateText":"text-generation", "train": "train-data-access", "saveModel":"save-model", "frozenModel": "load-model", "evaluateTest": "holdout",
        "kwip.data.union": "union", "kwip.data.join": "join", "kwip.data.aggregate": "summarize"
    ]

    static func isAvailable(entryID: String) -> Bool { catalogIDs.contains(entryID) }

    static func entryID(type: String, configuration: [String: JSONValue] = [:]) -> String? {
        let rawIdentifier: String
        if type.hasPrefix("wf-t-") {
            guard let decoded = try? WorkflowCanvasIDs.type(from: type) else { return nil }
            let supportedKinds: Set<String> = ["modelBlock", "modelPrimitive", "dataOperator", "optimizerRecipe", "lossRecipe",
                "initializationRecipe", "gradientPolicyRecipe", "trainabilityRecipe", "schedulerRecipe", "datasetRecipe", "batchRecipe",
                "gameSession", "modelDefinition", "trainingData", "runSetup", "estimator", "train", "saveModel", "generateText", "frozenModel", "evaluateTest"]
            guard supportedKinds.contains(decoded.kind) else { return nil }
            if decoded.kind == "modelBlock" { return "module-composition" }
            rawIdentifier = decoded.identifier
        } else {
            rawIdentifier = type
        }
        let identifier=canonicalType(rawIdentifier)
        let destination: String?
        if RetiredTrainingRecipeMigration.componentIDs.contains(identifier) {
            destination = "train-data-access"
        } else if identifier == "train", case .object = configuration["selfPlay"] {
            destination=SelfPlayPractice.lessonID
        } else if identifier == "train", case .object(let fine)=configuration["fineTuning"] {
            destination=fine["task"] == .string("typedDecisions") ? TypedDecisionPracticeExample.exampleID:FineTuningPracticeExample.exampleID
        } else if identifier == "output", configuration["task"] == .string("gamePolicyValue") {
            destination=SelfPlayPractice.lessonID
        } else if identifier == "frozenModel", configuration["source"] == .string("huggingFace"), let lesson=reviewedHubLesson(configuration) {
            destination=lesson
        } else if identifier == "estimator" && PrebuiltModelConfiguration.isPrebuilt(configuration) {
            destination=PrebuiltModelsPractice.lessonID
        } else if (identifier == "estimator" && DocumentClassificationConfiguration.isDocument(configuration)) || (identifier == "kwip.data.textEncode" && configuration["mode"] == .string("documentFeatures")) || (identifier == "frozenModel" && configuration["source"] == .string("documentPackage")) {
            destination=DocumentClassificationPracticeExample.exampleID
        } else if identifier == "estimator" {
            switch configuration["backendID"] {
            case .string("tensorflow"): destination = "tensorflow"
            case .string("sklearn"), nil: destination = "sklearn"
            default: destination = nil
            }
        } else if identifier.range(of: #"^composite:[^@]+@[0-9]+\.[0-9]+\.[0-9]+$"#, options: .regularExpression) != nil {
            destination = "module-composition"
        } else if let exact = destinations[identifier] {
            destination = exact
        } else if dataOperators.contains(identifier) {
            destination = "data-preparation-course"
        } else {
            destination = bundledIDs.contains(identifier) ? identifier : nil
        }
        guard let destination, isAvailable(entryID: destination) else { return nil }
        return destination
    }
    private static func canonicalType(_ type:String)->String {
        let aliases=["workflow.train":"train","workflow.frozenModel":"frozenModel","workflow.saveModel":"saveModel","workflow.evaluateTest":"evaluateTest","workflow.gameSession":"gameSession","workflow.generateText":"generateText","workflow.logisticRegression":"estimator","workflow.kerasClassifier":"estimator"]
        return aliases[type] ?? type
    }
    private static func reviewedHubLesson(_ configuration:[String:JSONValue])->String? {
        // Only explicit known acquisition profiles establish a task. Local folders and
        // saved runs need their verified resource to establish support, never path guesses.
        switch configuration["repository"]?.stringValue {
        case "convaiinnovations/laya" where configuration["subfolder"] == .string("typed-decisions"):return TypedDecisionPracticeExample.exampleID
        case "openai/whisper-tiny":return SpeechTranscriptionPracticeExample.exampleID
        case "MIT/ast-finetuned-audioset-10-10-0.4593":return AudioClassificationPracticeExample.exampleID
        default:return nil
        }
    }
    static func entryID(payload:WorkflowNodePayload)->String? {
        switch payload {
        case .train(let config):return entryID(type:"train",configuration:config)
        case .frozenModel(let config):return entryID(type:"frozenModel",configuration:config)
        case .saveModel(let config):return entryID(type:"saveModel",configuration:config)
        case .estimator(let config):return entryID(type:"estimator",configuration:config)
        case .dataOperator(let type,let config):return entryID(type:type,configuration:config)
        case .modelPrimitive(let type,_,let config):return entryID(type:type,configuration:config)
        default:return entryID(type:WorkflowCanvasAdapter.icon(payload))
        }
    }
    /// Context options for shared tools; reading links do not assert runtime support.
    static func relatedEntryIDs(type:String)->[String] {
        switch canonicalType(type) {
        case "train":return ["train-data-access","fine-tuning","typed-decisions","self-play"]
        case "frozenModel":return ["load-model","fine-tuning","document-classification","prebuilt-models","transcribe-audio","classify-audio","typed-decisions"]
        case "saveModel":return ["save-model","fine-tuning","document-classification","prebuilt-models","typed-decisions","self-play"]
        case "estimator":return ["prebuilt-models","document-classification","sklearn","tensorflow"]
        default:return entryID(type:type).map{[$0]} ?? []
        }
    }

}
