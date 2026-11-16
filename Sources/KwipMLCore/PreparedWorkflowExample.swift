import Foundation

/// Keeps an example's authored workflow and bundled observations together.
/// Resource names are bundle-relative; private source paths never enter documents.
public enum PreparedWorkflowExample {
    case preparationPractice, parallel, tensorLayout, softmax, embedding, matrix, bilinear, lazyLinear, conv1d, conv2d, conv3d, convTranspose1d, convTranspose2d, convTranspose3d, depthwiseConv2d, separableConv2d, guided, joined, aggregated, groupPartition, temporalPartition
    case backend(String)
    case component(String)
    public static var silu: Self { .component("silu") }
    public static var leakyReLU: Self { .component("leaky_relu") }
    public static var elu: Self { .component("elu") }
    public static var maxPool1d: Self { .component("max_pool1d") }
    public static var maxPool2d: Self { .component("max_pool2d") }
    case layerNorm(normalizeWholeSample: Bool)
    case gelu(approximateTanh: Bool)

    public struct SourceResource {
        public let address:WorkflowNodeAddress
        public let resource:String
    }
    public var sourceResources:[SourceResource] {
        if case .component(let id)=self,id==TypedDecisionPracticeExample.exampleID {return [.init(address:TypedDecisionPracticeExample.source,resource:"typed-decision-tickets")]}
        if case .component(let id)=self,id==PrebuiltModelsPractice.lessonID {return [.init(address:PrebuiltModelsPractice.source,resource:"prebuilt-customers")]}
        if case .component(let id)=self,[SelfPlayPractice.lessonID,GameEpisodePractice.lessonID,GameRuleConfiguration.lessonID,OCRPracticeExample.exampleID,SpeechTranscriptionPracticeExample.exampleID,AudioClassificationPracticeExample.exampleID,TypedDecisionPracticeExample.exampleID].contains(id) {return []}
        if case .component(let id) = self, id == TableOutputPracticeExample.exampleID {return TableOutputPracticeExample.sourceResources}
        if case .component(let id) = self, id == SortPracticeExample.exampleID {return SortPracticeExample.sourceResources}
        if case .component(let id) = self, id == UniquePracticeExample.exampleID {return UniquePracticeExample.sourceResources}
        if case .component(let id) = self, id == PreprocessPracticeExample.cleanupExampleID {return [.init(address:WorkflowExamples.sourceAddress,resource:PreprocessPracticeExample.cleanupObservationsResource)]}
        if case .component(let id) = self, id == ReshapeTablePracticeExample.exampleID || id == ReshapeTablePracticeExample.wideExampleID {return [.init(address:WorkflowExamples.sourceAddress,resource:id == ReshapeTablePracticeExample.wideExampleID ? ReshapeTablePracticeExample.wideResource:ReshapeTablePracticeExample.observationsResource)]}
        if case .component(let id) = self, id == SplitTextPracticeExample.exampleID {return [.init(address:WorkflowExamples.sourceAddress,resource:SplitTextPracticeExample.observationsResource)]}
        if case .component(let id)=self,id==DocumentClassificationPracticeExample.exampleID {return [.init(address:DocumentClassificationPracticeExample.labels,resource:"document-classification-labels"),.init(address:DocumentClassificationPracticeExample.unseen,resource:"document-classification-unseen")]}
        if case .component(let id) = self, id == ModelOutputPracticeExample.exampleID {return [.init(address:WorkflowExamples.sourceAddress,resource:ModelOutputPracticeExample.observationsResource)]}
        if case .component(let id) = self, id == UnionPracticeExample.exampleID {return UnionPracticeExample.sourceResources}
        if case .component(let id) = self, id == SummarizePracticeExample.exampleID { return [.init(address:WorkflowExamples.sourceAddress,resource:SummarizePracticeExample.observationsResource)] }
        if case .component(let id) = self, id == PreprocessPracticeExample.exampleID { return [.init(address:WorkflowExamples.sourceAddress,resource:PreprocessPracticeExample.observationsResource)] }
        if case .component(let id) = self, id == FormulaTemporalPracticeExample.exampleID { return [.init(address:WorkflowExamples.sourceAddress,resource:FormulaTemporalPracticeExample.observationsResource)] }
        if case .component(let id) = self, id == FormulaPracticeExample.exampleID { return [.init(address:WorkflowExamples.sourceAddress,resource:FormulaPracticeExample.observationsResource)] }
        if case .component(let id) = self, id == PartitionPracticeExample.exampleID { return [.init(address:WorkflowExamples.sourceAddress,resource:PartitionPracticeExample.observationsResource)] }
        if case .component(let id) = self, id == FilterPracticeExample.exampleID { return [.init(address: WorkflowExamples.sourceAddress, resource: FilterPracticeExample.observationsResource)] }
        if case .component(let id)=self,id==FineTuningPracticeExample.exampleID {return [.init(address:WorkflowExamples.sourceAddress,resource:"fine-tuning-observations")]}
        if case .component(let id) = self, [SequenceTrainingPracticeExample.exampleID,TextGenerationPracticeExample.exampleID,RotaryPositionPracticeExample.componentID].contains(id) { return [.init(address:WorkflowExamples.sourceAddress,resource:TextEncodingPracticeExample.trainingResource)] }
        if case .component(let id) = self, id == TextEncodingPracticeExample.trainingExampleID { return [.init(address:WorkflowExamples.sourceAddress,resource:TextEncodingPracticeExample.trainingResource)] }
        if case .component(let id) = self, id == SequenceWindowsPracticeExample.exampleID { return [.init(address:WorkflowExamples.sourceAddress,resource:SequenceWindowsPracticeExample.observationsResource)] }
        if case .component(let id) = self, id == TextEncodingPracticeExample.exampleID { return [.init(address: WorkflowExamples.sourceAddress, resource: TextEncodingPracticeExample.observationsResource)] }
        if case .component(let id) = self, id == ColumnsPracticeExample.exampleID { return [.init(address: WorkflowExamples.sourceAddress, resource: ColumnsPracticeExample.observationsResource)] }
        if case .component(let id) = self, [FileInputViewPracticeExample.exampleID,PrepareMediaPracticeExample.exampleID].contains(id) { return [] }
        if case .component(let id) = self, let lesson = try? DataEngineeringPractice.lesson(id) {
            return [.init(address: .init(domain: .data, localID: lesson.sourceNode), resource: (lesson.source as NSString).deletingPathExtension)]
        }
        if case .groupPartition = self {return [.init(address:WorkflowExamples.sourceAddress,resource:"repeated-measurements")]}
        if case .temporalPartition = self {return [.init(address:WorkflowExamples.sourceAddress,resource:"partition-observations")]}
        if case .aggregated = self {return [.init(address:WorkflowExamples.sourceAddress,resource:"repeated-measurements")]}
        if case .joined = self {
            return [.init(address:WorkflowExamples.sourceAddress,resource:"join-signals"),
                    .init(address:WorkflowExamples.joinContextAddress,resource:"join-context")]
        }
        return [.init(address:WorkflowExamples.sourceAddress,resource:observationsResource)]
    }

    public var observationsResource: String {
        if case .component(let id) = self, id == TableOutputPracticeExample.exampleID {return "table-output-practice"}
        if case .component(let id) = self, id == SortPracticeExample.exampleID {return "sort-practice"}
        if case .component(let id) = self, id == UniquePracticeExample.exampleID {return "unique-practice"}
        if case .component(let id) = self, id == PreprocessPracticeExample.cleanupExampleID {return PreprocessPracticeExample.cleanupObservationsResource}
        if case .component(let id) = self, id == ReshapeTablePracticeExample.exampleID {return ReshapeTablePracticeExample.observationsResource}
        if case .component(let id) = self, id == ReshapeTablePracticeExample.wideExampleID {return ReshapeTablePracticeExample.wideResource}
        if case .component(let id) = self, id == SplitTextPracticeExample.exampleID {return SplitTextPracticeExample.observationsResource}
        if case .component(let id) = self, id == ModelOutputPracticeExample.exampleID {return ModelOutputPracticeExample.observationsResource}
        if case .component(let id) = self, id == PreprocessPracticeExample.exampleID { return PreprocessPracticeExample.observationsResource }
        if case .component(let id) = self, id == SummarizePracticeExample.exampleID { return SummarizePracticeExample.observationsResource }
        if case .component(let id) = self, id == FormulaTemporalPracticeExample.exampleID { return FormulaTemporalPracticeExample.observationsResource }
        if case .component(let id) = self, id == FormulaPracticeExample.exampleID { return FormulaPracticeExample.observationsResource }
        if case .component(let id) = self, id == PartitionPracticeExample.exampleID { return PartitionPracticeExample.observationsResource }
        if case .component(let id) = self, id == FilterPracticeExample.exampleID { return FilterPracticeExample.observationsResource }
        if case .component(let id)=self,id==FineTuningPracticeExample.exampleID {return "fine-tuning-observations"}
        if case .component(let id) = self, [SequenceTrainingPracticeExample.exampleID,TextGenerationPracticeExample.exampleID,RotaryPositionPracticeExample.componentID].contains(id) { return TextEncodingPracticeExample.trainingResource }
        if case .component(let id) = self, id == TextEncodingPracticeExample.trainingExampleID { return TextEncodingPracticeExample.trainingResource }
        if case .component(let id) = self, id == SequenceWindowsPracticeExample.exampleID { return SequenceWindowsPracticeExample.observationsResource }
        if case .component(let id) = self, id == TextEncodingPracticeExample.exampleID { return TextEncodingPracticeExample.observationsResource }
        if case .component(let id) = self, id == ColumnsPracticeExample.exampleID { return ColumnsPracticeExample.observationsResource }
        if case .component(let id) = self, let lesson = try? DataEngineeringPractice.lesson(id) { return (lesson.source as NSString).deletingPathExtension }
        if case .component(let id) = self, GradientLearningTask(rawValue:id) != nil {return "gradient-observations"}
        if case .preparationPractice = self { return PreparationPracticeExample.observationsResource }
        if case .component(let id) = self, let resource=BundledLossComponents.entries[id]?.learning?.sourceResource {return resource}
        if case .component(let id) = self, let resource=BundledSpatialPatchComponents.entries[id]?.learning?.sourceResource {return resource}
        if case .component(let id) = self, let resource=BundledTensorRoutingComponents.entries[id]?.learning?.sourceResource {return resource}
        if case .component(let id) = self, let resource = (BundledTensorCompositionComponents.entries[id]?.learning?.sourceResource ?? BundledConvolutionComponents.entries[id]?.learning?.sourceResource ?? BundledEmbeddingComponents.entries[id]?.learning?.sourceResource ?? BundledTensorMathComponents.entries[id]?.learning?.sourceResource ?? BundledTensorLayoutComponents.entries[id]?.learning?.sourceResource ?? BundledLayerComponents.entries[id]?.learning?.sourceResource ?? BundledPoolingComponents.entries[id]?.learning?.sourceResource ?? BundledActivationComponents.entries[id]?.learning?.sourceResource ?? BundledNormalizationComponents.entries[id]?.learning?.sourceResource ?? BundledOptimizerComponents.entries[id]?.learning?.sourceResource) { return resource }
        if case .embedding = self { return "embedding-observations" }
        return "workflow-observations"
    }

    public func document(device: String = "auto") throws -> WorkflowDocument {
        switch self {
        case .preparationPractice: return try PreparationPracticeExample.document(device: device)
        case .backend(let id):
            if id == "sklearn" { return try WorkflowExamples.preparedSklearnClassifier() }
            if id == "tensorflow" { return try WorkflowExamples.preparedTensorFlowClassifier() }
            throw DataFlowContractError("Unknown backend example.")
        case .separableConv2d: return try WorkflowExamples.preparedSeparableConv2dClassifier(device: device)
        case .depthwiseConv2d: return try WorkflowExamples.preparedDepthwiseConv2dClassifier(device: device)
        case .convTranspose3d: return try WorkflowExamples.preparedConvTranspose3dClassifier(device: device)
        case .convTranspose2d: return try WorkflowExamples.preparedConvTranspose2dClassifier(device: device)
        case .convTranspose1d: return try WorkflowExamples.preparedConvTranspose1dClassifier(device: device)
        case .conv3d: return try WorkflowExamples.preparedConv3dClassifier(device: device)
        case .conv2d: return try WorkflowExamples.preparedConv2dClassifier(device: device)
        case .component(let id):
            if id == MixtureExpertsPractice.lessonID {return try MixtureExpertsPractice.document(device:device)}
            if id == TrainDataAccessPracticeExample.exampleID { return try TrainDataAccessPracticeExample.document(device: device) }
            if id == TableOutputPracticeExample.exampleID {return try TableOutputPracticeExample.document()}
            if id == SortPracticeExample.exampleID {return try SortPracticeExample.document()}
            if id == UniquePracticeExample.exampleID {return try UniquePracticeExample.document()}
            if id == PreprocessPracticeExample.cleanupExampleID {return try PreprocessPracticeExample.cleanupDocument()}
            if id == ReshapeTablePracticeExample.exampleID {return try ReshapeTablePracticeExample.document()}
            if id == ReshapeTablePracticeExample.wideExampleID {return try ReshapeTablePracticeExample.document(wide:true)}
            if id == SplitTextPracticeExample.exampleID {return try SplitTextPracticeExample.document()}
            if id == ModelOutputPracticeExample.exampleID {return try ModelOutputPracticeExample.document(device:device)}
            if id == UnionPracticeExample.exampleID {return try UnionPracticeExample.document()}
            if id == SummarizePracticeExample.exampleID { return try SummarizePracticeExample.document() }
            if id == PreprocessPracticeExample.exampleID { return try PreprocessPracticeExample.document() }
            if id == FormulaTemporalPracticeExample.exampleID { return try FormulaTemporalPracticeExample.document() }
            if id == FormulaPracticeExample.exampleID { return try FormulaPracticeExample.document() }
            if id == PartitionPracticeExample.exampleID { return try PartitionPracticeExample.document() }
            if id == FilterPracticeExample.exampleID { return try FilterPracticeExample.document() }
            if id == FineTuningPracticeExample.exampleID {return try FineTuningPracticeExample.document(device:device)}
            if id == TextGenerationPracticeExample.exampleID {return try TextGenerationPracticeExample.document(device:device)}
            if id == SequenceTrainingPracticeExample.exampleID {return try SequenceTrainingPracticeExample.document(device:device)}
            if id == TextEncodingPracticeExample.trainingExampleID {return try TextEncodingPracticeExample.trainingDocument(device:device)}
            if id == SequenceWindowsPracticeExample.exampleID {return try SequenceWindowsPracticeExample.document()}
            if id == TextEncodingPracticeExample.exampleID {return try TextEncodingPracticeExample.document()}
            if id == ColumnsPracticeExample.exampleID { return try ColumnsPracticeExample.document() }
            if id == DocumentClassificationPracticeExample.exampleID {return try DocumentClassificationPracticeExample.document()}
            if id == SelfPlayPractice.lessonID {return try SelfPlayPractice.document()}
            if id == GameEpisodePractice.lessonID {return try GameEpisodePractice.document()}
            if id == GameRuleConfiguration.lessonID {return try GameRuleConfiguration.document()}
            if id == PrebuiltModelsPractice.lessonID {return try PrebuiltModelsPractice.document()}
            if id == TypedDecisionPracticeExample.exampleID {return try TypedDecisionPracticeExample.document()}
            if id == AudioClassificationPracticeExample.exampleID {return try AudioClassificationPracticeExample.document()}
            if id == SpeechTranscriptionPracticeExample.exampleID {return try SpeechTranscriptionPracticeExample.document()}
            if id == OCRPracticeExample.exampleID {return try OCRPracticeExample.document()}
            if id == PrepareMediaPracticeExample.exampleID {return try PrepareMediaPracticeExample.document()}
            if id == FileInputViewPracticeExample.exampleID { return try FileInputViewPracticeExample.document() }
            if id == TableInputViewPracticeExample.exampleID { return try TableInputViewPracticeExample.document() }
            if DataEngineeringPractice.isExample(id) { return try DataEngineeringPractice.document(id) }
            if let task=GradientLearningTask(rawValue:id) {return try GradientWorkflowExamples.document(task:task,device:device)}
            if BundledLossComponents.entries[id] != nil {return try BundledLossComponents.example(id,device:device)}
            if id=="transformer" {return try WorkflowExamples.preparedEditableTransformerClassifier(device:device)}
            if BundledAttentionComponents.entries[id] != nil {return try BundledAttentionComponents.example(id,device:device)}
            if BundledRecurrentComponents.entries[id] != nil {return try BundledRecurrentComponents.example(id,device:device)}
            if BundledSchedulerComponents.entries[id] != nil {return try BundledSchedulerComponents.example(id,device:device)}
            if BundledOptimizerComponents.entries[id] != nil {return try BundledOptimizerComponents.example(id,device:device)}
            if BundledSpatialPatchComponents.entries[id] != nil {return try BundledSpatialPatchComponents.example(id,device:device)}
            if BundledTensorRoutingComponents.entries[id] != nil {return try BundledTensorRoutingComponents.example(id,device:device)}
            if BundledTensorCompositionComponents.entries[id] != nil { return try BundledTensorCompositionComponents.example(id, device: device) }
            if BundledConvolutionComponents.entries[id] != nil { return try BundledConvolutionComponents.example(id, device: device) }
            if BundledEmbeddingComponents.entries[id] != nil { return try BundledEmbeddingComponents.example(id, device: device) }
            if BundledTensorMathComponents.entries[id] != nil { return try BundledTensorMathComponents.example(id, device: device) }
            if BundledTensorLayoutComponents.entries[id] != nil { return try BundledTensorLayoutComponents.example(id, device: device) }
            if BundledLayerComponents.entries[id] != nil { return try BundledLayerComponents.example(id, device: device) }
            if BundledNormalizationComponents.entries[id] != nil { return try BundledNormalizationComponents.example(id, device: device) }
            if BundledActivationComponents.entries[id] != nil { return try BundledActivationComponents.example(id, device: device) }
            return try BundledPoolingComponents.example(id, device: device)
        case .conv1d: return try WorkflowExamples.preparedConv1dClassifier(device: device)
        case .lazyLinear: return try WorkflowExamples.preparedLazyLinearClassifier(device: device)
        case .bilinear: return try WorkflowExamples.preparedBilinearClassifier(device: device)
        case .groupPartition: return try WorkflowExamples.preparedGroupSplitClassifier(device:device)
        case .temporalPartition: return try WorkflowExamples.preparedTemporalSplitClassifier(device:device)
        case .aggregated: return try WorkflowExamples.preparedAggregatedClassifier(device:device)
        case .joined: return try WorkflowExamples.preparedJoinedClassifier(device:device)
        case .guided: return try WorkflowGuidedBuildPlan.parallelClassifier().document(completedSteps: 1)
        case .parallel: return try WorkflowExamples.preparedParallelClassifier(device: device)
        case .tensorLayout: return try WorkflowExamples.preparedTensorLayoutClassifier(device: device)
        case .softmax: return try WorkflowExamples.preparedSoftmaxClassifier(device: device)
        case .matrix: return try WorkflowExamples.preparedMatrixClassifier(device: device)
        case .embedding: return try WorkflowExamples.preparedEmbeddingClassifier(device: device)
        case .layerNorm(let whole): return try WorkflowExamples.preparedLayerNormClassifier(device: device, normalizeWholeSample: whole)
        case .gelu(let approximate): return try WorkflowExamples.preparedGELUClassifier(device: device, approximateTanh: approximate)
        }
    }
}
