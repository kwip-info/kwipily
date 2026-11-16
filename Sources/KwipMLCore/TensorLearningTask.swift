import Foundation

/// Shared task identity and policy for measured tensor constructions.
public enum TensorLearningTask {
    /// Presentation only; saved operation IDs and assessment identities stay stable.
    public static func title(for operation:String)->String? {
        if let task=DataLoadingLearningTask(rawValue:operation) {return task.title}
        if let task=ProgramExportLearningTask(rawValue:operation) {return task.title}
        if let task=ModuleCompositionLearningTask(rawValue:operation) {return task.title}
        if let task=GradientLearningTask(rawValue:operation) {return task.title}
        if let task=InitializationLearningTask(rawValue:operation) {return task.title}
        if let task=CheckpointLearningTask(rawValue:operation) {return task.title}
        if let task=LossLearningTask(rawValue:operation) {return task.title}
        if let task=AttentionLearningTask(rawValue:operation) {return task.title}
        if let task=RecurrentLearningTask(rawValue:operation) {return task.title}
        if let task=SchedulerLearningTask(rawValue:operation) {return task.title}
        if let task=OptimizerLearningTask(rawValue:operation) {return task.title}
        if let task=StatefulNormalizationLearningTask(rawValue:operation) {return task.title}
        if let task=SpatialPatchLearningTask(rawValue:operation) {return task.title}
        if let task=TensorRoutingLearningTask(rawValue:operation) {return task.title}
        if let task=RemainingActivationLearningTask(rawValue:operation) {return task.title}
        if let task=TensorCompositionLearningTask(rawValue:operation) {return task.title}
        if let task=AdaptivePoolingLearningTask(rawValue:operation) {return task.title}
        if let task=ConfigurableActivationLearningTask(rawValue:operation) {return task.title}
        if let task=PoolingLearningTask(rawValue:operation) {return task.title}
        if let task=ConvolutionLearningTask(rawValue:operation) {return task.title}
        if operation == LazyLinearAssessment.operation {return LazyLinearAssessment.title}
        if let task=BilinearLearningTask(rawValue:operation) {return task.title}
        if let task=MatrixLearningTask(rawValue:operation) {return task.title}
        if let task=EmbeddingLearningTask(rawValue:operation) {return task.title}
        if let task=EmbeddingBagLearningTask(rawValue:operation) {return task.title}
        if let task=LayerNormLearningTask(rawValue:operation) {return task.title}
        switch operation {
        case "reshape":return "Reshape tensors"
        case "transpose":return "Transpose tensor axes"
        case "flatten":return "Flatten tensor dimensions"
        case "softmax-last-axis":return "Softmax · normalize rows"
        case "softmax-first-sample-axis":return "Softmax · normalize columns"
        default:return nil
        }
    }
    public static let operations:Set<String> = Set(DataLoadingLearningTask.allCases.map(\.rawValue)).union(ProgramExportLearningTask.allCases.map(\.rawValue)).union(ModuleCompositionLearningTask.allCases.map(\.rawValue)).union(EmbeddingBagLearningTask.allCases.map(\.rawValue)).union(GradientLearningTask.allCases.map(\.rawValue)).union(InitializationLearningTask.allCases.map(\.rawValue)).union(LossLearningTask.allCases.map(\.rawValue)).union(CheckpointLearningTask.allCases.map(\.rawValue)).union(["multihead-attention-connections","transformer-source-target","transformer-encoder-context","transformer-decoder-memory","scaled-attention-similarity","rnn-sequence-memory","lstm-separate-memory","gru-reset-memory","rnn-cell-step","lstm-cell-step","gru-cell-step","step-lr-intervals","cosine-lr-curve","linear-lr-warmup","plateau-lr-improvement","one-cycle-lr-batches","sgd-momentum-steps","adam-running-averages","adamw-separate-decay","rmsprop-squared-memory","adagrad-accumulated-steps","batch-norm1d-current-group","batch-norm1d-remembered-summary","batch-norm2d-current-group","batch-norm2d-remembered-summary","instance-norm1d-individual-patterns","instance-norm1d-remembered-summary","instance-norm2d-individual-grids","instance-norm2d-remembered-summary","group-norm-channel-teams","group-norm-channel-adjustments","unfold-overlapping-windows","unfold-padded-spaced-windows","fold-overlap-addition","fold-uncovered-positions","split-size-order","split-sections-order","chunk-uneven-parts","gather-repeated-positions","einsum-combine-tables","einsum-diagonal-view","prelu-shared-slope","prelu-channel-slopes","logsoftmax-sample-scores","dropout-training-pattern","dropout-evaluation-pattern","concat-sample-axis","add-shared-pattern","stack-new-direction","permute-sample-order","avg-pool2d-windows","adaptive-avg-pool1d-bins","adaptive-avg-pool2d-bins","adaptive-max-pool2d-bins","silu-values","leaky-relu-slope","elu-alpha","max-pool1d-windows","max-pool2d-windows","max-pool3d-windows","avg-pool1d-windows","conv-transpose1d-shared-kernel","conv-transpose2d-shared-kernel","conv-transpose3d-shared-kernel","conv1d-shared-kernel","conv2d-shared-kernel","conv3d-shared-kernel","lazy-linear-inferred-width","reshape","transpose","flatten","softmax-last-axis","softmax-first-sample-axis","layer-norm-rows","layer-norm-sample","layer-norm-affine","embedding-gather","embedding-padding-frequency","embedding-renormalize","matrix-contraction","matrix-broadcast-gradients","bilinear-with-bias","bilinear-without-bias"])
    public static func softmaxAxis(for operation:String)->Int? {
        switch operation {
        case "softmax-last-axis":return -1
        case "softmax-first-sample-axis":return 1
        default:return nil
        }
    }
    public static func plan(operation:String,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        if let task=DataLoadingLearningTask(rawValue:operation) {return try DataLoadingAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=ProgramExportLearningTask(rawValue:operation) {return try ProgramExportAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=ModuleCompositionLearningTask(rawValue:operation) {return try task.assessment(seed:seed,runtimeVersion:runtimeVersion)}
        if let task=GradientLearningTask(rawValue:operation) {return try GradientLearningAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=InitializationLearningTask(rawValue:operation) {return try InitializationAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=CheckpointLearningTask(rawValue:operation) {return try CheckpointAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=LossLearningTask(rawValue:operation) {return try LossLearningAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=AttentionLearningTask(rawValue:operation) {return try AttentionAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=RecurrentLearningTask(rawValue:operation) {return try RecurrentAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=SchedulerLearningTask(rawValue:operation) {return try SchedulerAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=OptimizerLearningTask(rawValue:operation) {return try OptimizerAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=StatefulNormalizationLearningTask(rawValue:operation) {return try StatefulNormalizationAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=SpatialPatchLearningTask(rawValue:operation) {return try SpatialPatchAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=TensorRoutingLearningTask(rawValue:operation) {return try TensorRoutingAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=RemainingActivationLearningTask(rawValue:operation) {return try RemainingActivationAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=TensorCompositionLearningTask(rawValue:operation) {return try TensorCompositionAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=AdaptivePoolingLearningTask(rawValue:operation) {return try AdaptivePoolingAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=ConfigurableActivationLearningTask(rawValue:operation) {return try ConfigurableActivationAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=PoolingLearningTask(rawValue:operation) {return try PoolingAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=ConvolutionLearningTask(rawValue:operation) {return try ConvolutionAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if operation == LazyLinearAssessment.operation {return try LazyLinearAssessment.plan(runtimeVersion:runtimeVersion,seed:seed)}
        if let task=BilinearLearningTask(rawValue:operation) {return try BilinearAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=MatrixLearningTask(rawValue:operation) {return try MatrixAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=EmbeddingLearningTask(rawValue:operation) {return try EmbeddingAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=EmbeddingBagLearningTask(rawValue:operation) {return try EmbeddingBagAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let task=LayerNormLearningTask(rawValue:operation) {return try LayerNormAssessment.plan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        if let axis=softmaxAxis(for:operation) {return try SoftmaxAssessment.plan(axis:axis,runtimeVersion:runtimeVersion,seed:seed)}
        return try TensorTransformAssessment.plan(operation:operation,runtimeVersion:runtimeVersion,seed:seed)
    }
}
