import XCTest
@testable import KwipMLCore

final class TensorLearningTaskTitleTests:XCTestCase {
    func testEveryTaskHasAReadableTitleWithoutChangingItsIdentity() throws {
        var expected=["unfold-overlapping-windows":"Turn nearby cells into patch columns","unfold-padded-spaced-windows":"Extract spaced patches with zero borders","fold-overlap-addition":"Rebuild a grid by adding overlapping parts","fold-uncovered-positions":"See where patch parts leave gaps","split-size-order":"Split and reorder table sections","split-sections-order":"Choose unequal table sections","chunk-uneven-parts":"Find the final uneven part","gather-repeated-positions":"Pick repeated table positions","einsum-combine-tables":"Combine two tables","einsum-diagonal-view":"Read a table’s diagonal","concat-sample-axis":"Combine along an existing direction","add-shared-pattern":"Add a shared pattern","stack-new-direction":"Keep two sources in a new direction","permute-sample-order":"Reorder the sample directions","prelu-shared-slope":"PReLU · one learned negative multiplier","prelu-channel-slopes":"PReLU · a multiplier per channel","logsoftmax-sample-scores":"LogSoftmax · compare relative scores","dropout-training-pattern":"Dropout · temporarily remove measurements","dropout-evaluation-pattern":"Dropout · switch from training to evaluation","avg-pool2d-windows":"Average a map · boundary windows","adaptive-avg-pool1d-bins":"Choose a length · shared average bins","adaptive-avg-pool2d-bins":"Choose a map height · keep its width","adaptive-max-pool2d-bins":"Choose a map height · winning positions","silu-values":"SiLU · smooth signed values","leaky-relu-slope":"LeakyReLU · negative slope","elu-alpha":"ELU · negative saturation","max-pool1d-windows":"MaxPool1d · winning positions","max-pool2d-windows":"MaxPool2d · winning positions","max-pool3d-windows":"MaxPool3d · winning positions","avg-pool1d-windows":"Average windows · partial boundaries","lazy-linear-inferred-width":"LazyLinear · infer the feature width","reshape":"Reshape tensors","transpose":"Transpose tensor axes","flatten":"Flatten tensor dimensions","softmax-last-axis":"Softmax · normalize rows","softmax-first-sample-axis":"Softmax · normalize columns","layer-norm-rows":"LayerNorm · normalize rows","layer-norm-sample":"LayerNorm · normalize each sample","layer-norm-affine":"LayerNorm · learned scale and bias","embedding-gather":"Embedding · gather and accumulate","embedding-padding-frequency":"Embedding · padding and frequency","embedding-renormalize":"Embedding · forward weight changes","conv1d-shared-kernel":"Conv1d · shared local windows","conv-transpose1d-shared-kernel":"ConvTranspose1d · overlapping output windows","conv-transpose2d-shared-kernel":"ConvTranspose2d · overlapping output windows","conv-transpose3d-shared-kernel":"ConvTranspose3d · overlapping output windows","conv2d-shared-kernel":"Conv2d · shared local windows","conv3d-shared-kernel":"Conv3d · shared local windows","bilinear-with-bias":"Bilinear interactions · learned bias","bilinear-without-bias":"Bilinear interactions · without bias","matrix-contraction":"Matrix products · rows and columns","matrix-broadcast-gradients":"Matrix products · shared heads and gradients"]
        for task in StatefulNormalizationLearningTask.allCases {expected[task.rawValue]=task.title}
        for task in OptimizerLearningTask.allCases {expected[task.rawValue]=task.title}
        for task in SchedulerLearningTask.allCases {expected[task.rawValue]=task.title}
        expected.merge(["rnn-sequence-memory":"RNN · Remember a sequence of measurements","lstm-separate-memory":"LSTM · Carry hidden and cell memory","gru-reset-memory":"GRU · Control which earlier information is reused","rnn-cell-step":"RNNCell · Continue one observation step","lstm-cell-step":"LSTMCell · Continue one observation step","gru-cell-step":"GRUCell · Continue one observation step"]) {_,new in new}
        expected.merge(["multihead-attention-connections":"MultiheadAttention · Choose which values matter","transformer-source-target":"Transformer · Combine a source with a requested sequence","transformer-encoder-context":"TransformerEncoder · Compare positions within a sequence","transformer-decoder-memory":"TransformerDecoder · Use a source to build a requested sequence","scaled-attention-similarity":"Scaled attention · Mix values using similarity"]) {_,new in new}
        expected.merge(["cross-entropy-class-choice":"CrossEntropyLoss · Choose a class","bce-independent-labels":"BCEWithLogitsLoss · Judge independent labels","mse-error-size":"MSELoss · Give larger mistakes more importance","l1-absolute-error":"L1Loss · Measure the distance from a target","nll-log-probability-choice":"NLLLoss · Use prepared class probabilities","kl-distribution-comparison":"KLDivLoss · Compare two distributions","ctc-sequence-alignment":"CTCLoss · Match a sequence without position labels","triplet-relative-distance":"TripletMarginLoss · Bring matching examples closer","checkpoint-restore-weights":"Keep and reuse learned settings","checkpoint-continue-training":"Continue from where training stopped"]) {_,new in new}
        expected.merge(["initialization-starting-values":"Choose starting values","initialization-starting-spread":"Match the starting spread"]) {_,new in new}
        expected.merge([
            "dataset-indexed-observations":"Revisit numbered observations",
            "dataset-aligned-tensors":"Keep inputs and targets together",
            "dataset-selected-view":"Choose an ordered view",
            "dataset-finite-stream":"Read a finite stream",
            "loader-complete-and-short-groups":"Make complete and short groups",
            "loader-repeatable-mixing":"Mix rows reproducibly",
            "loader-aligned-sequences":"Keep sequence lengths aligned",
            "loader-continue-grouping":"Continue the same grouping",
            "program-export-computation":"Save the computation",
            "program-export-size-range":"Allow a safe input-size range",
            "gradient-selected-influence":"Ask what changes a result",
            "gradient-combine-small-groups":"Combine small groups before adjusting settings",
            "trainability-keep-settings":"Keep one layer's settings fixed",
            "trainability-next-run":"Choose what can learn next",
            "bag-fixed-sum":"Combine a fixed group",
            "bag-packed-mean":"Average groups of different sizes",
            "bag-packed-max":"Keep the strongest values",
            "bag-fixed-weighted-sum":"Give items different influence",
            "bag-packed-weighted-sum":"Weight groups of different sizes",
            "bag-padding-norm":"Ignore padding and limit vector length",
            "composition-build-reuse":"Build and reuse a block",
            "composition-order-name":"Order and name the work"
        ]) {_,new in new}
        XCTAssertEqual(Set(expected.keys),TensorLearningTask.operations)
        for (operation,title) in expected {
            XCTAssertEqual(TensorLearningTask.title(for:operation),title)
            let plan=try TensorLearningTask.plan(operation:operation,runtimeVersion:"2.14.1",seed:17)
            XCTAssertEqual(plan.id,(DataLoadingLearningTask(rawValue:operation) != nil || ProgramExportLearningTask(rawValue:operation) != nil || GradientLearningTask(rawValue:operation) != nil || EmbeddingBagLearningTask(rawValue:operation) != nil || ModuleCompositionLearningTask(rawValue:operation) != nil || InitializationLearningTask(rawValue:operation) != nil || LossLearningTask(rawValue:operation) != nil || CheckpointLearningTask(rawValue:operation) != nil || AttentionLearningTask(rawValue:operation) != nil || RecurrentLearningTask(rawValue:operation) != nil || StatefulNormalizationLearningTask(rawValue:operation) != nil || OptimizerLearningTask(rawValue:operation) != nil || SchedulerLearningTask(rawValue:operation) != nil || SpatialPatchLearningTask(rawValue:operation) != nil || TensorRoutingLearningTask(rawValue:operation) != nil || RemainingActivationLearningTask(rawValue:operation) != nil || TensorCompositionLearningTask(rawValue:operation) != nil || ConfigurableActivationLearningTask(rawValue: operation) != nil || operation.hasPrefix("softmax") || operation.hasPrefix("layer-norm") || operation.hasPrefix("embedding") || operation.hasPrefix("matrix") || operation.hasPrefix("bilinear") || operation.hasPrefix("lazy-linear") || operation.hasPrefix("conv") || operation.hasPrefix("adaptive-") || operation.hasPrefix("max-pool") || operation.hasPrefix("avg-pool")) ? "assessment."+operation : "assessment.tensor-"+operation)
        }
        XCTAssertNil(TensorLearningTask.title(for:"unknown"))
    }
}
