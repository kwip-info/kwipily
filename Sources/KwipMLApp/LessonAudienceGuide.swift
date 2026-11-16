import KwipMLCore
import Foundation

/// Presentation guidance only: immutable assessment requirements remain unchanged.
struct LessonAudienceGuide {
    struct Term { let name: String; let meaning: String }
    let goal: String
    let steps: [String]
    let terms: [Term]
    private static func bagControls(_ task:EmbeddingBagLearningTask)->String {
        let packed = task != .fixedSum && task != .fixedWeightedSum
        let weighted = task == .fixedWeightedSum || task == .packedWeightedSum
        let mode = task == .packedMean ? "Mean" : task == .packedMax ? "Maximum" : "Sum"
        return "Set Maximum items per group to 3 on Bag Input. On EmbeddingBag, set Table rows to 4 and Vector dimensions to 3. Choose Group layout: " + (packed ? "Packed groups" : "Fixed rows") + " on both components. Set Combine selected vectors to " + mode + ". " + (weighted ? "Enable" : "Disable") + " Use item weights on both components. " + (task == .packedWeightedSum ? "Enable" : "Disable") + " Include final boundary on both components. Keep Freeze learned weights, Sparse gradients and Scale gradients by frequency disabled. " + (task == .paddingNorm ? "Set Padding row to 0, Maximum norm to 1 and Norm order to 2." : "Leave Padding row and Maximum norm unset; keep Norm order at 2.")
    }
    private static func bagMeaning(_ task:EmbeddingBagLearningTask)->String {
        switch task {
        case .fixedSum:return "Add selected vectors within each observation group. A repeated category contributes again; different groups stay separate."
        case .packedMean:return "Average each group's actual items. Source lengths say how many cells are used. Empty groups keep their output row and return zeros."
        case .packedMax:return "Keep the largest value separately at each vector position; do not select just one whole vector. Empty groups keep their output row and return zeros."
        case .fixedWeightedSum:return "Multiply each selected vector by its real item weight before adding within the group. A weight controls how much that item contributes."
        case .packedWeightedSum:return "Pair each real item weight with its category ID. The final boundary marks the end of all items. Empty groups keep their output row; unused source cells have zero influence."
        case .paddingNorm:return "Padding is a reserved table row excluded from the combination and table influence. Norm means vector length; order 2 is the usual straight-line distance from zero. Shorten selected rows longer than the limit before combining; unused rows stay unchanged. This forward table change is separate from a training update."
        }
    }
    private static func adaptivePoolingControls(_ task:AdaptivePoolingLearningTask)->String {
        switch task {
        case .averageTwo:return "Set Window [height, width] to 3, 3; Stride [height, width] to 2, 3; and Padding [height, width] to 1, 1. Enable Include partial final window and Include padding in average divisor. Leave Custom divisor empty to use the window count. Expect four rows and three columns per channel."
        case .adaptiveAverageOne:return "Set Output length to 3. Each channel has five positions; neighboring averaging regions can share boundary inputs."
        case .adaptiveAverageTwo:return "Set Output height to 3 and leave Output width empty to keep its four input columns. Each output averages its assigned rectangle; neighboring regions can share rows."
        case .adaptiveMaximumTwo:return "Set Output height to 3, leave Output width empty to keep its four input columns, and enable Expose winning indices. Each region selects its largest value; equal values choose the first position in reading order."
        }
    }
    private static func normalizationGrouping(_ task:StatefulNormalizationLearningTask)->String {
        if task.componentID.hasPrefix("batch") {return "Compare each channel across the current observations and positions; different channels have separate summaries."}
        if task.componentID.hasPrefix("instance") {return "Compare positions within each observation and channel; other observations and channels stay separate."}
        return "Compare two equal teams of channels within each observation; different observations stay separate."
    }
    private static func normalizationControls(_ task:StatefulNormalizationLearningTask)->String {
        task.controlInstructions + (task.affine ? " In Normalization parameters, choose Initialize / reset scale to 1 and shift to 0, then enter the requested scale and offset (shown as Bias) for each channel. Enable Inspect this block’s learned-adjustment influence." : "")
    }
    private static func normalizationCheck(_ task:StatefulNormalizationLearningTask)->String {
        "Choose Check construction and compare output, evaluation and measured influence tables. " + (task.remembered ? "The checker processes two supplied calibration groups to build remembered summaries, then uses them for new examples. Those summaries must remain unchanged during evaluation." : "Use the current input's summaries during both training and evaluation.")
    }
    static let routingExplanation="Split and Chunk divide a table into named parts; reconnect the part you mean rather than treating all parts as one output. Gather reads the provided whole-number positions and can reuse a value several times.\n\nEinsum’s arrangement presets combine paired rows and columns or read a diagonal. Directions keep separate observations apart. Measured input influence adds when a source value is reused; position selectors are not learned values. No code or handwritten derivatives are required."
    private static func routingBuild(_ task:TensorRoutingLearningTask)->String {
        switch task {
        case .splitSize,.splitSections:return "Connect the number Input to Split. Add two Concatenate blocks to join the named parts in the requested order; connect the final result to Model Output."
        case .chunk:return "Connect number Input to Chunk, then the selected part to Model Output."
        case .gather:return "Connect the number Input to source and Position selectors to indices; connect Gather to Model Output."
        case .einsumCombine:return "Connect the first table to a and the second table to b; connect Einsum to Model Output."
        case .einsumDiagonal:return "Connect number Input to a, then Einsum to Model Output."
        }
    }
    private static func attentionBuild(_ task:AttentionLearningTask)->String {
        "Add Input for each supplied role: "+task.sourceRoles.joined(separator:", ")+". Connect each to its matching attention port. Connect "+task.maskReferenceRoles[0]+" to Attention mask query and "+task.maskReferenceRoles[1]+" to key; connect mask to "+task.maskInputPort+(task == .multihead ? " and padding to key_padding_mask":"")+". Connect out to result Output"+(task == .multihead ? " and weights to a separate weights Output":"")+"."
    }
    private static func lossControls(_ task:LossLearningTask)->String {
        task.instructions.components(separatedBy:"Configure: ").last?.components(separatedBy:"\n\nCheck:").first ?? task.instructions
    }
    private static func lossBuild(_ task:LossLearningTask)->String {
        "Connect the loss recipe out to Objective check.objective. "+(task == .triplet ? "The supplied prediction has anchor, match and nonmatch in that order; it needs no separate target input. ":"Connect supplied roles "+task.sourceRoles.filter {$0 != "prediction"}.joined(separator:", ")+" to their matching Objective check inputs. ")+(task.usesLogSoftmax ? "Connect prediction through the visible LogSoftmax, with Normalization direction −1, to Objective check.prediction. ":"Connect prediction directly to Objective check.prediction. ")+"Connect the Individual scores, Combined score and Input influence outputs to their matching measured results."
    }
    static func requirements(_ text: String) -> String { text }
    /// The visual task describes the same public construction requirements. The
    /// original specification and independent numerical grading remain unchanged.
    static func requirements(operation: String, fallback: String) -> String {
        if let task=DataLoadingLearningTask(rawValue:operation) {return task.instructions}
        if let task=ProgramExportLearningTask(rawValue:operation) {return task.instructions}
        if let task=EmbeddingBagLearningTask(rawValue:operation) {
            return "**Build**\nA bag is one observation's group of category IDs. An ID is a whole number selecting a table row. A vector is an ordered list of numbers from that row. Place Bag Input, EmbeddingBag and Model Output. Connect indices to indices, offsets to offsets when Group layout is Packed groups, and per_sample_weights to the matching input when Use item weights is enabled. Connect out to Model Output. The checker supplies real source rows, lengths, item weights and the table; no code or hand-entered answers are needed.\n\n**Configure**\n"+bagControls(task)+" Apply each configuration before checking.\n\n**What should happen**\n"+bagMeaning(task)+" An offset marks where a group starts in a packed list. Equal offsets preserve empty groups.\n\n**Check**\nChoose Check construction. Compare each group's combined vector, the table before and after the operation, and measured influence. A gradient describes how a small change affects the result. Whole-number IDs and offsets have no gradient available; this is different from zero influence. Real item weights can have gradients with Sum. Private assessment keeps source and result tables hidden."
        }
        if let task=ModuleCompositionLearningTask(rawValue:operation) {
            let build=task == .buildReuse ? "Open the supplied Block in its own tab. Connect A to B and both required branches. Keep the supplied fixed sign reversal and subtraction, then connect the model's input and output." : "Open each supplied Block in its own tab. Connect A to B inside each Block, then connect all three Block results in Model."
            let configure=task == .buildReuse ? "Keep each private copy's learned settings separate. Leave the supplied fixed layers and behavior check unchanged." : "Choose the sequence, ordered collection or named collection for the matching Block. Inspect stored child order and names separately from the connections that determine the calculation."
            return "**Build**\n"+build+"\n\n**Configure**\n"+configure+" Apply each Block draft before checking. Discard returns to the applied construction.\n\n**Check**\nChoose Check construction. Influence means how a small change affects a result; the checker measures it automatically along with results and saved reuse. Guided and practice modes can show reference differences: zero means agreement for that measurement. Private assessment keeps its measured tables hidden."
        }
        if let task=GradientLearningTask(rawValue:operation) {
            return "**Build**\nConnect Gradient policy, Trainability and Supplied SGD out to the matching Train inputs. The checker supplies the model and observations.\n\n**Configure**\n"+task.controlInstructions+" Apply the structured controls.\n\n**Check**\nChoose Check construction. Compare measured differences from the independent reference; zero means match. A missing derivative is reported separately from zero influence."
        }
        if let task=InitializationLearningTask(rawValue:operation) {
            return "**Build**\nConnect Starting settings out to Train initialization and Supplied SGD out to Train optimizer. The checker supplies the model and rows.\n\n**Configure**\n"+task.controlInstructions+" Apply the typed owner and algorithm controls.\n\n**Check**\nChoose Check construction. Compare differences from the independent reference: zero means match. Repeatability and saved reuse are measured separately."
        }
        if let task=CheckpointLearningTask(rawValue:operation) {
            return "**Configure**\n"+task.controlInstructions+" Apply the controls. The checker supplies the architecture, rows and saved lesson run; do not add blocks or file paths.\n\n**What should happen**\n"+task.explanation+"\n\n**Check**\nChoose Check construction. KwipML automatically saves and reopens the parent, runs your chosen mode and compares the measured tables. The saved parent must stay unchanged."
        }
        if let task=LossLearningTask(rawValue:operation) {
            return task.instructions.replacingOccurrences(of:"Build: ",with:"**Build**\n").replacingOccurrences(of:"Configure: ",with:"**Configure**\n").replacingOccurrences(of:"Check: ",with:"**Check**\n").replacingOccurrences(of:"Understanding: ",with:"**Understanding**\n")+" The checker measures scores and input influence automatically."
        }
        if let task=AttentionLearningTask(rawValue:operation) {
            return "**Build**\n"+attentionBuild(task)+"\n\n**Configure**\n"+task.controlInstructions+"\n\n**Check**\nChoose Check construction. Compare position results"+(task == .multihead ? " and connection weights":"")+" and each supplied source's influence. The checker supplies learned tables; do not enter weights or answers."
        }
        if let task=RecurrentLearningTask(rawValue:operation) {
            return "**Build**\nAdd one Input for each supplied role: "+task.sourceRoles.joined(separator:", ")+". Connect input to in and each state to its matching port. Connect out to the result Output and hidden/cell to their matching separate Output roles.\n\n**Configure**\n"+task.controlInstructions+"\n\n**Check**\nChoose Check construction. Compare the ordered measurement results and the final state. The checker supplies varied learned tables and measures how input, starting state and every learned value influence all connected results."
        }
        if let task=SchedulerLearningTask(rawValue:operation) {
            return "**Build**\nAdd \(task.componentTitle), Supplied SGD and Train. Connect SGD out → Train optimizer, then \(task.componentTitle) out → Train scheduler. These wires carry training instructions.\n\n**Configure**\n"+task.controlInstructions+" Keep the supplied SGD settings unchanged.\n\n**Check**\nChoose Check construction. Compare the rate used for each group, the next rate, and the learned settings. An epoch is one complete pass; this exercise has two groups per pass. The checker saves halfway and checks that reopening continues with the same adjustments."
        }
        if let task=OptimizerLearningTask(rawValue:operation) {
            return "**Build**\nAdd \(task.componentTitle) and Train. Connect optimizer out → Train optimizer. This wire carries a recipe; the checker supplies the learned settings and gradient sequence.\n\n**Configure**\nSelect \(task.componentTitle). "+task.controlInstructions+"\n\n**Check**\nChoose Check construction. Compare starting settings, supplied gradients, updated settings and remembered quantities over four steps. A parameter without a gradient receives no update. Learning rate zero keeps its settings unchanged. No code or hidden-state editing is required."
        }
        if let task=StatefulNormalizationLearningTask(rawValue:operation) {
            return "**Build**\nConnect the supplied Input → \(task.componentTitle) → Model Output. "+normalizationGrouping(task)+"\n\n**Configure**\nSelect \(task.componentTitle). "+normalizationControls(task)+"\n\n**Check**\n"+normalizationCheck(task)+" Influence means how a small change affects a result; KwipML measures it automatically. No code or handwritten derivatives are required."
        }
        if let task=SpatialPatchLearningTask(rawValue:operation) {
            return "**Build**\nConnect the supplied Input → \(task.isUnfold ? "Unfold":"Fold") → Model Output. Keep observations and channels separate.\n\n**Configure**\nSelect the patch block. "+task.controlInstructions+"\n\n**Check**\nChoose Check construction and compare output and input-influence tables. A source used repeatedly receives added influence. Declared borders and uncovered grid positions contribute zero. Fold adds overlapping entries; it does not average them or automatically undo Unfold. No code or handwritten derivatives are required."
        }
        if let task=TensorRoutingLearningTask(rawValue:operation) {
            return "**Build**\n"+routingBuild(task)+"\n\n**Configure**\nSelect the requested block. "+task.controlInstructions+"\n\n**Check**\nChoose Check construction. Output and number-source influence are measured automatically. Compare their tables; repeated source uses add influence. Position selectors are fixture-owned whole numbers, not values to train."
        }
        if let task=RemainingActivationLearningTask(rawValue:operation) {return task.instructions}
        if let task=TensorCompositionLearningTask(rawValue:operation) {
            return "**Build**\n"+(task.isBinary ? "Add two Inputs. Connect left to a and right to b, then the block to Model Output.":"Connect Input → Permute → Model Output.")+"\n\n**Configure**\n"+task.controlInstructions+"\n\n**Check**\nChoose Check construction. Output and each source’s small-change influence are measured automatically; compare the separate tables."
        }
        if let task=AdaptivePoolingLearningTask(rawValue:operation) {
            return "**Build**\nConnect Input → \(BundledPoolingComponents.definition(task.componentID).title) → Model Output. Keep observations and channels separate.\n\n**Configure**\nSelect the pooling block. "+adaptivePoolingControls(task)+"\n\n**Check**\nChoose Check construction. Output values, input influence and winning indices when enabled are measured automatically. Inspect the separate tables. A gradient describes how a small input change affects the result; no code or handwritten derivatives are required."
        }
        if let task=ConfigurableActivationLearningTask(rawValue:operation) {
            return "**Build**\nUse an array with two rows of three numbers per observation.\n\n**Configure**\n"+task.instructions+"\n\n**Check**\nChoose Check construction and inspect the automatically measured output and input-influence tables."
        }
        if let task=PoolingLearningTask(rawValue:operation) {
            return "**Build**\nConnect Input → \(task.isAverage ? "AvgPool1d":"MaxPool\(task.dimensions)d") → Model Output.\n\n**Configure**\n"+task.instructions+"\n\n**Check**\nChoose Check construction. Output, input influence and winning indices when applicable are measured automatically; inspect each table."
        }
        if let task=ConvolutionLearningTask(rawValue:operation),task.isTransposed {
            return "**Build**\nConnect Input → ConvTranspose\(task.dimensions)d → Model Output. Each observation has two channels and three positions in every spatial direction. A channel is a separate measurement field; a kernel is a small shared window of weights. The shared kernel scatters each input into an output window; contributions add where windows overlap.\n\n**Configure**\nSelect ConvTranspose\(task.dimensions)d. Set Input and Output channels to 2, Kernel size to 3, Stride to 2, Dilation to 1, Groups to 1, Padding to 0 and Output padding to 1 in every spatial direction. Stride moves the window’s start; dilation spaces positions within it; Groups 1 connects all input channels to each output channel. Padding controls boundary extent. Enable Learn bias; the lesson supplies kernel weights and bias automatically.\n\n**Check**\nExpect eight positions in each spatial direction. Output padding chooses extra output extent; it does not append input values or reverse the original calculation. An extra boundary position with no window contribution still contains bias. Choose Check construction; influence on inputs, kernel and bias is measured automatically. No code or handwritten derivatives are required."
        }
        if let task=ConvolutionLearningTask(rawValue:operation) {
            return "**Build**\nAdd Input, Conv\(task.dimensions)d and Model Output. Connect them in that order. Each observation has two channels and four positions in each of its \(task.dimensions) spatial directions. The same shared window is used at every position.\n\n**Configure**\nSelect Conv\(task.dimensions)d. Set Input channels and Output channels to 2, Kernel size to 2 in each spatial direction, Stride and Dilation to 1, Groups to 1, and enable Learn bias. Use Explicit zero padding with amount 0; Valid with amount 0 is equivalent. The lesson supplies kernel weights and bias automatically.\n\n**Check**\nPredict two output channels and three positions per spatial direction. Choose Check construction to inspect the measured output, input influence and shared kernel/bias influence. A gradient describes how a small change affects the result; KwipML checks it automatically. No code or handwritten derivatives are required. Equivalent valid names and preserving layouts do not determine the grade."
        }
        if operation == LazyLinearAssessment.operation {
            return "**Configure**\nAdd Input, LazyLinear and Model Output. Connect Input → LazyLinear → Model Output. Set Output features to four. Either Learn bias choice is accepted. Each observation has two groups of three measurements.\n\n**What should happen**\nKeep the two groups and produce four values for each group. The layer infers three input features and creates four sets of three weights, plus four offsets if Learn bias is on.\n\n**Check your construction**\nChoose Check construction. KwipML measures the inferred width, output shape and created weights. It also repeats the calculation and reloads the same saved weights to verify unchanged results and no fresh random draws. This check does not train the layer or judge prediction quality."
        }
        let configure: String
        let result: String
        switch operation {
        case "reshape":
            configure = "Connect Input → Reshape → Model Output. Set Sample dimensions to [3,2]. Each input observation is [2,3]: two rows of three numbers. The output must have three rows of two numbers."
            result = "Keep the six numbers in the same reading order: read across each row before moving to the next. Do not combine different observations. Reshape changes the layout, not the values."
        case "transpose":
            configure = "Connect Input → Transpose → Model Output. Set First axis to 1 and Second axis to 2. Each observation changes from [2,3] to [3,2]. Axis 0 identifies observations and must stay separate."
            result = "The rows become columns. A number's old row becomes its new column, and its old column becomes its new row. This moves numbers differently from Reshape even though both produce [3,2]."
        case "flatten":
            configure = "Connect Input → Flatten → Model Output. Set First axis to 1 and Last axis to -1, which means the final direction. Each [2,3] observation becomes [6]: one row of six numbers."
            result = "Keep the same reading order: finish one row before starting the next. Do not combine different observations; axis 0 stays separate."
        case "softmax-last-axis":
            configure = "Connect Input → Softmax → Model Output. Set Normalization axis to -1, the last direction. Each observation is [2,3]; the three values in each row compete with one another."
            result = "Each row's three output shares must add up to 1. A larger input score gets a larger share. Keep the shape [2,3]. Do not combine different observations."
        case "softmax-first-sample-axis":
            configure = "Connect Input → Softmax → Model Output. Set Normalization axis to 1. Each observation is [2,3]; the two values in each column compete with one another."
            result = "Each column's two output shares must add up to 1. A larger input score gets a larger share. Keep the shape [2,3]. Do not combine different observations."
        case "layer-norm-rows", "layer-norm-sample", "layer-norm-affine":
            let wholeSample = operation == "layer-norm-sample"
            let withParameters = operation == "layer-norm-affine"
            let group = wholeSample ? "[2,3], so all six values in one observation form a group" : "[3], so each row of three values forms a separate group"
            let parameters = withParameters
                ? "Enable learned scale and bias. In the parameter editor, enter scale [0.5,-1,2] and bias [1,-0.5,0.25]. Use the same three settings for every row and observation. Enable **Inspect this component’s parameter gradients** in the parameter editor so the checker can measure how small changes to these settings affect the result."
                : "Disable learned scale and bias; this task only centers and scales the groups."
            configure = "Connect Input → LayerNorm → Model Output. Set Trailing sample dimensions to \(group). Set Epsilon to 0.125. \(parameters)"
            result = "Keep each observation's shape [2,3] and keep observations separate. For each group, subtract its average. Measure spread by averaging the squared differences, add 0.125, then take the square root and divide by that result. Epsilon is the extra amount inside this calculation; it prevents a zero divisor."
        case "embedding-gather", "embedding-padding-frequency", "embedding-renormalize":
            let options: String
            let policy: String
            switch operation {
            case "embedding-padding-frequency":
                options = "Set Padding row to 0 and enable Scale gradients by frequency. Leave Maximum norm unset."
                policy = "Row 0 keeps its supplied values in the output, but its learned numbers receive no update contribution in this exercise. For other rows, repeated selections share their contributions by the number of times that row appears in the whole input. The lookup leaves the table values unchanged."
            case "embedding-renormalize":
                options = "Leave Padding row unset. Set Maximum norm to 1 and Norm order to 2. Disable Scale gradients by frequency."
                policy = "Any selected row longer than the limit is shortened to that limit before lookup. Maximum norm still changes selected rows when Freeze is enabled in a general model; keep Freeze disabled for this graded task. Length means the square root of the sum of its squared values; the technical specification records the small numerical safeguard. Unselected rows stay unchanged. The checker treats shortening as a separate table update, rather than differentiating that update."
            default:
                options = "Leave Padding row and Maximum norm unset. Disable Scale gradients by frequency."
                policy = "Repeated selections reuse the same learned row. Their update contributions add together; unselected rows receive none. The lookup leaves the table values unchanged."
            }
            configure = "Connect Input → Embedding → Model Output. The lesson supplies the lookup table; it is not a separate connector. Set Table rows to 4 rows and Vector dimensions to 3 values per row. \(options) Set Norm order to 2, and keep Sparse gradients and Freeze learned weights disabled."
            result = "Category indices are whole-number selectors that choose table rows. Keep all selector directions and add a final direction of size 3 for the selected values. \(policy) Inspect the output and table-after-lookup evidence. These whole-number selectors are labels and have no input gradient; the checker measures influence on table values instead."
        case "matrix-contraction", "matrix-broadcast-gradients":
            let sharing = operation == "matrix-broadcast-gradients"
                ? "Additional grouping directions may share values when they have equal sizes or size 1. This is broadcasting: size 1 is reused for each matching position. Observations remain independent."
                : "Match each observation's two tables. Additional groups are not an invitation to mix observations."
            configure = "Add the two supplied Inputs, Matrix multiply and Model Output. Connect the left table to input a and the right table to input b, then connect the result to Model Output. The left table's final size must match the right table's next-to-last size. Do not transpose, scale or normalize the tables."
            result = "Each result cell multiplies matching numbers from a left row and a right column, then adds those products. \(sharing) The display shows representative shapes; the checker also tries other compatible sizes. Reused values collect all their update contributions, with each gradient keeping its original table shape."
        case "bilinear-with-bias", "bilinear-without-bias":
            let withBias = operation == "bilinear-with-bias"
            let bias = withBias
                ? "Enable bias is this task's mode: Learned bias is fixed on. Connect the supplied Shared bias input to bias. Start the other Bilinear task if you want to work without bias."
                : "Disable bias is this task's mode: Learned bias is fixed off. Do not add a bias input. Start the other Bilinear task if you want to work with bias."
            configure = "Add the supplied Left operand, Right operand and Shared weights Inputs, Bilinear and Model Output. Connect them to a, b and weight, then connect the result to Model Output. \(bias)"
            result = "Bilinear combines pairs of features using the supplied interaction weights. The two input tables' observation and grouping sizes must match exactly; no size-1 expansion is allowed. The weight table has directions for output features, left features and right features, with no observation direction.\(withBias ? " Bias provides one offset per output feature." : " There is no added offset.") The checker varies compatible feature widths. Input gradients keep observations separate; shared weight\(withBias ? " and bias" : "") contributions add across their uses."
        default:
            return fallback
        }
        return """
        **Configure**
        \(configure)

        **What should happen**
        \(result)

        **Check your construction**
        Choose Check construction. KwipML checks output values and the required gradients automatically. A gradient describes how a small change affects the result. You do not need to write code, calculate derivatives or enter gradient answers. If a check fails, adjust the indicated setting or connection and try again. Equivalent valid constructions are accepted by their behavior.
        """
    }
    struct CriterionFeedback {
        let title: String
        let nextStep: String
    }
    /// Reviewed presentation actions, independent of private fixture values and
    /// the immutable grader's technical evidence. Unknown checks stay unknown.
    static func criterionFeedback(operation: String, criterionID: String) -> CriterionFeedback? {
        if let task=DataLoadingLearningTask(rawValue:operation) {
            let title=DataLoadingLessonReference.label(criterionID)
            guard title != criterionID else {return nil}
            return .init(title:title,nextStep:task.controlInstructions+" Apply your changes and check both named recipe connections. Choose Check construction again to compare selected rows, aligned desired outcomes and actual groups.")
        }
        if let task=ProgramExportLearningTask(rawValue:operation) {
            let title=ProgramExportLessonReference.label(criterionID)
            guard title != criterionID else {return nil}
            return .init(title:title,nextStep:task.controlInstructions+" Apply your changes, then choose Check construction again. Compare the reloaded results and the requested input-size rules; the saved parent must stay unchanged.")
        }
        if let task=EmbeddingBagLearningTask(rawValue:operation) {
            let titles=["output":"Combined vector for each group","tableAfterForward":"Table values after combining","tableGradient":"Influence on table values","sourceWeightsGradient":"Influence on real item weights"]
            guard let title=titles[criterionID],criterionID != "sourceWeightsGradient" || task == .fixedWeightedSum || task == .packedWeightedSum else {return nil}
            return .init(title:title,nextStep:bagControls(task)+" Check matching source connectors and keep every observation group separate. "+bagMeaning(task)+" Choose Check construction again. The checker measures influence automatically; do not enter calculated answers.")
        }
        if let task=ModuleCompositionLearningTask(rawValue:operation) {
            guard task.measuredShapes[criterionID] != nil else {return nil}
            return .init(title:ModuleCompositionLearningTask.label(criterionID),nextStep:task.instructions+" Check the connections inside each Block and in Model. Apply every pending Block draft, then choose Check construction again. The checker measures results and influence automatically; do not enter calculated answers.")
        }
        if let task=GradientLearningTask(rawValue:operation) {
            let labels=["selectedResultDifference":"Selected result", "inputInfluenceDifference":"Influence of supplied inputs", "parameterInfluenceDifference":"Influence of the first weight array", "gradientBufferDifference":"Existing gradients remain unchanged", "analysisStateDifference":"Analysis preserves model state", "windowSchedule":"Groups and adjustment windows", "windowObjectiveDifference":"Fairly combined error score", "windowGradientDifference":"Combined adjustment direction", "windowResetDifference":"Reset before the next window", "updatedSettingsDifference":"Settings after each adjustment", "fixedSettingsDifference":"Frozen settings stay fixed", "learningSettingsDifference":"Other settings keep learning", "ownerInventoryDifference":"Actual parameter flags and optimizer membership", "recordingAvailability":"Recorded influence and unavailable recording", "restoredStartingDifference":"Child starts from saved settings", "childOwnerInventoryDifference":"Selected next-run parameter flags", "freshMemoryDifference":"Fresh optimizer memory", "childSettingsDifference":"Actual child settings and predictions", "parentByteDifference":"Saved parent stays unchanged"]
            guard let title=labels[criterionID] else {return nil}
            return .init(title:title,nextStep:task.controlInstructions+" Check the named policy connections, then choose Check construction again. Frozen settings, evaluation mode and unavailable recording are different controls.")
        }
        if let task=LossLearningTask(rawValue:operation) {
            let titles=["training-objective":"Combined score during training","evaluation-objective":"Combined score on held-out examples","target-contract":"Desired outcomes and their supplied roles","reduction":"Individual scores before combining","input-gradients":"Influence of the original model scores"]
            guard let title=titles[criterionID] else {return nil}
            let detail=criterionID == "target-contract" ? "Keep each supplied role connected to its matching input; whole-number labels and lengths must remain whole numbers. "+(task == .triplet ? "The three prediction positions are anchor, match and nonmatch; no separate target is required. ":"") : criterionID == "reduction" ? "Compare Individual scores separately from Combined score; Keep individual scores does not skip Objective check's combination. " : criterionID == "input-gradients" ? "Input influence describes how a small change to an original model score changes the combined result; the checker measures it for you. " : "Compare the selected loss recipe and Combined score. Use the same fixed settings for training and held-out examples. "
            return .init(title:title,nextStep:lossControls(task)+" "+detail+(task.usesLogSoftmax ? "Keep the visible LogSoftmax before Objective check; do not normalize twice. ":"")+"Choose Check construction again and compare the measured tables. No code or handwritten answers are required.")
        }
        if let task=InitializationLearningTask(rawValue:operation) {
            guard ["startingSettings","predictions","firstAdjustment","repeatIsolation","savedReuse"].contains(criterionID) else {return nil}
            return .init(title:InitializationLessonReference.label(criterionID),nextStep:task.controlInstructions+" Check the selected learned-array owners and both Train connections. Apply the controls, then choose Check construction again. Tables show differences from an independent reference; zero means match. Saved reuse checks learned weights, rather than claiming full training continuation.")
        }
        if let task=CheckpointLearningTask(rawValue:operation) {
            guard task.measuredShapes[criterionID] != nil else {return nil}
            return .init(title:CheckpointLessonReference.label(criterionID),nextStep:task.controlInstructions+" Apply the controls, then choose Check construction again. Fine-tuning begins a new job from saved weights; exact resume preserves the previous job’s training memory and completes its remaining passes.")
        }
        if let task=AttentionLearningTask(rawValue:operation) {
            let title:String
            if criterionID == "blockedConnections" {title="Requested blocked connections"}
            else if criterionID == "blockedSourcePositions" {title="Available source positions"}
            else if criterionID == "result" {title="Mixture at each requested position"}
            else if criterionID == "weights",task == .multihead {title="Share assigned to each allowed source"}
            else if task.sourceRoles.map({$0+"Gradient"}).contains(criterionID) {title="Influence on supplied "+String(criterionID.dropLast(8))}
            else if task.parameterShapes.keys.map({"parameter."+$0+"Gradient"}).contains(criterionID) {title="Influence on learned table · "+String(criterionID.dropFirst(10).dropLast(8))}
            else {return nil}
            return .init(title:title,nextStep:"Check source roles and the Attention mask connections. Block query position 0 from key position 1; preserve the other allowed connections. Check the requested comparison and normalization controls, then choose Check construction again.")
        }
        if let task=RecurrentLearningTask(rawValue:operation) {
            let labels=["result":task.isCell ? "Current hidden result":"Result at every time position","hidden":"Final hidden state","cell":"Final cell memory","inputGradient":"Influence on supplied measurements","hiddenGradient":"Influence on starting hidden state","cellGradient":"Influence on starting cell memory"]
            let title:String?
            if task.outputRoles.contains(criterionID) || task.sourceRoles.map({$0+"Gradient"}).contains(criterionID) {title=labels[criterionID]}
            else if task.parameterShapes.keys.map({"parameter."+$0+"Gradient"}).contains(criterionID) {
                title=criterionID.contains("weight_ih") ? "Influence on the measurement table":criterionID.contains("weight_hh") ? "Influence on the memory table":criterionID.contains("bias_ih") ? "Influence on the measurement offset":"Influence on the memory offset"
            } else {title=nil}
            guard let title else {return nil}
            return .init(title:title,nextStep:task.controlInstructions+" Check the input, hidden and any cell source roles; keep time positions in their original order. Connect every named result to its matching Output role, then choose Check construction again. Learned tables and output influence belong to the checker; do not enter weights or answers.")
        }
        if let task=SchedulerLearningTask(rawValue:operation) {
            let titles=["usedRates":"Adjustment size actually used","nextRates":"Adjustment size prepared for the next group","scheduleTicks":"When the schedule changed","updatedParameters":"Learned settings after each group","momentumBuffer":"Changes remembered between groups","resumedUsedRates":"Adjustment sizes after reopening","resumedNextRates":"Next adjustment sizes after reopening","resumedTicks":"Schedule position after reopening","resumedParameters":"Learned settings after reopening","resumedMomentum":"Remembered changes after reopening"]
            guard let title=titles[criterionID] else {return nil}
            return .init(title:title,nextStep:task.controlInstructions+" Check both recipe connections and keep Supplied SGD unchanged. Epoch schedules change after a complete pass; OneCycle changes after each group. Choose Check construction again. The checker owns starting settings, validation results and saved memory.")
        }
        if let task=OptimizerLearningTask(rawValue:operation) {
            let common=["updatedParameters":"Learned settings after each step","statePresence":"Remembered quantities created when needed","updateCounts":"Gradient applications for each parameter"]
            let specific:[String:String]
            switch task {
            case .sgd:specific=["momentumBuffer":"Remembered momentum for each setting"]
            case .adam,.adamW:specific=["stepCounts":"Recorded optimizer step counts","firstAverage":"Running average of signed gradients","squaredAverage":"Running average of squared gradients","largestSquaredAverage":"Largest remembered squared averages"]
            case .rmsprop:specific=["stepCounts":"Recorded optimizer step counts","squaredAverage":"Recent squared-gradient memory","gradientAverage":"Recent signed-gradient average","momentumBuffer":"Remembered RMSprop adjustments"]
            case .adagrad:specific=["stepCounts":"Recorded optimizer step counts","accumulatedSquares":"Accumulated squared gradients"]
            }
            guard let title=common[criterionID] ?? specific[criterionID] else {return nil}
            let detail=criterionID == "statePresence" || criterionID == "stepCounts" || criterionID == "updateCounts" ? "A parameter without a gradient keeps its settings and existing state; only actual gradient applications advance its count. Adagrad creates its accumulated squares at initialization; other remembered quantities appear when needed. SGD has no stored step counter. " : task == .adamW ? "Weight decay shrinks settings separately; it must not enter the remembered gradient averages. " : "Compare the supplied gradients and remembered quantities row by row, keeping Weights and Bias in their declared order. "
            return .init(title:title,nextStep:task.controlInstructions+" "+detail+"Choose Check construction again; updates are measured automatically.")
        }
        if let task=StatefulNormalizationLearningTask(rawValue:operation) {
            let titles=["output":"Centered and scaled output values","evaluationOutput":"Values when evaluating examples","inputGradient":"Influence on original input values","scaleGradient":"Influence on each channel's scale","biasGradient":"Influence on each channel's offset","runningMean":"Remembered center of each channel","runningVariance":"Remembered spread of each channel","batchesTracked":"Recorded calibration-group count","postEvalRunningMean":"Remembered centers stay unchanged","postEvalRunningVariance":"Remembered spreads stay unchanged","postEvalBatchesTracked":"Calibration count stays unchanged"]
            guard let title=titles[criterionID],task.affine || !["scaleGradient","biasGradient"].contains(criterionID),task.remembered || !["runningMean","runningVariance","batchesTracked","postEvalRunningMean","postEvalRunningVariance","postEvalBatchesTracked"].contains(criterionID) else {return nil}
            let action:String
            switch criterionID {
            case "scaleGradient","biasGradient":
                action="Influence on learned adjustments is measured automatically. "+normalizationControls(task)+" Each channel reuses its scale and offset across its positions; those uses add influence."
            case "inputGradient":
                action="Input influence follows the selected comparison group. "+normalizationGrouping(task)+" "+(task.remembered ? "Remembered summaries remain fixed while the new input changes.":"Current summaries change when values in their group change.")+" Check Safety value and learned-adjustment settings."
            case "runningMean","runningVariance":
                action="Enable Track running statistics and set Running-stat update weight to 0.25. "+(task.componentID.hasPrefix("batch") ? "Use Moving for Averaging method. ":"")+"The checker builds each channel's remembered center and spread from two supplied calibration groups; do not enter summary answers."
            case "batchesTracked":
                action="The checker supplies two calibration groups. BatchNorm records processed training groups; InstanceNorm does not increment this counter. Check the requested block and Track running statistics."
            case "postEvalRunningMean","postEvalRunningVariance","postEvalBatchesTracked":
                action="Evaluation uses remembered summaries without updating them. The recorded center, spread and calibration count must stay unchanged. Keep Track running statistics enabled; the checker controls the calibration and evaluation sequence."
            case "evaluationOutput":
                action=normalizationControls(task)+" "+(task.remembered ? "Evaluation uses the remembered center and spread from calibration, rather than replacing them with the new input's summaries.":"Tracking is off, so evaluation still uses current-input summaries. "+normalizationGrouping(task))
            default:
                action=normalizationGrouping(task)+" "+normalizationControls(task)+" Compare the channel grouping, Safety value and scale/offset choices with the measured output."
            }
            return .init(title:title,nextStep:action+" Choose Check construction again.")
        }
        if let task=SpatialPatchLearningTask(rawValue:operation) {
            guard ["output","inputGradient"].contains(criterionID) else {return nil}
            return .init(title:criterionID == "output" ? (task.isUnfold ? "Values in patch columns":"Added values in the output grid"):"Influence on original input positions",nextStep:task.controlInstructions+" Compare the measured tables. Repeated uses add influence; border-only and uncovered positions contribute zero. Choose Check construction again.")
        }
        if let task=TensorRoutingLearningTask(rawValue:operation) {
            let titles=["output":"Values in the requested part or arrangement","inputGradient":"Influence on original number positions","leftGradient":"Influence on the first table","rightGradient":"Influence on the second table"]
            guard let title=titles[criterionID],(task == .einsumCombine ? criterionID != "inputGradient" : !["leftGradient","rightGradient"].contains(criterionID)) else {return nil}
            return .init(title:title,nextStep:task.controlInstructions+" Choose Check construction again; influence is measured automatically.")
        }
        if let task=RemainingActivationLearningTask(rawValue:operation) {
            let titles=["output":"Values from the selected block","inputGradient":"Influence on each input value","slopeGradient":"Influence on learned negative multipliers","trainingOutput":"Values when training is enabled","trainingInputGradient":"Input influence when training is enabled"]
            guard let title=titles[criterionID],criterionID != "slopeGradient" || task.isPReLU,!criterionID.hasPrefix("training") || task == .dropoutEvaluation else {return nil}
            return .init(title:title,nextStep:task.controlInstructions+" Choose Check construction again; influence is measured automatically.")
        }
        if let task=TensorCompositionLearningTask(rawValue:operation) {
            let titles=["output":"Values and source order","leftGradient":"Influence on the first source","rightGradient":"Influence on the second source"]
            guard let title=titles[criterionID],criterionID != "rightGradient" || task.isBinary else {return nil}
            return .init(title:title,nextStep:task.controlInstructions+" Choose Check construction again.")
        }
        if let task=AdaptivePoolingLearningTask(rawValue:operation) {
            let titles=["output":"Values from each region","inputGradient":"Influence on contributing inputs","indices":"Winning positions in the input grid"]
            guard let title=titles[criterionID],criterionID != "indices" || !task.isAverage else {return nil}
            return .init(title:title,nextStep:"Select the requested pooling block. "+adaptivePoolingControls(task)+" Choose Check construction again; the checker measures influence automatically.")
        }
        if let task=ConfigurableActivationLearningTask(rawValue:operation) {
            guard ["output","inputGradient"].contains(criterionID) else {return nil}
            return .init(title:criterionID == "output" ? "Signed transformed values":"Influence on each input value",nextStep:"Select the requested activation. \(task == .silu ? "SiLU has no adjustable setting.":task == .leakyReLU ? "Set Negative slope to 0.2.":"Set Alpha to 2.") Keep each array position and observation separate; Check construction measures influence automatically, including the zero convention.")
        }
        if let task=PoolingLearningTask(rawValue:operation) {
            let titles=["output":"Window output values","inputGradient":"Influence on contributing inputs","indices":"Winning input positions"]
            guard let title=titles[criterionID] else {return nil}
            return .init(title:title,nextStep:"Select the pooling block. Set Window length or Window dimensions to 3, Stride 2, Padding 1 and enable Include partial final window. \(task.isAverage ? "Enable Include padding in average divisor; the final partial window divides by 2.":"Set Dilation 1 and enable Expose winning indices; padding cannot win and equal values choose the first position.") Choose Check construction again.")
        }
        if let task=ConvolutionLearningTask(rawValue:operation),task.isTransposed {
            let titles=["output":"Added scatter-window contributions", "inputGradient":"Influence on scattered input values", "kernelGradient":"Influence on shared scatter weights", "biasGradient":"Influence on every output offset"]
            guard let title=titles[criterionID] else {return nil}
            return .init(title:title,nextStep:"Check two channels, Kernel size 3, Stride 2, Dilation 1, Groups 1, Padding 0, Output padding 1 and enabled bias. Overlapping windows add; extra extent does not add input data. Measure again with Check construction.")
        }
        if ConvolutionLearningTask(rawValue:operation) != nil {
            let titles=["output":"Shared-window output", "inputGradient":"Influence on overlapping input positions", "kernelGradient":"Influence on shared kernel weights", "biasGradient":"Influence on shared output offsets"]
            guard let title=titles[criterionID] else {return nil}
            return .init(title:title,nextStep:"Check both channel counts 2, Kernel size 2 in every spatial direction, Stride/Dilation 1, Groups 1, zero padding and enabled bias. Keep the supplied input connected through the requested convolution to Model Output; Check construction measures influence automatically.")
        }
        if operation == LazyLinearAssessment.operation {
            let titles = ["output-shape":"Two groups with four results each", "inferred-width":"Three input features inferred", "weight-shape":"Four sets of three weights", "parameter-count":"Created parameter inventory", "repeat-state":"Repeated calculation keeps the same state", "reload-state":"Saved weights reload unchanged", "rng-preserved":"No fresh random draws on repeat or reload"]
            guard let title=titles[criterionID] else {return nil}
            return .init(title:title,nextStep:"Select LazyLinear and choose four Output features. Keep the two groups of three input features; avoid flattening them together before this layer. Check construction uses the same created weights for state comparisons.")
        }
        let title: String
        let action: String
        switch operation {
        case "reshape", "transpose", "flatten":
            switch criterionID {
            case "element-order": title = "Number order and output shape"
            case "gradient-routing": title = "Influence returns to the matching input"
            default: return nil
            }
            switch operation {
            case "reshape": action = "Select Reshape and set Sample dimensions to [3,2]. Keep the six numbers in reading order within each observation."
            case "transpose": action = "Select Transpose and set First axis to 1 and Second axis to 2. Rows become columns; axis 0 keeps observations separate."
            default: action = "Select Flatten and set First axis to 1 and Last axis to -1. Keep the numbers in reading order within each observation."
            }
        case "softmax-last-axis", "softmax-first-sample-axis":
            switch criterionID {
            case "axis-probabilities": title = "Shares within the intended score group"
            case "coupled-gradient": title = "Influence within each score group"
            default: return nil
            }
            let setting = operation == "softmax-last-axis" ? "Normalization axis to -1" : "Normalization axis to 1"
            let group = operation == "softmax-last-axis" ? "three values in each row" : "two values in each column"
            action = "Select Softmax and set \(setting). The \(group) should compete together and their output shares should add up to 1. Keep different observations separate. Changing one score affects the other shares in its group."
        case "layer-norm-rows", "layer-norm-sample", "layer-norm-affine":
            switch criterionID {
            case "normalized-values": title = "Centered and scaled output groups"
            case "coupled-input-gradient": title = "Influence within each normalization group"
            case "shared-scale-gradient" where operation == "layer-norm-affine": title = "Influence on shared scale settings"
            case "shared-bias-gradient" where operation == "layer-norm-affine": title = "Influence on shared offset settings"
            default: return nil
            }
            let shape = operation == "layer-norm-sample" ? "[2,3]" : "[3]"
            let parameters = operation == "layer-norm-affine"
                ? "Enable learned scale and bias. In the parameter editor, enter scale [0.5,-1,2] and bias [1,-0.5,0.25]; reuse those settings for every row. Enable **Inspect this component’s parameter gradients** so the checker can measure the scale and offset settings."
                : "Disable learned scale and bias."
            action = "Select LayerNorm and set Trailing sample dimensions to \(shape) and Epsilon to 0.125. \(parameters) Keep observations separate. Values in the same group share the group's average and spread, so a change to one can affect the others."
        case "embedding-gather", "embedding-padding-frequency", "embedding-renormalize":
            switch criterionID {
            case "output": title = "Selected lookup-table rows"
            case "tableAfterForward": title = "Lookup table after row selection"
            case "tableGradient": title = "Influence on shared lookup rows"
            default: return nil
            }
            let policy: String
            switch operation {
            case "embedding-padding-frequency":
                policy = "Set Padding row to 0 and enable Scale gradients by frequency. Leave Maximum norm unset. Padding keeps its supplied output values; repeated non-padding selections share their influence by occurrence count."
            case "embedding-renormalize":
                policy = "Leave Padding row unset, disable Scale gradients by frequency, and set Maximum norm to 1 with Norm order 2. Only selected rows are shortened when they exceed the length limit."
            default:
                policy = "Leave Padding row and Maximum norm unset, disable Scale gradients by frequency, and keep Norm order 2. Repeated selectors reuse the same row; their influence on that shared row adds together."
            }
            action = "Select Embedding and set Table rows to 4 and Vector dimensions to 3. \(policy) Keep Sparse gradients and Freeze learned weights disabled. Inspect the selected rows and table-after-lookup values."
        case "matrix-contraction", "matrix-broadcast-gradients":
            switch criterionID {
            case "output": title = "Row–column products"
            case "leftGradient": title = "Influence on the left table"
            case "rightGradient": title = "Influence on the right table"
            default: return nil
            }
            let sharing = operation == "matrix-broadcast-gradients"
                ? "A size-1 grouping direction reuses its values; those uses contribute to the same original value."
                : "Keep each observation's pair of tables together."
            action = "Check Matrix multiply connections: left table to a, right table to b, then result to Model Output. The left table's final size must match the right table's next-to-last size. Keep observations separate. \(sharing) Compare the measured output shapes with Required shapes & rules."
        case "bilinear-with-bias", "bilinear-without-bias":
            switch criterionID {
            case "output": title = "Combined feature-pair output"
            case "leftGradient": title = "Influence on the first input"
            case "rightGradient": title = "Influence on the second input"
            case "weightGradient": title = "Influence on shared interaction weights"
            case "biasGradient" where operation == "bilinear-with-bias": title = "Influence on shared offsets"
            default: return nil
            }
            let bias = operation == "bilinear-with-bias" ? "Enable bias and connect the supplied offset." : "Disable bias."
            action = "Check Bilinear's first input, second input and supplied interaction-weight connections. \(bias) Observation and grouping sizes must match exactly. Reused interaction weights receive influence from each position that uses them. Compare the measured shapes with Required shapes & rules."
        default: return nil
        }
        return CriterionFeedback(title: title, nextStep: action + " Choose Check construction again. KwipML checks the output and small-change influence automatically; adjust the block settings and connections in the canvas.")
    }
    static func make(operation: String) -> Self {
        if let task=DataLoadingLearningTask(rawValue:operation) {
            var terms:[Term]=[.init(name:"Observation",meaning:"One input row together with its desired outcome. Its ID follows both when rows move.")]
            switch task {
            case .indexed,.tensor,.subset:break
            case .iterable:terms.append(.init(name:"Finite stream",meaning:"A bounded sequence of observations that restarts from the same snapshot for each pass."))
            case .batches:terms.append(.init(name:"Batch",meaning:"A group of observations processed together. The final group may be shorter."))
            case .shuffle:terms.append(.init(name:"Seed",meaning:"A chosen whole number that makes the same mixing order repeatable."))
            case .collation:terms.append(.init(name:"Padding",meaning:"Extra zeros that let shorter sequences share a group. True lengths keep the original positions distinguishable."))
            case .continuation:terms += [.init(name:"Checkpoint",meaning:"A saved training position with learned settings and memory needed to continue the same run."),.init(name:"Prefetch",meaning:"How many upcoming groups each separate reader prepares ahead."),.init(name:"Bag",meaning:"One observation’s group of items. Different bags can have different lengths; Whole bags keeps each group and its desired outcome together, including empty groups.")]
            }
            return .init(goal:task.title,steps:["Place Dataset and DataLoader beside the supplied Train.",task.controlInstructions,"Connect each out to the matching named Train input, apply changes, then Check construction."],terms:terms)
        }
        if let task=ProgramExportLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:[task.controlInstructions,"Apply the artifact controls, then Check construction.","Compare actual reloaded results and input-size checks."],terms:[.init(name:"Executable program",meaning:"A saved calculation together with the learned settings it uses."),.init(name:"Batch",meaning:"The number of observations processed together. Export rules determine which group sizes are accepted."),.init(name:"Strict capture",meaning:"Additional checks while tracing the calculation. Turning it off does not guarantee unsupported work can be exported."),.init(name:"Deferred runtime checks",meaning:"Prefer checking eligible size rules inside the saved calculation when it runs. Off keeps the normal checking policy, including any later input checks.")])
        }
        if let task=EmbeddingBagLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:["Place Bag Input, EmbeddingBag and Model Output. Connect each enabled source role to its matching input, then connect out to Model Output.",bagControls(task)+" Apply the controls.",bagMeaning(task)+" Choose Check construction; measured influence is automatic and private assessment hides its tables."],terms:[.init(name:"Bag",meaning:"One observation's group of categories; different groups stay separate."),.init(name:"Category ID",meaning:"A whole number naming a row in the shared table."),.init(name:"Vector",meaning:"An ordered list of numbers, such as one selected table row."),.init(name:"Offset",meaning:"The position where a group starts in a packed list. Equal offsets represent an empty group."),.init(name:"Gradient",meaning:"How a small change affects a result. Whole-number IDs and offsets have no gradient available."),.init(name:"Item weight",meaning:"A real number controlling one selected item's contribution with Sum."),.init(name:"Source length",meaning:"How many cells in one source row belong to the group; later cells are unused."),.init(name:"Padding",meaning:"A reserved table row excluded from the combination and from table learning changes.")])
        }
        if let task=ModuleCompositionLearningTask(rawValue:operation) {
            let build=task == .buildReuse ? "Open the supplied Block and its private copies. Connect A to B and both required branches; preserve the fixed sign reversal and subtraction." : "Open each supplied Block. Connect A to B, then connect all three Block results in Model."
            let configure=task == .buildReuse ? "Keep the private copies' learned settings separate and leave the supplied behavior check unchanged. Apply each Block draft." : "Choose the matching container kind. Inspect stored child order and names separately from connected work. Apply each Block draft."
            return .init(goal:task.title,steps:[build,configure,"Choose Check construction. The checker measures results, influence and saved reuse automatically."],terms:[.init(name:"Block",meaning:"A reusable group of connected work."),.init(name:"Private copy",meaning:"A copy with its own learned settings."),.init(name:"Stored order",meaning:"The order in which a Block keeps its children."),.init(name:"Connected order",meaning:"Which result feeds the next calculation."),.init(name:"Gradient",meaning:"How a small change affects a result; the checker measures this influence automatically."),.init(name:"Reference difference",meaning:"A measured value minus an independently calculated value. Zero means agreement for that measurement.")])
        }
        if let task=GradientLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:["Connect Gradient policy, Trainability and Supplied SGD to their matching Train inputs.",task.controlInstructions+" Apply the structured controls.","Choose Check construction. Compare real measured differences; zero means match. Private assessment hides its tables."],terms:[.init(name:"Learned setting",meaning:"A weight or bias number training can adjust."),.init(name:"Gradient",meaning:"How a small change affects a result; training uses this direction to choose adjustments."),.init(name:"SGD optimizer",meaning:"The supplied instructions for adjusting learned settings from gradients."),.init(name:"Accumulation window",meaning:"Several small observation groups combined before one adjustment."),.init(name:"Frozen",meaning:"A learned array cannot be adjusted. Influence through it can still exist."),.init(name:"Evaluation mode",meaning:"Prediction behavior for layers such as Dropout; it does not freeze parameter flags."),.init(name:"Unavailable",meaning:"The requested influence could not be recorded; this does not mean it equals zero.")])
        }
        if let task=InitializationLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:["Connect Starting settings to Train initialization and Supplied SGD to Train optimizer.",task.controlInstructions+" Apply the controls.","Choose Check construction. Compare starting values, predictions and first adjustments; zero difference means match. Repeat and saved-reuse checks protect the same learned arrays."],terms:[.init(name:"Learned settings",meaning:"Numbers training adjusts. Weights combine incoming values; bias shifts a result."),.init(name:"SGD optimizer",meaning:"The supplied training instructions that adjust learned settings using the measured direction of change."),.init(name:"Gradient",meaning:"Sensitivity: how a small change to a learned setting changes the error score. Training uses it to choose an adjustment."),.init(name:"Seed",meaning:"A repeatable starting choice for random numbers."),.init(name:"Normal",meaning:"Starting values around a chosen center, with a chosen spread."),.init(name:"Fan",meaning:"The count of incoming or outgoing connections in a stored weight array."),.init(name:"Gain",meaning:"A multiplier for the starting spread."),.init(name:"Slope",meaning:"The share of a negative value that Leaky ReLU keeps."),.init(name:"Reference difference",meaning:"Measured value minus an independently computed expectation. Zero means match.")])
        }
        if let task=CheckpointLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:[task.controlInstructions,"Apply the controls. The checker supplies a real saved lesson run; you do not connect graph blocks or enter file paths.","Choose Check construction and compare restored settings, training memory, completed epochs and the unchanged saved parent."],terms:[.init(name:"Checkpoint",meaning:"Saved learned settings and the training memory needed to continue a job."),.init(name:"Weights",meaning:"Settings the model learned from its training examples."),.init(name:"Training memory",meaning:"Remembered information used to choose later adjustments."),.init(name:"Epoch",meaning:"One complete pass through the training rows."),.init(name:"Batch",meaning:"A group of rows handled together."),.init(name:"Seed",meaning:"A choice that makes starting random decisions repeatable."),.init(name:"Fine-tuning",meaning:"A new job that reuses saved learned settings and starts new training memory."),.init(name:"Exact resume",meaning:"Continuing the same compatible CPU job, including its saved training memory.")])
        }
        if let task=LossLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:[lossBuild(task),lossControls(task),"Choose Check construction; compare individual scores, the combined score, desired roles and measured input influence."],terms:[.init(name:"Loss",meaning:"A score comparing a model result with the desired outcome; a smaller score means less error under that comparison."),.init(name:"Target",meaning:"The desired outcome used to judge a result."),.init(name:"Observation",meaning:"One independent example; its values stay separate from other examples."),.init(name:"Combine scores",meaning:"Mean averages scores, Sum adds them, and Keep individual scores preserves the separate cells."),.init(name:"Gradient",meaning:"How a small change to an original input affects the combined score; shown as Input influence."),.init(name:"Class",meaning:"One possible answer when choosing among alternatives."),.init(name:"Probability",meaning:"A share from zero to one expressing how likely an answer is."),.init(name:"Log-probability",meaning:"A probability recorded on a compressed number scale: probability 1 becomes 0 and smaller probabilities become negative numbers. The result is not the probability itself. LogSoftmax performs the conversion; no logarithm calculation is required."),.init(name:"Packed targets",meaning:"Used sequence labels placed together in one list. Target lengths say how many labels belong to each observation, keeping their boundaries clear."),.init(name:"Alignment",meaning:"Matching desired labels to the time positions that could produce them."),.init(name:"Anchor",meaning:"The example compared with a matching and a nonmatching example in a distance task.")])
        }
        if let task=AttentionLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:[attentionBuild(task),task.controlInstructions,"Check construction; compare position results, connection shares when shown, and source or learned-table influence."],terms:[.init(name:"Query",meaning:"A position asking which information is useful."),.init(name:"Key",meaning:"A position compared with a query to decide how much it matters."),.init(name:"Value",meaning:"Information mixed into the result after the comparisons."),.init(name:"Mask",meaning:"Blocked connections: 1 blocks and 0 allows. Position numbers start at 0."),.init(name:"Head",meaning:"One comparison group that can focus on a different relationship."),.init(name:"Sequence",meaning:"Values arranged in position order."),.init(name:"Normalization",meaning:"Balancing a row's values by its center and spread."),.init(name:"Gradient",meaning:"How a small change affects the result; the checker measures this influence.")])
        }
        if let task=RecurrentLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:["Connect input and starting hidden"+(task.isLSTM ? " and cell":"")+" state to their matching ports; add every named Output.",task.controlInstructions,"Check construction and compare time-ordered results, final states and measured influence."],terms:[.init(name:"Sequence",meaning:"Measurements arranged in time order."),.init(name:"State",meaning:"Information carried from an earlier step; each observation has its own."),.init(name:"Hidden state",meaning:"The summary used to create the next result."),.init(name:"Cell state",meaning:"LSTM's separate memory, controlled by what is kept, added and revealed."),.init(name:"Gate",meaning:"A number between zero and one choosing how much information passes."),.init(name:"Learned table",meaning:"Shared numbers that combine measurements and earlier memory at each step."),.init(name:"Gradient",meaning:"How a small change affects the results; the checker measures this influence."),.init(name:"Observation",meaning:"One independent example; its measurements and state stay separate.")])
        }
        if let task=SchedulerLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:["Connect Supplied SGD to Train optimizer and \(task.componentTitle) to Train scheduler.",task.controlInstructions,"Check construction, inspect adjustment sizes and learned settings, then compare the continuation after reopening."],terms:[.init(name:"Learning rate",meaning:"The size of each adjustment to learned settings."),.init(name:"Batch",meaning:"A group of observations processed together. This exercise uses two groups per pass."),.init(name:"Epoch",meaning:"One complete pass through the observations."),.init(name:"Scheduler",meaning:"Instructions that change adjustment size at a chosen time."),.init(name:"Optimizer",meaning:"Instructions that use measured changes to adjust learned settings."),.init(name:"Momentum",meaning:"Carrying part of an earlier change into the next adjustment."),.init(name:"Validation",meaning:"Checking a model on examples it did not use to learn."),.init(name:"Checkpoint",meaning:"Saved learned settings and remembered training state used to continue later.")])
        }
        if let task=OptimizerLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:["Add \(task.componentTitle) and Train. Connect optimizer out to Train optimizer; this is a training-control recipe, not model data.",task.controlInstructions,"Choose Check construction and compare starting settings, supplied gradients, new settings and remembered quantities over four steps."],terms:[.init(name:"Weight",meaning:"A learned setting that affects a model's result."),.init(name:"Bias",meaning:"A learned offset added to a model's calculation; it is another parameter the optimizer can adjust."),.init(name:"Parameter",meaning:"A learned value, or a group of learned values such as the two weights in this exercise."),.init(name:"Gradient",meaning:"How changing a learned setting affects the model's error. The checker supplies it; a missing gradient means that parameter is not updated."),.init(name:"Training step",meaning:"One adjustment of learned settings using the supplied gradients."),.init(name:"Optimizer recipe",meaning:"Settings that tell Train how to adjust its learned parameters."),.init(name:"Learning rate",meaning:"The size of an adjustment; zero leaves learned settings unchanged."),.init(name:"Remembered state",meaning:"Quantities an optimizer stores from earlier steps to guide later updates."),.init(name:"Weight decay",meaning:"Discouraging large settings; AdamW shrinks them separately from its gradient averages."),.init(name:"Square",meaning:"A number multiplied by itself; squared gradients help measure the size of changes."),.init(name:"Momentum",meaning:"Carrying part of an earlier change into the next step."),.init(name:"Running average",meaning:"A remembered blend of earlier and recent measurements."),.init(name:"Stability amount",meaning:"A small positive cushion in a divisor.")])
        }
        if let task=StatefulNormalizationLearningTask(rawValue:operation) {
            var terms:[Term]=[.init(name:"Observation",meaning:"One example, such as one record or grid."),.init(name:"Batch",meaning:"The current group of examples processed together."),.init(name:"Channel",meaning:"One parallel measurement field or grid; each channel has its own learned adjustment."),.init(name:"Normalization",meaning:"Subtract a group's center and divide by its spread to bring its numbers onto a consistent scale."),.init(name:"Spatial position",meaning:"A measurement's place along a line or in a grid."),.init(name:"Safety value",meaning:"A small positive cushion in the spread calculation that prevents division by zero."),.init(name:"Scale and shift",meaning:"Multiply a channel's normalized values by its scale, then add its shift, also called an offset or bias."),.init(name:"Gradient",meaning:"How a small change affects a result; KwipML measures this influence automatically."),.init(name:"Training",meaning:"The mode used while preparing a model; tracked summaries can update here. These lessons do not train weights with an optimizer."),.init(name:"Evaluation",meaning:"Using a model on examples; remembered summaries stay fixed in this mode.")]
            if task.remembered {terms += [.init(name:"Running statistics",meaning:"Stored summaries of each channel's center and spread from earlier examples."),.init(name:"Calibration",meaning:"Processing supplied example groups to build remembered summaries before evaluating new examples."),.init(name:"Update weight",meaning:"How much of the new summary is included in the remembered summary at each update.")]}
            if task.componentID == "group_norm" {terms.append(.init(name:"Channel group",meaning:"An equal team of channels whose values share one center and spread within an observation."))}
            return .init(goal:task.title,steps:["Connect the supplied Input → \(task.componentTitle) → Model Output. Select \(task.componentTitle).",normalizationControls(task),normalizationCheck(task)],terms:terms)
        }
        if let task=SpatialPatchLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:["Connect the supplied Input, \(task.isUnfold ? "Unfold":"Fold") and Model Output. Select the patch block.",task.controlInstructions,"Choose Check construction; compare the measured output and input-influence tables. Fold adds overlapping parts and leaves uncovered positions zero."],terms:[.init(name:"Array",meaning:"Numbers arranged in directions such as rows and columns."),.init(name:"Observation",meaning:"One independent example; its numbers remain separate from other examples."),.init(name:"Channel",meaning:"A separate grid of measurements processed alongside other grids."),.init(name:"Patch",meaning:"A small rectangle selected from a grid."),.init(name:"Stride",meaning:"The spacing between patch starts, set by Spacing between patches."),.init(name:"Dilation",meaning:"The spacing between cells inside a patch, set by Spacing within a patch."),.init(name:"Padding",meaning:"Declared zero cells around a grid border."),.init(name:"Overlap",meaning:"Several patch entries landing on the same position; Fold adds their numbers."),.init(name:"Gradient",meaning:"How a small input change affects the result, measured automatically. Repeated uses add influence.")])
        }
        if let task=TensorRoutingLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:[routingBuild(task),task.controlInstructions,"Choose Check construction; compare the measured output and number-source influence tables."],terms:[.init(name:"Array",meaning:"Numbers arranged in directions such as rows and columns."),.init(name:"Port",meaning:"A named connection point on a block. Each Split or Chunk part has its own port."),.init(name:"Observation",meaning:"One independent example; direction 0 keeps examples separate."),.init(name:"Direction",meaning:"One way of moving through an array, such as across columns. Sample directions start at 1."),.init(name:"Selector",meaning:"A whole-number position beginning at 0; Gather uses it to read a source value."),.init(name:"Arrangement preset",meaning:"A named way to combine tables or read their diagonal, without typing notation."),.init(name:"Diagonal",meaning:"Table positions with the same row and column number, from top-left to bottom-right."),.init(name:"Gradient",meaning:"How a small source change affects the result, measured automatically. Repeated uses add influence.")])
        }
        if let task=RemainingActivationLearningTask(rawValue:operation) {
            var terms:[Term]=[.init(name:"Array",meaning:"Numbers arranged in directions, such as rows and columns."),.init(name:"Channel",meaning:"One measurement field; here each row contains three measurements."),.init(name:"Observation",meaning:"One independent example; its measurements remain separate from other examples."),.init(name:"Gradient",meaning:"How a small change affects a result; the checker measures this automatically.")]
            if task.isPReLU {terms += [.init(name:"Learned slope",meaning:"An adjustable multiplier for negative numbers. Positive numbers stay unchanged."),.init(name:"Shared parameter",meaning:"One learned number reused at several positions; their influences add together.")]}
            else if task.isDropout {terms += [.init(name:"Training",meaning:"The stage where a model adjusts its learned numbers."),.init(name:"Evaluation",meaning:"Using a model to assess examples without changing its learned numbers."),.init(name:"Drop probability",meaning:"The chance that a measurement is temporarily replaced with zero during training."),.init(name:"Seed",meaning:"A starting point for repeatable random draws in the same runtime and call order.")]}
            else {terms += [.init(name:"Probability",meaning:"A share between zero and one; the compared shares sum to one."),.init(name:"Logarithm",meaning:"A compressed scale for positive values. LogSoftmax stores probabilities on this scale, so its outputs are not probabilities themselves.")]}
            return .init(goal:task.title,steps:["Connect Input, the requested block and Model Output. Select the block.",task.controlInstructions,"Choose Check construction; compare the measured values and small-change influence tables."],terms:terms)
        }
        if let task=TensorCompositionLearningTask(rawValue:operation) {
            return .init(goal:task.title,steps:[task.isBinary ? "Connect the first Input to a and the second Input to b. Connect the requested block to Model Output.":"Add Transpose and choose Method → Reorder axes. Connect Input → Permute → Model Output.",task.controlInstructions,"Choose Check construction; compare output order and each source’s influence in the measured tables."],terms:[.init(name:"Array",meaning:"Numbers arranged in directions such as rows and columns."),.init(name:"Axis",meaning:"One direction in an array; axis 0 holds observations."),.init(name:"Observation",meaning:"One independent example; its values never mix with another example."),.init(name:"Broadcasting",meaning:"Reuse a size-1 sample direction to match another source, within each observation."),.init(name:"Gradient",meaning:"How a small source change affects the result; KwipML measures this automatically.")])
        }
        if let task=AdaptivePoolingLearningTask(rawValue:operation) {
            return .init(goal:task.isAverage ? "Choose regions to average and understand how shared inputs contribute.":"Choose regions to select maxima and locate their winning inputs.",steps:["Connect Input, the requested pooling block and Model Output. Select the pooling block.",adaptivePoolingControls(task),"Choose Check construction; compare the output, input influence and winning-position tables when available."],terms:[.init(name:"Channel",meaning:"A separate field of measurements processed independently."),.init(name:"Observation",meaning:"One example; the block keeps different examples separate."),.init(name:"Adaptive",meaning:"The requested output size determines which input regions are used."),.init(name:"Bin",meaning:"A group of input positions used for one output; neighboring groups can share inputs."),.init(name:"Window",meaning:"A rectangle of nearby cells combined into one result."),.init(name:"Stride",meaning:"How far a window moves down and across between results."),.init(name:"Padding",meaning:"Declared extra zero cells around the input boundary."),.init(name:"Divisor",meaning:"The number used to divide the sum when averaging."),.init(name:"Index",meaning:"A zero-based position counted row by row within one input channel."),.init(name:"Gradient",meaning:"How a small change affects the result, measured automatically; shared inputs receive added contributions.")])
        }
        if let task=ConfigurableActivationLearningTask(rawValue:operation) {
            return .init(goal:"Transform signed numbers independently and understand their small-change influence.",steps:["Connect Input, the requested activation and Model Output. Select the activation block.",task == .silu ? "SiLU has no adjustable setting; it smoothly scales each number.":task == .leakyReLU ? "Set Negative slope to 0.2; negative values are multiplied by that amount.":"Set Alpha to 2; very negative values approach −2.","Choose Check construction and inspect the output and input-influence tables."],terms:[.init(name:"Array",meaning:"Numbers arranged in named directions, such as rows and columns."),.init(name:"Activation",meaning:"A rule that transforms each number independently."),.init(name:"Slope",meaning:"How much a result changes when its input changes; Negative slope controls the negative branch of LeakyReLU."),.init(name:"Alpha",meaning:"ELU’s setting controlling the negative branch’s size."),.init(name:"Gradient",meaning:"How a small change affects the result, measured automatically."),.init(name:"Observation",meaning:"One example; these activations keep examples separate.")])
        }
        if let task=PoolingLearningTask(rawValue:operation) {
            return .init(goal:task.isAverage ? "Average nearby values and understand partial boundary counts.":"Select local winners and trace where they came from.",steps:["Connect Input, the requested pooling block and Model Output; Select the pooling block.","Set Window length or Window dimensions to 3, Stride 2 and Padding 1; enable Include partial final window.",task.isAverage ? "Enable Include padding in average divisor. Check construction and compare the final partial window’s divisor 2 with full-window divisor 3.":"Set Dilation 1 and enable Expose winning indices. Check construction and inspect values, winning positions and influence."],terms:[.init(name:"Window",meaning:"Nearby positions considered together."),.init(name:"Channel",meaning:"A separate field of measurements pooled independently."),.init(name:"Stride",meaning:"How far a window moves between results."),.init(name:"Padding",meaning:"Declared boundary positions: zeros for Average, unable to win for Max."),.init(name:"Partial window",meaning:"A final window with fewer positions; Include partial final window keeps it."),.init(name:"Dilation",meaning:"Spacing within a Max window; this task uses 1."),.init(name:"Index",meaning:"A zero-based winning position inside one input channel."),.init(name:"Gradient",meaning:"How a small change affects the result; repeated uses add influence automatically.")])
        }
        if let task=ConvolutionLearningTask(rawValue:operation),task.isTransposed {
            return .init(goal:"Scatter a shared window from each input position and add the overlapping contributions into a larger output.",steps:["Connect Input, ConvTranspose\(task.dimensions)d and Model Output; select the transposed convolution to configure it.","Set two channels, Kernel size 3, Stride 2, Dilation 1, Groups 1, Padding 0, Output padding 1 and enabled bias.","Predict eight output positions per direction; choose Check construction and inspect overlap, bias and measured influence."],terms:[.init(name:"Channel",meaning:"A separate measurement field at each position."),.init(name:"Spatial direction",meaning:"Positions along a line, across a table or through a volume."),.init(name:"Kernel",meaning:"Shared weights that spread each input into a small output window."),.init(name:"Shared weights",meaning:"The same window weights are reused for every input position and observation."),.init(name:"Stride",meaning:"Distance between starts of scattered windows."),.init(name:"Dilation",meaning:"Spacing within the shared kernel window."),.init(name:"Groups",meaning:"Which input channels connect to which output channels; Groups 1 connects all of them."),.init(name:"Padding",meaning:"Control the output boundary extent; this task uses 0."),.init(name:"Overlap",meaning:"Contributions from several windows add at the same output position."),.init(name:"Output padding",meaning:"Select extra output extent without appending input data; bias still applies there."),.init(name:"Gradient",meaning:"How a small change affects the result; the checker measures this automatically.")])
        }
        if let task=ConvolutionLearningTask(rawValue:operation) {
            return .init(goal:"Move one shared local window across an observation and see how overlapping positions contribute to the result.",steps:["Add Input, Conv\(task.dimensions)d and Model Output; connect them in that order. Select the convolution block to configure it.","Configure two input/output channels, Kernel size 2, Stride and Dilation 1, Groups 1, zero padding and enabled bias.","Predict three positions in each spatial direction, then choose Check construction and inspect the output and influence tables."],terms:[.init(name:"Channel",meaning:"A separate field of measurements at each spatial position."),.init(name:"Spatial direction",meaning:"Positions along a line, across a table or through a volume; this lesson has \(task.dimensions)."),.init(name:"Kernel",meaning:"A small window of weights used to combine nearby positions and channels."),.init(name:"Shared weights",meaning:"The same kernel numbers are reused at every window position and observation."),.init(name:"Stride",meaning:"How far the window moves between output positions."),.init(name:"Dilation",meaning:"Spacing between input positions selected inside a window."),.init(name:"Padding",meaning:"Extra boundary positions; this task adds none."),.init(name:"Gradient",meaning:"How a small change affects the result. The checker measures this influence automatically.")])
        }
        if operation == LazyLinearAssessment.operation {
            return .init(goal:"Let a layer infer its input width from each group of measurements, then check that its created weights remain stable.",steps:["Add Input, LazyLinear and Model Output; connect them in that order.","Select LazyLinear, choose four Output features and either Learn bias setting.","Predict two groups of four results, then choose Check construction and inspect the shape and state checks."],terms:[.init(name:"Inferred",meaning:"Determined from the connected array instead of typed into a setting."),.init(name:"Weights",meaning:"Learned numbers that control how input measurements contribute to each result."),.init(name:"Initialization",meaning:"Creating the starting weights once, before they are trained."),.init(name:"Seed",meaning:"A starting number for repeatable random initialization; initialization order also matters."),.init(name:"State",meaning:"The stored weights and optional bias values that determine the layer’s results."),.init(name:"Feature",meaning:"One measurement in the last direction of an array.")])
        }
        let goal: String
        let action: String
        switch operation {
        case "reshape":
            goal = "Arrange each observation's numbers into a new shape while keeping their order. Think of moving the same cells into a different table layout."
            action = "Add Reshape and choose Method → Set dimensions. Connect Input → Reshape → Model Output, then set the sample dimensions from the task requirements."
        case "transpose":
            goal = "Swap two directions in an array. For a table, swapping rows and columns changes where each number appears; it does not calculate new numbers."
            action = "Add Transpose and choose Method → Swap axes. Connect Input → Transpose → Model Output, then choose the two sample directions to swap."
        case "flatten":
            goal = "Turn several directions of an array into one longer row. Keep every number and its order, and keep observations separate."
            action = "Add Reshape and choose Method → Flatten axes. Connect Input → Flatten → Model Output. Set the inclusive first and last sample axes required by this task."
        case "softmax-last-axis", "softmax-first-sample-axis":
            goal = "Turn a group of scores into shares that add up to 1. A larger score gets a larger share. Choose which numbers compete with one another."
            action = "Connect Input → Softmax → Model Output. Choose the direction required by the task, then compare each group's shares."
        case "layer-norm-rows", "layer-norm-sample", "layer-norm-affine":
            goal = "Bring a group of numbers onto a consistent scale. First subtract the group's average, then divide by its spread. Decide which numbers belong to each group."
            let capture = operation == "layer-norm-affine" ? " Enable **Inspect this component’s parameter gradients** in the parameter editor to measure how small changes to the scale and offsets affect the result." : ""
            action = "Connect Input → LayerNorm → Model Output. Choose the group shape. If requested, enter the scale and offset values in Configuration." + capture
        case "embedding-gather", "embedding-padding-frequency", "embedding-renormalize":
            goal = "Use whole-number row labels to look up rows in a table. The result contains the selected rows, including repeated selections."
            action = "Connect the supplied row selectors to Embedding, then connect its output. The lesson supplies its learned table automatically. Apply only the lookup options requested by the task."
        case "matrix-contraction", "matrix-broadcast-gradients":
            goal = "Combine one table's rows with another table's columns. Multiply matching pairs and add them to get each new cell."
            action = "Connect both supplied tables to Matrix multiply, then connect its output. Check that the joined row and column lengths match."
        case "bilinear-with-bias", "bilinear-without-bias":
            goal = "Combine two sets of features so their pairs can affect the result. Unlike simply joining two rows, this operation learns how pairs interact."
            action = "Connect both supplied inputs and interaction weights to Bilinear, then connect its output. Include the offset only when the task asks for it."
        default:
            goal = "Build the requested transformation and compare its measured output with your prediction."
            action = "Connect the permitted components using the task's input and output requirements."
        }
        let matrixTerms: [Term] = MatrixLearningTask(rawValue: operation) == nil ? [] : [
            Term(name: "Matrix", meaning: "A table of numbers arranged in rows and columns."),
            Term(name: "Array", meaning: "Numbers organized into one or more directions; a table has two directions."),
            Term(name: "Directions", meaning: "Ways to index an array, such as rows, columns or groups of tables."),
            Term(name: "Contraction", meaning: "Multiply matching entries along a shared direction and add them; that direction disappears from the result."),
            Term(name: "Broadcasting", meaning: "Reuse a table along a grouping direction when sizes match or one size is 1. Observations remain separate."),
            Term(name: "Change sensitivity", meaning: "How a small change to an input affects the result. A gradient records that influence and keeps the input’s shape.")]
        let embeddingTerms: [Term] = EmbeddingLearningTask(rawValue: operation) == nil ? [] : [
            Term(name: "Lookup row", meaning: "A row selected from the shared table by its whole-number label."),
            Term(name: "Vector", meaning: "An ordered list of numbers; here, one selected row with three values."),
            Term(name: "Category index", meaning: "A whole-number row label, not a measurement to multiply or differentiate."),
            Term(name: "Repeat accumulation", meaning: "Repeated selections reuse the same row, so their influence contributions add together."),
            Term(name: "Frequency", meaning: "How many times a row is selected across the entire input; frequency scaling divides its total influence by that count."),
            Term(name: "Padding", meaning: "A reserved row whose supplied values remain visible while its gradient contribution is suppressed."),
            Term(name: "Freeze", meaning: "Prevent learned table updates; a configured Maximum norm can still change selected rows during lookup."),
            Term(name: "Norm", meaning: "Vector length. Order 2 uses the square root of the sum of squared values.")]
        return Self(goal: goal, steps: [
            "Read the required input and output shapes below. Predict what will happen to a few numbers before running.",
            action,
            "Select a block to configure it. Use Check construction to measure your result, then inspect the feedback and output table. KwipML checks gradients for you; no handwritten derivatives or code are required.",
            "If the result differs from your prediction, trace where the numbers went and change one setting or connection at a time."
        ], terms: matrixTerms + embeddingTerms + [
            Term(name: "Tensor", meaning: "An array of numbers. A table is a two-direction tensor; more directions can represent channels, time steps or other groups."),
            Term(name: "Shape & axis", meaning: "Shape gives the size of each direction. An axis is one of those directions. Axis 0 separates observations in these exercises."),
            Term(name: "Batch", meaning: "A group of observations processed together. Operations should keep them separate unless the task explicitly says otherwise."),
            Term(name: "Gradient", meaning: "How much a small change would affect the final score. These exercises also check whether that influence reaches the correct inputs or shared values."),
            Term(name: "Fixture", meaning: "Small example data supplied by the lesson, so your construction can be checked without finding or importing a dataset.")
        ])
    }
}
