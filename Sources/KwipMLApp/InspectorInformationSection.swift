import SwiftUI
import KwipMLCore

/// Shared inspector template slot for explanation, separate from editable controls.
struct InspectorInformationSection<Content: View>: View {
    @Environment(\.inspectedComponentType) private var inheritedType
    @Environment(\.inspectedComponentConfiguration) private var inheritedConfiguration
    var componentType: String? = nil
    @ViewBuilder let content: () -> Content
    var body: some View {
        InspectorSection("Information",symbol:"info.circle",identifier:"inspector.information",initiallyExpanded:false) {
            VStack(alignment:.leading,spacing:8) {
                content()
                if let type = inheritedType ?? componentType {
                    ComponentLessonLink(type: type, configuration: inheritedConfiguration)
                }
            }
                .font(.caption).foregroundStyle(MLTheme.muted)
                .fixedSize(horizontal:false,vertical:true).textSelection(.enabled)
                .frame(maxWidth:.infinity,alignment:.leading)
        }
    }
}

/// Bundle-owned numerical guidance stays in Information, beside its explanation.
struct ComponentInformationContent: View {
    let type: String
    var body: some View {
        if WorkflowModelInput.componentIDs.contains(type) {
            LessonRichText(text:"**Model starting point**\nThis boundary supplies the actual batched tensor to connected layers. Floating inputs use float32; integer inputs use int64 for tokens or category indices. Sample dimensions exclude the leading batch axis. It does not read files or tokenize text.\n\n**Named inputs**\nModel definition manages input names and floating/integer kind. Train maps prepared table columns to those names. Multiple inputs remain separate tensors, with independent shapes. The existing saved identity and tensor wires are preserved.\n\n**Captured values**\nResults shows retained values from the selected run. Declared shape is a contract, not evidence of execution. Disconnect a visible wire before replacing it; no configuration-recipe connections are created.")
        }

        Text(ComponentStyle.purpose(type))
        if TensorOperationAuthoring.componentIDs.contains(type) {TensorOperationInformationView(type:type)}
        if AttentionConfiguration.componentIDs.contains(type) {AttentionInformationView(type:type)}
        if type=="embedding_bag" {
            LessonRichText(text:"**Group layout**\nFixed rows gives each observation the same number of category IDs. Packed groups uses a flat indices array and offsets marking group starts; repeated offsets preserve empty groups. Include final boundary expects an extra ending offset. Source and component settings must agree.\n\n**Combine selected vectors**\nMean averages non-padding rows, Sum adds them, and Maximum chooses each coordinate’s largest value. Use item weights adds a real incoming weight for every selected item and works with Sum only. These input weights are data, separate from the learned Embedding weights matrix.\n\n**Padding and norms**\nPadding is excluded from the combination and lookup gradients. Negative padding counts backward from the table end. Norm limiting can change selected learned rows before combination even when they are frozen. Bag norm limiting requires CPU; it never silently falls back from MPS.\n\n**Gradients**\nMaximum cannot use frequency scaling or sparse gradients. Sparse and frequency scaling cannot coexist. Current Adam training refuses trainable sparse gradients; trainable frequency scaling requires CPU. Valid form values do not replace selected-runtime admission.\n\n**Connections and saved state**\nDisconnect offsets or item weights explicitly before removing those inputs. Existing project, group identity and whole-bag boundaries remain. Shared weights initialization and trainability apply to the learned table; an explicit policy overrides the layer’s Freeze default. Changing table dimensions never silently resizes a checkpoint.")
        }
        if type=="embedding" {
            LessonRichText(text:"**Index lookup**\nInteger category IDs select complete rows from the learned weight table. Input dimensions are preserved and Vector dimensions is appended. No bias or tokenizer is added.\n\n**Padding row**\nEnable Use padding row to reserve one row from lookup gradients. Negative indices count backward from the table end. Padding values are not silently replaced with zeros; explicit initialization or checkpoints may supply them.\n\n**Norm limit**\nLimit vector norm shortens selected rows before lookup using Maximum norm and Norm order. This forward change can occur even with frozen weights. Unselected rows stay unchanged.\n\n**Gradients and training**\nFrequency scaling divides each row’s gradient contribution by its occurrence count. It cannot be combined with Sparse gradients. Sparse learning requires compatible optimizer/device support; the current Adam training path refuses trainable sparse gradients. Valid configuration alone does not establish execution support.\n\n**Weights**\nThere is one Embedding weights matrix and no bias. Freeze learned weights sets the layer default. An explicit Weights → Trainability override takes precedence and is shown beside the freeze control; choose Layer default there before editing Freeze. Retained Train overrides remain authoritative where present. Initializers apply only to fresh training; changing dimensions never silently resizes saved checkpoints.")
        }
        if RecurrentCellConfiguration.componentIDs.contains(type) {
            LessonRichText(text:"**Single step**\nInput uses observation × input features. Each call processes one step; Recurrent handles a whole ordered sequence. Hidden and cell state use observation × hidden features.\n\n**Explicit starting state**\nStart from zeros resets state for every call. Use state connections requires hidden, and LSTM Cell also requires cell. State travels on tensor connections; the tool never remembers previous calls implicitly. Existing canvas cycle rules still apply.\n\n**Captured output**\nOutput and hidden are aliases of the same step values, not two different computations. LSTM Cell also exposes separate cell memory. Inspect each named captured output in Results.\n\n**Weights and biases**\nInput weights multiply current features; recurrent weights multiply incoming hidden state. Each has its own optional bias. LSTM and GRU pack different gates, so clear authored parameter policies explicitly before changing method. Removing bias requires clearing its authored policies first. Connected or exposed hidden/cell ports cannot be discarded.\n\n**Saved state**\nStarting values apply to fresh learned parameters. Compatible checkpoints restore learned weights; activation state is separate. Width or method edits never resize or silently transfer saved weights. Cells do not have sequence layers, directions, inter-layer dropout or projection.")
        }
        if RecurrentSequenceConfiguration.componentIDs.contains(type) {
            LessonRichText(text:"**Sequence axes**\nInput values use observation × time × input features. Sequence output keeps observations and time; its final width is hidden features, or projected features for LSTM, doubled when Bidirectional is enabled. No batch axis is reinterpreted.\n\n**Explicit starting state**\nStart from zeros resets state on every call. Use state connections requires hidden, and LSTM also requires cell. Hidden uses observation × (layers × directions) × state width. LSTM cell uses hidden width even when its hidden summary is projected. State is carried by tensor connections, never secretly remembered between calls.\n\n**Layers and directions**\nLayers stacks sequence readers. Bidirectional reads both directions; forward and reverse parameters are distinct. Carrying a reverse summary between chunks is not equivalent to reading a whole reversed sequence.\n\n**Between-layer dropout**\nApplies only during training between two or more stacked layers. A single layer requires zero; lower dropout to zero explicitly before reducing to one layer.\n\n**LSTM projection**\nProjection features zero disables projection. A positive width must be smaller than hidden features. Projection changes the hidden/output width without shrinking cell memory. The current runtime requires CPU for projected LSTM.\n\n**Named parameters and saved state**\nWeights and biases identifies input, recurrent and projection parameters by layer and direction. Gate layouts differ between RNN, LSTM and GRU; clear authored parameter policies explicitly before changing method. Removing biases, layers, a direction or projection also requires clearing affected policies. Connected or exposed state ports cannot be discarded. Architecture changes require exactly compatible checkpoints; learned weights are never resized or transferred silently.")
        }
        if type=="bilinear" {
            LessonRichText(text:"**Two tensor inputs**\nThe final widths of a and b are independent. Leading observation and sample dimensions must match exactly; Bilinear does not broadcast them. Output features replaces the final dimension.\n\n**Learned interactions**\nEach output combines pairs of features, one from each input: y[o] = ΣᵢΣⱼ a[i] W[o,i,j] b[j] + bias[o]. The same weights apply at each leading position.\n\n**Parameter count**\nOutput features × First input features × Second input features, plus Output features when bias is enabled. The indicator reflects the current configuration draft, not captured training. Each width is 1–512; the existing 2 million parameter and whole-model budgets still apply.\n\n**Saved state**\nStarting values apply to fresh models. Checkpoints restore exactly compatible saved weights. Width changes never resize learned state; disabling bias requires explicitly removing any authored bias policy first.")
        }
        if LinearConfiguration.componentIDs.contains(type) {
            LessonRichText(text:"**Final-dimension projection**\nMaps the last tensor dimension to Output features. Leading observation and sample dimensions are preserved; the same weights and bias apply at every position.\n\n**Input sizing**\nSet explicitly requires the connected final dimension to equal Input features. Infer from input resolves that dimension before training and initializes Lazy Linear's parameters before the optimizer. The indicator describes inference, not a measured width. Returning to explicit sizing restores an unapplied explicit width when available; otherwise it starts from the original default of two.\n\n**Weights and biases**\nPolicies belong to named parameters. Turning off bias requires explicitly removing any authored bias policy first. Inferred parameters do not yet have an authoritative weight shape in the editor, so rank-dependent initializers must be changed explicitly before switching.\n\n**Saved state**\nChanging widths or sizing changes the architecture. Start a fresh run or use an exactly compatible checkpoint; existing weights are never silently resized or reinterpreted.")
        }
        if type == "dropout" {
            LessonRichText(text:"**During training**\nRandomly masks individual tensor values. For drop probability below one, retained values are divided by one minus the probability to preserve their expected magnitude. Zero leaves values unchanged; one returns all zeros.\n\n**During evaluation**\nPasses values through unchanged. Dropout has no learned weights or biases.\n\n**Reproducibility**\nUses Train’s seed and execution mode. Each training call draws another mask from that seeded sequence. The percentage describes a probability, not a measured count; a small tensor can lose more or fewer values. Captured Results show only recorded tensor values.")
        }
        if let variant=NormalizationConfiguration.Variant(rawValue:type) {
            LessonRichText(text:"**Learned adjustments**\nLearn scale enables a learned multiplier; Use bias adds a learned offset. Weights and biases controls their starting values and trainability. Hidden options retain their saved settings; removing a parameter with an explicit policy requires you to remove that policy first.\n\n**Epsilon**\nA positive safety value is added before scaling by the standard deviation. Scientific notation is supported; changing epsilon can change the output.")
            if variant.method == .layer {
                LessonRichText(text:"**Normalized shape**\nRows define ordered trailing sample dimensions and must match the connected tensor exactly. The leading batch dimension stays separate. Add or remove dimensions to change the normalized extent; this does not reshape the tensor.")
            } else if variant.method == .group {
                LessonRichText(text:"**Channel groups**\nGroups must divide Channels exactly. Each group normalizes its own values within an observation and needs more than one value. Group normalization has no running statistics.")
            } else {
                LessonRichText(text:"**Running statistics**\nTracked statistics update during training and are used during evaluation. When tracking is off, the current input supplies statistics in both contexts. Batch normalization combines observations and spatial positions per channel; Instance normalization keeps observations separate.\n\n**Update weight**\nMoving summaries weight each new statistic by this value; it is not the optimizer’s momentum. Batch also offers a cumulative summary, which uses the number of tracked batches instead. Hidden update settings remain saved. Running buffers belong to the checkpoint; method or tracking changes do not transfer compatible state automatically.")
            }
        }
        if let learning = BundledNormalizationComponents.entries[type]?.learning,
           let precision = learning.paragraphs.last {
            Divider()
            Text("Numerical precision").fontWeight(.semibold)
            Text(precision)
        }
    }
}


private struct TensorOperationInformationView:View {
    let type:String
    private var paragraphs:[String] {
        if let paragraphs=BundledTensorLayoutComponents.entries[type]?.learning?.paragraphs {return paragraphs}
        if let paragraphs=BundledTensorCompositionComponents.entries[type]?.learning?.paragraphs {return paragraphs}
        if let paragraphs=BundledTensorRoutingComponents.entries[type]?.learning?.paragraphs {return paragraphs}
        if let paragraphs=BundledSpatialPatchComponents.entries[type]?.learning?.paragraphs {return paragraphs}
        if let paragraphs=BundledTensorMathComponents.entries[type]?.learning?.paragraphs {return paragraphs}
        return []
    }

    var body:some View {
        ForEach(Array(paragraphs.enumerated()),id:\.offset) {_,paragraph in LessonRichText(text:paragraph)}
    }
}
