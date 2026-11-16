import SwiftUI
import AppKit
import KwipMLCore

/// One visual identity shared by the palette, graph, and inspectors.
enum ComponentStyle {
    static func systemSymbol(_ type:String)->String? {componentSystemSymbol(type)}
    static func tint(_ type: String) -> Color { ComponentVisualFamily.resolve(type:type).primary }
    static func purpose(_ type: String) -> String {
        if let presentation=BundledDataLoadingComponents.entries[type]?.presentation {return presentation.purpose}
        if let presentation=BundledInitializationComponents.entries[type]?.presentation {return presentation.purpose}
        if let presentation=BundledLossComponents.entries[type]?.presentation {return presentation.purpose}
        if let presentation=BundledAttentionComponents.entries[type]?.presentation {return presentation.purpose}
        if let presentation = BundledSpatialPatchComponents.entries[type]?.presentation { return presentation.purpose }
        if let presentation = BundledTensorRoutingComponents.entries[type]?.presentation { return presentation.purpose }
        if let presentation = BundledTensorCompositionComponents.entries[type]?.presentation { return presentation.purpose }
        if let presentation = BundledTensorLayoutComponents.entries[type]?.presentation { return presentation.purpose }
        if let presentation = BundledLayerComponents.entries[type]?.presentation { return presentation.purpose }
        if let presentation = BundledNormalizationComponents.entries[type]?.presentation { return presentation.purpose }
        if let presentation = BundledActivationComponents.entries[type]?.presentation { return presentation.purpose }
        if let presentation = BundledPoolingComponents.entries[type]?.presentation { return presentation.purpose }
        if RecurrentCellConfiguration.componentIDs.contains(type) {return "Read one feature vector per observation and update explicitly connected memory."}
        switch type {
        case "mixture_experts":return "Route observation or sequence-position features through editable independent expert graphs. Top-k, weighting, masks and balancing remain explicit; all experts stay resident."
        case "kwip.recurrentCell": return "Process one step with RNN Cell, LSTM Cell or GRU Cell and explicit hidden/cell state."
        case "kwip.recurrent": return "Process ordered sequences with RNN, LSTM or GRU. Configure explicit hidden/cell state, layers, directions and applicable method settings."
        case "kwip.tensor.axes": return "Swap two sample axes or reorder every sample axis; preserve batch and values."
        case "kwip.tensor.parts": return "Split tensors into explicit sizes or request chunks; inspect the actual parts."
        case "kwip.reshape": return "Set sample dimensions or flatten an inclusive axis range while preserving batch and values."
        case "kwip.linear": return "Learn an affine projection with explicit input width or width inferred from the connected tensor. Configure output features, bias and named parameter policies."
        case "kwip.normalization": return "Normalize intermediate tensors with RMSNorm, LayerNorm, BatchNorm1d, BatchNorm2d, InstanceNorm1d, InstanceNorm2d or GroupNorm, preserving their distinct statistics and learned adjustment controls."
        case "kwip.convolution": return "Apply standard, transposed or depthwise convolution with per-axis geometry and learned weights. Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Depthwise Conv2d."
        case "kwip.pooling": return "Reduce spatial windows or choose adaptive output sizes. Inspect captured values and winning indices."
        case "kwip.activation": return "Choose the activation function and its applicable settings. Inspect captured tensor values in Results."
        case "workflow.frozenModel": return "Load a verified saved run, document classifier package, local model folder or pinned Hugging Face snapshot with task-specific capabilities."
        case "workflow.gameSession": return "Bind authored two-player rules to frozen numeric policies and run bounded episodes, with complete Summary, Episodes and Moves tables. No learning or search."
    case "workflow.generateText": return "Continue text with verified next-token weights and the captured tokenizer. Greedy or seeded sampling produces a bounded, inspectable table."
        case "workflow.saveModel": return "Save a verified trained model as a package or checkpoint, or an imported model as a Hugging Face folder, at a chosen destination on this Mac."
        case "workflow.evaluateTest": return "Evaluate the selected archived Validation or Test population with a verified trained model. Batch size controls memory; preview size controls display only. Inspecting results is recorded locally."
        case "kwip.data.join": return "Match two tables through ordered key pairs. Choose missing-key behavior, expected key relationships and an output budget before splitting."
        case "kwip.data.sort": return "Order rows by explicit keys, direction and missing-value placement."
        case "kwip.data.unique": return "Separate retained key records from duplicates."
        case "kwip.data.reshapeTable": return "Reshape a table between long rows and wide columns."
        case "kwip.data.splitText": return "Split delimited or fixed-width text into columns or rows."
        case "kwip.data.union": return "Append tables in input order with explicit name or position alignment. Duplicates are retained."
        case "kwip.data.aggregate": return "Summarize observations by chosen groups using counts, numeric summaries or joined text. Inspect missing values and group membership before splitting."
        case "kwip.data.filterRows": return "Route rows through typed conditions, expressions or text patterns into matching and nonmatching tables."
        case "kwip.data.gameRules": return "Author bounded game state, actions and rules; validate definitions and inspect complete test/transition tables."
        case "kwip.data.decide": return "Score typed questions over records with a verified model. Complete Decisions and Scores are standard tables."
        case "kwip.data.classifyAudio": return "Classify captured mono waveforms with verified AST weights into complete Predictions and Scores tables."
        case "kwip.data.transcribeAudio": return "Transcribe captured mono waveforms with a connected verified Whisper model into Transcripts and Segments tables."
        case "kwip.data.extractText": return "Extract English printed text as Documents and Regions tables with source geometry."
        case "kwip.data.prepareMedia": return "Prepare captured images or PCM audio as explicit typed tensors, preserving source identity."
        case "kwip.data.sequenceWindows": return "Create aligned next-token examples within each document and partition."
        case "kwip.data.textEncode": return "Encode text into typed token sequences or Training-fitted TF-IDF document features."
        case "kwip.data.columns": return "Select, order, rename and cast columns in one grid."
        case "kwip.data.selectRename": return "Choose ordered columns and explicit output names."
        case "kwip.data.filter": return "Keep rows with a typed comparison; inspect removed-row counts."
        case "kwip.data.sourceCSV": return "Read a selected table source through a private verified snapshot."
        case "kwip.data.sourceFiles": return "Read local files, documents, image pixels and explicitly selected PCM WAV samples into a captured table."
        case "kwip.data.view": return "Materialize complete upstream data for bounded paging and search."
        case "kwip.data.cast": return "Convert named columns to explicit data types."
        case "kwip.data.sampleRows": return "Select an exact seeded row budget without replacement. Preserve observation identities and source order; count and seed control reproducible subsets."
        case "kwip.data.preprocess": return "Prepare separate Training, Validation and Test tables with shared training-fitted column rules, or use fixed table cleanup."
        case "kwip.data.formula": return "Create and transform column values with ordered formulas and guided text patterns."
        case "kwip.data.partition": return "Create Training, Validation and Test tables using random, whole-entity or chronological partitions."
        case "kwip.data.split": return "Create reproducible training and validation partitions."
        case "kwip.data.groupSplit": return "Keep whole entities together across training, validation and test. Allocate seeded percentages of groups; inspect actual row proportions in Results."
        case "kwip.data.temporalSplit": return "Partition observations by explicit time cutoffs. Exclude embargo gaps and optionally require disjoint entities; inspect actual boundaries in Results."
        case "kwip.data.standardScalerFit": return "Fit means and scales using training observations only."
        case "kwip.data.imputerFit": return "Learn missing-value replacements from training observations only."
        case "kwip.data.oneHotEncoderFit": return "Learn a training-only vocabulary and explicit unknown-category policy."
        case "kwip.data.transform": return "Apply the same training-fitted state to training, validation and test partitions without refitting held-out observations."
        case "reshape": return "Change sample dimensions while preserving element count and order; keep the batch axis."
        case "transpose": return "Exchange two sample axes and their coordinates; keep the batch axis."
        case "flatten": return "Merge sample axes into one feature vector in order; keep the batch axis."
        case "index_input": return "Mark where token or category-index tensors enter the model. Integer values are not continuous measurements."
        case "embedding_bag": return "Combine category vectors by group. Configure fixed or packed groups, Mean/Sum/Maximum, optional item weights, padding, norms and learned weights."
        case "embedding": return "Look up learned vectors from integer category IDs. Configure table dimensions, padding, norm limits, gradients and weights."
        case "matmul": return "Multiply rows by columns through explicit a/b operands. Sample axes may broadcast; observations remain independent. Use separate transpose or reshape components to change the operands."
        case "input": return "Mark where the model receives a batched tensor. Model definition owns named-input contracts; Train maps prepared columns to them."
        case "parallel_features": return "Edit parallel tensor paths in a collapsible canvas container with grouped inputs and outputs."
        case "separable_conv2d": return "An editable Block combining independent spatial filters with a pointwise channel mixer. Open it to inspect both layers and their linked settings."
        case "depthwise_conv2d": return "Apply independent spatial filter banks to each input channel. Channel multiplier chooses filters per channel; output channel c × multiplier + m belongs to input c. A later pointwise convolution can mix channels."
        case "conv_transpose3d": return "Scatter shared volume kernels over [batch, channels, depth, height, width]. Configure stride, padding and output padding triples in [depth, height, width] order. Matched linear operators are adjoints, not generally inverses."
        case "conv_transpose2d": return "Scatter shared spatial kernels over [batch, channels, height, width]. Configure paired stride, padding and output padding in [height, width] order. Matched linear operators are adjoints, not generally inverses."
        case "conv_transpose1d": return "Scatter shared kernels along [batch, channels, length]. Configure groups, stride, padding and output padding to select output geometry. Transposed convolution is an adjoint operation, not an inverse."
        case "conv3d": return "Apply shared volume kernels to [batch, channels, depth, height, width]. Spatial triples use [depth, height, width]; configure groups, bias and padding. Same padding requires stride [1, 1, 1]."
        case "conv2d": return "Apply shared spatial kernels to [batch, channels, height, width]. Spatial pairs are [height, width]; configure groups, bias and padding. Same padding requires stride [1, 1]."
        case "conv1d": return "Apply shared kernels along length in [batch, channels, length]. Configure channels, kernel, stride, dilation, groups, bias and padding. Same padding requires stride 1."
        case "lazy_linear": return "Infer input width from the connected tensor and learn an affine projection. Configure output width and bias; parameters initialize before training. Changing upstream width requires compatible new weights."
        case "bilinear": return "Learn pairwise interactions between tensor inputs a and b. Configure their independent final feature widths, output features, optional bias and named weight policies. Leading dimensions must match."
        case "linear": return "Learn a weighted transformation from one feature width to another."
        case "relu": return "Keep positive values; replace negative values with zero."
        case "rotary_position": return "Rotate Query and Key by sequence position; inline numbering and scale, no configuration-input edge."
        case "rms_norm": return "Scale trailing sample dimensions by root mean square without subtracting the mean, with optional learned scale and no bias or running statistics."
        case "layer_norm": return "Center and scale trailing sample dimensions using current-input population statistics, with optional learned scale and bias. Keep observations separate; this does not fit dataset statistics."
        case "softmax": return "Normalize each slice along a sample axis into probabilities that sum to one. Negative axes count from the end; the batch stays separate."
        case "gelu": return "Weight each value by its Gaussian cumulative probability. Choose exact GELU or its tanh approximation."
        case "tanh": return "Smoothly compress values into the range −1 to 1."
        case "concat": return "Join two branches along their feature dimension."
        case "add": return "Add matching features from two equal-width branches."
        case "output": return "Expose the model's final predictions for training and evaluation."
        default: return "Configure this component in the inspector."
        }
    }
    static func shortTitle(_ type: String, fallback: String) -> String {
        if let presentation=BundledDataLoadingComponents.entries[type]?.presentation {return presentation.shortTitle}
        if let presentation=BundledInitializationComponents.entries[type]?.presentation {return presentation.shortTitle}
        if let presentation=BundledLossComponents.entries[type]?.presentation {return presentation.shortTitle}
        if let presentation=BundledAttentionComponents.entries[type]?.presentation {return presentation.shortTitle}
        if let presentation = BundledSpatialPatchComponents.entries[type]?.presentation { return presentation.shortTitle }
        if let presentation = BundledTensorRoutingComponents.entries[type]?.presentation { return presentation.shortTitle }
        if let presentation = BundledTensorCompositionComponents.entries[type]?.presentation { return presentation.shortTitle }
        if let presentation = BundledTensorLayoutComponents.entries[type]?.presentation { return presentation.shortTitle }
        if let presentation = BundledLayerComponents.entries[type]?.presentation { return presentation.shortTitle }
        if let presentation = BundledNormalizationComponents.entries[type]?.presentation { return presentation.shortTitle }
        if let presentation = BundledActivationComponents.entries[type]?.presentation { return presentation.shortTitle }
        if let presentation = BundledPoolingComponents.entries[type]?.presentation { return presentation.shortTitle }
        switch type {
        case "kwip.tensor.axes": return "Transpose"
        case "kwip.tensor.parts": return "Split tensor"
        case "kwip.reshape": return "Reshape"
        case "conv_transpose1d": return "ConvT 1D"
        case "conv_transpose2d": return "ConvT 2D"
        case "conv_transpose3d": return "ConvT 3D"
        case "depthwise_conv2d": return "Depthwise"
        case "separable_conv2d": return "Separable"
        case "rms_norm": return "RMSNorm"
        case "layer_norm": return "LayerNorm"
        case "concat": return "Concat"
        case "kwip.data.filterRows": return "Filter"
        case "kwip.data.gameRules": return "Game definition"
        case "kwip.data.decide": return "Decide"
        case "kwip.data.classifyAudio": return "Classify audio"
        case "kwip.data.transcribeAudio": return "Transcribe audio"
        case "kwip.data.extractText": return "Extract text"
        case "kwip.data.prepareMedia": return "Prepare media"
        case "kwip.data.sequenceWindows": return "Sequence windows"
        case "kwip.data.textEncode": return "Text encoding"
        case "kwip.data.columns": return "Columns"
        case "kwip.data.selectRename": return "Select / name"
        case "kwip.data.filter": return "Filter"
        case "kwip.data.sourceCSV": return "Table Input"
        case "kwip.data.sourceFiles": return "File Input"
        case "kwip.data.view": return "View"
        case "kwip.data.cast": return "Cast"
        case "kwip.data.sampleRows": return "Sample rows"
        case "kwip.data.join": return "Join tables"
        case "kwip.data.sort": return "Sort"
        case "kwip.data.unique": return "Unique"
        case "kwip.data.reshapeTable": return "Reshape Table"
        case "kwip.data.splitText": return "Split Text"
        case "kwip.data.union": return "Union"
        case "kwip.data.aggregate": return "Summarize"
        case "kwip.data.preprocess": return "Preprocess"
        case "kwip.data.formula": return "Formula"
        case "kwip.data.partition": return "Partition"
        case "kwip.data.split": return "Split"
        case "kwip.data.groupSplit": return "Group split"
        case "kwip.data.temporalSplit": return "Time split"
        case "kwip.data.standardScalerFit": return "Fit scaler"
        case "kwip.data.imputerFit": return "Fit imputer"
        case "kwip.data.oneHotEncoderFit": return "Fit encoder"
        case "kwip.data.transform": return "Transform"
        default: return fallback
        }
    }
}

/// System-symbol fallbacks remain available when generated assets cannot load.
private func componentSystemSymbol(_ type: String) -> String? {
    if type == "separable_conv2d" || type == "kwip.block" { return "square.stack.3d.up" }
    if type.hasPrefix("composite:") { return "square.stack.3d.up" }
    switch type {
    case "workflow.modelDefinition": return "point.3.connected.trianglepath.dotted"
    case "workflow.trainingData": return "tablecells.badge.ellipsis"
    case "workflow.runSetup": return "slider.horizontal.3"
    case "workflow.train": return "gearshape"
    case "workflow.frozenModel": return "tray.and.arrow.down"
    case "workflow.gameSession": return "arrow.triangle.2.circlepath"
    case "workflow.generateText": return "text.bubble"
    case "workflow.saveModel": return "square.and.arrow.down"
    case "workflow.evaluateTest", "objective_check": return "checkmark.seal"
    case "index_input": return "number.square"
    case "prelu": return "chart.xyaxis.line"
    case "log_softmax": return "chart.bar"
    case "dropout": return "square.grid.3x3"
    case "matmul": return "multiply.square"
    case "embedding": return "tablecells"
    case "kwip.data.formula": return "function"
    case "kwip.data.filterRows": return "line.3.horizontal.decrease"
    case "kwip.data.columns": return "tablecells"
    case "kwip.data.selectRename": return "rectangle.split.3x1"
    case "kwip.data.filter": return "line.3.horizontal.decrease"
    case "kwip.data.view": return "tablecells.badge.ellipsis"
    case "kwip.data.sourceCSV": return "tablecells"
    case "kwip.data.gameRules": return "list.bullet.rectangle"
    case "kwip.data.decide": return "arrow.triangle.branch"
    case "kwip.data.classifyAudio": return "waveform"
    case "kwip.data.transcribeAudio": return "waveform"
    case "kwip.data.extractText": return "text.viewfinder"
    case "kwip.data.prepareMedia": return "waveform.path"
    case "kwip.data.sourceFiles": return "folder"
    case "kwip.data.cast": return "arrow.left.arrow.right"
    case "kwip.data.sampleRows": return "line.3.horizontal.decrease.circle"
    case "kwip.data.partition", "kwip.data.split": return "arrow.triangle.branch"
    case "kwip.data.groupSplit": return "person.3.sequence"
    case "kwip.data.temporalSplit": return "clock.arrow.circlepath"
    case "kwip.data.standardScalerFit": return "slider.horizontal.3"
    case "kwip.data.imputerFit": return "square.dashed.inset.filled"
    case "kwip.data.oneHotEncoderFit": return "square.grid.3x3"
    case "kwip.data.transform": return "function"
    case "split": return "rectangle.split.2x1"
    case "chunk": return "rectangle.split.3x1"
    case "gather": return "scope"
    case "unfold": return "rectangle.split.3x3"
    case "fold": return "square.stack.3d.up"
    case "einsum": return "sum"
    case "reshape": return "rectangle.split.3x1"
    case "transpose": return "arrow.left.arrow.right"
    case "flatten": return "line.3.horizontal"
    default: return nil
    }
}

struct ComponentIcon: View {
    let type: String
    var size: CGFloat = 32
    @Environment(\.colorScheme) private var colorScheme
    private var family: ComponentVisualFamily { ComponentVisualFamily.resolve(type:type) }
    private var glyphColor: Color { colorScheme == .dark ? family.primary : family.secondary }
    @ViewBuilder var body: some View {
        if let image = ComponentArtworkCatalog.image(for:type) {
            Image(nsImage:image).resizable().renderingMode((type == "workflow.logisticRegression" || type == "workflow.kerasClassifier") ? .template : .original).interpolation(.high).scaledToFit()
                .foregroundStyle(glyphColor)
                .frame(width:size,height:size).accessibilityHidden(true)
        } else {
            legacyIcon
        }
    }
    private var legacyIcon: some View {
        Group {
            if let image = ComponentImageAssets.images[type] {
                Image(nsImage: image).resizable().renderingMode(.template).interpolation(.high).scaledToFit().foregroundStyle(glyphColor)
            } else if let symbol = ComponentStyle.systemSymbol(type) {
                Image(systemName: symbol).font(.system(size: size * 0.48, weight: .medium))
                    .foregroundStyle(glyphColor)
            } else {
                fallbackGlyph
            }
        }
        .frame(width:size * 0.65,height:size * 0.65)
        .frame(width:size,height:size)
        .background(ComponentIconFrame(silhouette:family.silhouette).fill(family.secondary.opacity(colorScheme == .dark ? 0.23 : 0.08)))
        .overlay(ComponentIconFrame(silhouette:family.silhouette).stroke(glyphColor.opacity(0.8),lineWidth:size >= 32 ? 1:0.75))
        .accessibilityHidden(true)
    }

    private var fallbackGlyph: some View {
        Canvas { context, canvasSize in
            let tint = glyphColor
            let width = max(1.15, canvasSize.width / 22)
            func p(_ x: Double, _ y: Double) -> CGPoint {
                CGPoint(x: canvasSize.width * x, y: canvasSize.height * y)
            }
            func line(_ points: [CGPoint], opacity: Double = 1) {
                var path = Path()
                if let first = points.first {
                    path.move(to: first)
                    for point in points.dropFirst() { path.addLine(to: point) }
                }
                context.stroke(path, with: .color(tint.opacity(opacity)), style: StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round))
            }
            func dot(_ point: CGPoint) {
                let radius = canvasSize.width * 0.038
                context.fill(Path(ellipseIn: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)), with: .color(tint))
            }
            switch BundledPoolingComponents.entries[type]?.presentation.icon ?? type {
            case "layer_norm":
                for y in [0.32,0.68] {
                    line([p(0.18,y),p(0.82,y)],opacity:0.35)
                    line([p(0.5,y-0.12),p(0.5,y+0.12)])
                    for x in [0.28,0.72] {dot(p(x,y))}
                }
            case "softmax":
                line([p(0.2,0.8),p(0.8,0.8)],opacity:0.3)
                for (x,y) in [(0.3,0.55),(0.5,0.25),(0.7,0.65)] {line([p(x,0.8),p(x,y)])}
            case "bilinear":
                for y in [0.3, 0.7] {
                    line([p(0.2,y),p(0.5,0.5),p(0.8,y)])
                    dot(p(0.2,y)); dot(p(0.8,y))
                }
                dot(p(0.5,0.5))
            case "depthwise_conv2d":
                for y in [0.28,0.72] {
                    dot(p(0.2,y))
                    line([p(0.27,y),p(0.48,y),p(0.72,y-0.1)])
                    line([p(0.48,y),p(0.72,y+0.1)])
                    dot(p(0.78,y-0.1));dot(p(0.78,y+0.1))
                }
            case "conv_transpose3d":
                line([p(0.18,0.2),p(0.38,0.2),p(0.38,0.4),p(0.18,0.4),p(0.18,0.2)])
                line([p(0.45,0.52),p(0.7,0.52),p(0.7,0.82),p(0.45,0.82),p(0.45,0.52)])
                line([p(0.45,0.52),p(0.6,0.4),p(0.85,0.4),p(0.85,0.7),p(0.7,0.82)])
                line([p(0.7,0.52),p(0.85,0.4)])
                line([p(0.32,0.35),p(0.52,0.58)],opacity:0.6)
                dot(p(0.55,0.62));dot(p(0.62,0.72))
            case "conv_transpose2d":
                line([p(0.2,0.2),p(0.42,0.2),p(0.42,0.42),p(0.2,0.42),p(0.2,0.2)])
                line([p(0.47,0.47),p(0.82,0.47),p(0.82,0.82),p(0.47,0.82),p(0.47,0.47)])
                line([p(0.47,0.65),p(0.82,0.65)],opacity:0.4)
                line([p(0.65,0.47),p(0.65,0.82)],opacity:0.4)
                line([p(0.35,0.35),p(0.56,0.56)])
                dot(p(0.56,0.56));dot(p(0.73,0.73))
            case "conv_transpose1d":
                for x in [0.3, 0.7] {
                    dot(p(x,0.25))
                    for offset in [-0.15, 0.0, 0.15] {
                        line([p(x,0.32),p(x+offset,0.72)],opacity:0.6)
                        dot(p(x+offset,0.78))
                    }
                }
            case "conv3d":
                line([p(0.17,0.35),p(0.58,0.35),p(0.58,0.8),p(0.17,0.8),p(0.17,0.35)])
                line([p(0.17,0.35),p(0.4,0.17),p(0.82,0.17),p(0.82,0.61),p(0.58,0.8)])
                line([p(0.58,0.35),p(0.82,0.17)])
                line([p(0.37,0.35),p(0.37,0.8)],opacity:0.4)
                line([p(0.17,0.57),p(0.58,0.57)],opacity:0.4)
                dot(p(0.27,0.46));dot(p(0.48,0.68))
            case "conv2d":
                for coordinate in [0.2, 0.4, 0.6, 0.8] {
                    line([p(coordinate, 0.2), p(coordinate, 0.8)], opacity: 0.35)
                    line([p(0.2, coordinate), p(0.8, coordinate)], opacity: 0.35)
                }
                line([p(0.2, 0.2), p(0.6, 0.2), p(0.6, 0.6), p(0.2, 0.6), p(0.2, 0.2)])
                dot(p(0.3, 0.3)); dot(p(0.5, 0.5))
            case "pooling":
                for (x,y) in [(0.2,0.65),(0.4,0.3),(0.6,0.5),(0.8,0.7)] {
                    line([p(x,0.8),p(x,y)],opacity:0.5)
                }
                line([p(0.15,0.18),p(0.65,0.18),p(0.65,0.85),p(0.15,0.85),p(0.15,0.18)],opacity:0.6)
                dot(p(0.4,0.3));line([p(0.4,0.3),p(0.82,0.18)])
            case "conv1d":
                for y in [0.3, 0.7] { line([p(0.16, y), p(0.84, y)], opacity: 0.4) }
                for x in [0.35, 0.5, 0.65] { dot(p(x, 0.3)); dot(p(x, 0.7)) }
                line([p(0.3, 0.18), p(0.7, 0.18), p(0.7, 0.82), p(0.3, 0.82), p(0.3, 0.18)])
            case "linear", "lazy_linear":
                for y1 in [0.33, 0.67] {
                    for y2 in [0.23, 0.5, 0.77] { line([p(0.27, y1), p(0.73, y2)], opacity: 0.55) }
                    dot(p(0.27, y1))
                }
                for y in [0.23, 0.5, 0.77] { dot(p(0.73, y)) }
            case "relu", "tanh", "gelu":
                line([p(0.18, 0.72), p(0.82, 0.72)], opacity: 0.3)
                line([p(0.5, 0.18), p(0.5, 0.84)], opacity: 0.3)
                if type == "relu" { line([p(0.19, 0.72), p(0.48, 0.72), p(0.81, 0.22)]) }
                else if type == "gelu" {
                    var points: [CGPoint] = []
                    for x: Double in stride(from: -2.5, through: 2.5, by: 0.1) {
                        let probability: Double = 0.5 * (1.0 + Foundation.erf(x / 1.4142135623730951))
                        let value: Double = x * probability
                        points.append(p(0.5 + x * 0.128, 0.65 - value * 0.16))
                    }
                    line(points)
                } else {
                    line(stride(from: -2.5, through: 2.5, by: 0.1).map { p(0.5 + $0 * 0.128, 0.5 - Foundation.tanh($0) * 0.27) })
                }
            case "concat":
                line([p(0.18, 0.25), p(0.39, 0.25), p(0.56, 0.43), p(0.79, 0.43)])
                line([p(0.18, 0.75), p(0.39, 0.75), p(0.56, 0.57), p(0.79, 0.57)])
                line([p(0.72, 0.34), p(0.82, 0.5), p(0.72, 0.66)])
            case "add":
                line([p(0.19, 0.27), p(0.35, 0.27), p(0.42, 0.38)], opacity: 0.7)
                line([p(0.19, 0.73), p(0.35, 0.73), p(0.42, 0.62)], opacity: 0.7)
                line([p(0.48, 0.5), p(0.74, 0.5)])
                line([p(0.61, 0.37), p(0.61, 0.63)])
            case "input":
                line([p(0.65, 0.23), p(0.8, 0.23), p(0.8, 0.77), p(0.65, 0.77)], opacity: 0.6)
                line([p(0.18, 0.5), p(0.64, 0.5)])
                line([p(0.48, 0.34), p(0.65, 0.5), p(0.48, 0.66)])
            case "output":
                line([p(0.4, 0.23), p(0.2, 0.23), p(0.2, 0.77), p(0.4, 0.77)], opacity: 0.6)
                line([p(0.39, 0.5), p(0.82, 0.5)])
                line([p(0.67, 0.34), p(0.83, 0.5), p(0.67, 0.66)])
            default:
                line([p(0.25, 0.25), p(0.75, 0.25), p(0.75, 0.75), p(0.25, 0.75), p(0.25, 0.25)])
            }
        }
    }
}

/// Decode generated placeholders once; new modules retain accessible fallback glyphs.
private enum ComponentImageAssets {
    static let images: [String: NSImage] = {
        var result: [String: NSImage] = [:]
        for type in ["input", "linear", "relu", "tanh", "gelu", "concat", "add", "output", "reshape", "transpose", "flatten", "index_input", "embedding"] {
            let url = Bundle.module.url(forResource: "\(type)-v1", withExtension: "png")
                ?? Bundle.module.url(forResource: "\(type)-v1", withExtension: "png", subdirectory: "ComponentIcons")
            if let url, let image = NSImage(contentsOf: url) { result[type] = image }
        }
        for (type, name) in ["kwip.data.selectRename": "data-select", "kwip.data.filter": "data-filter", "kwip.data.sourceCSV": "data-source", "kwip.data.cast": "data-cast", "kwip.data.partition": "data-split", "kwip.data.split": "data-split", "kwip.data.standardScalerFit": "data-fit", "kwip.data.transform": "data-transform", "kwip.data.imputerFit": "data-imputer", "kwip.data.oneHotEncoderFit": "data-encoder"] {
            let url = Bundle.module.url(forResource: "\(name)-v1", withExtension: "png")
                ?? Bundle.module.url(forResource: "\(name)-v1", withExtension: "png", subdirectory: "ComponentIcons")
            if let url, let image = NSImage(contentsOf: url) { result[type] = image }
        }
        return result
    }()
}
