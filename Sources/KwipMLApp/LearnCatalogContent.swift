import SwiftUI
import KwipMLCore

/// App-level discovery only. Launches retain the existing managed lesson/project ownership.
struct LearnCatalogContent: View {
    @ObservedObject var store: WorkspaceStore
    @ObservedObject var affineLearning: AffineLearningStore
    @ObservedObject var normalizationLearning: NormalizationLearningStore
    @ObservedObject var tensorLearning: TensorLearningStore
    let section: String
    let search: String
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.lessonRailPresented) private var inLessonRail
    var typeFilter = "All types"
    var frameworkFilter = "All frameworks"
    var openEntry: (String) -> Void = { _ in }

    static let embeddingBagPracticeTasks=EmbeddingBagLearningTask.allCases
    static let embeddingBagPracticeSummary="Combine category IDs for each observation into one learned vector. Compare fixed groups and groups of different sizes, repeated items, empty groups, item weights, mean, sum, max and padding. An offset marks a group boundary. Build with Bag Input and Embedding bag; no code is required."
    static let compositionPracticeTasks=ModuleCompositionLearningTask.allCases
    static let compositionExampleIDs=ModuleCompositionLearningTask.allCases.map {"prepared-"+$0.rawValue}
    static let compositionPracticeSummary="Build reusable Blocks and compare private copies. Choose stored order and names, then use connections to say what work runs. Compare results, influence and saved reuse. No code is required."
    static let gradientPracticeTasks=GradientLearningTask.allCases
    static let gradientExampleIDs=GradientLearningTask.allCases.map {"prepared-"+$0.rawValue}
    static let initializationPracticeTasks=InitializationLearningTask.allCases
    static let initializationPracticeSummary="Choose the numbers learned settings start with. Compare two model layers, repeat the same draw, and inspect the first training adjustment and saved-weight reuse. A seed repeats random choices; no code is required."
    static let initializationExampleIDs=InitializationLearningTask.allCases.map {"prepared-"+$0.rawValue}
    static let checkpointPracticeTasks = CheckpointLearningTask.allCases
    static let checkpointPracticeSummary = "Keep a model’s learned settings, reuse them in a new job, or continue the same training job from its saved checkpoint. Compare actual saved settings, adjustment memory and completed passes. No code or file paths are required."
    static let lossPracticeTasks = LossLearningTask.allCases
    static let lossPracticeSummary = "Compare produced results with desired outcomes. A loss is a score that guides training: lower is better for the comparison you choose. Targets are the desired outcomes. Try class choices, independent labels, values, sequences and distances; inspect individual scores and how changing an input changes the result."

    private let categories = ["Foundations", "Data preparation", "Tensors & linear algebra", "Layers & activations", "Training & evaluation"]

    struct ComponentExampleEntry: Identifiable {
        let componentID: String
        let item: LearnCatalogItem
        let learning: any BundledComponentLearningContent
        var id: String { componentID }
    }
    /// Published bundle learning uses one discovery/rendering/launch contract.
    /// Concrete families keep their independently generated manifest records.
    static var componentExampleEntries: [ComponentExampleEntry] {
        let registry = try? ComponentRegistry.builtIn()
        func entry(_ id: String, _ authored: any BundledComponentLearningContent) -> ComponentExampleEntry {
            let baseLearning: any BundledComponentLearningContent = (BundledOptimizerComponents.entries[id] != nil || BundledSchedulerComponents.entries[id] != nil || BundledLossComponents.entries[id] != nil) ? InlineTrainLearningContent(authored: authored) : authored
            let learning:any BundledComponentLearningContent = RecurrentAuthoringConfiguration.componentIDs.contains(id) ? ReviewedRecurrentLearningContent(authored:baseLearning):baseLearning
            let lossTerms = LossLearningTask.allCases.first(where:{$0.componentID == id}).map { task in
                [task.rawValue,task.title,learning.exampleID,"loss scoring model results desired outcomes target roles class labels values sequences distance anchor match nonmatch input lengths target lengths"].joined(separator:" ")
            } ?? ""
            let item = LearnCatalogItem(id: id, title: learning.title, category: (BundledOptimizerComponents.entries[id] != nil || BundledSchedulerComponents.entries[id] != nil || BundledLossComponents.entries[id] != nil) ? "Training & evaluation" : (TensorLayoutRuntimeCapability.operations.contains(id) || BundledTensorMathComponents.entries[id] != nil || BundledTensorCompositionComponents.entries[id] != nil || BundledTensorRoutingComponents.entries[id] != nil || BundledSpatialPatchComponents.entries[id] != nil) ? "Tensors & linear algebra" : "Layers & activations",
                kind: "Example", backend: "PyTorch",
                searchText: ([id, lossTerms, AttentionLearningTask.allCases.first(where:{$0.componentID == id})?.componentTitle ?? "", RecurrentLearningTask.allCases.first(where:{$0.componentID == id})?.componentTitle ?? "", registry?.component(type: id)?.title ?? id, registry?.component(type: id)?.category ?? "", learning.title, learning.subtitle, learning.formula, learning.disclosureTitle] + learning.paragraphs).joined(separator: " "))
            return ComponentExampleEntry(componentID: id, item: item, learning: learning)
        }
        let pooling = BundledPoolingComponents.entries.compactMap { id, record -> ComponentExampleEntry? in
            guard record.published, let learning = record.learning else { return nil }
            return entry(id, learning)
        }
        let activations = BundledActivationComponents.entries.compactMap { id, record -> ComponentExampleEntry? in
            guard record.published, let learning = record.learning else { return nil }
            return entry(id, learning)
        }
        let normalization = BundledNormalizationComponents.entries.compactMap { id, record -> ComponentExampleEntry? in
            guard record.published, let learning = record.learning else { return nil }
            return entry(id, learning)
        }
        let layers = BundledLayerComponents.entries.compactMap { id, record -> ComponentExampleEntry? in
            guard record.published, let learning = record.learning else { return nil }
            return entry(id, learning)
        }
        let tensorLayout = BundledTensorLayoutComponents.entries.compactMap { id, record -> ComponentExampleEntry? in
            guard record.published, let learning = record.learning else { return nil }
            return entry(id, learning)
        }
        let tensorMath = BundledTensorMathComponents.entries.compactMap { id, record -> ComponentExampleEntry? in
            guard record.published, let learning = record.learning else { return nil }
            return entry(id, learning)
        }
        let embeddings = BundledEmbeddingComponents.entries.compactMap { id, record -> ComponentExampleEntry? in
            guard record.published, let learning = record.learning else { return nil }
            return entry(id, learning)
        }
        let convolution = BundledConvolutionComponents.entries.compactMap { id, record -> ComponentExampleEntry? in
            guard record.published, let learning = record.learning else { return nil }
            return entry(id, learning)
        }
        let tensorComposition = BundledTensorCompositionComponents.entries.compactMap {id,record -> ComponentExampleEntry? in
            guard record.published,let learning=record.learning else {return nil}
            return entry(id,learning)
        }
        let tensorRouting = BundledTensorRoutingComponents.entries.compactMap {id,record -> ComponentExampleEntry? in
            guard record.published,let learning=record.learning else {return nil}
            return entry(id,learning)
        }
        let spatialPatches = BundledSpatialPatchComponents.entries.compactMap {id,record -> ComponentExampleEntry? in
            guard record.published,let learning=record.learning else {return nil}
            return entry(id,learning)
        }
        let optimizers = BundledOptimizerComponents.entries.compactMap {id,record -> ComponentExampleEntry? in
            guard record.published,let learning=record.learning else {return nil}
            return entry(id,learning)
        }
        let attention = BundledAttentionComponents.entries.compactMap {id,record -> ComponentExampleEntry? in
            guard id != "attention_mask",record.published,let learning=record.learning else {return nil}
            return entry(id,learning)
        }
        let recurrent = BundledRecurrentComponents.entries.compactMap {id,record -> ComponentExampleEntry? in
            guard record.published,let learning=record.learning else {return nil}
            return entry(id,learning)
        }
        let schedulers = BundledSchedulerComponents.entries.compactMap {id,record -> ComponentExampleEntry? in
            guard record.published,let learning=record.learning else {return nil}
            return entry(id,learning)
        }
        let losses = BundledLossComponents.entries.compactMap {id,record -> ComponentExampleEntry? in
            guard record.published,let learning=record.learning else {return nil}
            return entry(id,learning)
        }
        return (losses + attention + recurrent + schedulers + optimizers + spatialPatches + tensorRouting + tensorComposition + pooling + activations + normalization + layers + tensorLayout + tensorMath + embeddings + convolution).sorted { $0.componentID < $1.componentID }
    }
    /// A specialized card can offer several mode choices for one migrated bundle.
    /// Keep the authored bundle discoverable through its metadata without rendering
    /// a second generic card for that same stable component identity.
    static var displayedComponentExampleEntries: [ComponentExampleEntry] {
        let specialized = Set(items.map(\.id)).union((1...3).compactMap { dimension in items.contains { $0.id == "transpose\(dimension)d" } ? "conv_transpose\(dimension)d" : nil }).union(items.contains { $0.id == "norm-example" } ? ["layer_norm"] : []).union(items.contains { $0.id == "tensor-example" } ? TensorLayoutRuntimeCapability.operations : [])
        return componentExampleEntries.filter { !specialized.contains($0.componentID) && !(items.contains { $0.id == "lazy-linear" } && $0.componentID == "lazy_linear") && !(items.contains { $0.id == "matrix-example" } && $0.componentID == "matmul") && !(items.contains { $0.id == "embedding-example" } && ["embedding", "index_input"].contains($0.componentID)) }
    }

    var body: some View {
        let matches = LearningLibraryPresentation.matching(section: section, search: search, type: typeFilter, framework: frameworkFilter)
        LazyVStack(alignment: .leading, spacing: 8) {
            if matches.isEmpty {
                Label("No matching learning material", systemImage: "magnifyingglass").font(.headline)
                Text("Try another topic, search, type or framework.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            ForEach(categories, id: \.self) { category in
                let entries = matches.filter { $0.category == category }
                if !entries.isEmpty {
                    Text(category).font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(colorScheme == .dark ? LearningLibraryPresentation.family(category).primary : LearningLibraryPresentation.family(category).secondary).padding(.top, 10)
                    LazyVStack(spacing: 0) {
                        ForEach(entries) { item in
                            LearningDiscoveryRow(item: item, reading:store.lessonReading) { openEntry(item.id) }
                        }
                    }.overlay(Rectangle().stroke(MLTheme.rule))
                }
            }
        }
    }

    @ViewBuilder func entryContent(_ id: String) -> some View {
        switch id {
        case "train-data-access": TrainDataAccessLearningView(store: store)
        case "table-output", "data02-output": TableOutputLearningView(store:store)
        case "sort": SortLearningView(store:store)
        case "unique": UniqueLearningView(store:store)
        case "reshapeTable": ReshapeTableLearningView(store:store)
        case "splitText": SplitTextLearningView(store:store)
        case "model-input": ModelInputLearningView(store:store)
        case "model-output": ModelOutputLearningView(store:store)
        case "union": UnionLearningView(store:store)
        case "summarize": SummarizeLearningView(store:store)
        case "preprocess": PreprocessLearningView(store:store)
        case "formula", "data02-calculate", "data02-regex-extract", "data02-regex-replace": FormulaLearningView(store:store)
        case "partition": PartitionLearningView(store:store)
        case "filter": FilterLearningView(store: store)
        case "text-encoding": TextEncodingLearningView(store:store)
        case "text-generation":TextGenerationLearningView(store:store)
        case "sequence-windows":SequenceWindowsLearningView(store:store)
        case "columns": ColumnsLearningView(store: store)
        case "prebuilt-models": PrebuiltModelsLearningView(store:store)
        case "document-classification": DocumentClassificationLearningView(store:store)
        case "self-play":SelfPlayLearningView(store:store)
        case "game-episodes":GameEpisodeLearningView(store:store)
        case "game-rules":GameRuleLearningView(store:store)
        case "typed-decisions":TypedDecisionLearningView(store:store)
        case "classify-audio":AudioClassificationLearningView(store:store)
        case "transcribe-audio":SpeechTranscriptionLearningView(store:store)
        case "extract-text":OCRLearningView(store:store)
        case "prepare-media": PrepareMediaLearningView(store:store)
        case "file-input": FileInputLearningView(store: store)
        case "data-preparation-course": DataPreparationLearningView(store: store)
        case "data02-expression", "data02-regex-match": DataEngineeringPracticeView(store: store, lessonID: id)
        case "spatial-practice": SpatialLearningPracticeView(store: store)
        case "iteration-transfer": IterationTransferLearningView(store: store)
        case "mixture-experts": TrainingLessonReferenceView(entryID:MixtureExpertsPractice.lessonID){Button("Open MoE practice"){_=store.loadComponentWorkflowExample(MixtureExpertsPractice.lessonID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)}
        case "fine-tuning": FineTuningLearningView(store:store)
        case "load-model": LoadModelLearningView(store:store)
        case "save-model": SaveModelLearningView(store:store)
        case "tensorflow": entry_tensorflow
        case "sklearn": entry_sklearn
        case "tensor-layout": entry_tensor_layout
        case "activation": ActivationLearningView()
        case "dropout": DropoutLearningView(store:store,tensorLearning:tensorLearning)
        case "recurrent-cell-overview": RecurrentCellOverviewLearningView()
        case "recurrent-overview": RecurrentOverviewLearningView()
        case "linear-overview": LinearOverviewLearningView()
        case "normalization-overview": NormalizationOverviewLearningView()
        case "convolution-overview": ConvolutionLearningView()
        case "pooling-overview": PoolingLearningView()
        case "softmax": entry_softmax
        case "layernorm": entry_layernorm
        case "data-loading": entry_data_loading
        case "program-export": entry_program_export
        case "embedding-bag": entry_embedding_bag
        case "embedding": entry_embedding
        case "optimizer-updates": entry_historical_train_configuration
        case "attention-connections": entry_attention_connections
        case "recurrent-memory": entry_recurrent_memory
        case "scheduler-rates": entry_historical_train_configuration
        case "module-composition": entry_module_composition
        case "module-composition-examples": entry_module_composition_examples
        case "gradient-controls": entry_historical_train_configuration
        case "gradient-examples": entry_historical_train_configuration
        case "initialization-starting-settings": entry_historical_train_configuration
        case "initialization-examples": entry_historical_train_configuration
        case "checkpoint-lifecycle": entry_checkpoint_lifecycle
        case "checkpoint-examples": entry_checkpoint_examples
        case "loss-results": entry_historical_train_configuration
        case "channel-normalization": entry_channel_normalization
        case "spatial-patches": entry_spatial_patches
        case "tensor-routing": entry_tensor_routing
        case "tensor-composition": entry_tensor_composition
        case "adaptive-pooling": entry_adaptive_pooling
        case "pooling": entry_pooling
        case "configurable-activations": entry_configurable_activations
        case "remaining-activations": entry_remaining_activations
        case "convolution": entry_convolution
        case "transposed-convolution": entry_transposedConvolution
        case "matrix": entry_matrix
        case "lazy-linear-lesson": entry_lazy_linear_lesson
        case "bilinear": entry_bilinear
        case "affine": entry_affine
        case "normalization": PreprocessLearningView(store:store)
        case "parallel": entry_parallel
        case "join": entry_join
        case "aggregate": entry_aggregate
        case "holdout": entry_holdout
        case "sigmoid-lesson": entry_sigmoid_lesson
        case "tanh-lesson": entry_tanh_lesson
        case "relu-lesson": entry_relu_lesson
        case "gelu": entry_gelu
        case "tensor-example": entry_tensor_example
        case "softmax-example": entry_softmax_example
        case "norm-example": entry_norm_example
        case "embedding-example": entry_embedding_example
        case "matrix-example": entry_matrix_example
        case "bilinear-example": entry_bilinear_example
        case "lazy-linear": entry_lazy_linear
        case "conv1d": entry_conv1d
        case "conv2d": entry_conv2d
        case "conv3d": entry_conv3d
        case "separable": entry_separable
        case "transpose3d": entry_transpose3d
        case "transpose2d": entry_transpose2d
        case "transpose1d": entry_transpose1d
        case "guided": entry_guided
        case "depthwise": entry_depthwise
        default:
            if let entry = Self.componentExampleEntries.first(where: { $0.componentID == id }) {
                BundledComponentLearningEntry(store: store, componentID: entry.componentID, learning: entry.learning)
            }
        }
    }

    @ViewBuilder private var entry_tensorflow: some View {
        TensorFlowFoundationLearningView(store: store)
    }

    @ViewBuilder private var entry_sklearn: some View {
        SklearnFoundationLearningView(store: store)
    }

    @ViewBuilder private var entry_tensor_layout: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Tensor layout lab").font(.title3.weight(.semibold))
            Text("Rearrange a table of numbers without losing values. Place Reshape for dimensions or Flatten axes, or place Transpose, then check the measured output. KwipML also checks how changes travel through the construction. Drafts and verified results are saved on this Mac.").font(.caption).foregroundStyle(MLTheme.muted)
            HStack {
                ForEach(["reshape","transpose","flatten"],id:\.self) {operation in
                    Button(operation.capitalized) {_ = tensorLearning.start(operation:operation)}.disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_softmax: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Softmax lab").font(.title3.weight(.semibold))
            Text("Turn scores into shares that add up to 1. Choose whether the numbers in each row or column compete, then inspect the output. Keep each observation separate; KwipML checks the construction for you.").font(.caption).foregroundStyle(MLTheme.muted)
            HStack {
                Button("Normalize rows") {_ = tensorLearning.start(operation:"softmax-last-axis")}
                Button("Normalize columns") {_ = tensorLearning.start(operation:"softmax-first-sample-axis")}
            }.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_layernorm: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("LayerNorm lab").font(.title3.weight(.semibold))
            Text("Bring groups of numbers onto a consistent scale. Choose whether to center and scale each row or the whole observation, then compare the measured values. Scale multiplies the result; bias adds an offset.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(LayerNormLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_data_loading:some View {
        VStack(alignment:.leading,spacing:10) {
            Text(TrainDataAccessLessonContent.legacyTitle).font(.title3.weight(.semibold))
            LessonRichText(text: TrainDataAccessLessonContent.legacyExplanation)
            Button("Open ordinary Train data access practice") { _ = store.loadComponentWorkflowExample(TrainDataAccessPracticeExample.exampleID, device: "cpu") }
                .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                .accessibilityIdentifier("learning.dataLoading.open-train-practice")

        }
    }
    @ViewBuilder private var entry_historical_train_configuration:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Training configuration · historical lesson").font(.title3.weight(.semibold))
            LessonRichText(text: "Configure the optimizer, learning-rate schedule, objective, starting values and learning controls directly on **Train**. Current workflows connect model and data to Train; Run settings are inline on Train.\n\nSaved connected-recipe lessons and their recorded assessment results remain available in **Saved lessons**. This historical page does not start a new connected-recipe lab or award completion for the current Train practice.")
            Button("Open Train configuration practice") { openEntry("train-data-access") }
                .buttonStyle(FlatButtonStyle()).accessibilityIdentifier("learning.historical-train.open-current-practice")
        }
    }
    @ViewBuilder private var entry_program_export:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Save executable computation").font(.title3.weight(.semibold))
            Text("Configure an artifact action over a supplied saved model. Check real portable reload and accepted input sizes.").font(.caption).foregroundStyle(MLTheme.muted)
            ForEach(ProgramExportLearningTask.allCases,id:\.rawValue) {task in
                Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}.accessibilityIdentifier("learning.programExport.start."+task.rawValue).disabled(store.isBusy)
            }
        }
    }

    @ViewBuilder private var entry_embedding_bag: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Grouped categories lab").font(.title3.weight(.semibold))
            Text(Self.embeddingBagPracticeSummary).font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:200),alignment:.leading)],alignment:.leading) {
                ForEach(Self.embeddingBagPracticeTasks,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title)
                        .accessibilityIdentifier("learning.embedding-bag.start."+task.rawValue).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_embedding: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Embedding lab").font(.title3.weight(.semibold))
            Text("Use whole-number labels to look up rows in a supplied table. Inspect repeated selections, an ignored padding row and limits on row size. Connect the lookup on the canvas; no code is required.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(EmbeddingLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_configurable_activations: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Individual feature responses lab").font(.title3.weight(.semibold))
            Text("An activation changes each number separately. Build a small flow, compare positive and negative values, and explore how Negative slope or Alpha changes the response. Inspect measured output and influence without writing code.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), alignment: .leading)], alignment: .leading) {
                ForEach(ConfigurableActivationLearningTask.allCases, id: \.rawValue) { task in
                    Button(task.title) { _ = tensorLearning.start(operation: task.rawValue) }
                        .accessibilityLabel("Start " + task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_remaining_activations: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Learned responses and training behavior lab").font(.title3.weight(.semibold))
            Text("Adjust learned multipliers for negative measurements, compare relative scores, and see why temporary removal belongs to training. Build and configure the blocks visually; the checker measures values and influence.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(RemainingActivationLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_attention_connections:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Sequence comparisons lab").font(.title3.weight(.semibold))
            Text("Choose Attention and Transformer tools from Layers. Configure heads, normalization and role-specific masks; connect supplied sequences and an Attention mask, then inspect captured tensors and their influence.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(AttentionLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}.accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_recurrent_memory:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Sequences and explicit memory lab").font(.title3.weight(.semibold))
            Text("Place Recurrent from Layers and choose the exercise’s RNN, LSTM or GRU method; for single-step exercises place Recurrent Cell and choose RNN Cell, LSTM Cell or GRU Cell. Read measurements in time order, carry explicit starting state, and inspect results and final memories. The checker provides learned tables and measures each influence.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(RecurrentLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}.accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_module_composition:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Build, reuse and order Blocks lab").font(.title3.weight(.semibold))
            Text(Self.compositionPracticeSummary).font(.caption).foregroundStyle(MLTheme.muted)
            LessonTextPanel(title:"Parallel features on the main canvas",text:"Place Parallel features to work inside an expandable container. Its chevron shows or hides the same components and connections. Select a child to configure it; shared fields remain on the container. Move a component across the frame to change membership; moving it out preserves its current values. Moves that would create a recursive block dependency are refused. Collapse and layout do not change a captured computation. The advanced labs below retain their dedicated exercise tabs.")
            Button("Open inline container practice") {_=store.loadPreparedWorkflowExample(device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                .accessibilityIdentifier("learning.composition.inline-practice")
            ForEach(Self.compositionPracticeTasks,id:\.rawValue) {task in
                Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                    .accessibilityIdentifier("learning.composition.start."+task.rawValue).disabled(store.isBusy)
            }
        }
    }
    @ViewBuilder private var entry_module_composition_examples:some View {
        VStack(alignment:.leading,spacing:10) {
            if !inLessonRail {
            Text("Reusable Blocks · worked examples").font(.title3.weight(.semibold))
            Text("Open connected Blocks with explicit child order and names. Inspect each private copy, then choose Check construction to measure results, influence, adjustments and saved reuse.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            ForEach(Self.compositionPracticeTasks,id:\.rawValue) {task in
                Button(task.title+" · worked example") {_ = tensorLearning.start(operation:"prepared-"+task.rawValue,mode:.guided)}
                    .accessibilityIdentifier("learning.composition.example.prepared-"+task.rawValue).disabled(store.isBusy)
            }
        }
    }

    @ViewBuilder private var entry_checkpoint_examples:some View {
        VStack(alignment:.leading,spacing:10) {
            if !inLessonRail {
            Text("Save and reopen · worked examples").font(.title3.weight(.semibold))
            Text("Open a configured exercise, then choose Check construction to create a real saved parent and measure its reuse. Explore the settings and compare the actual result tables.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            ForEach(CheckpointLearningExamples.all) {example in
                Button(example.title) {_ = tensorLearning.start(operation:example.id,mode:.guided)}
                    .accessibilityIdentifier("learning.checkpoint.example."+example.id).disabled(store.isBusy)
            }
        }
    }

    @ViewBuilder private var entry_checkpoint_lifecycle:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Saved runs and continuation lab").font(.title3.weight(.semibold))
            Text(Self.checkpointPracticeSummary).font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(Self.checkpointPracticeTasks,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).accessibilityIdentifier("learning.checkpoint.start."+task.rawValue).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_channel_normalization: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Channel patterns and remembered summaries lab").font(.title3.weight(.semibold))
            Text("Center numbers and adjust their spread. Compare channels across a group, within each observation, or in teams; then use remembered summaries for new observations.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(StatefulNormalizationLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}.accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_spatial_patches: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Patch columns and overlapping grids lab").font(.title3.weight(.semibold))
            Text("Turn small grid rectangles into columns, change their spacing and rebuild grids by adding overlapping parts. Build visually and compare measured output and input influence; uncovered positions stay zero.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(SpatialPatchLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_tensor_routing: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Parts, positions and table arrangements lab").font(.title3.weight(.semibold))
            Text("Divide a table into named parts, select repeated positions or combine tables using arrangement presets. Build visually and inspect measured output and number-source influence.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(TensorRoutingLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_tensor_composition: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Combine and reorder arrays lab").font(.title3.weight(.semibold))
            Text("Join sources, reuse a pattern, keep sources in a new direction or reorder directions. Configure the blocks visually and compare measured values and influence.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(TensorCompositionLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_adaptive_pooling: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Choose regions and output sizes lab").font(.title3.weight(.semibold))
            Text("Average a rectangular window or choose how many output positions you want. Inspect shared regions, boundary counts and winning positions. Build and configure each flow on the canvas without writing code.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), alignment: .leading)], alignment: .leading) {
                ForEach(AdaptivePoolingLearningTask.allCases, id: \.rawValue) { task in
                    Button(task.title) { _ = tensorLearning.start(operation: task.rawValue) }
                        .accessibilityLabel("Start " + task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_pooling: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Window selection and averages lab").font(.title3.weight(.semibold))
            Text("Move a small window along a line, across a grid or through a volume. Select its largest value or calculate its average, then inspect which input positions contribute. Build the connections and configure boundary behavior without writing code.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), alignment: .leading)], alignment: .leading) {
                ForEach(PoolingLearningTask.allCases, id: \.rawValue) { task in
                    Button(task.title) { _ = tensorLearning.start(operation: task.rawValue) }
                        .accessibilityLabel("Start " + task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_convolution: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Shared-window convolution lab").font(.title3.weight(.semibold))
            Text("On the canvas, place Convolution and choose Standard with the lesson’s Dimensions.").font(.caption).foregroundStyle(ComponentVisualFamily.layers.secondary)
            Text("Move a small window along a line, across a grid or through a volume. Reuse the same weights at every position, combine separate measurement channels and inspect the output. The lesson supplies the weights; build and configure the connections without writing code.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(ConvolutionLearningTask.allCases.filter { !$0.isTransposed },id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_transposedConvolution: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Overlapping output windows lab").font(.title3.weight(.semibold))
            Text("On the canvas, place Convolution and choose Transposed with the lesson’s Dimensions.").font(.caption).foregroundStyle(ComponentVisualFamily.layers.secondary)
            Text("Spread each input through a shared window into a larger output. Add contributions where windows overlap, and inspect how window spacing, output extent and bias affect the result. Build and configure the connections without writing code.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), alignment: .leading)], alignment: .leading) {
                ForEach(ConvolutionLearningTask.allCases.filter { $0.isTransposed }, id: \.rawValue) { task in
                    Button(task.title) { _ = tensorLearning.start(operation: task.rawValue) }
                        .accessibilityLabel("Start " + task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_matrix: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Matrix multiplication lab").font(.title3.weight(.semibold))
            Text("Combine table rows with table columns by multiplying matching pairs and adding them. Connect two inputs, inspect the result and explore when one table is reused across several groups.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(MatrixLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_lazy_linear_lesson: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Let a layer infer its feature width").font(.title3.weight(.semibold))
            LessonRichText(text:"Place **Linear** from Layers and choose **Infer from input** in Configuration. Build a layer for two groups of three measurements and produce four results per group. Inspect the inferred width and created weights, then verify that repeating and reloading keep the same state. This is an independently measured shape/state exercise; it does not train a predictor.")
            Button("Start inferred-width lesson") {_ = tensorLearning.start(operation:LazyLinearAssessment.operation)}.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_bilinear: some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Bilinear interactions lab").font(.title3.weight(.semibold))
            Text("In a workflow, place Bilinear from Layers. Configuration sets the independent a/b feature widths, output features and Use bias; Weights and biases sets named parameter policies. The exercises below expose supplied weights so you can study the pairwise calculation directly.").font(.caption).foregroundStyle(MLTheme.muted)
            LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                ForEach(BilinearLearningTask.allCases,id:\.rawValue) {task in
                    Button(task.title) {_ = tensorLearning.start(operation:task.rawValue)}
                        .accessibilityLabel("Start "+task.title).disabled(store.isBusy)
                }
            }
        }
    }

    @ViewBuilder private var entry_affine: some View {
        AffineLearningCourseCard(lesson: affineLearning, workspace: store)
    }

    @ViewBuilder private var entry_parallel: some View {
        HStack(alignment: .top) {
            if !inLessonRail {
            VStack(alignment: .leading, spacing: 8) {
                Text("Parallel classifier").font(.title3.weight(.semibold))
                Text("Two learnable branches combine to classify bundled signal observations. Includes CSV preparation, a held-out split and training-only scaling on one canvas.").font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                Text("Model input settings · Linear / Linear → ReLU / Tanh → Concat → Linear → Model Output").font(.system(size: 11, design: .monospaced)).foregroundStyle(MLTheme.teal)
            }
            }
            Spacer()
            Button("Open example") { store.loadPreparedWorkflowExample() }.disabled(store.isBusy)
                .accessibilityLabel("Open parallel classifier example")
        }
    }

    @ViewBuilder private var entry_join: some View {
        HStack(alignment:.top) {
            if !inLessonRail {
                VStack(alignment:.leading,spacing:8) {
                    Text("Join observations and train").font(.title3.weight(.semibold))
                    LessonRichText(text:DataExampleLessonCopy.joinedIntroduction)
                }
            }
            Spacer()
            Button("Open joined example") {store.loadJoinedWorkflowExample()}.disabled(store.isBusy)
                .accessibilityLabel("Open joined classifier example")
        }
    }

    @ViewBuilder private var entry_aggregate: some View {
        HStack(alignment:.top) {
            if !inLessonRail {
                VStack(alignment:.leading,spacing:8) {
                    Text("Summarize repeated measurements").font(.title3.weight(.semibold))
                    LessonRichText(text:DataExampleLessonCopy.aggregatedIntroduction)
                }
            }
            Spacer()
            Button("Open aggregation example") {store.loadAggregatedWorkflowExample()}.disabled(store.isBusy)
                .accessibilityLabel("Open aggregated classifier example")
        }
    }

    @ViewBuilder private var entry_holdout: some View {
        HStack(alignment:.top) {
            if !inLessonRail {
            VStack(alignment:.leading,spacing:8) {
                Text("Protect subjects and future observations").font(.title3.weight(.semibold))
                Text("Keep repeated measurements from each subject in one partition, or use explicit chronological cutoffs for dated independent observations. Training statistics are fitted only on training rows.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                Text("Source → group or chronological split → training-only scaling → classifier").font(.system(size:11,design:.monospaced)).foregroundStyle(MLTheme.teal)
                Text("Train using training and validation only, then choose the saved checkpoint in Saved checkpoint and run Evaluate with Test selected. Its protected test data never trains the model; Validation is available for model comparisons.").font(.caption).foregroundStyle(MLTheme.muted)
                DisclosureGroup("Evaluation: workflow and mathematics") { EvaluationGuideView().padding(.top, 8) }
                    .accessibilityIdentifier("learning.evaluation.guide")
            }
            }
            Spacer()
            VStack(alignment:.trailing,spacing:8) {
                Button("Open subject holdout example") {store.loadGroupSplitWorkflowExample()}.accessibilityLabel("Open group partition classifier example")
                Button("Open chronological example") {store.loadTemporalSplitWorkflowExample()}.accessibilityLabel("Open chronological partition classifier example")
            }.disabled(store.isBusy)
        }
    }

    private var entry_sigmoid_lesson: some View {
        activationConstructionLesson(identity: .sigmoid, title: "Transform values with Sigmoid", text: "Turn each incoming number into a smooth value between zero and one. Build the connections, preserve feature order and inspect measured outputs. **Choose Sigmoid in Activation’s Function setting.** This function has no additional computation settings. These transformed values do not automatically tell you how likely an outcome is.")
    }
    private var entry_tanh_lesson: some View {
        activationConstructionLesson(identity: .tanh, title: "Transform values with Tanh", text: "Turn each incoming number into a smooth signed value between minus one and one. Zero stays zero. Build the connections, preserve feature order and inspect measured outputs. **Choose Tanh in Activation’s Function setting.** This function has no additional computation settings. Each value changes independently; rows do not become probabilities.")
    }
    private var entry_relu_lesson: some View {
        activationConstructionLesson(identity: .relu, title: "Transform values with ReLU", text: "Replace each negative number with zero and keep zero and positive numbers unchanged. Build the connections, preserve feature order and inspect measured outputs. **ReLU has no computation settings in this adapter.** Other branches keep their original inputs. Applying ReLU twice gives the same result.")
    }
    private func activationConstructionLesson(identity: AffineLessonIdentity, title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.title3.weight(.semibold))
            LessonRichText(text: text)
            Text("Guided construction · independent practice · assessed outputs")
                .font(.system(size: 11)).foregroundStyle(MLTheme.teal)
            if affineLearning.isActive && affineLearning.lessonTask?.identity == identity {
                Button("Resume lesson") { store.activeSection = "Learn" }
            } else {
                Button("Start lesson") {
                    affineLearning.start(identity: identity, mode: .guided)
                    if affineLearning.isActive { store.activeSection = "Learn" }
                }.disabled(store.isBusy || affineLearning.isActive)
                    .help(affineLearning.isActive ? "Leave the active lesson before starting another project." : "Start a new managed construction lesson")
            }
            Text("Checking Guided or Practice does not award assessment completion. The classifier example is separate from this forward-transformation task.")
                .font(.caption).foregroundStyle(MLTheme.muted)
        }
    }

    @ViewBuilder private var entry_gelu: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 8) {
                if !inLessonRail {
                Text("GELU classifier").font(.title3.weight(.semibold))
                Text("A sequential two-feature classifier with a smooth Gaussian gate. Train on bundled data, inspect activations, and export the model and weights.").font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                Text("Model input settings · Linear 2→8 → GELU → Linear 8→2 → Model Output").font(.system(size: 11, design: .monospaced)).foregroundStyle(MLTheme.teal)
                Text("Choose a starting mode; both remain editable in the activation's configuration.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 8) {
                Button("Open exact GELU example") { store.loadGELUWorkflowExample() }
                    .accessibilityLabel("Open exact GELU example")
                Button("Open approximate GELU example") { store.loadGELUWorkflowExample(approximateTanh:true) }
                    .accessibilityLabel("Open approximate GELU example")
            }.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_tensor_example: some View {
        HStack(alignment:.top) {
            if !inLessonRail {
            VStack(alignment:.leading,spacing:8) {
                Text("Tensor layout classifier").font(.title3.weight(.semibold))
                Text("Project twelve features, reshape to 3 × 4, transpose to 4 × 3, then flatten before the classifier. Inspect how coordinates and gradients move; these operations add no learned parameters or nonlinear capacity.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                Text("Practice each operation in Tensor layout lab above, then edit this trainable flow and export its code and weights.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            }
            Spacer()
            Button("Open tensor layout example") {store.loadTensorLayoutWorkflowExample()}.disabled(store.isBusy)
                .accessibilityLabel("Open tensor layout example")
        }
    }

    @ViewBuilder private var entry_softmax_example: some View {
        HStack(alignment:.top) {
            VStack(alignment:.leading,spacing:8) {
                if !inLessonRail {
                Text("Softmax axis classifier").font(.title3.weight(.semibold))
                Text("Arrange six learned scores into two groups of three, then normalize each group. Change the axis to compare which values compete. A final Linear layer produces class scores for training.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                Text("Softmax couples values within a slice; it is not an elementwise activation or a general fix for exploding gradients.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            Spacer()
            Button("Open Softmax example") {store.loadSoftmaxWorkflowExample()}.disabled(store.isBusy)
                .accessibilityLabel("Open Softmax example")
        }
    }

    @ViewBuilder private var entry_norm_example: some View {
        HStack(alignment:.top) {
            VStack(alignment:.leading,spacing:8) {
                if !inLessonRail {
                Text("LayerNorm scope comparison").font(.title3.weight(.semibold))
                Text("Compare normalization within each group of three learned features with normalization across all six. Both preserve observations and learn coordinatewise scale and bias before GELU and the classifier.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                Text("Preprocess learns training-set scaling statistics; LayerNorm recomputes statistics within each observation. Inspect and edit both on the same canvas.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            Spacer()
            VStack {
                Button("Open grouped LayerNorm example") {store.loadLayerNormWorkflowExample()}
                Button("Open whole-sample LayerNorm example") {store.loadLayerNormWorkflowExample(normalizeWholeSample:true)}
            }.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_embedding_example: some View {
        HStack(alignment:.top,spacing:24) {
            VStack(alignment:.leading,spacing:8) {
                if !inLessonRail {
                Text("Categorical embedding classifier").font(.title3.weight(.semibold))
                Text("Map integer categories to learned vectors, flatten the paired representation and train a classifier. Inspect exact category IDs and held-out predictions in the bottom Results dock.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                Text("Synthetic teaching data repeats category combinations; it demonstrates shared lookup weights, not generalization to unseen categories. The Embedding lab above explores the underlying gather and scatter-add mathematics.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            Spacer()
            Button("Open embedding example") {store.loadEmbeddingWorkflowExample()}.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_matrix_example: some View {
        HStack(alignment:.top,spacing:24) {
            if !inLessonRail {
            VStack(alignment:.leading,spacing:8) {
                Text("Matrix interaction classifier").font(.title3.weight(.semibold))
                Text("Learn two matrix operands from the same observations. Multiply across two heads with a shared right operand, then combine their features into class scores. Edit shapes and operand connections on Build.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                Text("A matrix interaction example, not a complete attention architecture. The Matrix multiplication lab explores both operand gradients.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            }
            Spacer()
            Button("Open matrix example") {store.loadMatrixWorkflowExample()}.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_bilinear_example: some View {
        HStack(alignment:.top,spacing:24) {
            VStack(alignment:.leading,spacing:8) {
                if !inLessonRail {
                Text("Bilinear interaction classifier").font(.title3.weight(.semibold))
                Text("Project observations into separate 3- and 5-feature branches. A learned [2, 3, 5] weight tensor combines their pairwise interactions into two class scores. Edit First input features · a, Second input features · b, Output features and Use bias in Configuration. Inspect named parameter policies under Weights and biases.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                Text("Leading sample dimensions must match exactly. This example uses 30 interaction weights and 2 biases; increasing either input width multiplies the parameter cost.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            Spacer()
            Button("Open bilinear example") {store.loadBilinearWorkflowExample()}.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_lazy_linear: some View {
        HStack(alignment:.top,spacing:24) {
            VStack(alignment:.leading,spacing:8) {
                if !inLessonRail {
                Text("Inferred-width classifier").font(.title3.weight(.semibold))
                Text("Project two measurements into six features, reshape to [N, 2, 3], then let Lazy Linear infer three input features and produce four per position. Flatten to eight features; a second Lazy Linear infers eight and produces two class scores.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                }
                DisclosureGroup("Understand inferred dimensions and initialization") {
                    VStack(alignment:.leading,spacing:8) {
                        Text("Y[n, s, o] = Σᵢ X[n, s, i] W[o, i] + b[o]").font(.system(size:12,design:.monospaced)).textSelection(.enabled)
                        Text("The first lazy layer shares W[4, 3] and b[4] across observations and both positions: 16 parameters. Flatten preserves values and exposes width 8; the final layer has 2 × 8 + 2 = 18 parameters. The earlier projection adds 18, totaling 52 with bias enabled.")
                        Text("Lazy describes when input width becomes known, not a different affine operation. These projections and reshapes still compose into an affine model. To learn nonlinear boundaries, add an activation and compare held-out results.")
                        Text("KwipML resolves the connected shape and initializes lazy weights on the selected device before creating the optimizer. Eager layers initialize first, so an equivalent all-eager model need not start with identical weights under the same seed. Changing upstream width requires compatible new state; trained weights never resize silently.")
                        Text("Try changing the first output width from 4 to 5: flatten becomes width 10 and the final input adapts. Inspect both connections and Code before running. Undo restores the earlier architecture. Toggle Use bias to change which parameter tensors exist.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top,6)
                }
            }
            Spacer()
            Button("Open inferred-width example") {store.loadLazyLinearWorkflowExample()}.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_conv1d: some View {
        HStack(alignment:.top,spacing:24) {
            VStack(alignment:.leading,spacing:8) {
                if !inLessonRail {
                Text("Grouped convolution classifier").font(.title3.weight(.semibold))
                Text("Project two measurements into fourteen features, reshape to [N, 2, 7], and apply four shared kernels in two channel groups. ReLU adds a nonlinear response; flatten exposes 28 features for the final two class scores.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                }
                DisclosureGroup("Understand shared kernels, groups and output length") {
                    VStack(alignment:.leading,spacing:8) {
                        Text("Y[n,o,t] = b[o] + Σᵢ Σₖ W[o,i,k] Xpad[n, g·(Cin/G)+i, t·s+k·d]").font(.system(size:12,design:.monospaced)).textSelection(.enabled)
                        Text("For output channel o, g = floor(o / (Cout/G)); i runs over Cin/G input channels and k over K kernel positions. PyTorch uses cross-correlation: the kernel is not reversed. The same W applies at every position and to every observation, so the number of positions does not multiply the parameter count.")
                        Text("Lout = floor((L + Pleft + Pright − d·(K−1) − 1) / s) + 1").font(.system(size:12,design:.monospaced)).textSelection(.enabled)
                        Text("Here L=7, K=3, stride s=1 and dilation d=1. Same padding adds one zero on each side and keeps seven positions. With two groups, each input channel feeds two output channels: W has shape [4, 1, 3], with 12 weights and 4 optional biases. Projection adds 42 parameters and the final classifier adds 58, totaling 116 with convolution bias enabled.")
                        Text("These positions are constructed by a learned projection of ordinary measurements. This is an architectural demonstration of local shared computation, not evidence that the bundled observations contain time-series structure. Real sequence inputs need a meaningful ordering and a suitable preparation flow.")
                        Text("Try groups=1: each kernel sees both channels and its weight shape becomes [4, 2, 3]. Compare parameter counts and held-out results. Changing padding or stride can change length; inspect the connector shape and update the final Linear input width after flatten. Toggle Use bias in Convolution to remove four convolution parameters. Changed architecture requires a compatible checkpoint and a fresh preview cache.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top,6)
                }
            }
            Spacer()
            Button("Open grouped convolution example") {store.loadConv1dWorkflowExample()}.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_conv2d: some View {
        HStack(alignment:.top,spacing:24) {
            VStack(alignment:.leading,spacing:8) {
                if !inLessonRail {
                Text("Spatial convolution on a constructed grid").font(.title3.weight(.semibold))
                Text("Project two measurements into 40 features, reshape to [N, 2, 4, 5], and apply four grouped kernels of height 2 and width 3. ReLU and flatten produce 80 responses for the final classifier.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                }
                DisclosureGroup("Understand spatial axes, shared weights and padding") {
                    VStack(alignment:.leading,spacing:8) {
                        Text("Y[n,o,h,w] = b[o] + Σᵢ Σᵤ Σᵥ W[o,i,u,v] Xpad[n,g·(Cin/G)+i,h·sH+u·dH,w·sW+v·dW]").font(.system(size:12,design:.monospaced)).textSelection(.enabled)
                        Text("Each output channel belongs to group g = floor(o/(Cout/G)). The kernel slides across height and width without reversing: this is cross-correlation. Its weights are shared across all spatial positions and observations, so larger grids add computation, not more kernel parameters.")
                        Text("For each axis: Lout = floor((L + Pbefore + Pafter − d·(K−1) − 1)/s) + 1.").font(.system(size:12,design:.monospaced)).textSelection(.enabled)
                        Text("Same padding with stride [1,1] keeps the 4×5 grid. The even height-2 kernel adds zero rows above and one below; width-3 adds one column on each side. Groups=2 gives weight shape [4,1,2,3]: 24 weights and four optional biases. Flatten turns [N,4,4,5] into [N,80].")
                        Text("This grid is constructed from ordinary measurements by a learned projection; it is not an image source or evidence of natural spatial structure. Real image inputs require meaningful channel and pixel ordering, consistent scaling and an appropriate preparation flow.")
                        Text("Try groups=1 and compare 48 kernel weights with the original 24. Change stride or padding, inspect the resulting connector shapes, then update the classifier width after flatten. Spatial settings always use [height,width] order. Architecture changes require compatible weights and refreshed previews.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top,6)
                }
            }
            Spacer()
            Button("Open spatial convolution example") {store.loadConv2dWorkflowExample()}.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_conv3d: some View {
        HStack(alignment:.top,spacing:24) {
            VStack(alignment:.leading,spacing:8) {
                if !inLessonRail {
                Text("Volume convolution on constructed features").font(.title3.weight(.semibold))
                Text("Project two measurements into 48 features, reshape to [N, 2, 2, 3, 4], and apply four grouped kernels of shape [2, 2, 3]. ReLU and flatten expose 96 responses to the final classifier.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                }
                DisclosureGroup("Understand depth, height, width and shared volume kernels") {
                    VStack(alignment:.leading,spacing:8) {
                        Text("Y[n,o,z,y,x] = b[o] + Σᵢ Σᵤ Σᵥ Σ𝓌 W[o,i,u,v,w] Xpad[n,g·(Cin/G)+i,z·sD+u·dD,y·sH+v·dH,x·sW+w·dW]").font(.system(size:12,design:.monospaced)).textSelection(.enabled)
                        Text("This is three-dimensional cross-correlation: kernels are not reversed. For output channel o, g = floor(o/(Cout/G)); each group sees Cin/G input channels. The same kernel weights apply to every location and observation. Depth is a spatial axis here, not a batch or channel axis.")
                        Text("For each axis: Lout = floor((L + Pbefore + Pafter − d·(K−1) − 1)/s) + 1.").font(.system(size:12,design:.monospaced)).textSelection(.enabled)
                        Text("All spatial settings use [depth,height,width]. Same padding with stride [1,1,1] keeps the 2×3×4 volume. Kernel[2,2,3] adds zero front/one back, zero top/one bottom, and one left/one right. Groups=2 yields W[4,1,2,2,3]: 48 weights and four optional biases. Flatten converts [N,4,2,3,4] to [N,96].")
                        Text("These volumes are learned constructions from ordinary measurements, not medical scans or video frames. Real volumetric inputs require meaningful axis ordering, consistent spacing and scaling, and an appropriate preparation flow. This example demonstrates architecture mechanics, not domain suitability.")
                        Text("Try groups=1: the kernel weight count doubles to 96. Change one stride or dilation entry and inspect which spatial dimension changes; update the final Linear width after flatten. Larger volumes increase computation and padding memory even when the shared weight count is unchanged.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top,6)
                }
            }
            Spacer()
            Button("Open volume convolution example") {store.loadConv3dWorkflowExample()}.disabled(store.isBusy)
        }
    }

    @ViewBuilder private var entry_separable: some View {
        SeparableConv2dLearningEntry(store: store)
    }

    @ViewBuilder private var entry_transpose3d: some View {
        ConvTranspose3dLearningEntry(store: store)
    }

    @ViewBuilder private var entry_transpose2d: some View {
        ConvTranspose2dLearningEntry(store: store)
    }

    @ViewBuilder private var entry_transpose1d: some View {
        ConvTranspose1dLearningEntry(store: store)
    }

    @ViewBuilder private var entry_guided: some View {
        ForEach(GuidedLessonCopy.overview) {step in
            LessonTextPanel(title:step.title,text:step.blocks[0].text)
        }
        Button("Start guided build in canvas") { store.startCourse() }.disabled(store.isBusy)
            .accessibilityLabel("Start guided build in canvas")
    }

    @ViewBuilder private var entry_depthwise: some View {
        DepthwiseConv2dLearningEntry(store: store)
    }

    private static let baseItems: [LearnCatalogItem] = [
        .init(id: "train-data-access", title: TrainDataAccessLessonContent.title, category: "Training & evaluation", kind: "Practice", backend: "PyTorch", searchText: "Train configuration optimizer learning rate scheduler objective loss target mapping scalarization initialization gradients trainability data access batches Indexed rows Tensor tuple Selected indices Finite stream row identity alignment batch size mixing shuffle seed drop short group reader processes prefetch timeout collation prepared tensors materialized editable unscored practice"),
        .init(id:"program-export",title:"Executable computation lab",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:ProgramExportLearningTask.allCases.map { $0.rawValue+" "+$0.title }.joined(separator:" ")+" program export reload input sizes static bounded"),
        .init(id:"embedding-bag",title:"Grouped categories lab",category:"Layers & activations",kind:"Practice",backend:"PyTorch",searchText:embeddingBagPracticeSummary+" "+EmbeddingBagLearningTask.allCases.map(\.rawValue).joined(separator:" ")+" bag_input embedding_bag offsets real source weights"),
        .init(id:"module-composition",title:"Build, reuse and order Blocks lab",category:"Layers & activations",kind:"Practice",backend:"PyTorch",searchText:([compositionPracticeSummary,"Module Sequential ModuleList ModuleDict stored order connected order names private copies influence guided practice assessment"]+compositionPracticeTasks.map {$0.rawValue+" "+$0.title}).joined(separator:" ")),
        .init(id:"module-composition-examples",title:"Reusable Blocks worked examples",category:"Layers & activations",kind:"Example",backend:"PyTorch",searchText:([compositionPracticeSummary,"Module Sequential ModuleList ModuleDict names stored order connected work private copies influence"]+compositionExampleIDs+compositionPracticeTasks.map {$0.title}).joined(separator:" ")),

        .init(id:"gradient-controls",title:"Influence and learned settings lab",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:(["gradient trainability influence accumulation reset frozen trainable no_grad eval guided practice private"]+gradientPracticeTasks.map {$0.title+" "+$0.rawValue}).joined(separator:" ")),
        .init(id:"gradient-examples",title:"Influence and learned settings worked examples",category:"Training & evaluation",kind:"Example",backend:"PyTorch",searchText:(gradientExampleIDs+gradientPracticeTasks.map {$0.title}).joined(separator:" ")),
        .init(id:"initialization-starting-settings",title:"Starting learned settings lab",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:([initializationPracticeSummary,"initialization Xavier Kaiming Normal Zeros seed gain fan slope guided practice assessment"]+initializationPracticeTasks.map {[$0.rawValue,$0.title].joined(separator:" ")}).joined(separator:" ")),
        .init(id:"initialization-examples",title:"Starting settings worked examples",category:"Training & evaluation",kind:"Example",backend:"PyTorch",searchText:([initializationPracticeSummary]+initializationExampleIDs+initializationPracticeTasks.map { $0.title }).joined(separator:" ")),

        .init(id:"checkpoint-examples",title:"Save and reopen · worked examples",category:"Training & evaluation",kind:"Example",backend:"PyTorch",searchText:([checkpointPracticeSummary,"checkpoint weights save load restore fine-tune exact resume torch.save torch.load"]+CheckpointLearningExamples.all.map {[$0.id,$0.task.rawValue,$0.title].joined(separator:" ")}).joined(separator:" ")),
        .init(id:"checkpoint-lifecycle",title:"Saved runs and continuation lab",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:([checkpointPracticeSummary,"checkpoint weights save load restore fine-tune exact resume torch.save torch.load guided practice assessment"]+checkpointPracticeTasks.map {[$0.rawValue,$0.title].joined(separator:" ")}).joined(separator:" ")),
        .init(id:"loss-results",title:"Scoring model results lab",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:([lossPracticeSummary,"loss target desired outcomes CrossEntropyLoss BCEWithLogitsLoss MSELoss L1Loss NLLLoss KLDivLoss CTCLoss TripletMarginLoss class label value sequence distance guided practice assessment"]+lossPracticeTasks.map {[$0.componentID,$0.rawValue,$0.title].joined(separator:" ")}).joined(separator:" ")),
        .init(id:"attention-connections",title:"Sequence comparisons lab",category:"Layers & activations",kind:"Practice",backend:"PyTorch",searchText:"MultiheadAttention Transformer TransformerEncoder TransformerDecoder scaled dot product attention query key value mask head sequence influence guided practice assessment"),
        .init(id:"recurrent-memory",title:"Sequences and explicit memory lab",category:"Layers & activations",kind:"Practice",backend:"PyTorch",searchText:"RNN LSTM GRU RNNCell LSTMCell GRUCell sequence state hidden cell gate time supplied memory influence guided practice assessment"),
        .init(id:"scheduler-rates",title:"Changing adjustment sizes lab",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"scheduler learning rate StepLR cosine LinearLR warmup plateau OneCycle epoch batch checkpoint resume guided practice assessment"),
        .init(id:"optimizer-updates",title:"Learned settings and training updates lab",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"optimizer SGD Adam AdamW RMSprop Adagrad momentum running averages separate decay squared memory accumulated steps learning rate gradients training recipe state updates guided practice assessment"),
        .init(id:"channel-normalization",title:"Channel patterns and remembered summaries lab",category:"Layers & activations",kind:"Practice",backend:"PyTorch",searchText:"batch_norm1d batch_norm2d instance_norm1d instance_norm2d group_norm channels normalization current group independent observations remembered summaries running statistics scale shift influence guided practice assessment"),
        .init(id:"spatial-patches",title:"Patch columns and overlapping grids lab",category:"Tensors & linear algebra",kind:"Practice",backend:"PyTorch",searchText:"spatial patches unfold fold extraction packed columns overlap addition uncovered zeros grid borders patch size stride dilation padding measured input influence no code guided practice assessment"),
        .init(id:"tensor-routing",title:"Parts, positions and table arrangements lab",category:"Tensors & linear algebra",kind:"Practice",backend:"PyTorch",searchText:"split chunk gather einsum named parts sections uneven repeated positions selectors diagonal multiply table arrangements preset observations output influence no code guided practice assessment"),
        .init(id:"tensor-composition",title:"Combine and reorder arrays lab",category:"Tensors & linear algebra",kind:"Practice",backend:"PyTorch",searchText:"tensor composition concat concatenate add broadcasting shared pattern stack new direction permute reorder observations sources measured influence no code guided practice assessment"),
        .init(id: "adaptive-pooling", title: "Choose regions and output sizes lab", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "adaptive pooling AvgPool2d AdaptiveAvgPool1d AdaptiveAvgPool2d AdaptiveMaxPool2d averages bins output height width length keep input custom divisor indices overlap measured influence guided private assessment no code"),
        .init(id: "configurable-activations", title: "Individual feature responses lab", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "SiLU LeakyReLU ELU Negative slope Alpha activation individual signed values shape measured influence guided private assessment no code"),
        .init(id:"remaining-activations",title:"Learned responses and training behavior lab",category:"Layers & activations",kind:"Practice",backend:"PyTorch",searchText:"PReLU learned slopes channels LogSoftmax relative scores probability logarithm Dropout masks training evaluation seeds measured influence no code guided practice assessment"),
        .init(id: "pooling", title: "Window selection and averages lab", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "pooling MaxPool1d MaxPool2d MaxPool3d AvgPool1d window winner indices averages stride padding partial boundaries channels measured influence guided private assessment no code"),
        .init(id: "transposed-convolution", title: "Overlapping output windows lab", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "transposed convolution ConvTranspose1d ConvTranspose2d ConvTranspose3d shared windows scatter overlap channels stride padding output extent bias measured influence guided private assessment no code"),
        .init(id:"convolution",title:"Shared-window convolution lab",category:"Layers & activations",kind:"Practice",backend:"PyTorch",searchText:"convolution Conv1d Conv2d Conv3d shared local windows channels line grid volume kernel weights bias measured output input influence guided private assessment no code"),
        .init(id:"lazy-linear-lesson",title:"Infer feature width with LazyLinear",category:"Layers & activations",kind:"Practice",backend:"PyTorch",searchText:"LazyLinear inferred input width output shape weights initialization seed repeat reload state preserved assessment"),
        .init(id: "sigmoid-lesson", title: "Transform values with Sigmoid", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "sigmoid smooth elementwise independent values saturation construction guided practice assessed outputs feature order zero one"),
        .init(id: "tanh-lesson", title: "Transform values with Tanh", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "tanh smooth elementwise independent values saturation construction guided practice assessed outputs feature order signed zero minus one saturation"),
        .init(id: "relu-lesson", title: "Transform values with ReLU", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "relu smooth elementwise independent values saturation construction guided practice assessed outputs feature order negative zero positive clipping independent out-of-place"),
        .init(id:"mixture-experts",title:"Route observations through experts",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"MoE mixture experts router top k masks balancing conditional graph training routing output"),
        .init(id:"fine-tuning",title:"Fine-tune a loaded model",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"Train loaded GPT-2 Llama full selected layers LoRA adapter tokenizer save merge continuation"),
        .init(id:"load-model",title:"Load and inspect a model",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"Load model Saved run local folder Hugging Face pinned snapshot safetensors tokenizer token mask bounded generation export"),
        .init(id:"save-model",title:"Save a trained model",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"Save model trained model Saved checkpoint portable package checkpoint pt private destination export input contract completed run validation test unscored"),
        .init(id: "iteration-transfer", title: "Iterate, compare and reuse your work", category: "Training & evaluation", kind: "Guided", backend: "PyTorch", searchText: "iteration iterate Train run settings device batch epoch seed limits experiment grid history checkpoint resume fine tune export inference preparation reusable Block subflow files pinned revision diagnostics repair unscored practice"),
        .init(id: "spatial-practice", title: "Moving windows and local patterns", category: "Layers & activations", kind: "Guided", backend: "PyTorch", searchText: "convolution pooling transposed depthwise separable window stride padding dilation channels max average synthetic unscored practice"),
        .init(id:"union",title:"Union monthly tables in order",category:"Data preparation",kind:"Practice",backend:"Polars",searchText:"Union append stack tables inputs order name position alignment missing shared identical numeric widening duplicates"),
        .init(id:"summarize",title:"Summarize orders and join item text",category:"Data preparation",kind:"Practice",backend:"Polars",searchText:"Summarize aggregate grouping count sum mean min max join text separator newline concatenate orders"),
        .init(id:"table-output",title:"Table Output",category:"Data preparation",kind:"Practice",backend:"Portable",searchText:"Table Output complete local files CSV TSV Parquet JSON NDJSON Excel xlsx typed schema atomic replace permission destination image assets receipt unscored"),
        .init(id:"sort",title:"Sort · ordered keys",category:"Data preparation",kind:"Practice",backend:"Portable",searchText:"Sort ordered keys ascending descending stable ties missing first last Unicode casefold working population unscored"),
        .init(id:"unique",title:"Unique · retained rows and duplicates",category:"Data preparation",kind:"Practice",backend:"Portable",searchText:"Unique duplicates deduplicate compound keys first last missing equal distinct stop casefold identity partition unscored"),
        .init(id:"reshapeTable",title:"Reshape Table · Long and Wide",category:"Data preparation",kind:"Practice",backend:"Portable",searchText:"Reshape Table pivot unpivot melt Long Wide identifiers keys monthly aggregation duplicate missing expansion budget unscored"),
        .init(id:"splitText",title:SplitTextLessonContent.title,category:"Data preparation",kind:"Practice",backend:"Portable",searchText:"Split Text delimited fixed width columns rows quoted fields escape Unicode code points UTF8 bytes ordinal conversion unscored"),
        .init(id:"model-output",title:ModelOutputPracticeExample.title,category:"Foundations",kind:"Practice",backend:"PyTorch",searchText:"Model Output boundary Tensor Definition shape dtype assertions expected output per sample variable batch capture coordinates unscored"),
        .init(id:"preprocess",title:PreprocessLessonContent.title,category:"Data preparation",kind:"Practice",backend:"Polars",searchText:"Preprocess text cleanup Unicode NFC NFKC controls punctuation decimal digits whitespace trim collapse lowercase uppercase empty null fixed drop stop missing values imputation mean median most frequent encoding categories scaling training validation test fixed rules"),
        .init(id:"formula",title:FormulaLessonContent.title,category:"Data preparation",kind:"Practice",backend:"Polars",searchText:"Formula calculated columns expression regex extract replace match text captures null ordered grid Date DateTime Time temporal Parse Format timezone Extract part Difference ISO custom precision microsecond DST ambiguity invalid"),
        .init(id:"partition",title:PartitionLessonContent.title,category:"Data preparation",kind:"Practice",backend:"Polars",searchText:"Partition random stratification entity time embargo training validation test exclusions"),
        .init(id: "filter", title: FilterLessonContent.title, category: "Data preparation", kind: "Practice", backend: "Polars", searchText: "Filter rows simple conditions all any expression text pattern regex matching nonmatching missing null Boolean case search full editable unscored practice"),
        .init(id:"text-generation",title:"Generate bounded text",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"Generate text next token prompt greedy sampling top k top p temperature EOS context saved checkpoint tokenizer CPU GPU unscored"),
        .init(id:"sequence-windows",title:"Create next-token sequence windows",category:"Data preparation",kind:"Practice",backend:"Polars",searchText:"Sequence windows next token targets stride context document fragments order padding loss mask token IDs unscored practice"),
        .init(id:"model-input",title:"Where tensors enter the model",category:"Foundations",kind:"Practice",backend:"PyTorch",searchText:"Model input boundary numeric integer indices tokens named inputs batch shape dtype data tensor training mapping"),
        .init(id:"text-encoding",title:"Encode text into token sequences",category:"Data preparation",kind:"Practice",backend:"Polars",searchText:"Text encoding tokens vocabulary training heldout Unicode character word whitespace IDs named model inputs numeric sequence padding unscored practice"),
        .init(id: "columns", title: ColumnsLessonContent.title, category: "Data preparation", kind: "Practice", backend: "Polars", searchText: "Columns column grid keep select exclude reorder output name rename cast type decimal integer string text Boolean error null schema editable unscored practice"),
        .init(id:"prebuilt-models",title:"Explore behavior segments",category:"Training & evaluation",kind:"Practice",backend:"scikit-learn",searchText:"prebuilt models estimator classification regression clustering kmeans mixture random forest ridge elastic net dbscan segments profiles save reload"),
        .init(id:"document-classification",title:"Classify private documents",category:"Training & evaluation",kind:"Practice",backend:"scikit-learn",searchText:"local private documents classification TF-IDF sparse logistic multiclass labels probabilities confusion matrix save reload unseen offline"),
        .init(id:"self-play",title:"Learn from bounded self-play",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"game policy value joint training search replay checkpoint resume evaluation"),
        .init(id:"game-episodes",title:"Run bounded game episodes",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"game rules policy frozen model episodes legal mask sampling inference"),
        .init(id:"game-rules",title:"Author and test game rules",category:"Data preparation",kind:"Practice",backend:"Portable",searchText:"game rules environment state legal actions transition reward observation tests replay simulation"),
        .init(id:"typed-decisions",title:"Train typed decisions for support tickets",category:"Training & evaluation",kind:"Practice",backend:"PyTorch",searchText:"Laya ModernBERT choice ordinal yes no soft targets calibration CPU fine tuning typed decisions"),
        .init(id:"classify-audio",title:"Classify sounds in a recording",category:"Data preparation",kind:"Practice",backend:"PyTorch",searchText:"AST AudioSet audio classification sounds waveform mono windows multi label scores CPU model offline"),
        .init(id:"transcribe-audio",title:"Transcribe a private recording",category:"Data preparation",kind:"Practice",backend:"PyTorch",searchText:"Whisper audio speech transcription STT ASR waveform mono sample rate chunks source segments model offline"),
        .init(id:"extract-text",title:"Read text from scanned images",category:"Data preparation",kind:"Practice",backend:"PyTorch",searchText:"Extract text OCR English printed scanned images PDF text regions confidence geometry optional pack offline document classification"),
        .init(id:"prepare-media",title:"Prepare image pixels and audio windows",category:"Data preparation",kind:"Practice",backend:"Polars",searchText:"Prepare media image tensor PCM WAV audio waveform channels resample window mask layout dtype int64 float32 padding source preserved"),
        .init(id: "file-input", title: "Read files, documents and image pixels", category: "Data preparation", kind: "Practice", backend: "Polars", searchText: "File Input local folder directory text PDF Word DOCX RTF ODT images pixels frames metadata file identity units view sample full no OCR editable unscored practice"),
        .init(id: "data-preparation-course", title: "Prepare tables and protect evaluation", category: "Data preparation", kind: "Guided", backend: "PyTorch", searchText: "table cleanup schema cast missing sample join aggregate split entity time training validation test scaler imputer categories encoder checkpoint holdout unscored practice"),
        .init(id: "tensorflow", title: "TensorFlow foundation", category: "Training & evaluation", kind: "Example", backend: "TensorFlow", searchText: "TensorFlow foundation Training & evaluation Example TensorFlow Dense linear matrices saved model checks "),
        .init(id: "sklearn", title: "scikit-learn foundation", category: "Training & evaluation", kind: "Example", backend: "scikit-learn", searchText: "scikit-learn foundation Training & evaluation Example scikit-learn Logistic regression affine probability coefficients "),
        .init(id: "tensor-layout", title: "Tensor layout lab", category: "Tensors & linear algebra", kind: "Practice", backend: "PyTorch", searchText: "Tensor layout lab Tensors & linear algebra Practice PyTorch  Tensor layout lab Rearrange a table of numbers without losing values. Place Reshape for dimensions or Flatten axes, or place Transpose, then check the measured output. KwipML also checks how changes travel through the construction. Drafts and verified results are saved on this Mac. reshape transpose flatten "),
        .init(id:"dropout",title:"Dropout",category:"Layers & activations",kind:"Practice",backend:"PyTorch",searchText:"Conditioning Dropout drop probability chance masks training evaluation seeds repeatability elementwise gradients measured assessed"),
        .init(id:"recurrent-cell-overview",title:"Recurrent Cell",category:"Layers & activations",kind:"Guided",backend:"PyTorch",searchText:"Recurrent Cell RNN Cell LSTM Cell GRU Cell single step hidden cell explicit state weights biases"),
        .init(id:"recurrent-overview",title:"Recurrent",category:"Layers & activations",kind:"Guided",backend:"PyTorch",searchText:"Recurrent RNN LSTM GRU sequence time hidden cell explicit state layers bidirectional dropout projection"),
        .init(id:"linear-overview",title:"Linear",category:"Layers & activations",kind:"Guided",backend:"PyTorch",searchText:"Linear Lazy Linear affine input sizing explicit inferred features weights bias"),
        .init(id:"normalization-overview",title:"Normalization",category:"Layers & activations",kind:"Guided",backend:"PyTorch",searchText:"Normalization RMSNorm root mean square no centering LayerNorm BatchNorm InstanceNorm GroupNorm channels groups epsilon affine scale bias running statistics averaging momentum normalized shape"),
        .init(id:"convolution-overview",title:"Convolution",category:"Layers & activations",kind:"Guided",backend:"PyTorch",searchText:"Convolution Standard Transposed Depthwise Dimensions channels groups multiplier kernel stride dilation padding output padding learned weights bias"),
        .init(id:"pooling-overview",title:"Pooling",category:"Layers & activations",kind:"Guided",backend:"PyTorch",searchText:"Pooling Method Sizing Dimensions Max Average fixed window adaptive output stride padding dilation ceil indices divisor output size"),
        .init(id:"activation",title:"Activation",category:"Layers & activations",kind:"Guided",backend:"PyTorch",searchText:"Activation Function ReLU Leaky ReLU PReLU ELU GELU SiLU Sigmoid Tanh Softmax LogSoftmax nonlinearity negative slope approximation"),
        .init(id: "softmax", title: "Softmax lab", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "Softmax lab Layers & activations Practice PyTorch  Softmax lab Turn scores into shares that add up to 1. Choose whether the numbers in each row or column compete, then inspect the output. Keep each observation separate; KwipML checks the construction for you. Normalize rows softmax-last-axis Normalize columns softmax-first-sample-axis "),
        .init(id: "layernorm", title: "LayerNorm lab", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "LayerNorm lab Layers & activations Practice PyTorch  LayerNorm lab Bring groups of numbers onto a consistent scale. Choose whether to center and scale each row or the whole observation, then compare the measured values. Scale multiplies the result; bias adds an offset. Start  "),
        .init(id: "embedding", title: "Embedding lab", category: "Layers & activations", kind: "Practice", backend: "PyTorch", searchText: "Embedding lab Layers & activations Practice PyTorch  Embedding lab Use whole-number labels to look up rows in a supplied table. Inspect repeated selections, an ignored padding row and limits on row size. Connect the lookup on the canvas; no code is required. Start  "),
        .init(id: "matrix", title: "Matrix multiplication lab", category: "Tensors & linear algebra", kind: "Practice", backend: "PyTorch", searchText: "Matrix multiplication lab Tensors & linear algebra Practice PyTorch  Matrix multiplication lab Combine table rows with table columns by multiplying matching pairs and adding them. Connect two inputs, inspect the result and explore when one table is reused across several groups. Start  "),
        .init(id: "bilinear", title: "Bilinear interactions lab", category: "Tensors & linear algebra", kind: "Practice", backend: "PyTorch", searchText: "Bilinear interactions lab Tensors & linear algebra Practice PyTorch  Bilinear interactions lab Combine two inputs so pairs of features can influence the result. Weights control the importance of each pair. Connect the supplied inputs and weights, then inspect the measured output. Start  "),
        .init(id: "affine", title: "Affine constructions", category: "Foundations", kind: "Practice", backend: "PyTorch", searchText: "Affine constructions Foundations Practice PyTorch  Compose affine maps Recover an affine map Build the backward map Carry fitted statistics into inference Construct exact GELU Explore approximate GELU composition recovery transpose basis input gradient bias weights"),
        .init(id: "normalization", title: "Normalization constructions", category: "Foundations", kind: "Practice", backend: "PyTorch", searchText: "Normalization constructions Foundations Practice PyTorch  standardization variance mean scale training statistics measured construction"),
        .init(id: "parallel", title: "Parallel classifier", category: "Training & evaluation", kind: "Example", backend: "PyTorch", searchText: "Parallel classifier Training & evaluation Example PyTorch  Parallel classifier Two learnable branches combine to classify bundled signal observations. Includes CSV preparation, a held-out split and training-only scaling on one canvas. Input → Linear / Linear → ReLU / Tanh → Concat → Linear → Model Output Open example Open parallel classifier example "),
        .init(id: "join", title: "Join observations and train", category: "Data preparation", kind: "Example", backend: "PyTorch", searchText: "Join observations and train Data preparation Example PyTorch  Join observations and train Match two CSV tables by observation identity, then split, scale using training statistics, and train a classifier. The context table is deliberately reordered: keys, not row positions, establish matches. Two sources → one-to-one join → split → training-only scaling → classifier Use Matching keys to pair columns and Output columns to keep, order and name the joined fields. Apply, run and inspect the table in Results. This example uses independent observations; repeated entities require group-aware splitting. Open joined example Open joined classifier example "),
        .init(id: "aggregate", title: "Summarize repeated measurements", category: "Data preparation", kind: "Example", backend: "PyTorch", searchText: "Summarize repeated measurements Data preparation Example PyTorch  Summarize repeated measurements Summarize three measurements for each of 64 independent subjects before splitting. Mean signal and context become model features; each subject appears in only one partition. 192 measurements → 64 subject summaries → split → training-only scaling → classifier This fixture has consistent labels. Inspect minimum label, maximum label and measurement count in Results; these reducers do not automatically reject or repair conflicting labels in other datasets. Open aggregation example Open aggregated classifier example "),
        .init(id: "holdout", title: "Protect subjects and future observations", category: "Training & evaluation", kind: "Example", backend: "PyTorch", searchText: "Protect subjects and future observations Training & evaluation Example PyTorch  Protect subjects and future observations Keep repeated measurements from each subject in one partition, or use explicit chronological cutoffs for dated independent observations. Training statistics are fitted only on training rows. Source → group or chronological split → training-only scaling → classifier Train using training and validation only, then choose the saved checkpoint in Saved checkpoint and run Evaluate with Test selected. Its protected test data never trains the model; Validation is available for model comparisons. Evaluation: workflow and mathematics learning.evaluation.guide Open subject holdout example Open group partition classifier example Open chronological example Open chronological partition classifier example "),
        .init(id: "gelu", title: "GELU classifier", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "GELU classifier Layers & activations Example PyTorch  GELU classifier A sequential two-feature classifier with a smooth Gaussian gate. Train on bundled data, inspect activations, and export the model and weights. Input → Linear 2→8 → GELU → Linear 8→2 → Model Output Choose a starting mode; both remain editable in the activation's configuration. Open exact GELU example Open exact GELU example Open approximate GELU example Open approximate GELU example "),
        .init(id: "tensor-example", title: "Tensor layout classifier", category: "Tensors & linear algebra", kind: "Example", backend: "PyTorch", searchText: "Tensor layout classifier Tensors & linear algebra Example PyTorch  Tensor layout classifier Project twelve features, reshape to 3 × 4, transpose to 4 × 3, then flatten before the classifier. Inspect how coordinates and gradients move; these operations add no learned parameters or nonlinear capacity. Practice each operation in Tensor layout lab above, then edit this trainable flow and export its code and weights. Open tensor layout example Open tensor layout example "),
        .init(id: "softmax-example", title: "Softmax axis classifier", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Softmax axis classifier Layers & activations Example PyTorch  Softmax axis classifier Arrange six learned scores into two groups of three, then normalize each group. Change the axis to compare which values compete. A final Linear layer produces class scores for training. Softmax couples values within a slice; it is not an elementwise activation or a general fix for exploding gradients. Open Softmax example Open Softmax example "),
        .init(id: "norm-example", title: "LayerNorm scope comparison", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "LayerNorm scope comparison Layers & activations Example PyTorch  LayerNorm scope comparison Compare normalization within each group of three learned features with normalization across all six. Both preserve observations and learn coordinatewise scale and bias before GELU and the classifier. Preprocess learns training-set scaling statistics; LayerNorm recomputes statistics within each observation. Inspect and edit both on the same canvas. Open grouped LayerNorm example Open whole-sample LayerNorm example "),
        .init(id: "embedding-example", title: "Categorical embedding classifier", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Categorical embedding classifier Layers & activations Example PyTorch  Categorical embedding classifier Map integer categories to learned vectors, flatten the paired representation and train a classifier. Inspect exact category IDs and held-out predictions in the bottom Results dock. Synthetic teaching data repeats category combinations; it demonstrates shared lookup weights, not generalization to unseen categories. The Embedding lab above explores the underlying gather and scatter-add mathematics. Open embedding example "),
        .init(id: "matrix-example", title: "Matrix interaction classifier", category: "Tensors & linear algebra", kind: "Example", backend: "PyTorch", searchText: "Matrix interaction classifier Tensors & linear algebra Example PyTorch  Matrix interaction classifier Learn two matrix operands from the same observations. Multiply across two heads with a shared right operand, then combine their features into class scores. Edit shapes and operand connections on Build. A matrix interaction example, not a complete attention architecture. The Matrix multiplication lab explores both operand gradients. Open matrix example "),
        .init(id: "bilinear-example", title: "Bilinear interaction classifier", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Bilinear interaction classifier Layers & activations Example PyTorch  Bilinear interaction classifier Project observations into separate 3- and 5-feature branches. A learned [2, 3, 5] weight tensor combines their pairwise interactions into two class scores. Edit First input features · a, Second input features · b, Output features and Use bias in Configuration. Inspect named parameter policies under Weights and biases. Leading sample dimensions must match exactly. This example uses 30 interaction weights and 2 biases; increasing either input width multiplies the parameter cost. Open bilinear example "),
        .init(id: "lazy-linear", title: "Inferred-width classifier", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Linear Infer from input Lazy Linear affine projection reshape flatten bias connected width classifier initialization"),
        .init(id: "conv1d", title: "Grouped convolution classifier", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Grouped convolution classifier Layers & activations Example PyTorch  Grouped convolution classifier Project two measurements into fourteen features, reshape to [N, 2, 7], and apply four shared kernels in two channel groups. ReLU adds a nonlinear response; flatten exposes 28 features for the final two class scores. Understand shared kernels, groups and output length Y[n,o,t] = b[o] + Σᵢ Σₖ W[o,i,k] Xpad[n, g·(Cin/G)+i, t·s+k·d] For output channel o, g = floor(o / (Cout/G)); i runs over Cin/G input channels and k over K kernel positions. PyTorch uses cross-correlation: the kernel is not reversed. The same W applies at every position and to every observation, so the number of positions does not multiply the parameter count. Lout = floor((L + Pleft + Pright − d·(K−1) − 1) / s) + 1 Here L=7, K=3, stride s=1 and dilation d=1. Same padding adds one zero on each side and keeps seven positions. With two groups, each input channel feeds two output channels: W has shape [4, 1, 3], with 12 weights and 4 optional biases. Projection adds 42 parameters and the final classifier adds 58, totaling 116 with convolution bias enabled. These positions are constructed by a learned projection of ordinary measurements. This is an architectural demonstration of local shared computation, not evidence that the bundled observations contain time-series structure. Real sequence inputs need a meaningful ordering and a suitable preparation flow. Try groups=1: each kernel sees both channels and its weight shape becomes [4, 2, 3]. Compare parameter counts and held-out results. Changing padding or stride can change length; inspect the connector shape and update the final Linear input width after flatten. Toggle Use bias in Convolution to remove four convolution parameters. Changed architecture requires a compatible checkpoint and a fresh preview cache. Open grouped convolution example "),
        .init(id: "conv2d", title: "Spatial convolution on a constructed grid", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Spatial convolution on a constructed grid Layers & activations Example PyTorch  Spatial convolution on a constructed grid Project two measurements into 40 features, reshape to [N, 2, 4, 5], and apply four grouped kernels of height 2 and width 3. ReLU and flatten produce 80 responses for the final classifier. Understand spatial axes, shared weights and padding Y[n,o,h,w] = b[o] + Σᵢ Σᵤ Σᵥ W[o,i,u,v] Xpad[n,g·(Cin/G)+i,h·sH+u·dH,w·sW+v·dW] Each output channel belongs to group g = floor(o/(Cout/G)). The kernel slides across height and width without reversing: this is cross-correlation. Its weights are shared across all spatial positions and observations, so larger grids add computation, not more kernel parameters. For each axis: Lout = floor((L + Pbefore + Pafter − d·(K−1) − 1)/s) + 1. Same padding with stride [1,1] keeps the 4×5 grid. The even height-2 kernel adds zero rows above and one below; width-3 adds one column on each side. Groups=2 gives weight shape [4,1,2,3]: 24 weights and four optional biases. Flatten turns [N,4,4,5] into [N,80]. This grid is constructed from ordinary measurements by a learned projection; it is not an image source or evidence of natural spatial structure. Real image inputs require meaningful channel and pixel ordering, consistent scaling and an appropriate preparation flow. Try groups=1 and compare 48 kernel weights with the original 24. Change stride or padding, inspect the resulting connector shapes, then update the classifier width after flatten. Spatial settings always use [height,width] order. Architecture changes require compatible weights and refreshed previews. Open spatial convolution example "),
        .init(id: "conv3d", title: "Volume convolution on constructed features", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Volume convolution on constructed features Layers & activations Example PyTorch  Volume convolution on constructed features Project two measurements into 48 features, reshape to [N, 2, 2, 3, 4], and apply four grouped kernels of shape [2, 2, 3]. ReLU and flatten expose 96 responses to the final classifier. Understand depth, height, width and shared volume kernels Y[n,o,z,y,x] = b[o] + Σᵢ Σᵤ Σᵥ Σ𝓌 W[o,i,u,v,w] Xpad[n,g·(Cin/G)+i,z·sD+u·dD,y·sH+v·dH,x·sW+w·dW] This is three-dimensional cross-correlation: kernels are not reversed. For output channel o, g = floor(o/(Cout/G)); each group sees Cin/G input channels. The same kernel weights apply to every location and observation. Depth is a spatial axis here, not a batch or channel axis. For each axis: Lout = floor((L + Pbefore + Pafter − d·(K−1) − 1)/s) + 1. All spatial settings use [depth,height,width]. Same padding with stride [1,1,1] keeps the 2×3×4 volume. Kernel[2,2,3] adds zero front/one back, zero top/one bottom, and one left/one right. Groups=2 yields W[4,1,2,2,3]: 48 weights and four optional biases. Flatten converts [N,4,2,3,4] to [N,96]. These volumes are learned constructions from ordinary measurements, not medical scans or video frames. Real volumetric inputs require meaningful axis ordering, consistent spacing and scaling, and an appropriate preparation flow. This example demonstrates architecture mechanics, not domain suitability. Try groups=1: the kernel weight count doubles to 96. Change one stride or dilation entry and inspect which spatial dimension changes; update the final Linear width after flatten. Larger volumes increase computation and padding memory even when the shared weight count is unchanged. Open volume convolution example "),
        .init(id: "separable", title: "Separable convolution", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Separable convolution Layers & activations Example PyTorch  factorization depthwise pointwise channel mixing independent weights"),
        .init(id: "transpose3d", title: "Volume transposed convolution", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Volume transposed convolution Layers & activations Example PyTorch  volume scatter overlap output geometry"),
        .init(id: "transpose2d", title: "Spatial transposed convolution", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Spatial transposed convolution Layers & activations Example PyTorch  spatial scatter overlap output geometry"),
        .init(id: "transpose1d", title: "Sequence transposed convolution", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Sequence transposed convolution Layers & activations Example PyTorch  scatter add adjoint output geometry"),
        .init(id: "guided", title: "Guided canvas build", category: "Foundations", kind: "Guided", backend: "PyTorch", searchText: "guided classifier signal context label Table Input Columns Partition Preprocess View Model input Linear Activation Concatenate Model definition Train explicit placement data mapping run settings validation unscored"),
        .init(id: "depthwise", title: "Depthwise convolution", category: "Layers & activations", kind: "Example", backend: "PyTorch", searchText: "Depthwise convolution Layers & activations Example PyTorch channel isolation multiplier kernels shared weights "),
    ]
    static var dataEngineeringItems: [LearnCatalogItem] {
        ((try? DataEngineeringPractice.catalog())?.lessons ?? []).filter { !["data02-calculate","data02-regex-extract","data02-regex-replace","data02-output"].contains($0.id) }.map { lesson in
            LearnCatalogItem(id: lesson.id, title: lesson.title, category: "Data preparation", kind: "Guided", backend: "Portable", searchText: ([lesson.id, lesson.operatorID, lesson.title, lesson.objective, lesson.terms, lesson.prediction, lesson.inspect, lesson.limits, "expression regex calculated column delimited output unscored guided practice"] + lesson.steps).joined(separator: " "))
        }
    }
    /// Read-only routing compatibility for saved lesson/deep-link identities; absent from discovery.
    static let historicalEntryIDs: Set<String> = ["normalization","data02-output", "data02-calculate", "data02-regex-extract", "data02-regex-replace", "data-loading", "loss-results", "optimizer-updates", "scheduler-rates", "gradient-controls", "gradient-examples", "initialization-starting-settings", "initialization-examples"]
    static let historicalItems: [LearnCatalogItem] = baseItems.filter { historicalEntryIDs.contains($0.id) }.map(enrich)
    static let items: [LearnCatalogItem] = baseItems.filter { !historicalEntryIDs.contains($0.id) }.map(enrich) + dataEngineeringItems
    private static func enrich(_ item: LearnCatalogItem) -> LearnCatalogItem {
        var enriched = item
        if ConstructionLessonCatalog.contains(item.id) {enriched.kind="Construction"}
        let taskTitles: [String]
        switch item.id {
        case "configurable-activations": taskTitles = ConfigurableActivationLearningTask.allCases.map(\.title)
        case "remaining-activations": taskTitles = RemainingActivationLearningTask.allCases.map(\.title)
        case "optimizer-updates": taskTitles = OptimizerLearningTask.allCases.map(\.title)
        case "channel-normalization": taskTitles = StatefulNormalizationLearningTask.allCases.map(\.title)
        case "spatial-patches": taskTitles = SpatialPatchLearningTask.allCases.map(\.title)
        case "tensor-routing": taskTitles = TensorRoutingLearningTask.allCases.map(\.title)
        case "tensor-composition": taskTitles = TensorCompositionLearningTask.allCases.map(\.title)
        case "adaptive-pooling": taskTitles = AdaptivePoolingLearningTask.allCases.map(\.title)
        case "pooling": taskTitles = PoolingLearningTask.allCases.map(\.title)
        case "layernorm": taskTitles = LayerNormLearningTask.allCases.map(\.title)
        case "embedding-bag": taskTitles = EmbeddingBagLearningTask.allCases.map(\.title)
        case "embedding": taskTitles = EmbeddingLearningTask.allCases.map(\.title)
        case "convolution": taskTitles = ConvolutionLearningTask.allCases.filter { !$0.isTransposed }.map(\.title)
        case "transposed-convolution": taskTitles = ConvolutionLearningTask.allCases.filter { $0.isTransposed }.map(\.title)
        case "matrix": taskTitles = MatrixLearningTask.allCases.map(\.title)
        case "bilinear": taskTitles = BilinearLearningTask.allCases.map(\.title)
        default: taskTitles = []
        }
        enriched.searchText += " " + taskTitles.joined(separator: " ")
        return enriched
    }
}

/// Search uses authored content, never a fabricated completion/coverage denominator.
struct LearnCatalogItem: Identifiable {
    let id: String
    let title: String
    let category: String
    var kind: String
    let backend: String
    var searchText: String

    func matches(section: String, search: String) -> Bool {
        let allowed: Bool
        switch section {
        case "Library", "Overview", "Learning paths": allowed = true
        case "Practice": allowed = kind == "Practice" || kind == "Guided"
        case "Examples": allowed = kind == "Example"
        case "PyTorch", "scikit-learn", "TensorFlow": allowed = backend == section
        default: allowed = category == section
        }
        let terms = search.split(whereSeparator: { $0.isWhitespace })
        let haystack = [title, category, kind, backend, searchText].joined(separator: " ")
        return allowed && terms.allSatisfy { haystack.localizedCaseInsensitiveContains(String($0)) }
    }
}

/// Legacy lesson explanations inherit the same visible, square container treatment.
private struct LearnCatalogDisclosureStyle: DisclosureGroupStyle {
    func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Button { configuration.isExpanded.toggle() } label: {
                HStack(spacing: 8) {
                    Image(systemName: configuration.isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .semibold)).frame(width: 10)
                    configuration.label.font(.system(size: 12, weight: .semibold))
                    Spacer(minLength: 0)
                }.padding(10).frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
            }.buttonStyle(.plain).background(MLTheme.background)
                .accessibilityValue(configuration.isExpanded ? "Expanded" : "Collapsed")
            if configuration.isExpanded {
                Divider()
                configuration.content.frame(maxWidth: .infinity, alignment: .leading).padding(12)
            }
        }.frame(maxWidth: .infinity, alignment: .leading).overlay(Rectangle().stroke(MLTheme.rule))
    }
}

/// Current-authoring copy over immutable bundle metadata and worked-example identities.
/// Historical assessments still consume their original generated learning contracts.
private struct InlineTrainLearningContent:BundledComponentLearningContent {
    let authored:any BundledComponentLearningContent
    var contentVersion:Int {authored.contentVersion+1}
    var updatedAt:String {"2026-10-14"}
    var exampleID:String {authored.exampleID}
    var sourceResource:String {authored.sourceResource}
    var title:String {authored.title}
    var subtitle:String {authored.subtitle}
    var disclosureTitle:String {authored.disclosureTitle}
    var formula:String {authored.formula}
    var buttonTitle:String {authored.buttonTitle}
    var paragraphs:[String] {
        authored.paragraphs.map { paragraph in
            paragraph
                .replacingOccurrences(of:"Connect this recipe to Train’s optimizer input; it does not change the model’s forward connections.",with:"Choose this optimizer in Train’s configuration; it does not change the model’s forward connections.")
                .replacingOccurrences(of:"The connected recipe owns its learning rate; Run setup’s rate applies only when Train has no optimizer recipe.",with:"Train’s optimizer configuration owns its learning rate; the Run settings rate applies when no explicit optimizer is configured.")
                .replacingOccurrences(of:"Connect this recipe to Train’s scheduler input and connect an explicit optimizer recipe too.",with:"Choose this schedule and an explicit optimizer in Train’s configuration.")
                .replacingOccurrences(of:"Connect the recipe to Train’s objective input. The recipe controls scoring;",with:"Choose this objective in Train’s configuration. The objective controls scoring;")
                .replacingOccurrences(of:"The matching visual lesson lets you build the scoring connections and inspect this input influence without writing code or calculating derivatives.",with:"Inspect the objective and its target roles on Train, then compare recorded outcomes without writing code or calculating derivatives.")
                .replacingOccurrences(of:"Private lessons measure actual update results and remembered quantities; they do not accept a correct-looking name alone.",with:"This worked example is editable and unscored; inspect actual updates and recorded outcomes after running.")
                .replacingOccurrences(of:"Private lessons measure actual rates, learned adjustments and timing; a correct-looking name does not pass by itself.",with:"This worked example is editable and unscored; compare actual rates, updates and timing after running.")
                .replacingOccurrences(of:"The scoring recipe uses",with:"Train’s objective uses")
        }
    }
}

/// Current discovery wording; legacy task/receipt contracts retain their original text.
private struct ReviewedRecurrentLearningContent:BundledComponentLearningContent {
    let authored:any BundledComponentLearningContent
    var contentVersion:Int {authored.contentVersion+1}
    var updatedAt:String {"2026-10-18"}
    var exampleID:String {authored.exampleID}
    var sourceResource:String {authored.sourceResource}
    var title:String {authored.title}
    var subtitle:String {authored.subtitle}
    var disclosureTitle:String {authored.disclosureTitle}
    var formula:String {authored.formula}
    var buttonTitle:String {authored.buttonTitle}
    var paragraphs:[String] {authored.paragraphs.map {$0.replacingOccurrences(of:"Guided construction, optional-hint practice and private result checks use changed examples to test understanding. Input influence means how a small change in a source or starting summary affects the result; the lesson also checks how learned settings influence every required output.",with:"This editable classifier example is unscored. Inspect captured sequence and state outputs after running. Separate Construction assessments measure behavior on independent inputs and keep private fixture tables hidden; reading or running this example does not award their completion.")}}
}
