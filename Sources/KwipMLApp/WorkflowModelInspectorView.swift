import SwiftUI
import KwipMLCore

struct WorkflowModelInspectorView:View {
    @ObservedObject var context:WorkflowModelNodeContext
    @StateObject private var parameters:LayerParametersInspectorContext
    @State private var activationPending=false
    var makeMixtureValidation:(()->MixtureValidationSession?)?
    let inputLayout:BatchedTensorLayout?
    let operationInputs:[String:BatchedTensorLayout]
    let operationIssue:String?
    let operationContracts:[String:TypedTensorContract]
    init(context:WorkflowModelNodeContext,inputLayout:BatchedTensorLayout?=nil,operationInputs:[String:BatchedTensorLayout]=[:],operationContracts:[String:TypedTensorContract]=[:],operationIssue:String?=nil,makeMixtureValidation:(()->MixtureValidationSession?)?=nil) {
        self.makeMixtureValidation=makeMixtureValidation
        self.operationIssue=operationIssue;self.operationInputs=operationInputs;self.operationContracts=operationContracts
        self.inputLayout=inputLayout
        self.context=context;_parameters=StateObject(wrappedValue:.init(context:context))
    }
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorConfigurationSection {
                TextField("Component label",text:Binding(get:{context.authoredName},set:{_=context.rename($0)})).flatField().disabled(context.isBusy)
                if let member=try? WorkflowInlineContainer.parse(in:context.authority.owner.document,address:context.authority.address),
                   let scope=try? WorkflowMixtureContainer.split(member.childID),
                   let chain=(try? WorkflowMixtureContainer.chains(in:context.authority.owner.document,root:member.root))?.first(where:{$0.id==scope.chain}) {
                    Text("MoE · "+chain.name+" chain").font(.caption).foregroundStyle(MLTheme.rose)
                }
                if let definition=context.definition {
                    Text("\(definition.title) · \(definition.reference.version.major).\(definition.reference.version.minor).\(definition.reference.version.patch)").font(.caption).foregroundStyle(MLTheme.muted)
                    Button(context.isInlineContainer ? "Expand / collapse container":"Open reusable block") {_=context.openBlock()}.buttonStyle(FlatButtonStyle()).disabled(!context.canOpenBlock)
                    if context.isInlineContainer {
                        InspectorSection("Inputs and outputs",symbol:"arrow.left.arrow.right",identifier:"container.interface",initiallyExpanded:false) {
                            WorkflowContainerInterfaceView(context:context)
                        }
                    }
                    if context.isFileBackedBlock {
                        Button(context.hasPrivateDraft ? "Resume private draft" : "Edit a private copy…") { _ = context.editPrivateCopy() }
                            .buttonStyle(FlatButtonStyle()).disabled(!context.canEditPrivateCopy)
                        Button("Check for saved update") { context.reviewFlowUpdate() }
                            .buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
                        if let proposal = context.flowAdoption,
                           let before = proposal.before.flowDocument?.dependencies.first(where: { $0.definition == definition.reference }),
                           let after = proposal.after.flowDocument?.dependencies.first(where: { $0.definition == definition.reference }) {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Saved revision \(before.revision.revision) → \(after.revision.revision)")
                                    .font(.caption).foregroundStyle(.primary)
                                Text("Updates every instance of this shared definition in this flow. Existing runs remain unchanged. You can undo adoption.")
                                    .font(.caption).foregroundStyle(MLTheme.muted)
                                HStack {
                                    Button("Adopt revision") { _ = context.adoptReviewedFlowUpdate() }
                                    Button("Keep current") { context.dismissFlowUpdate() }
                                }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
                            }
                        }
                    }
                }
                if let type=context.primitiveComponentID,ActivationConfiguration.componentIDs.contains(type),
                   let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    ActivationConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"activation"),key:"activation",pending:value)}) {function,values,expectedType,expected in
                            context.setActivation(function,configuration:values,expectedType:expectedType,expected:expected)
                        }
                    LayerParametersInspector(editor:parameters).disabled(activationPending)
                } else if let type=context.primitiveComponentID,PoolingConfiguration.componentIDs.contains(type),
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    PoolingConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy,
                        onPending:{value in context.setNumericPending(origin:context.numericOrigin(key:"pooling"),key:"pooling",pending:value)}) {variant,values,expectedType,expected in
                            context.setPooling(variant,configuration:values,expectedType:expectedType,expected:expected)
                        }
                } else if let type=context.primitiveComponentID,ConvolutionConfiguration.componentIDs.contains(type),
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    ConvolutionConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"convolution"),key:"convolution",pending:value)}) {variant,values,expectedType,expected in
                            context.setConvolution(variant,configuration:values,expectedType:expectedType,expected:expected)
                        }

                    LayerParametersInspector(editor:parameters).disabled(activationPending)
                } else if let type=context.primitiveComponentID,AttentionConfiguration.componentIDs.contains(type),
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    AttentionConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,inputLayouts:operationInputs,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"attention"),key:"attention",pending:value)}) {values,expected in
                            context.setAttention(configuration:values,expected:expected)
                        }

                    LayerParametersInspector(editor:parameters).disabled(activationPending)
                } else if let type=context.primitiveComponentID,type=="embedding_bag",
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    EmbeddingBagConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"embedding_bag"),key:"embedding_bag",pending:value)}) {values,expected in
                            context.setEmbeddingBag(configuration:values,expected:expected)
                        }

                    LayerParametersInspector(editor:parameters).disabled(activationPending)
                } else if let type=context.primitiveComponentID,type=="embedding",
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    EmbeddingConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"embedding"),key:"embedding",pending:value)}) {values,expected in
                            context.setEmbedding(configuration:values,expected:expected)
                        }

                    LayerParametersInspector(editor:parameters).disabled(activationPending)
                } else if let type=context.primitiveComponentID,type=="bilinear",
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    BilinearConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"bilinear"),key:"bilinear",pending:value)}) {values,expected in
                            context.setBilinear(configuration:values,expected:expected)
                        }

                    LayerParametersInspector(editor:parameters).disabled(activationPending)
                } else if let type=context.primitiveComponentID,RecurrentAuthoringConfiguration.componentIDs.contains(type),
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    RecurrentConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"recurrent"),key:"recurrent",pending:value)}) {method,values,expectedType,expected in
                            context.setRecurrent(method,configuration:values,expectedType:expectedType,expected:expected)
                        }

                    LayerParametersInspector(editor:parameters).disabled(activationPending)
                } else if let type=context.primitiveComponentID,TensorOperationAuthoring.componentIDs.contains(type),
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    TensorOperationConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,inputLayouts:operationInputs,inputLabels:TensorOperationInputLabels.workflow(context.authority.owner.document,address:context.authority.address,registry:context.authority.owner.modelRegistry),inputContracts:operationContracts,sourceIssue:operationIssue,boundFields:context.inlineBoundFields,connectedInputs:context.tensorOperationConnections.inputs,connectedOutputs:context.tensorOperationConnections.outputs,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"tensor-operation"),key:"tensor-operation",pending:value)}) {method,values,expectedType,expected in
                            context.setTensorOperation(method,configuration:values,expectedType:expectedType,expected:expected)
                        }

                } else if let type=context.primitiveComponentID,ReshapeAuthoringConfiguration.componentIDs.contains(type),
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    ReshapeConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,inputLayout:inputLayout,allowedVariants:ReshapePalette.variants(allowedTypes:Set(context.authority.owner.modelRegistry.components.map(\.id))),boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"reshape"),key:"reshape",pending:value)}) {variant,values,expectedType,expected in
                            context.setReshape(variant,configuration:values,expectedType:expectedType,expected:expected)
                        }

                } else if let type=context.primitiveComponentID,LinearConfiguration.componentIDs.contains(type),
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    LinearConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"linear"),key:"linear",pending:value)}) {variant,values,expectedType,expected in
                            context.setLinear(variant,configuration:values,expectedType:expectedType,expected:expected)
                        }

                    LayerParametersInspector(editor:parameters).disabled(activationPending)
                } else if let type=context.primitiveComponentID,NormalizationConfiguration.componentIDs.contains(type),
                          let actual=context.authority.node,case .modelPrimitive(_,let version,let config)=actual.payload {
                    NormalizationConfigurationView(node:.init(id:actual.address.localID,type:type,name:actual.name,config:config,parameterSettings:actual.parameterSettings),
                        configurationVersion:version,registry:context.authority.owner.modelRegistry,boundFields:context.inlineBoundFields,
                        isBusy:context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers,
                        onPending:{value in activationPending=value;context.setNumericPending(origin:context.numericOrigin(key:"normalization"),key:"normalization",pending:value)}) {variant,values,expectedType,expected in
                            context.setNormalization(variant,configuration:values,expectedType:expectedType,expected:expected)
                        }

                    LayerParametersInspector(editor:parameters).disabled(activationPending)
                } else if context.primitiveComponentID == "mixture_experts" {
                    MixtureExpertsInspector(context:context,makeValidation:makeMixtureValidation)
                } else if context.primitiveComponentID == "dropout",let actual=context.authority.node,case .modelPrimitive(_,_,let config)=actual.payload {
                    DropoutConfigurationView(node:.init(id:actual.address.localID,type:"dropout",name:actual.name,config:config),
                        bound:context.inlineBoundFields.contains("p"),isBusy:context.isBusy,
                        onPending:{context.setNumericPending(origin:context.numericOrigin(key:"dropout"),key:"dropout",pending:$0)}) {value,expected in
                            context.setField("p",value:.number(value),expected:expected)
                        }
                } else if context.primitiveComponentID == "output" {
                    ModelOutputConfigurationView(context: context)
                } else {
                if let type=context.primitiveComponentID,WorkflowModelInput.componentIDs.contains(type) {
                    Text(type=="index_input" ? "Integer tensor · int64":"Floating tensor · float32")
                        .font(.caption.monospaced()).foregroundStyle(ComponentVisualFamily.boundaries.secondary)
                    let bindings=context.authority.owner.document.namedModelInputs ?? [:]
                    let names=Set(bindings.values.flatMap {$0.filter {$0.value==context.authority.address.localID}.map(\.key)}).sorted()
                    if !names.isEmpty {Text(names.joined(separator:" · ")).font(.caption).foregroundStyle(ComponentVisualFamily.boundaries.secondary)}
                    let document=context.authority.owner.document
                    if let definition=document.nodes.first(where:{node in
                        guard WorkflowModelBoundary.isDefinition(node) else {return false}
                        if document.namedModelInputs?[node.address.localID]?.values.contains(context.authority.address.localID)==true {return true}
                        return (try? WorkflowModelInput.boundary(for:node.address,in:document))?.address==context.authority.address
                    }) {
                        Button("Manage model inputs") {_=context.authority.owner.select(.init(nodeAddresses:[definition.address]),lease:context.authority.owner.lease())}.buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
                    }

                }
                ForEach(context.fields,id:\.key) {field in fieldEditor(field).disabled(context.isBusy || parameters.hasChanges || parameters.hasUncommittedNumbers || context.inlineBoundFields.contains(field.key))
                    .help(context.inlineBoundFields.contains(field.key) ? "Controlled by the containing block’s exposed settings.":field.title)}
                LayerParametersInspector(editor:parameters)
                }
            }.id(context.contextIdentity)
            InspectorInformationSection {
                if context.isInlineContainer {
                    if context.definition?.resolvedAppearance == .transformer {
                        LessonRichText(text:"**Editable Transformer**\nEncoder turns Source into memory; Decoder reads Target and that memory. These are real editable members on the same canvas. Select either member to configure its layers, masks, weights and biases. Both start with final normalization enabled.\n\n**Fresh architecture and saved weights**\nThis container initializes Encoder and Decoder independently. It is not a checkpoint-compatible rename of the older single Transformer module. Saved primitive projects and their checkpoints retain their original identity. Editing members changes architecture; use fresh training or an explicitly compatible checkpoint.\n\n**Learning**\nThe Transformer example uses this editable container. The measured Transformer lesson retains its original single-module construction and assessment identity; Encoder and Decoder have their own measured lessons.")
                        ComponentLessonLink(type:"transformer")
                    }
                    LessonRichText(text:"**Container boundaries**\nInputs and Outputs each form one group of named tensor ports. An input may feed several internal targets; an output names one internal source. These groups organize connections without running an extra computation.\n\n**Manage the interface**\nAdd a named input or output and choose its internal bindings, then Apply interface. Revert discards the entire draft. Disconnect external wires explicitly before removing a connected port. Existing port names remain stable.\n\n**Member identity**\nMembers remain real components with their own captured tensors and learned parameters. Expand or collapse on the same canvas; moves and semantic edits keep existing private-copy, Undo and checkpoint safeguards.\n\n**Container size**\nThe frame grows to fit moved members and keeps its size when they move back. Fit container to components tightens it deliberately. Drop near the right or bottom edge to make room; drag more than100canvas points beyond it to move a member out. Inner boundary ports inspect the same named tensor; edit bindings in Inputs and outputs.")
                }
                if case .modelPrimitive(let type, _, _) = context.authority.node?.payload {
                    ComponentInformationContent(type: type)
                    if type == "prelu" {Text("Initial slope supplies the layer default. A learned-slope starting-values override takes precedence; checkpoint continuation restores saved values instead.").font(.caption)}
                    if !context.inlineBoundFields.isEmpty {Text("Shared fields are controlled by the containing block’s exposed settings. Moving this component out keeps its current values.").font(.caption).foregroundStyle(MLTheme.muted)}
                    if type == "output" {
                        LessonRichText(text: "**Expected output**\nChecks compare the actual tensor; they never cast, reshape, normalize probabilities or select a class. The leading batch axis stays variable. Exact sample rank checks the ordered dimensions; Any dimension permits its size to vary. Any sample rank checks only the selected type.\n\n**Captured results**\nCoordinates show retained scalar values. Matrix shows a bounded rectangle of retained values; higher-rank captures offer axis and slice selection. Retained and total counts describe the actual capture; uncaptured values and slices are unavailable.")
                    }
                    if !parameters.choices.isEmpty {
                        LessonRichText(text:"**Weights and biases**\nStarting values apply once to fresh training. Layer default uses the native initializer. Normal chooses a center and spread; Xavier and Kaiming scale multidimensional weights by their fan. Train’s seed determines reproducible starting numbers. Checkpoint continuation restores learned values instead.\n\n**Trainability**\nLayer default preserves native behavior, including frozen embeddings. Frozen prevents that parameter from updating; it does not switch the layer to evaluation mode. Trainable explicitly enables updates.\n\n**Existing projects**\nRetained per-Train overrides take precedence for the same parameter. They remain editable under Train’s Legacy layer overrides. Shared layer settings apply to every Train using that model.")
                    }
                }
                if context.definition != nil {
                    Text("This instance has independent parameters and a pinned definition. Exposed fields can override definition defaults.")
                }
                if context.isLegacyBiasEnabled {
                    Text("Legacy configuration: bias is enabled. This editor preserves the original configuration version.")
                }
            }.id(context.contextIdentity)
            if !context.statusMessage.isEmpty {Text(context.statusMessage).font(.caption).foregroundStyle(MLTheme.muted)}
        }
    }
    @ViewBuilder private func fieldEditor(_ field:ComponentField)->some View {
        let expected=context.configuration
        if let value=context.value(for:field.key) {
            RegistryFieldControl(field:field,value:value,origin:context.numericOrigin(key:field.key),currentOrigin:{context.currentNumericOrigin(key:field.key)},onDraftChange:{origin,pending in context.setNumericPending(origin:origin,key:field.key,pending:pending)},componentID:context.primitiveComponentID,configuration:expected) {value in context.setField(field.key,value:value,expected:expected)}
            if context.definition != nil {
                Text(expected[field.key] == nil ? "Inherited from pinned definition":"Instance override").font(.caption2).foregroundStyle(MLTheme.muted)
                if expected[field.key] != nil {Button("Use definition default") {_=context.removeOverride(field.key,expected:expected)}.buttonStyle(.borderless)}
            }
        } else {
            Text("\(field.title): not configured").font(.caption).foregroundStyle(MLTheme.rose)
            Button("Set default for \(field.title)") {_=context.setField(field.key,value:field.defaultValue,expected:expected)}.buttonStyle(.borderless)
        }
    }
}
