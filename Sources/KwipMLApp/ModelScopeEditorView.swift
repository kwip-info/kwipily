import SwiftUI
import KwipMLCore

/// The same editor surface serves the parent model and its owned Block draft windows.
struct ModelScopeEditorView: View {
    @ObservedObject var store: ModelCanvasAdapter
    var showLibrary = true
    var showInspector = true
    var publishesHistoryFocus = true
    var onApply: (() -> Void)? = nil
    var onDiscard: (() -> Void)? = nil
    var onBeginEditing: (() -> Void)? = nil
    @State private var activationPending=false
    @State private var layerPending=false

    var body: some View {
        VStack(spacing: 0) {
            if !store.path.isEmpty {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(store.title).font(.system(size: 13, weight: .semibold)).lineLimit(1)
                    Text(store.breadcrumbTitles.joined(separator: " / "))
                        .font(.system(size: 10)).foregroundStyle(MLTheme.muted).lineLimit(1)
                        .help(store.breadcrumbTitles.joined(separator: " / "))
                }
                Spacer()
                if !store.editingDraft, let onBeginEditing { Button("Edit Block", action: onBeginEditing).disabled(store.owner.isBusy) }
                if let onDiscard { Button(store.owner.isFileBacked ? "Revert" : "Discard draft", action: onDiscard).disabled(!store.editingDraft || store.owner.isBusy) }
                if let onApply { Button(store.owner.isFileBacked ? "Save" : "Apply to parent", action: onApply).disabled(!store.editingDraft || store.isBusy || store.projectionError != nil) }
            }.font(.system(size: 11)).buttonStyle(.borderless).padding(11).background(MLTheme.panel)
            }
            if showLibrary {
                ComponentRibbonView(entries: ribbonEntries, isBusy: store.isBusy,
                                    onAdd: { store.addNode(type: $0, x: 100, y: 160) })
            }
            HSplitView {
                if showInspector { inspector.frame(minWidth: 240, idealWidth: 290, maxWidth: 420).workspacePane(.inspector) }
                VSplitView {
                    GraphCanvasView(store: store, publishesHistoryFocus: publishesHistoryFocus).id(store.contextKey).frame(minWidth: 350,minHeight:180)
                    if store.capturedOutput != nil {
                        capturedResults.frame(minHeight:100,idealHeight:180).workspacePane(.results)
                    }
                }
            }
            HStack {
                let notice = store.projectionError ?? store.owner.error ?? store.statusMessage
                if !notice.isEmpty { Text(notice).font(.system(size: 11)).lineLimit(3) }
                Spacer()
                if !store.path.isEmpty {
                    Text(store.owner.isFileBacked ? store.owner.editStatus : (store.editingDraft ? "Draft saved with parent · Apply to use in model" : "Pinned Block · open a draft to edit"))
                        .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                }
            }.padding(11).background(MLTheme.panel)
        }.background(MLTheme.background)
        .onChange(of:store.selectionGeneration) {_,_ in activationPending=false;layerPending=false}
    }
    private var capturedResults: some View {
        ThemedScrollView {
            VStack(alignment:.leading,spacing:8) {
                SectionLabel(title:"Results")
                if let selection = capturedSelection,
                   let capture = store.capturedSnapshot(for:selection.node) {
                    Text(store.graph.nodes.first(where:{$0.id == selection.node})?.name ?? selection.node)
                        .font(.caption.weight(.semibold))
                    Text("Captured revision \(capture.capturedRevision) · Run \(capture.runID.uuidString)")
                        .font(.caption2).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                    if store.editingDraft || !capture.matchesCurrentConstruction || !capture.matchesCurrentBindings {
                        Text("Captured from the applied model. Rerun after applying changes to update these samples.")
                            .font(.caption).foregroundStyle(.orange)
                    }
                    Text("Bounded activation samples. Current source files have not been revalidated.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                    TensorOutputPreviewView(preview:["outputs":capture.renderedPorts],selectedPort:selection.port,family:.resolve(type:store.graph.nodes.first(where:{$0.id==selection.node})?.type ?? ""))
                } else {
                    Text("Select a component or connection with captured output from a completed run.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }
            }.frame(maxWidth:.infinity,alignment:.leading).padding(12)
        }.background(MLTheme.panel).accessibilityIdentifier("model.child.results.bottom")
    }
    private var capturedSelection: (node:String,port:String?)? {
        if store.selectedNodeIDs.count == 1,store.capturedSelectionEdges.isEmpty,let node = store.selectedNodeIDs.first { return (node,nil) }
        if store.selectedNodeIDs.isEmpty,store.capturedSelectionEdges.count == 1,
           let edge = store.graph.edges.first(where:{store.capturedSelectionEdges.contains($0.id)}) { return (edge.source,edge.sourcePort) }
        return nil
    }
    private var ribbonEntries: [ComponentRibbonEntry] {
        var values = store.primitiveRegistry.components.filter { store.path.isEmpty || !["input", "output"].contains($0.id) }.map { component in
            ComponentRibbonEntry(id: component.id, title: component.title,
                                 shortTitle: ComponentStyle.shortTitle(component.id, fallback: component.title),
                                 purpose: ComponentStyle.purpose(component.id), category: component.category, typeIcon: component.id)
        }
        if store.canInsertRootBlock { values.append(ModelBlockPalette.entry) }
        return values
    }
    private var inspector: some View {
        ThemedScrollView {
            VStack(alignment: .leading, spacing: 14) {
                SectionLabel(title: "Component settings")
                if let id = store.selectedNodeID, let node = store.graph.nodes.first(where: { $0.id == id }) {
                    let lease = store.owner.lease(path: store.path)
                    let selectionGeneration=store.selectionGeneration
                    TextField("Component name", text: Binding(get: { node.name }, set: { value in
                        guard store.selectedNodeID == id, store.owner.lease(path: store.path) == lease, !store.isBusy else { return }
                        store.renameSelected(value, lease: lease)
                    })).flatField().disabled(store.isBusy)
                    if let instance = store.scope?.instances.first(where: { $0.id == id }) {
                        Text("Block · " + instance.definition.id).font(.caption).foregroundStyle(MLTheme.muted)
                        Button("Open Block") { store.openBlock(id) }.disabled(!store.canOpenBlock(id))
                    } else {
                        Text(store.registry.component(type: node.type)?.title ?? node.type).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    InspectorConfigurationSection {
                        if ActivationConfiguration.Function(rawValue:node.type) != nil {
                            ActivationConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.activationBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {function,configuration,expectedType,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setActivation(function,configuration:configuration,nodeID:id,lease:lease,
                                        expectedType:expectedType,expectedSourceConfig:expectedConfig)
                                }
                        } else if PoolingConfiguration.Variant(rawValue:node.type) != nil {
                            PoolingConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.poolingBoundFields(nodeID:id),isBusy:store.isBusy,
                                onPending:{activationPending=$0}) {variant,configuration,expectedType,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setPooling(variant,configuration:configuration,nodeID:id,lease:lease,
                                        expectedType:expectedType,expectedSourceConfig:expectedConfig)
                                }
                        } else if ConvolutionConfiguration.Variant(rawValue:node.type) != nil {
                            ConvolutionConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.convolutionBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {variant,configuration,expectedType,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setConvolution(variant,configuration:configuration,nodeID:id,lease:lease,
                                        expectedType:expectedType,expectedSourceConfig:expectedConfig)
                                }
                        } else if AttentionConfiguration.componentIDs.contains(node.type) {
                            AttentionConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.attentionBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {configuration,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setAttention(configuration:configuration,nodeID:id,lease:lease,
                                        expectedSourceConfig:expectedConfig)
                                }
                        } else if node.type=="embedding_bag" {
                            EmbeddingBagConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.embeddingBagBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {configuration,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setEmbeddingBag(configuration:configuration,nodeID:id,lease:lease,
                                        expectedSourceConfig:expectedConfig)
                                }
                        } else if node.type=="embedding" {
                            EmbeddingConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.embeddingBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {configuration,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setEmbedding(configuration:configuration,nodeID:id,lease:lease,
                                        expectedSourceConfig:expectedConfig)
                                }
                        } else if node.type=="bilinear" {
                            BilinearConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.bilinearBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {configuration,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setBilinear(configuration:configuration,nodeID:id,lease:lease,
                                        expectedSourceConfig:expectedConfig)
                                }
                        } else if RecurrentAuthoringConfiguration.method(for:node.type) != nil {
                            RecurrentConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.recurrentBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {method,configuration,expectedType,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setRecurrent(method,configuration:configuration,nodeID:id,lease:lease,
                                        expectedType:expectedType,expectedSourceConfig:expectedConfig)
                                }
                        } else if TensorOperationAuthoring.family(node.type) != nil {
                            TensorOperationConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,inputLayouts:TensorOperationAuthoring.incoming(nodeID:node.id,graph:store.graph,validation:store.validation),inputLabels:TensorOperationInputLabels.graph(store.graph,nodeID:node.id,registry:store.primitiveRegistry),inputContracts:TensorOperationAuthoring.incomingContracts(nodeID:node.id,graph:store.graph,validation:store.validation),
                                boundFields:store.tensorOperationBoundFields(nodeID:id),connectedInputs:store.tensorOperationConnections(nodeID:id).inputs,connectedOutputs:store.tensorOperationConnections(nodeID:id).outputs,isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {method,configuration,expectedType,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setTensorOperation(method,configuration:configuration,nodeID:id,lease:lease,
                                        expectedType:expectedType,expectedSourceConfig:expectedConfig)
                                }
                        } else if ReshapeAuthoringConfiguration.Variant(rawValue:node.type) != nil {
                            ReshapeConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,inputLayout:ReshapeAuthoringConfiguration.inputLayout(nodeID:node.id,graph:store.graph,validation:store.validation),allowedVariants:ReshapePalette.variants(allowedTypes:Set(store.primitiveRegistry.components.map(\.id))),
                                boundFields:store.reshapeBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {variant,configuration,expectedType,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setReshape(variant,configuration:configuration,nodeID:id,lease:lease,
                                        expectedType:expectedType,expectedSourceConfig:expectedConfig)
                                }
                        } else if LinearConfiguration.Variant(rawValue:node.type) != nil {
                            LinearConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.linearBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {variant,configuration,expectedType,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setLinear(variant,configuration:configuration,nodeID:id,lease:lease,
                                        expectedType:expectedType,expectedSourceConfig:expectedConfig)
                                }
                        } else if NormalizationConfiguration.Variant(rawValue:node.type) != nil {
                            NormalizationConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.primitiveRegistry,
                                boundFields:store.normalizationBoundFields(nodeID:id),isBusy:store.isBusy || layerPending,
                                onPending:{activationPending=$0}) {variant,configuration,expectedType,expectedConfig in
                                    guard store.selectionGeneration==selectionGeneration else {return false}
                                    return store.setNormalization(variant,configuration:configuration,nodeID:id,lease:lease,
                                        expectedType:expectedType,expectedSourceConfig:expectedConfig)
                                }
                        } else if node.type == "dropout" {
                            DropoutConfigurationView(node:node,bound:store.controlledParameter(nodeID:id,field:"p") != nil || store.derivedParameter(nodeID:id,field:"p") != nil,isBusy:store.isBusy,
                                onPending:{activationPending=$0}) {value,expected in
                                    guard !store.isBusy,store.selectionGeneration==selectionGeneration,
                                          store.graph.nodes.first(where:{$0.id==id})?.config==expected else {return false}
                                    return store.setConfigValue("p",value:.number(value),nodeID:id,lease:lease)
                                }
                        } else {
                            ForEach(store.registry.component(type: node.type)?.fields(configuration:node.config) ?? [], id: \.key) {field in
                                fieldControl(field, node: node, lease: lease)
                            }
                        }
                        if store.scope?.nodes.contains(where:{$0.id==node.id}) == true {
                            ModelScopeLayerParametersInspector(store:store,node:node,onPending:{layerPending=$0})
                                .disabled(activationPending)
                                .id(store.contextKey+node.id+String(store.selectionGeneration)+String(describing:node.config))
                        }
                    }.id(store.contextKey + id)
                    InspectorInformationSection {
                        if store.scope?.instances.contains(where: { $0.id == id }) == true {
                            Text("This instance has independent parameters and a pinned definition.")
                        } else { ComponentInformationContent(type: node.type) }
                    }.id(store.contextKey + id)
                } else {
                    Text("Select a component to configure it. Open a Block to edit its internal canvas.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                if store.activeDefinition != nil {
                    Divider()
                    CompositeBoundaryView(store: store)
                }
            }.padding(16)
        }
    }
    private func fieldControl(_ field: ComponentField, node: GraphNode, lease: ModelEditorLease) -> some View {
        let derived = store.derivedParameter(nodeID: node.id, field: field.key)
        let controlled = store.controlledParameter(nodeID: node.id, field: field.key)
        let identity = ModelInspectorFieldIdentity(contextKey: store.contextKey, node: node, field: field, lease: lease)
        let origin = identity.origin
        return VStack(alignment: .leading, spacing: 5) {
            RegistryFieldControl(field: field, value: node.config[field.key] ?? field.defaultValue,
                                 origin: origin, currentOrigin: {
                guard store.selectedNodeID == node.id, !store.isBusy, store.owner.lease(path: store.path) == lease else { return nil }
                return origin
            }, componentID: node.type, configuration: node.config) { store.setConfigValue(field.key, value: $0, nodeID: node.id, lease: lease) }
                .id(identity.viewID).disabled(store.isBusy || controlled != nil || derived != nil)
            if let derived {
                Text(store.projectionError == nil ? "Derived from \(derived.lhs) × \(derived.rhs). Edit the exposed values below." : "Derived value unresolved; repair the exposed values below.").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            }
            if let controlled {
                Text("Controlled by exposed “\(controlled.field.title)” below.").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            }
        }
    }
}
