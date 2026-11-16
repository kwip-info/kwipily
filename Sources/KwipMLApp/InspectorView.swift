import SwiftUI
import KwipMLCore

struct InspectorView: View {
    @ObservedObject var store: WorkspaceStore
    @State private var sourceID = ""
    @State private var inputPort = "in"
    var body: some View {
        ThemedScrollView {
            VStack(alignment:.leading,spacing:16) {
                if let node=store.selectedNode, let component=store.registry.component(type:node.type) {
                    HStack(spacing:10) {
                        ComponentIcon(type:node.type,size:34)
                        VStack(alignment:.leading,spacing:3) {
                            Text(component.title).font(.system(size:16,weight:.semibold))
                            Text(component.category.uppercased()).font(.system(size:9,weight:.medium)).tracking(1).foregroundStyle(ComponentStyle.tint(node.type))
                        }
                    }
                    Divider()
                    SectionLabel(title:"Your label")
                    TextField("Block name",text:Binding(get:{store.selectedNode?.name ?? ""},set:{store.renameNode($0)})).flatField().font(.system(size:12)).accessibilityLabel("Block name").disabled(store.isRunning || store.isPreviewing)
                    InspectorConfigurationSection {
                        if ActivationConfiguration.Function(rawValue:node.type) != nil {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            ActivationConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {function,configuration,expectedType,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setActivation(function,configuration:configuration,nodeID:node.id,expectedType:expectedType,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if PoolingConfiguration.Variant(rawValue:node.type) != nil {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            PoolingConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {variant,configuration,expectedType,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setPooling(variant,configuration:configuration,nodeID:node.id,expectedType:expectedType,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if ConvolutionConfiguration.Variant(rawValue:node.type) != nil {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            ConvolutionConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {variant,configuration,expectedType,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setConvolution(variant,configuration:configuration,nodeID:node.id,expectedType:expectedType,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if AttentionConfiguration.componentIDs.contains(node.type) {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            AttentionConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {configuration,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setAttention(configuration:configuration,nodeID:node.id,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if node.type=="embedding_bag" {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            EmbeddingBagConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {configuration,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setEmbeddingBag(configuration:configuration,nodeID:node.id,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if node.type=="embedding" {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            EmbeddingConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {configuration,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setEmbedding(configuration:configuration,nodeID:node.id,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if node.type=="bilinear" {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            BilinearConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {configuration,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setBilinear(configuration:configuration,nodeID:node.id,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if RecurrentAuthoringConfiguration.method(for:node.type) != nil {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            RecurrentConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {method,configuration,expectedType,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setRecurrent(method,configuration:configuration,nodeID:node.id,expectedType:expectedType,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if TensorOperationAuthoring.family(node.type) != nil {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            TensorOperationConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,inputLayouts:TensorOperationAuthoring.incoming(nodeID:node.id,graph:store.graph,validation:store.validation),inputLabels:TensorOperationInputLabels.graph(store.graph,nodeID:node.id,registry:store.registry),inputContracts:TensorOperationAuthoring.incomingContracts(nodeID:node.id,graph:store.graph,validation:store.validation),connectedInputs:Set(store.graph.edges.filter {$0.target==node.id}.map(\.targetPort)),connectedOutputs:Set(store.graph.edges.filter {$0.source==node.id}.map(\.sourcePort)),isBusy:store.isBusy) {method,configuration,expectedType,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setTensorOperation(method,configuration:configuration,nodeID:node.id,expectedType:expectedType,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if ReshapeAuthoringConfiguration.Variant(rawValue:node.type) != nil {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            ReshapeConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,inputLayout:ReshapeAuthoringConfiguration.inputLayout(nodeID:node.id,graph:store.graph,validation:store.validation),allowedVariants:ReshapePalette.variants(allowedTypes:Set(store.registry.components.map(\.id))),isBusy:store.isBusy) {variant,configuration,expectedType,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setReshape(variant,configuration:configuration,nodeID:node.id,expectedType:expectedType,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if LinearConfiguration.Variant(rawValue:node.type) != nil {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            LinearConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {variant,configuration,expectedType,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setLinear(variant,configuration:configuration,nodeID:node.id,expectedType:expectedType,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if NormalizationConfiguration.Variant(rawValue:node.type) != nil {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            NormalizationConfigurationView(node:node,configurationVersion:store.graph.schemaVersion,registry:store.registry,isBusy:store.isBusy) {variant,configuration,expectedType,expectedConfig in
                                guard store.historySessionRevision==historySession else {return false}
                                return store.setNormalization(variant,configuration:configuration,nodeID:node.id,expectedType:expectedType,
                                    expectedSourceConfig:expectedConfig,expectedProjectID:project)
                            }
                        } else if node.type == "dropout" {
                            let project=store.activeProjectID
                            let historySession=store.historySessionRevision
                            DropoutConfigurationView(node:node,isBusy:store.isBusy) {value,expected in
                                guard store.activeProjectID==project,store.historySessionRevision==historySession,
                                      store.selectedNode?.type=="dropout",store.selectedNode?.config==expected else {return false}
                                return store.setConfigValue("p",value:.number(value),nodeID:node.id)
                            }
                        } else if !component.fields(configuration:node.config).isEmpty {
                            ForEach(component.fields(configuration:node.config),id:\.key) { field in
                                RegistryFieldControl(field: field, value: node.config[field.key] ?? field.defaultValue, origin: "\(store.activeProjectID ?? ""):\(node.id):\(field.key)", currentOrigin: {
                                    guard !store.isBusy, store.selectedNodeID == node.id else { return nil }
                                    return "\(store.activeProjectID ?? ""):\(node.id):\(field.key)"
                                }, componentID: node.type, configuration: node.config) { store.setConfigValue(field.key, value: $0, nodeID: node.id) }
                                    .id(node.id + field.key).disabled(store.isBusy)
                            }
                        } else {
                            Text(node.type == "concat" ? "Concatenates along the feature axis." : node.type == "add" ? "Elementwise sum; both widths must match." : "No editable dimensions. Shape follows the connected tensor.").font(.system(size:11)).foregroundStyle(MLTheme.muted)
                        }
                    }.id(node.id)
                    InspectorInformationSection(componentType: node.type) { ComponentInformationContent(type: node.type) }.id(node.id)
                    Divider(); SectionLabel(title:"Tensor contract")
                    ForEach(component.inputs,id:\.self) { port in inputRow(node,port:port) }
                    if !component.outputs.isEmpty {
                        HStack(alignment:.top) { portLabel("out"); VStack(alignment:.leading,spacing:4) { Text(shape(node.id)).foregroundStyle(ComponentStyle.tint(node.type)); Text("\(store.graph.edges.filter {$0.source == node.id}.count) downstream connections").foregroundStyle(MLTheme.muted) } }.font(.system(size:10,design:.monospaced))
                    }
                    if !component.inputs.isEmpty {
                        SectionLabel(title:"Connect input")
                        FlatChoiceControl(title: "Connection source", selection: sourceID,
                                          options: store.graph.nodes.filter { $0.id != node.id && store.registry.component(type: $0.type)?.outputs.isEmpty == false }.map {
                                              FlatChoiceOption(id: $0.id, title: $0.name.isEmpty ? $0.type : $0.name)
                                          }, placeholder: "Choose source") { sourceID = $0 }
                            .disabled(store.isRunning || store.isPreviewing)
                        HStack(spacing:6) {
                            ForEach(component.inputs,id:\.self) { port in
                                Button(port) { inputPort=port }.buttonStyle(FlatButtonStyle(prominent:chosenPort(component) == port)).accessibilityLabel("Target input \(port)")
                            }
                            Spacer()
                            Button("Connect") { store.connect(source:sourceID,target:node.id,targetPort:chosenPort(component)) }.buttonStyle(FlatButtonStyle()).disabled(sourceID.isEmpty || store.isRunning || store.isPreviewing).accessibilityLabel("Connect selected source")
                        }
                    }
                    Divider(); SectionLabel(title:"Inspect this stage")
                    Text(store.previewWeightLabel).font(.system(size:10)).foregroundStyle(MLTheme.muted)
                    Button("Preview to here") { store.previewSelected() }.buttonStyle(FlatButtonStyle(prominent:true)).disabled(store.isRunning || store.isPreviewing || !store.validation.isValid).accessibilityLabel("Preview to here")
                    Button("Run from here") { store.resumeSelected() }.buttonStyle(FlatButtonStyle()).disabled(store.isRunning || store.isPreviewing || !store.canResumeSelected).accessibilityLabel("Run from here")
                    Text(store.previewStatus(for:node.id)).font(.system(size:10)).foregroundStyle(MLTheme.teal).fixedSize(horizontal:false,vertical:true)
                    Text("Inference only · captures a small batch. Training always executes the complete graph.").font(.system(size:10)).foregroundStyle(MLTheme.muted)
                    Divider()
                    Button("Delete block",role:.destructive) { store.deleteSelection() }.buttonStyle(FlatButtonStyle()).disabled(store.isRunning || store.isPreviewing)
                } else if let edge=store.graph.edges.first(where:{$0.id == store.selectedEdgeID}) {
                    SectionLabel(title:"Selected connection")
                    Text("\(nodeName(edge.source)) → \(nodeName(edge.target))").font(.system(size:14,weight:.medium)).fixedSize(horizontal:false,vertical:true)
                    Text("\(edge.sourcePort) → \(edge.targetPort) · \(shape(edge.source))").font(.system(size:11,design:.monospaced)).foregroundStyle(MLTheme.teal)
                    Button("Disconnect wire",role:.destructive) { store.deleteSelection() }.buttonStyle(FlatButtonStyle()).disabled(store.isRunning || store.isPreviewing)
                } else {
                    SectionLabel(title:"Flow overview")
                    Text("\(store.graph.nodes.count) blocks · \(store.graph.edges.count) connections").font(.system(size:13,weight:.medium))
                    Text("Select a block for its dimensions, tensor contract and stage preview.").font(.system(size:12)).foregroundStyle(MLTheme.muted)
                    Text("Data maps two features into each batch. The output has two class scores. Configure training and compute in Settings.").font(.system(size:11)).foregroundStyle(MLTheme.muted)
                }
                Divider(); validationSection
                if let error=store.runError { Text("Run error").font(.headline).foregroundStyle(.orange);Text(error).font(.system(size:11)).textSelection(.enabled) }
            }.padding(16)
        }.background(MLTheme.panel)
        .onChange(of:store.selectedNodeID) { _,_ in sourceID="";inputPort="in" }
    }
    private var validationSection: some View {
        VStack(alignment:.leading,spacing:8) {
            HStack { SectionLabel(title:"Validation");Spacer();Image(systemName:store.validation.isValid ? "checkmark.circle" : "exclamationmark.triangle").foregroundStyle(store.validation.isValid ? MLTheme.teal : .orange) }
            if store.validation.isValid { Text("Connections and dimensions are valid.").font(.system(size:11)).foregroundStyle(MLTheme.teal) }
            ForEach(store.validation.issues) { issue in
                Button { if let id=issue.nodeID { store.selectNode(id) } } label: { Text(issue.message).font(.system(size:11)).foregroundStyle(.orange).multilineTextAlignment(.leading).frame(maxWidth:.infinity,alignment:.leading) }.buttonStyle(.plain)
            }
        }
    }
    private func inputRow(_ node:GraphNode,port:String)->some View {
        let edge=store.graph.edges.first {$0.target == node.id && $0.targetPort == port}
        return HStack(alignment:.top) {
            portLabel(port)
            VStack(alignment:.leading,spacing:4) {
                Text(edge.map {"\(nodeName($0.source)) · \($0.sourcePort)"} ?? "Unconnected").foregroundStyle(edge == nil ? Color.orange : Color.primary).lineLimit(2)
                Text(edge.map {shape($0.source)} ?? "Choose a source below").foregroundStyle(MLTheme.muted)
            }
        }.font(.system(size:10))
    }
    private func portLabel(_ value:String)->some View { Text(value).font(.system(size:10,design:.monospaced)).frame(width:28,height:22).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule)) }
    private func chosenPort(_ definition:ComponentDefinition)->String { definition.inputs.contains(inputPort) ? inputPort : definition.inputs.first ?? "in" }
    private func nodeName(_ id:String)->String { guard let node=store.graph.nodes.first(where:{$0.id == id}) else {return "Missing block"};return node.name.isEmpty ? node.type : node.name }
    private func shape(_ id:String)->String { store.validation.tensorLayouts[id].map {"\($0.displayShape) · float32"} ?? "Shape unresolved" }
    private func hint(_ key:String)->String {
        switch key {case "features":return "Must match the number of mapped data columns.";case "in_features":return "Must equal the connected tensor’s feature width.";case "out_features":return "Learned output width; changes downstream dimensions.";default:return "Press Return to apply."}
    }
}
