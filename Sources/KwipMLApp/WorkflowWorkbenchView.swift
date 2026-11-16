import SwiftUI
import KwipMLCore

/// The mixed authoring surface. Its host owns project/session lifetime, execution,
/// navigation and child windows; this view owns only retained inspector contexts.
struct WorkflowWorkbenchView:View {
    @Environment(\.appearanceTheme) private var appearanceTheme
    @ObservedObject var owner:WorkflowAuthoringStore
    @ObservedObject var canvas:WorkflowCanvasAdapter
    let sources:any WorkflowDataSources
    var outputs:(any WorkflowDataOutputs)?=nil
    var executionHost:WorkflowWorkspaceSession?=nil
    var history:WorkflowHistoryStore?=nil
    var workspace:WorkspaceStore?=nil
    var showComponents=true
    var showInspector=true
    var canCaptureGradients=false
    var openHistory:(()->Void)?=nil
    @State private var resultsExpanded = true
    @State private var resultTab = "inspect"
    @State private var selectionReturn:WorkflowSelectionBookmark?
    @StateObject private var inspectorSession:WorkflowInspectorSession

    init(owner:WorkflowAuthoringStore,canvas:WorkflowCanvasAdapter,sources:any WorkflowDataSources,
         outputs:(any WorkflowDataOutputs)?=nil,executionHost:WorkflowWorkspaceSession?=nil,
         history:WorkflowHistoryStore?=nil,workspace:WorkspaceStore?=nil,showComponents:Bool=true,showInspector:Bool=true,
         canCaptureGradients:Bool=false,openHistory:(()->Void)?=nil) {
        self.owner=owner;self.canvas=canvas;self.sources=sources;self.outputs=outputs
        self.executionHost=executionHost;self.history=history;self.workspace=workspace
        self.openHistory=openHistory
        self.showComponents=showComponents;self.showInspector=showInspector;self.canCaptureGradients=canCaptureGradients
        let callback:((WorkflowNodeAddress,WorkflowEditorLease)->Void)? = canvas.onOpenBlock == nil ? nil : {address,lease in
            guard canvas.owner === owner,owner.isCurrent(lease),
                  let id=canvas.projection.displayIDByAddress[address],canvas.canOpenBlock(id) else {
                canvas.statusMessage="This block editor request is no longer current. Select the block again.";return
            }
            canvas.openBlock(id)
        }
        let privateCallback: ((WorkflowNodeAddress, WorkflowEditorLease) -> Void)? = canvas.onEditPrivateBlock == nil ? nil : { address, lease in
            guard canvas.owner === owner, owner.isCurrent(lease),
                  let id = canvas.projection.displayIDByAddress[address], canvas.canOpenBlock(id) else { return }
            canvas.onEditPrivateBlock?(address, lease)
        }
        _inspectorSession=StateObject(wrappedValue:WorkflowInspectorSession(owner:owner,sources:sources,outputs:outputs,onOpenBlock:callback,onEditPrivateBlock:privateCallback))
    }
    private var entries:[ComponentRibbonEntry] {
        canvas.creationEntries.map {entry in
            .init(id:entry.id,title:entry.component.title,shortTitle:ComponentStyle.shortTitle(entry.iconType,fallback:entry.component.title),purpose:ComponentStyle.purpose(entry.iconType),category:entry.component.category,typeIcon:entry.iconType)
        }
    }
    var body:some View {
        GeometryReader { viewport in
        VStack(spacing:0) {
            if let origin = owner.document.templateOrigin,
               let entry = try? WorkflowTemplates.catalog().first(where: { $0.id == origin.id }) {
                    HStack(spacing:12) {
                        Label(entry.title,systemImage:"square.grid.2x2").font(.caption.weight(.semibold))
                        Spacer(minLength:4)
                        Button("Set up with your data") {if let workspace {ApplicationNavigationWindows.shared.showTemplateSetup(workspace:workspace)}}.buttonStyle(FlatButtonStyle())
                            .accessibilityIdentifier("template.setup.open").disabled(workspace == nil)
                    }.padding(.horizontal,12).padding(.vertical,8).background(MLTheme.panel)
                        .fixedSize(horizontal:false,vertical:true)
                Divider()
            }
            if showComponents {
                ComponentRibbonView(entries:entries,isBusy:canvas.isBusy,onAdd:{type in
                    guard !canvas.isBusy else {return}
                    let point=canvas.paletteInsertionPoint(type:type,start:CGPoint(x:60,y:420))
                    canvas.addNode(type:type,x:point.x,y:point.y)
                },onDragStart:{type in
                    guard !canvas.isBusy else {return}
                    canvas.statusMessage="Drag \(entries.first {$0.id == type}?.title ?? "component") onto the canvas."
                })
                Divider()
            }
            if let recovery=canvas.savedDraftRecovery {
                HStack(spacing:10) {
                    Image(systemName:"square.stack.3d.up").foregroundStyle(appearanceTheme.accentColor)
                    Text("Editing paused · Apply or discard the saved \(recovery.title) block draft to continue.")
                        .font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
                    Spacer(minLength:8)
                    Button("Open saved draft") {_=canvas.openSavedDraft(recovery)}
                        .buttonStyle(FlatButtonStyle()).disabled(!canvas.canOpenSavedDraft(recovery))
                        .accessibilityIdentifier("workflow.savedDraft.open")
                }.padding(.horizontal,12).padding(.vertical,8).background(MLTheme.panel)
                Divider()
            }
            HSplitView {
                if showInspector {
                    VStack(spacing:0) {inspectorPanel
                        .environment(\.inspectedComponentType, selectedGraphNode?.type)
                        .environment(\.inspectedComponentConfiguration, selectedGraphNode.map {canvas.canvasConfiguration(nodeID:$0.id)} ?? [:])}.frame(minWidth:250,idealWidth:290,maxWidth:380).workspacePane(.inspector)
                }
                VSplitView {
                    GraphCanvasView(store:canvas,showsPortControls:false).id(owner.sessionID).frame(minWidth:380,minHeight:180)
                        .overlay {
                            if owner.document.nodes.isEmpty {
                                Button("Start from a template…") { ApplicationNavigationWindows.shared.showTemplates() }
                                    .buttonStyle(FlatButtonStyle()).accessibilityIdentifier("canvas.startFromTemplate")
                            }
                        }
                    resultsPanel.frame(minHeight:resultsExpanded ? 150 : 34,
                        idealHeight:resultsExpanded ? 240 : 34,maxHeight:resultsExpanded ? .infinity : 34).workspacePane(.results)
                }
            }
            Divider()
            Text(canvas.footerStatusMessage).font(.caption).foregroundStyle(MLTheme.muted)
                .textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
                .padding(.horizontal,10).padding(.vertical,6).background(MLTheme.panel)
        }
        .frame(width:viewport.size.width,height:viewport.size.height,alignment:.top)
        .clipped()
        }
        .task {await inspectorSession.runAppearanceLifetime()}
        .onDisappear {ApplicationNavigationWindows.shared.closeTemplateSetup(sessionID:owner.sessionID)}
        .onChange(of:canvas.portInspection) { _, request in
            if let request {
                let lease=owner.lease()
                _=owner.select(.init(nodeAddresses:[request.endpoint.address]),lease:lease)
                resultTab = "output"; resultsExpanded = true
            }
        }
        .onChange(of:resultsScope) { _, scope in
            if case .component(let address) = scope, executionHost?.attributedPreparationFailure?.matches(address) == true { resultTab = "diagnostics" }
            else { resultTab = resultTabs.first?.id ?? "output" }
            if scope == .global || scope == .multiple { selectionReturn = nil }
        }
    }
    private var resultsScope:WorkflowResultsScope { .resolve(selection:owner.selection,document:owner.document) }
    private var selectedGraphNode:GraphNode? {
        guard case .component(let address)=resultsScope,let id=canvas.projection.displayIDByAddress[address] else {return nil}
        return canvas.graph.nodes.first {$0.id == id}
    }
    private func hasPredictionTableOutput(_ address:WorkflowNodeAddress)->Bool {
        canvas.projection.portDescriptor(address:address,port:"predictions",direction:.output)?.semanticKind == .table
    }
    private var resultTabs:[(id:String,title:String)] {
        switch resultsScope {
        case .global: return [("activity","Activity"),("diagnostics","Diagnostics"),("performance","Performance")]
        case .component(let address):
            let metrics=hasPredictionTableOutput(address) || owner.document.nodes.contains{$0.address==address && SelfPlayConfiguration.isSelfPlay($0)}
            let effects=executionHost?.currentExecution.map {$0.request.plan.train==address && $0.derivativeResults != nil} ?? false
            return [("output","Output")] + (effects ? [("effects","Change effects")]:[]) + (metrics ? [("metrics","Metrics")]:[]) + [("ports","Ports & connections"),("diagnostics","Diagnostics")]
        case .connection: return [("output","Output"),("ports","Connection")]
        case .multiple:return [("components","Components"),("connections","Connections"),("diagnostics","Diagnostics")]
        case .unavailable:return []
        }
    }
    private var activeResultTab:String {resultTabs.contains(where:{$0.id == resultTab}) ? resultTab : resultTabs.first?.id ?? "output"}
    private var resultsTitle:String {
        switch resultsScope {
        case .global:return "Global results"
        case .component:
            let name=selectedGraphNode?.name.trimmingCharacters(in:.whitespacesAndNewlines) ?? ""
            return name.isEmpty ? "Component results" : name
        case .connection:return "Connection results"
        case .multiple:return "Multiple selection"
        case .unavailable:return "Selection unavailable"
        }
    }
    private var predictionOutputEndpoint:WorkflowEndpoint? {
        guard activeResultTab=="output" else {return nil}
        if let inspection=canvas.currentPortInspection {
            guard inspection.direction == .output,["predictions","profiles","confusion","routing","routingSample"].contains(inspection.endpoint.port) else {return nil};return inspection.endpoint
        }
        switch resultsScope {
        case .connection(let edge) where ["predictions","profiles","confusion","routing","routingSample"].contains(edge.from.port) && edge.kind == .table:return edge.from
        case .component(let address):
            guard hasPredictionTableOutput(address),canvas.projection.portDescriptor(address:address,port:"routing",direction:.output)==nil else {return nil}
            return .init(address:address,port:"predictions")
        default:return nil
        }
    }
    private var resultsPanel:some View {
        VStack(spacing:0) {
            HStack(spacing:12) {
                Button {resultsExpanded.toggle()} label: {
                    Label(resultsTitle,systemImage:resultsExpanded ? "chevron.down" : "chevron.up")
                        .lineLimit(1).truncationMode(.middle)
                }.buttonStyle(.plain).accessibilityLabel(resultsExpanded ? "Collapse results" : "Expand results")
                Spacer(minLength:4)
                if let bookmark = selectionReturn {
                    Button { restoreSelection(bookmark) } label: { Label("Back to selection", systemImage: "chevron.left") }
                        .buttonStyle(.borderless).disabled(canvas.isBusy || !bookmark.canRestore(projectID: owner.projectID, sessionID: owner.sessionID, document: owner.document))
                }
                if let openHistory {Button("Open History",action:openHistory).buttonStyle(.borderless)}
            }.font(.system(size:12,weight:.semibold)).padding(.horizontal,12).frame(height:34)
            if resultsExpanded {
                if !resultTabs.isEmpty {
                    ThemedScrollView(.horizontal) {
                        HStack(spacing:18) {
                            ForEach(resultTabs,id:\.id) { tab in
                                Button {resultTab=tab.id} label: {
                                    Text(tab.title).padding(.vertical,7)
                                        .overlay(alignment:.bottom) {if activeResultTab == tab.id {Rectangle().fill(appearanceTheme.accentColor).frame(height:2)}}
                                }.buttonStyle(.plain)
                                    .foregroundStyle(activeResultTab == tab.id ? appearanceTheme.accentColor : MLTheme.muted)
                                    .accessibilityAddTraits(activeResultTab == tab.id ? .isSelected : [])
                            }
                        }.font(.system(size:11,weight:.semibold)).padding(.horizontal,12)
                    }.fixedSize(horizontal:false,vertical:true)
                }
                Divider()
                if let endpoint=predictionOutputEndpoint,let query=executionHost?.predictionTableViewSession(for:endpoint) {
                    VStack(alignment:.leading,spacing:6) {
                        if canvas.currentPortInspection != nil {Button("Back to Ports & connections") {canvas.dismissPortInspection();resultTab="ports"}.buttonStyle(FlatButtonStyle())}
                        TableViewOutputView(session:query,executionBusy:owner.isBusy)
                    }.frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.topLeading).padding(12)
                } else if activeResultTab == "output", canvas.portInspection == nil,
                          case .component(let address)=resultsScope,let host=executionHost,hasPredictionTableOutput(address),
                          canvas.projection.portDescriptor(address:address,port:"routing",direction:.output)?.semanticKind == .table {
                    MixtureTrainingOutputView(host:host,address:address)
                        .frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.topLeading).padding(12)
                } else if activeResultTab == "output", canvas.portInspection == nil,
                   case .component(let address) = resultsScope,
                   let inspector = inspectorSession.selection, inspector.isCurrent,
                   case .data(let context) = inspector, context.address == address,
                   ((context.node?.operatorID == "kwip.data.outputDelimited" && APITableSettings.isExternal(context.node?.config ?? [:])) || (context.outputs ?? [:]).keys.contains(where:{context.capturedTableViewSession(port:$0) != nil}) || ["kwip.data.sourceCSV", "kwip.data.sourceFiles", "kwip.data.view", "kwip.data.columns", "kwip.data.filterRows", "kwip.data.join", "kwip.data.aggregate", "kwip.data.union", "kwip.data.sort", "kwip.data.unique", "kwip.data.reshapeTable", "kwip.data.splitText", "kwip.data.preprocess", "kwip.data.formula", "kwip.data.partition", "kwip.data.sampleRows"].contains(context.node?.operatorID ?? "")) {
                    DataNodeOutputsView(context: context,executionBusy:owner.isBusy)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading).padding(12)
                } else if activeResultTab == "output", case .connection(let edge) = resultsScope,
                          edge.from.address.domain == .data, [.table, .dataset].contains(edge.kind),
                          let snapshot = outputs?.snapshot(for: edge.from.address),
                          let value = snapshot.renderedPorts[edge.from.port] as? [String:Any] {
                    VStack(alignment: .leading, spacing: 6) {
                        if !snapshot.matchesCurrentConstruction || !snapshot.matchesCurrentBindings {
                            Label("Captured table is stale", systemImage: "clock.arrow.circlepath").font(.caption).foregroundStyle(.orange)
                        }
                        if let session = outputs?.capturedTableViewSession(for:edge.from) {
                            TableViewOutputView(session: session)
                        } else if let preview = value["preview"] as? [String:Any],
                                  preview["columns"] is [[String:Any]], preview["rows"] is [[Any]] {
                            CapturedDataGridView(grid: CapturedDataGrid(preview: preview), tableOnly: true, evidenceLabel: "Table", imageResolver: outputs?.imageArtifactResolver(for: edge.from.address))
                        } else { DataPortPreviewView(port: edge.from.port, output: value, imageResolver: outputs?.imageArtifactResolver(for: edge.from.address)) }
                    }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading).padding(12)
                } else if activeResultTab == "output",let target=tensorResultTarget,
                          let capture=executionHost?.modelSnapshot(for:target.address) {
                    VStack(alignment:.leading,spacing:6) {
                        TensorOutputPreviewView(preview:["outputs":capture.renderedPorts],selectedPort:target.port,
                            family:.resolve(type:canvas.canvasIconType(nodeID:(try? WorkflowCanvasIDs.nodeID(target.address)) ?? "")),
                            evidenceLabel:"Revision \(capture.capturedRevision) · Run \(capture.runID.uuidString)" +
                                ((!capture.matchesCurrentConstruction || !capture.matchesCurrentBindings) ? " · Stale relative to current source":""),fillsViewport:true,renderIdentity:capture.tensorRenderIdentity)
                            .equatable()
                    }.frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.topLeading).padding(12)
                } else {
                    GeometryReader {viewport in
                        ThemedScrollView {
                            VStack(alignment:.leading,spacing:12) { scopedResults }
                                .frame(maxWidth:.infinity,alignment:.leading).padding(12)
                                .environment(\.trainingChartViewportHeight,viewport.size.height-24)
                        }
                    }.id(resultsScopeIdentity + activeResultTab)
                }
            }
        }.background(MLTheme.panel)
    }
    /// Give captured tensors the dock's finite height, so the native table owns
    /// scrolling and its headers/gutter do not scroll away in an outer viewport.
    private var tensorResultTarget:(address:WorkflowNodeAddress,port:String?)? {
        guard canvas.currentPortInspection == nil else {return nil}
        switch resultsScope {
        case .component(let address) where address.domain == .model:return (address,nil)
        case .connection(let edge) where edge.from.address.domain == .model:return (edge.from.address,edge.from.port)
        default:return nil
        }
    }
    private var resultsScopeIdentity:String {
        switch resultsScope {
        case .global:return "global"
        case .component(let address):return address.domain.rawValue+":"+address.localID
        case .connection(let edge):return "edge:"+edge.id
        case .multiple:return "multiple:" + owner.selection.nodeAddresses.map { $0.domain.rawValue + ":" + $0.localID }.sorted().joined(separator: "|") + owner.selection.edgeIDs.sorted().joined(separator: "|")
        case .unavailable:return "unavailable"
        }
    }
    @ViewBuilder private var scopedResults:some View {
        switch resultsScope {
        case .global:
            if let host=executionHost {
                if activeResultTab == "diagnostics" {WorkflowResultsDiagnosticsView(host:host, onSelectFailure: { address in
                    inspectSelectedComponent(address); canvas.revealSelection()
                })}
                else if activeResultTab == "activity" {globalActivity(host)}
                else {WorkflowGlobalPerformanceView(host:host)}
            } else {resultsMessage("No current run", "Run the flow to capture output. Saved results remain available in History.")}
        case .component(let address):
            if activeResultTab == "ports" {WorkflowComponentPortsView(owner:owner,canvas:canvas,address:address)}
            else if activeResultTab == "effects",let result=executionHost?.currentExecution?.derivativeResults,let presentation=try? DerivativeResultsPresentation(results:result) {DerivativeResultsView(presentation:presentation)}
            else if activeResultTab == "metrics",let host=executionHost {
                if let session=host.selfPlay,session.address==address {SelfPlayMetricsView(rows:session.metrics,running:session.isBusy)}else {WorkflowComponentMetricsView(host:host,address:address)}
            }
            else if activeResultTab == "diagnostics" {
                componentDiagnostics(address)
            } else {componentOutput(address)}
        case .connection(let edge):
            if activeResultTab == "ports" {WorkflowConnectionResultsView(owner:owner,canvas:canvas,edge:edge)}
            else {
                Text(endpoint(edge.from)+" → "+endpoint(edge.to)).font(.caption.weight(.semibold)).textSelection(.enabled)
                Text("The current wire links these endpoints. A source capture does not establish that the destination consumed it.").font(.caption).foregroundStyle(MLTheme.muted)
                capturedOutput(edge.from.address,port:edge.from.port)
            }
        case .multiple:
            WorkflowResultsTableViewport {
                WorkflowMultipleResultsView(owner: owner, canvas: canvas, outputs: outputs, host: executionHost,
                    tab: activeResultTab, inspectComponent: inspectSelectedComponent, inspectConnection: inspectSelectedConnection)
            }
        case .unavailable:
            resultsMessage("Selection is no longer available", "Select a current component or clear the selection. Results from a different component are not substituted.")
        }
    }
    private func resultsMessage(_ title:String,_ detail:String)->some View {
        VStack(alignment:.leading,spacing:6) {
            Text(title).font(.system(size:13,weight:.semibold))
            Text(detail).font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
        }
    }
    @ViewBuilder private func globalActivity(_ host:WorkflowWorkspaceSession)->some View {
        if let session=host.selfPlay,session.isBusy {SelfPlayStatusView(session:session)}
        if host.modelLoader.isBusy {ModelOperationActivityView(store:host.modelLoader)}
        if let evaluation=host.evaluation,evaluation.request.matchesExecutionScope(in:owner,bindings:host.frozenBindings) {
            WorkflowResultsStatusStrip(title:"Evaluate · "+evaluation.request.plan.settings.partitionTitle,phase:evaluation.phase.rawValue,message:evaluation.statusMessage,isRunning:![.completed,.failed,.cancelled].contains(evaluation.phase))
            if evaluation.totalCount>0 {Text("\(evaluation.completedCount) / \(evaluation.totalCount) rows").font(.caption.monospacedDigit())}
        }
        if let preparation=host.predictionPreparation,preparation.isRunning {WorkflowDataPreparationResultsView(session:preparation,page:.activity).id(preparation.runID)}
        else if host.generationIsLatest,let generation=host.generation {WorkflowMessageActivityView(messages:[generation.statusMessage])}
        else if let preparation=host.currentDataPreparation {
            WorkflowDataPreparationResultsView(session:preparation,page:.activity).id(preparation.runID)
        } else if let execution=host.currentExecution {
            WorkflowActivityStatus(execution:execution)
            Divider()
            WorkflowRunConsoleView(console:execution.console,componentNames:Dictionary(uniqueKeysWithValues:execution.request.document.nodes.map {($0.address,$0.name.isEmpty ? WorkflowCanvasAdapter.icon($0.payload):$0.name)})).id(execution.runID)
        } else if let backend=host.currentBackendFit {WorkflowMessageActivityView(messages: backend.messages).id(backend.runID)}
        else {resultsMessage("No current activity", "Start a run to see session activity. Open History for saved runs.")}
    }
    @ViewBuilder private func componentOutput(_ address:WorkflowNodeAddress)->some View {
        if let inspection=canvas.currentPortInspection,inspection.endpoint.address == address {
            Button {
                canvas.dismissPortInspection()
                resultTab = "ports"
            } label: {
                Label("Back to Ports & connections", systemImage: "chevron.left")
            }.buttonStyle(FlatButtonStyle())
            exactPortInspection(inspection)
        } else if let host=executionHost,hasPredictionTableOutput(address),canvas.projection.portDescriptor(address:address,port:"routing",direction:.output)?.semanticKind == .table {
            MixtureTrainingOutputView(host:host,address:address).id(address)
        } else if let host=executionHost,hasPredictionTableOutput(address) {
            if let query=host.predictionTableViewSession(for:.init(address:address,port:"predictions")) {TableViewOutputView(session:query,executionBusy:owner.isBusy)}
            else if let evaluation=host.evaluation,evaluation.request.plan.evaluate==address,evaluation.phase == .completed,evaluation.predictionTableArtifact==nil {resultsMessage("Prediction table unavailable",evaluation.statusMessage)}
            else {resultsMessage("No current prediction table", "Run this component to publish its verified prediction table. Required working rows and optional capture follow the component’s retention policy.")}
        } else if let node=owner.document.nodes.first(where:{$0.address==address}),SelfPlayConfiguration.isSelfPlay(node),let host=executionHost {SelfPlayResultsView(host:host,address:address)
        } else if let node=owner.document.nodes.first(where:{$0.address==address}),case .gameSession=node.payload,let host=executionHost {GameSessionResultsView(host:host,address:address)
        } else if let node=owner.document.nodes.first(where:{$0.address==address}),case .generateText=node.payload,let host=executionHost {
            if let query=host.generationTableViewSession(for:.init(address:address,port:"table")) {TableViewOutputView(session:query,executionBusy:owner.isBusy)}
            else {resultsMessage("Generate text",host.generation?.statusMessage ?? "Connect a verified model and generate.")}
        } else if let node=owner.document.nodes.first(where:{$0.address==address}),case .saveModel=node.payload,let host=executionHost {
            if let url=host.externalModelSaveReceipts[address] {
                Label("Model saved",systemImage:"checkmark.circle").font(.headline).foregroundStyle(MLTheme.teal)
                Text(url.lastPathComponent).textSelection(.enabled)
                Text(savedModelPackageSummary(address, host:host)).font(.caption)
            } else if let receipt=host.modelSaveReceiptsByAddress[address] {
                Label("Model saved",systemImage:"checkmark.circle").font(.headline).foregroundStyle(MLTheme.teal)
                Text(receipt.destination.lastPathComponent).textSelection(.enabled)
                Text("Captured run · \(receipt.reference.runID)").font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                Text(receipt.format == .checkpoint ? "PyTorch checkpoint" : "Portable package · prepared inputs required").font(.caption)
            } else {resultsMessage("Save model",host.statusMessage)}
        } else if let backend=executionHost?.currentBackendFit,backend.plan.estimator == address {
            captureCaption(revision:backend.document.revision,runID:backend.runID,
                stale:!owner.document.matchesCapturedConstruction(backend.document) || executionHost?.bindings.revision != backend.bindingRevision)
            BackendFitResultsView(session:backend)
        } else if let inspector=inspectorSession.selection,inspector.isCurrent,
                  case .data(let context)=inspector,context.address == address {
            DataNodeOutputsView(context:context,executionBusy:owner.isBusy)
        } else {capturedOutput(address,port:nil)}
    }
    private func savedModelPackageSummary(_ address:WorkflowNodeAddress,host:WorkflowWorkspaceSession)->String {
        if host.isSelfPlaySource(address) {return "Paired policy/value inference package · architecture, weights and task provenance · nonresumable"}
        if let source=try? WorkflowTrainedModelSource.resolve(in:owner.document,consumer:address),source.isEstimator,
           let node=owner.document.nodes.first(where:{$0.address==source.address}),case .estimator(let config)=node.payload {
            if PrebuiltModelConfiguration.isPrebuilt(config) {return "Prebuilt model package · ordered prepared inputs required · verified model state"}
            if DocumentClassificationConfiguration.isDocument(config) {return "Classifier package · frozen text preparation and exact category order"}
            return "Verified model package · see its saved input contract"
        }
        return "Hugging Face folder · verified contents and matching tokenizer"
    }
    @ViewBuilder private func componentDiagnostics(_ address:WorkflowNodeAddress)->some View {
        if let inspector=inspectorSession.selection, inspector.isCurrent,
           case .data(let context)=inspector, context.address == address,
           DataFlowSourceOperators.contains(context.node?.operatorID ?? "") {
            ResultsSection(title:"Source inspection · saved parsing settings",symbol:"doc.text.magnifyingglass",initiallyExpanded:true) {
                SourceInspectionStatusView(status:context.sourceInspectionStatus)
                if context.sourceInspectionStatus == nil { Text("No current source inspection. Refresh the preview to check these parsing settings.").font(.caption).foregroundStyle(MLTheme.muted) }
            }
        }
        let id=canvas.projection.displayIDByAddress[address]
        let issues=canvas.validation.issues.filter {$0.nodeID != nil && $0.nodeID == id}
        ResultsSection(title:"Current configuration",symbol:"slider.horizontal.3",initiallyExpanded:true) {
            if issues.isEmpty {Text("No configuration findings are currently attributed to this component. Execution may discover additional constraints.").font(.caption).foregroundStyle(MLTheme.muted)}
            ForEach(issues) {issue in
                VStack(alignment:.leading,spacing:5) {
                    Text("Configuration finding").font(.caption.weight(.semibold))
                    LessonRichText(text:issue.message).font(.caption)
                }.padding(10).frame(maxWidth:.infinity,alignment:.leading).overlay(Rectangle().stroke(MLTheme.rule))
            }
        }
        if let failure=executionHost?.modelLoader.failure(address) {
            Text(failure).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
        } else if let generation=executionHost?.generation,generation.address==address,!generation.isBusy,generation.artifact==nil {
            Text("Generate text · "+generation.statusMessage).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
        } else if let evaluation=executionHost?.evaluation,evaluation.request.plan.evaluate==address,evaluation.phase == .failed {
            Text(evaluation.statusMessage).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
        } else if let failure = executionHost?.attributedPreparationFailure, failure.matches(address) {
            WorkflowPreparationFailureView(evidence: failure)
        } else {
            resultsMessage("Run diagnostics", "No execution failure is currently attributed to this component. Clear the canvas selection to see run-wide diagnostics.")
        }
    }
    @ViewBuilder private func capturedOutput(_ address:WorkflowNodeAddress,port:String?)->some View {
        if let node=owner.document.nodes.first(where:{$0.address==address}),WorkflowModelBoundary.isDefinition(node),port==nil || port=="definition" || port=="model" {
            WorkflowModelBoundaryResultsView(canvas:canvas,address:address)
        } else if port==nil || port=="model",let reference=executionHost?.modelLoader.reference(address) {
            VStack(alignment:.leading,spacing:8) {
                Label("Verified imported model",systemImage:"checkmark.shield").foregroundStyle(ComponentVisualFamily.boundaries.primary)
                Text(reference.metadata["architecture"]?.stringValue ?? "Model").font(.headline)
                Text("Identity · "+String(reference.identity.prefix(16))).font(.caption.monospaced()).textSelection(.enabled)
                Text(reference.metadata["task"] == .string("typedDecisions") ? "CPU · choice / ordinal / yes-no · matching question revision":reference.metadata["task"] == .string("audioClassification") ? "Inspect · Classify · matching audio processor":reference.metadata["task"] == .string("automaticSpeechRecognition") ? "Inspect · Transcribe · matching audio processor":reference.metadata["task"] == .string("prebuilt") ? "Prepared numeric features → frozen inference":reference.metadata["task"] == .string("documentClassification") ? "Text table → category + probabilities · Predict · Evaluate":"Inspect · Tokenize · Generate · Export").font(.caption).foregroundStyle(MLTheme.muted)
            }.padding(10).frame(maxWidth:.infinity,alignment:.leading).overlay(Rectangle().stroke(MLTheme.rule))
        } else if port=="model",let backend=executionHost?.currentBackendFit,backend.plan.estimator==address,backend.plan.configuration.hasTableOutputs,!backend.plan.configuration.isPrediction,backend.phase == .completed {
            Label(backend.plan.configuration.isPrebuilt ? "Verified Prebuilt model":"Verified document classifier",systemImage:"checkmark.shield").font(.headline).foregroundStyle(MLTheme.teal)
            Text(backend.plan.configuration.isPrebuilt ? "Frozen model with exact feature order. Portable package expects prepared numeric inputs; upstream preprocessing is not implicit.":"Frozen TF-IDF vocabulary/IDF, exact category order and fitted coefficients. Connect Save model to publish a portable package.").font(.caption).foregroundStyle(MLTheme.muted)
        } else if port=="model",let host=executionHost,let reference=host.frozenBindings.reference(for:address) {
            Label("Verified trained model",systemImage:"checkmark.shield").font(.headline).foregroundStyle(MLTheme.teal)
            Text("Run · "+reference.runID).font(.caption.monospaced()).textSelection(.enabled)
            Text("Connect Evaluate or Save model to use this completed model.").font(.caption).foregroundStyle(MLTheme.muted)
        } else if port=="table",let query=executionHost?.generationTableViewSession(for:.init(address:address,port:"table")) {TableViewOutputView(session:query,executionBusy:owner.isBusy)}
        else if let port,["predictions","profiles","confusion","routing","routingSample"].contains(port),let query=executionHost?.predictionTableViewSession(for:.init(address:address,port:port)) {TableViewOutputView(session:query,executionBusy:owner.isBusy)}
        else {capturedNonPredictionOutput(address,port:port)}
    }
    @ViewBuilder private func capturedNonPredictionOutput(_ address:WorkflowNodeAddress,port:String?)->some View {
        if let capture=executionHost?.modelSnapshot(for:address) {
            captureCaption(revision:capture.capturedRevision,runID:capture.runID,
                stale:!capture.matchesCurrentConstruction || !capture.matchesCurrentBindings)
            TensorOutputPreviewView(preview:["outputs":capture.renderedPorts],selectedPort:port,family:.resolve(type:canvas.canvasIconType(nodeID:(try? WorkflowCanvasIDs.nodeID(capture.address)) ?? "")))
        } else if let snapshot=outputs?.snapshot(for:address),snapshot.projectID == owner.projectID,snapshot.sessionID == owner.sessionID {
            captureCaption(revision:snapshot.capturedRevision,runID:snapshot.runID,
                stale:!snapshot.matchesCurrentConstruction || !snapshot.matchesCurrentBindings)
            ForEach(snapshot.renderedPorts.keys.sorted().filter {port == nil || $0 == port},id:\.self) { name in
                if let value=snapshot.renderedPorts[name] as? [String:Any] {DataPortPreviewView(port:name,output:value,imageResolver:outputs?.imageArtifactResolver(for:address))}
            }
        } else {resultsMessage("No captured output", "This component has no captured output in the latest run. Ports & connections shows its declared contracts; those are not measured results.")}
    }
    private func captureCaption(revision:UInt64,runID:UUID,stale:Bool)->some View {
        VStack(alignment:.leading,spacing:5) {
            Text("Captured revision \(revision) · Run \(runID.uuidString)").font(.caption2).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            Text("Captured evidence; current source files have not been revalidated.").font(.caption2).foregroundStyle(MLTheme.muted)
            if stale {Label("Output differs from the current canvas or source bindings",systemImage:"clock.arrow.circlepath").font(.caption).foregroundStyle(.orange)}
        }
    }
    @ViewBuilder private func exactPortInspection(_ inspection: WorkflowPortInspection) -> some View {
        let selected = inspection.endpoint
        let nodeID = canvas.projection.displayIDByAddress[selected.address]
        VStack(alignment: .leading, spacing: 12) {
            ResultsSection(title: "Port contract", symbol: "point.3.connected.trianglepath.dotted", initiallyExpanded: true) {
                HStack(alignment: .top, spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    SectionLabel(title: inspection.direction.rawValue + " port")
                    Text(endpoint(selected)).font(.system(size: 12, weight: .semibold)).textSelection(.enabled)
                }.frame(maxWidth: .infinity, alignment: .leading)
                VStack(alignment: .leading, spacing: 4) {
                    SectionLabel(title: "Declared contract")
                    Text(nodeID.map {canvas.canvasPortDescriptor(nodeID:$0,port:selected.port,direction:inspection.direction).accessibilityDescription} ?? "Port contract unavailable")
                        .font(.caption).textSelection(.enabled)
                }.frame(maxWidth: .infinity, alignment: .leading)
                }
                if inspection.direction == .input {
                    Text("The sources below are linked by the current canvas. Their captured outputs are not proof that this input consumed those values in this run.")
                        .font(.caption2).foregroundStyle(MLTheme.muted)
                }
            }
            let endpoints = canvas.inspectedSourceEndpoints
            if endpoints.isEmpty {
                Text("No source is connected to this input. No input values are inferred.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            ForEach(Array(endpoints.enumerated()), id: \.offset) { _, source in
                let snapshot = outputs?.snapshot(for:source.address) ?? executionHost?.snapshot(for:source.address)
                if let snapshot, snapshot.projectID == owner.projectID, snapshot.sessionID == owner.sessionID,
                   let value = snapshot.renderedPorts[source.port] as? [String:Any] {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(alignment: .top, spacing: 20) {
                            Text((inspection.direction == .input ? "Linked source output · " : "Captured output · ") + endpoint(source))
                                .font(.caption.weight(.semibold)).frame(maxWidth:.infinity,alignment:.leading)
                            Text("Revision \(snapshot.capturedRevision) · Run \(snapshot.runID.uuidString)")
                                .font(.caption2).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                        }
                        if !snapshot.matchesCurrentConstruction || !snapshot.matchesCurrentBindings {
                            Label("Stale relative to current construction or source bindings",systemImage:"clock.arrow.circlepath")
                                .font(.caption).foregroundStyle(.orange)
                        }
                        Text("Captured evidence; current source files have not been revalidated.").font(.caption2).foregroundStyle(MLTheme.muted)
                        if source.address.domain == .model {
                            TensorOutputPreviewView(preview:["outputs":[source.port:value]],selectedPort:source.port,family:.resolve(type:canvas.canvasIconType(nodeID:(try? WorkflowCanvasIDs.nodeID(source.address)) ?? "")))
                        } else {
                            DataPortPreviewView(port:source.port,output:value,imageResolver:outputs?.imageArtifactResolver(for:source.address))
                        }
                    }.padding(10).overlay(Rectangle().stroke(MLTheme.rule))
                } else {
                    Text("No captured output is available for " + endpoint(source) + ". Declared contracts are not measured data.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
        }
    }

    @ViewBuilder private var inspectorPanel:some View {
        if let inspectorError=inspectorSession.error {
            Text(inspectorError).font(.caption).foregroundStyle(appearanceTheme.accentColor).padding(16)
        } else if let inspector=inspectorSession.selection,inspector.isCurrent {
            switch inspector {
            case .data(let context):DataNodeInspectorView(context:context,showsOutputs:false).id(inspector.identity)
            case .model(let context):
                ThemedScrollView {VStack(alignment:.leading,spacing:16) {selectedToolHeader;WorkflowModelInspectorView(context:context,inputLayout:canvas.plannedInputLayout(for:context.authority.address),operationInputs:Dictionary(uniqueKeysWithValues:["in","a","b","source","indices","query","key"].compactMap {port in canvas.plannedInputLayout(for:context.authority.address,port:port).map {(port,$0)}}),operationContracts:Dictionary(uniqueKeysWithValues:["in","a","b","source","indices","query","key"].compactMap {port in canvas.plannedInputContract(for:context.authority.address,port:port).map {(port,$0)}}),operationIssue:canvas.modelConfigurationIssue(for:context.authority.address),makeMixtureValidation:{executionHost?.makeMixtureValidationSession()})}.padding(16)}.id(inspector.identity)
            case .control(let context):
                ThemedScrollView {
                    VStack(alignment:.leading,spacing:16) {
                        selectedToolHeader
                        if let node=context.authority.node,RetiredTrainingRecipeMigration.isRetired(node) {
                            RetiredTrainingRecipeInspectorView(node:node)
                        } else {
                            WorkflowControlInspectorView(context:context,configurationIssue:canvas.modelConfigurationIssue(for:context.authority.address),canCaptureGradients:canCaptureGradients,executionHost:executionHost,history:history)
                        }
                        if let host=executionHost,host.owner === owner {
                            if context.isEstimator {
                                Button(estimatorActionTitle(context.authority.node)) { _ = host.run(train: context.authority.address); resultTab = "output"; resultsExpanded = true }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy || owner.hasPendingInspectorEdits)
                            } else if context.isEvaluateTest {
                                WorkflowEvaluationLaunchView(context:context,host:host) {resultTab="output";resultsExpanded=true}
                            }
                        }
                    }.padding(16)
                }.id(inspector.identity)
            }
        } else {selectionSummary}
    }
    private func estimatorActionTitle(_ node:WorkflowNode?) -> String {
        guard let node,case .estimator(let values)=node.payload else {return "Run model"}
        if PrebuiltModelConfiguration.isPrebuilt(values) {switch values["operation"]?.stringValue {case "predict":return PrebuiltModelConfiguration.isCluster(values) ? "Assign new rows":"Predict rows";case "evaluate":return "Evaluate model";default:return PrebuiltModelConfiguration.isCluster(values) ? "Fit population":"Fit model"}}
        guard DocumentClassificationConfiguration.isDocument(values) else {return "Fit model"}
        switch values["operation"]?.stringValue {case "predict":return "Classify documents";case "evaluate":return "Evaluate documents";default:return "Fit classifier"}
    }
    @ViewBuilder private var selectedToolHeader:some View {
        if let node=selectedGraphNode {
            ToolInspectorHeader(title:RecurrentCellConfiguration.componentIDs.contains(canvas.canvasIconType(nodeID:node.id)) ? "Recurrent Cell":RecurrentSequenceConfiguration.componentIDs.contains(canvas.canvasIconType(nodeID:node.id)) ? "Recurrent":LinearConfiguration.componentIDs.contains(canvas.canvasIconType(nodeID:node.id)) ? "Linear":NormalizationConfiguration.componentIDs.contains(canvas.canvasIconType(nodeID:node.id)) ? "Normalization":ConvolutionConfiguration.componentIDs.contains(canvas.canvasIconType(nodeID:node.id)) ? "Convolution":PoolingConfiguration.componentIDs.contains(canvas.canvasIconType(nodeID:node.id)) ? "Pooling":ActivationConfiguration.componentIDs.contains(canvas.canvasIconType(nodeID:node.id)) ? "Activation":canvas.registry.component(type:node.type)?.title ?? node.type,
                authoredName:node.name,iconType:canvas.canvasIconType(nodeID:node.id))
        }
    }
    private var selectionSummary:some View {
        WorkflowSelectionInspectorView(owner: owner, canvas: canvas, workspace: workspace, inspectComponent: inspectSelectedComponent).id(owner.sessionID)
    }
    private func rememberSelection() {
        guard resultsScope == .multiple else { return }
        selectionReturn = .init(projectID: owner.projectID, sessionID: owner.sessionID, selection: owner.selection)
    }
    private func inspectSelectedComponent(_ address: WorkflowNodeAddress) {
        guard !canvas.isBusy, owner.authoringNode(at:address) != nil else { return }
        rememberSelection(); canvas.dismissPortInspection()
        _ = owner.select(.init(nodeAddresses: [address]), lease: owner.lease())
    }
    private func inspectSelectedConnection(_ edge: WorkflowEdge) {
        guard !canvas.isBusy, canvas.inlineEditableEdges.contains(edge) else { return }
        rememberSelection(); canvas.dismissPortInspection()
        _ = owner.select(.init(edgeIDs: [edge.id]), lease: owner.lease())
    }
    private func restoreSelection(_ bookmark: WorkflowSelectionBookmark) {
        guard !canvas.isBusy, bookmark.canRestore(projectID: owner.projectID, sessionID: owner.sessionID, document: owner.document) else { return }
        canvas.dismissPortInspection()
        if owner.select(bookmark.selection, lease: owner.lease()) { selectionReturn = nil }
    }
    private func endpoint(_ value:WorkflowEndpoint)->String {
        let node=owner.authoringNode(at:value.address)
        let name=node?.name.isEmpty == false ? node!.name : value.address.domain.rawValue+":"+value.address.localID
        return name+" · "+value.port
    }
}
