import SwiftUI
import KwipMLCore

/// Shared canvas composition; its host supplies assessment actions and measured result content.
struct TensorLearningWorkbenchView<Results:View>:View {
    @ObservedObject var session:TensorLearningDraftSession
    @ObservedObject var canvas:TensorLearningCanvasAdapter
    let showComponents:Bool
    let showInspector:Bool
    let task:String
    let results:Results
    var audienceGuide: LessonAudienceGuide? = nil
    var readableRequirements: String? = nil
    @Environment(\.lessonRailPresented) private var railPresented
    @State private var expanded=true

    var body:some View {
        VStack(spacing:0) {
            if showComponents {
                ComponentRibbonView(entries:canvas.registry.components.map {
                    .init(id:$0.id,title:$0.title,shortTitle:$0.title,purpose:$0.category,category:$0.category,typeIcon:$0.id)
                },isBusy:canvas.isBusy,onAdd:{type in
                    let point=canvas.paletteInsertionPoint(type:type,start:CGPoint(x:80,y:140))
                    canvas.addNode(type:type,x:point.x,y:point.y)
                }, restricted: true)
                Divider()
            }
            HSplitView {
                if showInspector {
                    ThemedScrollView {
                        VStack(alignment:.leading,spacing:14) {
                            if !railPresented, let guide = audienceGuide {
                                LessonTextPanel(title: "What you are building", text: guide.goal)
                                InspectorSection("Build it on the canvas", symbol: "point.3.connected.trianglepath.dotted", identifier: "lesson.canvas.steps") {
                                    ForEach(Array(guide.steps.enumerated()), id: \.offset) { index, step in
                                        HStack(alignment: .top, spacing: 8) {
                                            Text(String(index + 1)).font(.system(size: 12, weight: .semibold)).foregroundStyle(MLTheme.rose).frame(width: 14)
                                            LessonRichText(text: step)
                                        }
                                    }
                                }
                                InspectorSection("New terms explained", symbol: "book.closed", identifier: "lesson.terms", initiallyExpanded: false) {
                                    ForEach(guide.terms, id: \.name) { term in
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text(term.name).font(.system(size: 12, weight: .semibold))
                                            LessonRichText(text: term.meaning)
                                        }
                                    }
                                }
                            }
                            if !railPresented {
                            LessonTextPanel(title: canvas.isLossLesson ? "Objective & desired values":canvas.isTrainingControlLesson ? "Recipe & update rules":"Required shapes & rules", text: readableRequirements ?? task)
                            if readableRequirements != nil {
                                InspectorSection("Optional technical specification", symbol: "function", identifier: "lesson.technical.specification", initiallyExpanded: false) {
                                    LessonRichText(text: task)
                                }
                            }
                            InspectorSection(canvas.isLossLesson ? "Objective conventions":canvas.isTrainingControlLesson ? "Recipe conventions":"Axis conventions", symbol: "info.circle", identifier: "lesson.axes") {
                                LessonRichText(text: canvas.axisExplanation).foregroundStyle(MLTheme.muted)
                            }
                            }
                            if selectedDataRecipeOwnsSections {
                                inspector
                            } else {
                                InspectorConfigurationSection(isEnabled: !canvas.isBusy) { inspector }
                            }
                            if !canvas.isDataLoadingLesson && (canvas.isLossLesson || canvas.isTrainingControlLesson || canvas.isRecurrentLesson || canvas.isAttentionLesson) {InspectorInformationSection {
                                if let task=GradientLearningTask(rawValue:session.draft?.operation ?? "") {LessonRichText(text:task.instructions)}
                else if canvas.isInitializationLesson {LessonRichText(text:"The supplied model has two learned layers, first and second. Weights combine incoming values; bias shifts each result. These choices refer to actual learned arrays. Every fixture changes the layer widths, while keeping the same owner names. Zero in Results means the measured values match the independent reference.")}
                else if let task=LossLearningTask(rawValue:session.draft?.operation ?? "") {LessonRichText(text:task.instructions)}
                else if let task=RecurrentLearningTask(rawValue:session.draft?.operation ?? "") {LessonRichText(text:task.explanation)}
                else if let task=SchedulerLearningTask(rawValue:session.draft?.operation ?? "") {LessonRichText(text:task.explanation)}
                else if let task=OptimizerLearningTask(rawValue:session.draft?.operation ?? "") {LessonRichText(text:task.explanation)}
                            }}
                        }.padding(14)
                    }.frame(minWidth:250,idealWidth:290,maxWidth:400).background(MLTheme.panel)
                }
                VSplitView {
                    GraphCanvasView(store:canvas).frame(minWidth:380,minHeight:180)
                    VStack(spacing:0) {
                        HStack {
                            Button {expanded.toggle()} label: {Label("Results",systemImage:expanded ? "chevron.down":"chevron.up")}.buttonStyle(.plain)
                            Spacer()
                            Text(canvas.isDataLoadingLesson ? "Selected rows · aligned outcomes · observed groups":canvas.isTrainingControlLesson ? "Training updates · measured settings and state":"Planned shapes · measured values appear after a check").font(.caption).foregroundStyle(MLTheme.muted)
                        }.padding(.horizontal,12).frame(height:34)
                        if expanded {Divider();ThemedScrollView {results.frame(maxWidth:.infinity,alignment:.leading).padding(12)}}
                    }.background(MLTheme.panel)
                        .frame(minHeight:expanded ? 150:34,idealHeight:expanded ? 230:34,maxHeight:expanded ? .infinity:34)
                        .accessibilityIdentifier("tensor.lesson.results.bottom")
                }
            }
            Divider()
            Text(session.errorMessage ?? canvas.statusMessage).font(.caption).textSelection(.enabled)
                .frame(maxWidth:.infinity,alignment:.leading).padding(10).background(MLTheme.panel)
        }.background(MLTheme.background)
    }
    /// Atomic data recipe controls own their two framed sections, just as the
    /// graph-free program action does. Do not nest them in a second container.
    private var selectedDataRecipeOwnsSections:Bool {
        canvas.isDataLoadingLesson && canvas.selectedNodeIDs.count == 1 && canvas.graph.nodes.contains {
            canvas.selectedNodeIDs.contains($0.id) && ["dataset_source","data_loader"].contains($0.type)
        }
    }
    @ViewBuilder private var inspector:some View {
        if canvas.selectedNodeIDs.count == 1,let node=canvas.graph.nodes.first(where:{canvas.selectedNodeIDs.contains($0.id)}),let component=canvas.registry.component(type:node.type) {
            if canvas.isDataLoadingLesson,["dataset_source","data_loader"].contains(node.type) {
                DataProgramLearningControlsView(canvas:canvas,nodeID:node.id).id(canvas.historyContextID+node.id)
            } else if canvas.isGradientLesson,node.type == "gradient_policy",let context=canvas.gradientPolicyInspector(nodeID:node.id) {
                GradientRequestsEditorView(context:context)
            } else if canvas.isGradientLesson,node.type == "trainability",let context=canvas.trainabilityInspector(nodeID:node.id) {
                TrainabilityOwnersEditorView(context:context)
            } else if canvas.isInitializationLesson,node.type == "initialization",let context=canvas.initializationInspector(nodeID:node.id) {
                InitializationRulesEditorView(context:context)
            } else {
            SectionLabel(title:"Configuration")
            Text(component.title).font(.headline)
            if node.type == "input",let operation=session.draft?.operation,let patch=SpatialPatchLearningTask(rawValue:operation) {
                Text(patch.isUnfold ? "Supplied lesson grids: 2 channels, 3 rows and 4 columns per observation. The checker varies observations and numbers; this block keeps the supplied data when renamed." : "Supplied packed patch columns: \(patch.sampleShape[0]) rows and \(patch.sampleShape[1]) columns per observation. Select Fold to choose the grid size and patch spacing.").font(.caption).foregroundStyle(MLTheme.muted)
            } else if node.type == "input",let role=node.config["fixture"]?.stringValue {
                Text("Supplied lesson data: "+role+" input. Renaming this block keeps the same data; change its connections to change where the numbers go.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            TextField("Component label",text:Binding(get:{node.name},set:{canvas.rename(nodeID:node.id,to:$0)})).flatField().disabled(canvas.isBusy)
            if ReshapeAuthoringConfiguration.componentIDs.contains(node.type) {
                Text("Method").font(.caption)
                FlatChoiceControl(title:"Method",selection:node.type,options:ReshapePalette.variants(allowedTypes:Set(canvas.registry.components.map(\.id))).map {.init(id:$0.rawValue,title:$0.title)},accent:ComponentVisualFamily.tensorLayout.secondary) {value in
                    guard let variant=ReshapeAuthoringConfiguration.Variant(rawValue:value) else {return}
                    _=canvas.setReshape(variant,nodeID:node.id,expectedType:node.type,expected:node.config)
                }.disabled(canvas.isBusy)
            }
            if let family=TensorOperationAuthoring.family(node.type),family.types.count>1 {
                Text("Method").font(.caption)
                FlatChoiceControl(title:"Method",selection:node.type,options:family.types.filter {canvas.registry.component(type:$0) != nil}.map {.init(id:$0,title:TensorOperationAuthoring.methodTitle($0))},accent:ComponentVisualFamily.tensorLayout.secondary) {value in
                    _=canvas.setTensorOperation(value,nodeID:node.id,expectedType:node.type,expected:node.config)
                }.disabled(canvas.isBusy)
            }
            if canvas.isFixedTrainingContext(node) {
                LessonTextPanel(title:node.type == "optimizer_sgd" ? "Supplied SGD controls":"Supplied training context",text:canvas.isDataLoadingLesson ? "The checker supplies the verified training source, desired outcomes, model and saved parent. Dataset chooses which original observations to access; DataLoader controls how they are grouped. Connect both recipes to their named inputs. Neither recipe replaces Train data.":canvas.isGradientLesson ? (node.type == "optimizer_sgd" ? "Supplied SGD uses learning rate 0.05, zero dampening and weight decay. Remembered movement is 0.5 only for the next-run exercise and zero otherwise. These controls stay fixed.":"The checker supplies two learned layers, training and held-out observations, and a genuine saved parent for the next-run exercise. Connect the gradient policy, trainability policy and supplied SGD to their named Train inputs. Configure only the two policies."):canvas.isInitializationLesson ? (node.type == "optimizer_sgd" ? "Learning rate 0.05 · no remembered movement or weight decay. These public controls stay fixed so the exercise measures starting values.":"The checker supplies two learned layers, training rows, held-out rows and saved state. Connect Starting settings to initialization and Supplied SGD to optimizer. Configure only Starting settings."):node.type == "optimizer_sgd" ? "Learning rate 0.1 · Momentum 0.5\nZero weight decay and dampening. Maximize and Look ahead are off.\nThese public controls stay fixed so the lesson measures your scheduler. Configure the scheduler to change the rate sequence.":"The checker supplies the starting settings, gradients, batch positions, validation measurements and saved state. Connect both recipes; there is no hidden-state editing.")
            }
            ForEach(canvas.isFixedTrainingContext(node) ? []:component.fields,id:\.key) {field in
                VStack(alignment:.leading,spacing:5) {
                    if canvas.isTrainingControlLesson || canvas.isLossLesson,field.kind == .boolean {
                        RegistryFieldControl(field:field,value:node.config[field.key] ?? field.defaultValue,origin:canvas.historyContextID+node.id+field.key,currentOrigin:{canvas.selectedNodeIDs == [node.id] && !canvas.isBusy ? canvas.historyContextID+node.id+field.key:nil},componentID:node.type,configuration:node.config) {value in
                            var config=node.config;config[field.key]=value
                            guard canvas.configure(nodeID:node.id,values:config) else {return false}
                            canvas.discardPending(nodeID:node.id,field:field.key);return true
                        }
                    } else if node.type == "bilinear",field.key == "bias" {
                        Text("Bias: "+(node.config[field.key]?.boolValue == true ? "enabled":"disabled")+" · fixed by this lesson").font(.caption).foregroundStyle(MLTheme.muted)
                    } else if field.allowsNull {
                        Toggle("Use "+field.title,isOn:Binding(get:{node.config[field.key] != .null},set:{enabled in
                            _ = canvas.setOptionalEnabled(nodeID:node.id,field:field.key,enabled:enabled)
                        })).toggleStyle(.checkbox).font(.caption)
                    }
                    if canvas.isTrainingControlLesson || canvas.isLossLesson,field.kind == .boolean {
                        EmptyView()
                    } else if node.type == "bilinear",field.key == "bias" {
                        EmptyView()
                    } else if field.allowsNull && node.config[field.key] == .null {
                        Text("Not set").font(.caption).foregroundStyle(MLTheme.muted)
                    } else if field.kind == .choice {
                        Text(field.title).font(.caption)
                        FlatChoiceControl(title:field.title,selection:node.config[field.key]?.stringValue ?? "",options:(field.choices ?? []).map {.init(id:$0.value,title:$0.title)}) { value in
                            guard canvas.selectedNodeIDs == [node.id] else {return}
                            var config=node.config;config[field.key] = .string(value)
                            if canvas.configure(nodeID:node.id,values:config) {canvas.discardPending(nodeID:node.id,field:field.key)}
                        }
                    } else if field.kind == .boolean {
                        Toggle(field.title,isOn:Binding(get:{node.config[field.key]?.boolValue ?? false},set:{value in
                            guard canvas.selectedNodeIDs == [node.id] else {return}
                            var config=node.config;config[field.key] = .bool(value)
                            if canvas.configure(nodeID:node.id,values:config) {canvas.discardPending(nodeID:node.id,field:field.key)}
                        })).toggleStyle(.checkbox).font(.caption)
                    } else {
                    Text(field.title).font(.caption)
                    TextField(field.title,text:canvas.configurationTextBinding(nodeID:node.id,field:field.key))
                        .flatField().onSubmit {_ = canvas.applyPending(nodeID:node.id,field:field.key)}
                    }
                    if canvas.pendingText(nodeID:node.id,field:field.key) != nil {
                        HStack {
                            Button("Apply") {_ = canvas.applyPending(nodeID:node.id,field:field.key)}
                            Button("Discard edit") {canvas.discardPending(nodeID:node.id,field:field.key)}
                        }.buttonStyle(.borderless)
                    }
                }.disabled(canvas.isBusy || ComponentFieldApplicability.inactiveReason(componentID:node.type,field:field.key,values:node.config) != nil).id(canvas.historyContextID+node.id+field.key)
                if let reason=ComponentFieldApplicability.inactiveReason(componentID:node.type,field:field.key,values:node.config) {Text(reason).font(.caption).foregroundStyle(MLTheme.muted)}
            }
            if ["layer_norm","prelu"].contains(node.type) || StatefulNormalizationConfiguration.componentIDs.contains(node.type) {LayerNormParameterEditor(canvas:canvas,node:node).id(canvas.historyContextID+node.id)}
            if component.fields.isEmpty && !canvas.isFixedTrainingContext(node) {Text(canvas.isTrainingControlLesson ? "Train uses checker-owned starting settings and gradients. Connect the optimizer recipe; there are no editable model values or hidden state here.":"This boundary uses the task's tensor contract.").font(.caption).foregroundStyle(MLTheme.muted)}
            Text(canvas.canvasSummary(node)).font(.system(size:11,design:.monospaced)).foregroundStyle(MLTheme.teal)
            }
        } else {Text("Select one component to configure it.").font(.caption).foregroundStyle(MLTheme.muted)}
        if !canvas.validation.issues.isEmpty {
            Divider()
            SectionLabel(title:"Construction checks")
            ForEach(canvas.validation.issues) {issue in Text(issue.message).font(.caption).foregroundStyle(MLTheme.rose)}
        }
    }
}
