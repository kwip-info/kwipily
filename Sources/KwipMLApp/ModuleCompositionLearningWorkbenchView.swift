import SwiftUI
import KwipMLCore

/// Window-local views borrow the lesson's one source owner and its transactional Block drafts.
struct ModuleCompositionLearningWorkbenchView<Results:View>:View {
    @ObservedObject var session:ModuleCompositionLearningCanvasSession
    let task:ModuleCompositionLearningTask
    let showInspector:Bool
    let results:Results
    var showsInstructions=true

    var body:some View {
        HSplitView {
            if showInspector && showsInstructions {
                ThemedScrollView {
                    VStack(alignment:.leading,spacing:14) {
                        LessonTextPanel(title:"What you are building",text:task.instructions)
                        InspectorSection("Build it on the canvas",symbol:"point.3.connected.trianglepath.dotted",identifier:"lesson.composition.steps") {
                            LessonRichText(text:steps)
                        }
                        InspectorSection("New terms explained",symbol:"book.closed",identifier:"lesson.composition.terms",initiallyExpanded:false) {
                            LessonRichText(text:"**Block** means a reusable group of connected work. A **private copy** has its own learned settings, even when it starts from the same Block.\n\n**Stored order** lists the children kept inside a Block. **Connected order** says which result feeds the next step. A sequence follows that connected order; an ordered or named collection still needs connections.\n\n**Influence** means how a small change affects a result. A **reference difference** is the measured value minus an independently calculated value: zero means they agree for that check.")
                        }
                    }.padding(14)
                }.frame(minWidth:250,idealWidth:290,maxWidth:400).background(MLTheme.panel)
            }
            VSplitView {
                VStack(spacing:0) {
                    tabs
                    Divider()
                    if let canvas=session.activeCanvas {
                        ModelScopeEditorView(store:canvas,showLibrary:false,showInspector:showInspector,publishesHistoryFocus:false,
                            onApply:{_ = session.apply()},onDiscard:{_ = session.discard()},
                            onBeginEditing:{session.open(session.activePath)})
                    } else {
                        Text("Open the lesson again to restore its canvas.").font(.caption).foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity,maxHeight:.infinity)
                    }
                    if !session.statusMessage.isEmpty {
                        Text(session.statusMessage).font(.caption).foregroundStyle(MLTheme.muted)
                            .frame(maxWidth:.infinity,alignment:.leading).padding(10).background(MLTheme.panel)
                    }
                }.frame(minWidth:380,minHeight:220)
                ThemedScrollView {
                    VStack(alignment:.leading,spacing:12) {
                        SectionLabel(title:"Results")
                        results
                    }.padding(14).frame(maxWidth:.infinity,alignment:.leading)
                }.frame(minHeight:140,idealHeight:240).background(MLTheme.panel)
                    .accessibilityIdentifier("lesson.composition.results.bottom")
            }
        }.background(MLTheme.background)
            .focusedSceneObject(session.historyProvider)
    }
    private var tabs:some View {
        ScrollView(.horizontal) {
            HStack(spacing:6) {
                ForEach(session.paths,id:\.self) {path in
                    HStack(spacing:4) {
                        Button(path.isEmpty ? "Model":path.joined(separator:" / ")) {session.open(path)}
                            .buttonStyle(FlatButtonStyle())
                            .foregroundStyle(path == session.activePath ? MLTheme.rose:MLTheme.muted)
                            .accessibilityIdentifier("lesson.composition.tab."+(path.isEmpty ? "root":path.joined(separator:"/")))
                        if !path.isEmpty {
                            Button {session.close(path)} label: {Image(systemName:"xmark").font(.system(size:9))}
                                .buttonStyle(.plain).accessibilityLabel("Close "+path.joined(separator:" / "))
                        }
                    }
                }
            }.padding(8)
        }.background(MLTheme.panel)
    }
    private var steps:String {
        if task == .buildReuse {
            return "1. Double-click the supplied Block to open its own tab. Open each private copy to inspect the work inside.\n\n2. Connect A to B and both required branches. Keep the supplied fixed sign reversal and subtraction. Apply each Block draft before returning to Model.\n\n3. Connect the model's input and output. Choose Check construction, then compare measured results, influence and saved reuse. Discard a draft to return to its applied construction."
        }
        return "1. Open each supplied Block in its own tab. Connect A to B and choose the sequence, ordered collection or named collection for the matching Block.\n\n2. Inspect stored child order and names separately from the connections. Apply the Block draft to use it in Model; Discard returns to the applied construction.\n\n3. Connect all three Block results in Model. Choose Check construction and compare connected work, native child names, influence and saved reuse."
    }
}
