import SwiftUI
import KwipMLCore

struct TensorFormativeResultsView:View {
    let projection:TensorFormativeProjection
    @State private var selectedFixture=""
    @State private var selectedObservation="0"
    @State private var output="output"
    private var fixture:TensorFormativeFixture? {projection.fixtures.first {$0.id == selectedFixture} ?? projection.fixtures.first}
    @ViewBuilder private func bagResults(_ fixture:TensorFormativeFixture)->some View {
        let choices=TensorMeasuredOutput.available(in:fixture)
        let selected=choices.first {$0.rawValue == output} ?? choices.first ?? .output
        Text("Each source row belongs to one observation group. Category IDs name table rows; they are not measured feature values. " + (fixture.sourceLayout == "packed" ? "Source lengths select the used cells. A length of zero keeps an empty group's position; later cells are unused." : "Every source cell belongs to its fixed-size group."))
            .font(.caption).foregroundStyle(MLTheme.muted)
        LazyVGrid(columns:[GridItem(.adaptive(minimum:200,maximum:320),alignment:.leading)],alignment:.leading,spacing:10) {
            FlatChoiceControl(title:"Source fixture",selection:fixture.id,options:projection.fixtures.enumerated().map {FlatChoiceOption(id:$0.element.id,title:"Fixture \($0.offset+1)")}) {selectedFixture=$0;selectedObservation="0"}
            FlatChoiceControl(title:"Measured result",selection:selected.rawValue,options:choices.map {FlatChoiceOption(id:$0.rawValue,title:$0.title)}) {output=$0}
        }
        VStack(alignment:.leading,spacing:10) {
            SectionLabel(title:"Supplied source · one row per observation group")
            LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading,spacing:16) {
                LearningTensorTable(tensor:fixture.input,title:"Category IDs · original source rows")
                if let lengths=fixture.sourceLengths {LearningTensorTable(tensor:lengths,title:"Used item count · one per group")}
                if let weights=fixture.sourceWeights {LearningTensorTable(tensor:weights,title:"Real item weights · paired source cells")}
            }
        }.padding(12).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
        if let table=fixture.sourceTable {
            VStack(alignment:.leading,spacing:10) {
                SectionLabel(title:"Shared table and result influence")
                Text("The table has category rows and vector coordinates, with no observation axis. A gradient measures how a small change affects the result. IDs and offsets have no gradient available.").font(.caption).foregroundStyle(MLTheme.muted)
                LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading,spacing:16) {
                    LearningTensorTable(tensor:table,title:"Supplied table · before the operation")
                    LearningTensorTable(tensor:fixture.upstream,title:"Supplied result influence · one row per group")
                }
            }.padding(12).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
        }
        if let measured=fixture.measured[selected.rawValue],let expected=fixture.expected[selected.rawValue] {
            VStack(alignment:.leading,spacing:10) {
                SectionLabel(title:selected.title)
                Text(selected == .sourceWeightsGradient ? "Influence follows the original source-cell positions. Unused packed cells have zero influence; an unavailable integer derivative is a different condition." : selected.isParameter ? "These are shared table rows, not observation rows. Repeated category selections contribute to the same learned row." : "Each row keeps its original observation-group position, including empty groups.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                Text("Shape · "+measured.shape.map(String.init).joined(separator:" × ")).font(.caption).textSelection(.enabled)
                LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading,spacing:16) {
                    LearningTensorTable(tensor:measured,title:"Measured · "+selected.title)
                    LearningTensorTable(tensor:expected,title:"Independent reference · "+selected.title)
                    if measured.shape == expected.shape {
                        let differences=zip(measured.values,expected.values).map {$0-$1}
                        if differences.allSatisfy(\.isFinite) {LearningTensorTable(tensor:.init(shape:measured.shape,values:differences),title:"Measured − reference",signed:true)}
                    }
                }
            }.padding(12).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
        }
    }
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            SectionLabel(title:"Inspect tensor coordinates")
            if let fixture,fixture.sourceLayout != nil {
                bagResults(fixture)
            } else if let fixture {
                let choices=TensorMeasuredOutput.available(in:fixture)
                let selected=choices.first {$0.rawValue == output} ?? choices.first ?? .output
                Text(fixture.sourceTable != nil ? "Table tensors have row and embedding-coordinate axes, with no batch axis. Table gradients accumulate repeated selections; padding and frequency policies may modify that accumulation." : selected.isParameter ? "Shared parameter gradients sum contributions across every observation and preceding sample axis in this fixture. These tensors have no batch axis." : "Choose one independent observation. Sample coordinates stay intact; the batch axis is not folded into rows or columns.").font(.caption).foregroundStyle(MLTheme.muted)
                let count=fixture.input.shape.first ?? 0
                let index=min(max(Int(selectedObservation) ?? 0,0),max(count-1,0))
                LazyVGrid(columns:[GridItem(.adaptive(minimum:200,maximum:320),alignment:.leading)],alignment:.leading,spacing:10) {
            FlatChoiceControl(title:"Tensor fixture",selection:fixture.id,options:projection.fixtures.enumerated().map {
                FlatChoiceOption(id:$0.element.id,title:"Fixture \($0.offset+1)")
            }) {selectedFixture=$0;selectedObservation="0"}
                if !selected.isParameter {
                FlatChoiceControl(title:"Observation",selection:String(index),options:(0..<count).map {
                    FlatChoiceOption(id:String($0),title:"Observation \($0+1)")
                }) {selectedObservation=$0}
                }
                FlatChoiceControl(title:"Measured tensor",selection:selected.rawValue,options:choices.map {FlatChoiceOption(id:$0.rawValue,title:$0.title)}) {output=$0}
                }
                if let measured=fixture.measured[selected.rawValue],let expected=fixture.expected[selected.rawValue] {
                    Text("Full tensor shape · measured \(measured.shape.map(String.init).joined(separator:" × ")) · expected \(expected.shape.map(String.init).joined(separator:" × "))").font(.caption).textSelection(.enabled)
                }
                LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading,spacing:16) {
                if !selected.isParameter,let input=fixture.input.batchObservation(at:index) {
                    LearningTensorTable(tensor:input,title:fixture.inputLabel+" · sample coordinates").id(fixture.id+"input"+String(index))
                }
                if let table=fixture.sourceTable {
                    LearningTensorTable(tensor:table,title:"Supplied table · before forward").id(fixture.id+"sourceTable")
                }
                if !selected.isParameter,let right=fixture.secondaryInput?.batchObservation(at:index) {
                    LearningTensorTable(tensor:right,title:fixture.secondaryInputLabel+" · sample coordinates").id(fixture.id+"secondaryInput"+String(index))
                }
                ForEach(fixture.sharedParameters.keys.sorted(),id:\.self) { key in
                    if let parameter=fixture.sharedParameters[key] {
                        LearningTensorTable(tensor:parameter,title:"Supplied "+key+" · shared across observations").id(fixture.id+"parameter"+key)
                    }
                }
                if selected == .weightGradient || (selected == .biasGradient && !fixture.sharedParameters.isEmpty) {
                    LearningTensorTable(tensor:fixture.input,title:fixture.inputLabel+" · all observations").id(fixture.id+"allLeft")
                    if let right=fixture.secondaryInput {
                        LearningTensorTable(tensor:right,title:fixture.secondaryInputLabel+" · all observations").id(fixture.id+"allRight")
                    }
                    LearningTensorTable(tensor:fixture.upstream,title:"Upstream g · all observations").id(fixture.id+"parameterUpstream")
                }
                if selected == .tableGradient {
                    LearningTensorTable(tensor:fixture.input,title:"Integer selectors · all observations").id(fixture.id+"allIndices")
                    LearningTensorTable(tensor:fixture.upstream,title:"Upstream g · all observations").id(fixture.id+"allUpstream")
                }
                if [.inputGradient,.leftGradient,.rightGradient].contains(selected),let upstream=fixture.upstream.batchObservation(at:index) {
                    LearningTensorTable(tensor:upstream,title:"Upstream g · output coordinates").id(fixture.id+"upstream"+String(index))
                }
                if let measuredTensor=fixture.measured[selected.rawValue],let expectedTensor=fixture.expected[selected.rawValue] {
                    if let measured=selected.displayTensor(measuredTensor,observation:index) {
                        LearningTensorTable(tensor:measured,title:"Measured · "+selected.title).id(fixture.id+output+"measured"+String(index))
                    }
                    if let expected=selected.displayTensor(expectedTensor,observation:index) {
                        LearningTensorTable(tensor:expected,title:"Expected · "+selected.title).id(fixture.id+output+"expected"+String(index))
                    }
                    if measuredTensor.shape == expectedTensor.shape,
                       let measured=selected.displayTensor(measuredTensor,observation:index),let expected=selected.displayTensor(expectedTensor,observation:index) {
                        let differences=zip(measured.values,expected.values).map {$0-$1}
                        if differences.allSatisfy(\.isFinite) {
                            LearningTensorTable(tensor:.init(shape:measured.shape,values:differences),title:"Measured − expected",signed:true).id(fixture.id+output+"difference"+String(index))
                        }
                    }
                }
                }
                Text(fixture.sourceTable != nil ? "Table row numbers and coordinates are zero-based. Compare supplied and post-forward weights to inspect norm changes; integer selectors are discrete and have no input gradient." : selected.isParameter ? "Coordinates match the selected component’s trailing parameter dimensions. Values are sums, not averages. Hover for full precision." : "Rows and columns use zero-based sample coordinates. Values are rounded for display; hover for full precision. Gradients route upstream coefficients back to the input coordinates that produced them.").font(.caption).foregroundStyle(MLTheme.muted)
            }
        }
    }
}

/// Source/grouping/action measurements use their real fields, without tensor-backward placeholders.
struct RecipeFormativeResultsView:View {
    let projection:RecipeFormativeProjection
    @State private var fixtureID=""
    @State private var output=""
    private var fixture:RecipeFormativeFixture? {projection.fixtures.first {$0.id==fixtureID} ?? projection.fixtures.first}
    private func label(_ key:String)->String {
        switch key {
        case "sourceRows":return "Supplied input rows"
        case "desiredRows":return "Desired outcomes · aligned with inputs"
        case "rowIDs":return "Original observation identities"
        case "sourceLengths":return "True sequence or bag lengths"
        case "sourceWeights":return "Real item weights"
        case "parameter.weight":return "Supplied learned weights"
        case "parameter.bias":return "Supplied learned offsets"
        default:return ProgramExportLearningTask(rawValue:projection.operation) != nil ? ProgramExportLessonReference.label(key):DataLoadingLessonReference.label(key)
        }
    }
    var body:some View {
        if let fixture {
            VStack(alignment:.leading,spacing:12) {
                InspectorSection("Observed source and results",symbol:"tablecells",identifier:"lesson.recipe.results") {
                    HStack(alignment:.top,spacing:12) {
                        FlatChoiceControl(title:"Source fixture",selection:fixture.id,options:projection.fixtures.enumerated().map {.init(id:$0.element.id,title:"Fixture \($0.offset+1)")}) {fixtureID=$0}
                        FlatChoiceControl(title:"Measured result",selection:fixture.measured[output] == nil ? fixture.measured.keys.sorted().first ?? "":output,options:fixture.measured.keys.sorted().map {.init(id:$0,title:label($0))}) {output=$0}
                    }
                    Text("Observation identities follow their original rows. A batch is a group processed together. A check value of 1 means the actual operation passed; zero difference means agreement. Private assessment keeps its source and reference tables hidden.").font(.caption).foregroundStyle(MLTheme.muted)
                    VStack(alignment:.leading,spacing:10) {
                        SectionLabel(title:"Supplied source")
                        ThemedScrollView(.horizontal) {
                            HStack(alignment:.top,spacing:16) {
                                ForEach(fixture.inputs.keys.filter {$0 != "seed"}.sorted(),id:\.self) {key in
                                    if let tensor=fixture.inputs[key] {LearningTensorTable(tensor:tensor,title:label(key)).frame(minWidth:240)}
                                }
                            }
                        }
                    }.padding(12).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                    let key=fixture.measured[output] == nil ? fixture.measured.keys.sorted().first ?? "":output
                    if let actual=fixture.measured[key],let reference=fixture.expected[key] {
                        VStack(alignment:.leading,spacing:10) {
                            SectionLabel(title:label(key))
                            ThemedScrollView(.horizontal) {
                                HStack(alignment:.top,spacing:16) {
                                    LearningTensorTable(tensor:actual,title:"Measured").frame(minWidth:260)
                                    LearningTensorTable(tensor:reference,title:"Independent reference").frame(minWidth:260)
                                }
                            }
                        }.padding(12).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                    }
                }
            }
        }
    }
}
