import SwiftUI
import KwipMLCore

/// One draft owns both grouped boundaries; editing never creates executable proxy nodes.
struct WorkflowContainerInterfaceView:View {
    @ObservedObject var context:WorkflowModelNodeContext
    @State private var baseline:CompositeDefinition?
    @State private var inputs:[CompositeInput]=[]
    @State private var outputs:[CompositeOutput]=[]
    @State private var inputName=""
    @State private var outputName=""
    @State private var error:String?
    private var changed:Bool {inputs != baseline?.inputs || outputs != baseline?.outputs}
    private var availableInputs:[(CompositePortReference,String)] {
        context.containerPortChoices.inputs.filter {candidate in
            !inputs.contains {$0.targets.contains(candidate.0)} && !(baseline?.body.edges.contains {$0.target==candidate.0.componentID && $0.targetPort==candidate.0.portID} ?? false)
        }
    }
    private var accent:Color {ComponentVisualFamily.resolve(type:context.definition?.resolvedAppearance?.rawValue ?? "kwip.block").secondary}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            boundaryHeader("Inputs",symbol:"arrow.right.to.line")
            ForEach(Array(inputs.enumerated()),id:\.element.id) {index,input in
                VStack(alignment:.leading,spacing:6) {
                    HStack {Text(input.id).fontWeight(.semibold);Spacer();Button {inputs.remove(at:index)} label:{Image(systemName:"minus.circle")}.accessibilityLabel("Remove input "+input.id)}
                    ForEach(Array(input.targets.enumerated()),id:\.offset) {targetIndex,target in
                        HStack {
                            Text(label(target,output:false)).fixedSize(horizontal:false,vertical:true)
                            Spacer(minLength:4)
                            Button {inputs[index].targets.remove(at:targetIndex)} label:{Image(systemName:"minus")}.accessibilityLabel("Remove target from "+input.id)
                        }
                    }
                    let choices=availableInputs
                    FlatChoiceControl(title:"Target for "+input.id,selection:"",options:choices.enumerated().map {.init(id:String($0.offset),title:$0.element.1)},placeholder:"Add target…",accent:accent) {value in
                        guard let row=Int(value),choices.indices.contains(row) else {return};inputs[index].targets.append(choices[row].0)
                    }
                }.padding(8).background(accent.opacity(0.06)).overlay(Rectangle().stroke(accent.opacity(0.25)))
            }
            TextField("New input name",text:$inputName).flatField().accessibilityLabel("New container input name")
            Button("Add input") {
                guard let target=availableInputs.first?.0 else {return}
                inputs.append(.init(inputName.trimmingCharacters(in:.whitespacesAndNewlines),targets:[target]));inputName=""
            }.buttonStyle(FlatButtonStyle()).disabled(!canAdd(inputName,existing:inputs.map(\.id)) || availableInputs.isEmpty)
            boundaryHeader("Outputs",symbol:"arrow.right.to.line.compact")
            ForEach(Array(outputs.enumerated()),id:\.element.id) {index,output in
                VStack(alignment:.leading,spacing:6) {
                    HStack {Text(output.id).fontWeight(.semibold);Spacer();Button {outputs.remove(at:index)} label:{Image(systemName:"minus.circle")}.accessibilityLabel("Remove output "+output.id)}
                    let choices=context.containerPortChoices.outputs
                    FlatChoiceControl(title:"Source for "+output.id,selection:choices.firstIndex(where:{$0.0==output.source}).map(String.init) ?? "",options:choices.enumerated().map {.init(id:String($0.offset),title:$0.element.1)},placeholder:"Choose source…",accent:accent) {value in
                        guard let row=Int(value),choices.indices.contains(row) else {return};outputs[index].source=choices[row].0
                    }
                }.padding(8).background(accent.opacity(0.06)).overlay(Rectangle().stroke(accent.opacity(0.25)))
            }
            TextField("New output name",text:$outputName).flatField().accessibilityLabel("New container output name")
            Button("Add output") {guard let source=context.containerPortChoices.outputs.last?.0 else {return};outputs.append(.init(outputName.trimmingCharacters(in:.whitespacesAndNewlines),source:source));outputName=""}
                .buttonStyle(FlatButtonStyle()).disabled(!canAdd(outputName,existing:outputs.map(\.id)) || context.containerPortChoices.outputs.isEmpty)
            HStack {
                Button("Apply interface") {
                    guard let baseline else {return}
                    if context.setContainerInterface(inputs:inputs,outputs:outputs,expected:baseline) {reload()} else {error=context.statusMessage}
                }.buttonStyle(FlatButtonStyle()).disabled(!changed).accessibilityIdentifier("container.interface.apply")
                Button("Revert") {context.authority.message=nil;reload()}.buttonStyle(FlatButtonStyle()).disabled(!changed).accessibilityIdentifier("container.interface.revert")
            }
            if let error {Text(error).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
        }.font(.caption).tint(accent).disabled(context.isBusy)
            .onAppear {reload()}
            .onChange(of:changed) {_,value in context.setNumericPending(origin:context.numericOrigin(key:"container-interface"),key:"container-interface",pending:value)}
            .onDisappear {context.setNumericPending(origin:context.numericOrigin(key:"container-interface"),key:"container-interface",pending:false)}
    }
    private func boundaryHeader(_ title:String,symbol:String)->some View {
        Label(title,systemImage:symbol).fontWeight(.semibold).foregroundStyle(accent).frame(maxWidth:.infinity,alignment:.leading).padding(.vertical,6)
    }
    private func label(_ port:CompositePortReference,output:Bool)->String {
        (output ? context.containerPortChoices.outputs:context.containerPortChoices.inputs).first {$0.0==port}?.1 ?? port.componentID+" · "+port.portID
    }
    private func canAdd(_ name:String,existing:[String])->Bool {
        let value=name.trimmingCharacters(in:.whitespacesAndNewlines)
        return !value.isEmpty && value.unicodeScalars.count<=128 && !existing.contains(value)
    }
    private func reload() {baseline=context.definition;inputs=baseline?.inputs ?? [];outputs=baseline?.outputs ?? [];inputName="";outputName="";error=nil;context.setNumericPending(origin:context.numericOrigin(key:"container-interface"),key:"container-interface",pending:false)}
}
