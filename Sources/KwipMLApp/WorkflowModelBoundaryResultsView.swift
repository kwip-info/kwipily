import SwiftUI
import KwipMLCore

/// An architecture definition is inspectable before training; it is not learned state.
struct WorkflowModelBoundaryResultsView:View {
    @ObservedObject var canvas:WorkflowCanvasAdapter
    let address:WorkflowNodeAddress
    private var members:[WorkflowNode] {
        let document=canvas.owner.document,region=WorkflowModelInput.ancestors(of:address,in:document)
        return document.nodes.filter {region.contains($0.address) && $0.address.domain == .model && $0.address != address}
    }
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {
                Label("Model definition",systemImage:"square.stack.3d.up").font(.system(size:12,weight:.semibold))
                Spacer()
                Text("Architecture · \(members.count) components").font(.caption).foregroundStyle(MLTheme.muted)
            }
            if let layout=canvas.plannedInputLayout(for:address) {
                Text("Declared output · N × "+layout.sample.dimensions.map(String.init).joined(separator:" × ")).font(.caption.monospaced()).foregroundStyle(MLTheme.muted)
            }
            VStack(spacing:0) {
                HStack {Text("Component");Spacer();Text("Type")}.font(.caption.weight(.semibold)).padding(8).background(MLTheme.panel)
                ForEach(members,id:\.address) {node in
                    let type=primitiveTitle(node)
                    HStack {Text(node.name.isEmpty ? type:node.name);Spacer();Text(type).foregroundStyle(MLTheme.muted)}
                        .font(.system(size:11)).padding(8).overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
                }
            }.overlay(Rectangle().stroke(MLTheme.rule))
            DisclosureGroup("Information") {
                Text("This connection describes the current architecture and its declared input/output, before learned weights exist. Train combines it with mapped data. Use Train’s Trained model output for Evaluate or Save model; those consumers require a verified completed run.")
                    .font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
            }.font(.caption)
        }
    }
    private func primitiveTitle(_ node:WorkflowNode)->String {
        switch node.payload {
        case .modelPrimitive(let type,_,_):return canvas.modelRegistry.component(type:type)?.title ?? type
        case .modelBlock:return "Reusable block"
        default:return "Model component"
        }
    }
}
