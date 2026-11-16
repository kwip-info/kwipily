import SwiftUI
import KwipMLCore

/// Settings for the existing router graph; no second graph or canvas is created.
struct MixtureRouterControls:View {
    @Binding var source:String
    let registry:ComponentRegistry
    let origin:String
    let pending:(String,Bool)->Void
    @State private var selected:String?
    @State private var error=""
    private var graph:GraphDocument? {try? MixtureRouterPresentation.graph(source)}
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            Text("Router layers").font(.headline)
            Text("The router reads the same features as the experts and produces one score per expert. Top-k selects from these scores; it is not an extra canvas output.").font(.caption).foregroundStyle(MLTheme.muted)
            if let graph {
                ForEach(graph.nodes,id:\.id) {node in
                    Button {selected=node.id} label: {
                        HStack(spacing:10) {
                            ComponentIcon(type:node.type,size:25)
                            VStack(alignment:.leading,spacing:3) {
                                Text(title(node)).font(.system(size:12,weight:.semibold))
                                Text(node.id).font(.system(size:10,design:.monospaced)).foregroundStyle(MLTheme.muted)
                            }
                            Spacer()
                            Image(systemName:selectedID(graph)==node.id ? "chevron.down":"chevron.right").font(.caption)
                        }.padding(12).frame(maxWidth:.infinity,alignment:.leading)
                            .background(selectedID(graph)==node.id ? MLTheme.rose.opacity(0.12):MLTheme.panel)
                            .overlay(Rectangle().stroke(MLTheme.rule))
                    }.buttonStyle(.plain).accessibilityLabel("Router layer · "+title(node)+" · "+node.id)
                }
                if let node=graph.nodes.first(where:{$0.id==selectedID(graph)}) {details(node,graph:graph)}
            } else {Text("The router source needs correction. Open Advanced router JSON to repair it.").font(.caption).foregroundStyle(MLTheme.rose)}
            if !error.isEmpty {Text(error).font(.caption).foregroundStyle(MLTheme.rose)}
        }
    }
    private func selectedID(_ graph:GraphDocument)->String {selected.flatMap{id in graph.nodes.contains(where:{$0.id==id}) ? id:nil} ?? graph.nodes.first(where:{!["input","output"].contains($0.type)})?.id ?? graph.nodes.first?.id ?? ""}
    private func title(_ node:GraphNode)->String {
        if node.type=="input" {return "Input features"}
        if node.type=="output" {return "Expert scores"}
        let type=registry.component(type:node.type)?.title ?? node.type
        return node.name.isEmpty || node.name==node.type ? type:node.name+" · "+type
    }
    private func details(_ node:GraphNode,graph:GraphDocument)->some View {
        VStack(alignment:.leading,spacing:12) {
            Text(title(node)+" settings").font(.headline)
            let connections=graph.edges.filter{$0.target==node.id || $0.source==node.id}
            ForEach(connections,id:\.id) {edge in
                Text(edge.source+"."+edge.sourcePort+" → "+edge.target+"."+edge.targetPort)
                    .font(.system(size:11,design:.monospaced)).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            }
            if let component=registry.component(type:node.type) {
                ForEach(component.fields,id:\.key) {field in
                    let identity=origin+"/"+node.id+"/"+field.key
                    RegistryFieldControl(field:field,value:node.config[field.key] ?? field.defaultValue,origin:identity,currentOrigin:{identity},onDraftChange:{_,isPending in pending(identity,isPending)},componentID:node.type,configuration:node.config) {value in
                        do {source=try MixtureRouterPresentation.setting(field.key,to:value,nodeID:node.id,source:source,registry:registry);error="";return true}
                        catch {self.error=error.localizedDescription;return false}
                    }
                }
                if component.fields.isEmpty {Text("This boundary has no editable layer settings.").font(.caption).foregroundStyle(MLTheme.muted)}
            }
        }.padding(14).overlay(Rectangle().stroke(MLTheme.rule))
    }
}
