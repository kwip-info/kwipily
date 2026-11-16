import SwiftUI
import KwipMLCore

struct ModuleCompositionInspectorView: View {
    @ObservedObject var adapter: ModelCanvasAdapter
    @StateObject private var context: ModuleCompositionInspectorContext
    @State private var keys: [String: String]
    init(adapter: ModelCanvasAdapter) {
        self.adapter = adapter
        _context = StateObject(wrappedValue: ModuleCompositionInspectorContext(adapter: adapter))
        _keys = State(initialValue: Dictionary(uniqueKeysWithValues: (adapter.activeDefinition?.composition?.children ?? []).compactMap { child in child.key.map { (child.localID, $0) } }))
    }
    var body: some View {
        if let definition = adapter.activeDefinition {
            InspectorSection("Container", symbol: "square.stack.3d.up", identifier: "inspector.container") {
                FlatChoiceControl(title: "Container kind", selection: definition.composition?.kind.rawValue ?? "",
                    options: CompositionPolicy.Kind.allCases.map { .init(id: $0.rawValue, title: ModuleCompositionPresentation.title($0)) },
                    placeholder: "Existing Block behavior") { value in
                        if let kind = CompositionPolicy.Kind(rawValue: value) { context.perform(.select(kind)) }
                    }
                if let policy = definition.composition {
                    HStack(spacing: 10) {
                        ComponentIcon(type: ModuleCompositionPresentation.iconType(policy.kind), size: 32)
                        Text(ModuleCompositionPresentation.summary(policy.kind)).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    HStack {
                        Text("Children · \(policy.children.count)").font(.caption.weight(.semibold))
                        Spacer()
                        Button("Refresh membership") { context.perform(.refresh) }.font(.caption)
                            .help("Keep surviving children and append newly added components. Connections are validated before saving.")
                    }
                    ForEach(Array(policy.children.enumerated()), id: \.element.localID) { index, child in
                        VStack(alignment: .leading, spacing: 7) {
                            HStack(spacing: 8) {
                                Text(String(index + 1)).font(.caption.monospacedDigit()).foregroundStyle(MLTheme.muted).frame(width: 20)
                                Text(childTitle(child.localID, in: definition)).font(.system(size: 11, weight: .medium)).lineLimit(2)
                                Spacer(minLength: 4)
                                moveButton(child.localID, to: index - 1, symbol: "arrow.up", disabled: index == 0)
                                moveButton(child.localID, to: index + 1, symbol: "arrow.down", disabled: index == policy.children.count - 1)
                            }
                            if let key = child.key {
                                HStack(spacing: 7) {
                                    TextField("Storage key", text: Binding(get: { keys[child.localID] ?? key }, set: { keys[child.localID] = $0 }))
                                        .flatField().accessibilityLabel("Storage key for " + childTitle(child.localID, in: definition))
                                    Button("Apply key") { context.perform(.rename(child.localID, keys[child.localID] ?? key)) }
                                        .font(.caption).disabled((keys[child.localID] ?? key) == key)
                                }
                            }
                        }.padding(9).frame(maxWidth: .infinity, alignment: .leading)
                            .overlay(Rectangle().stroke(MLTheme.rule))
                    }
                } else {
                    Text("Choose how this Block holds its children. Its current connections stay in place; a Sequence needs one complete chain.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }
                if let rejection = context.rejection {
                    Label(rejection, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if !context.validationNote.isEmpty {
                    Text(context.validationNote).font(.caption).foregroundStyle(MLTheme.muted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Text("Container changes stay in this Block draft until you Apply. Discard restores the saved Block.")
                    .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            }.disabled(!adapter.boundaryContextIsActive)
        }
    }
    private func childTitle(_ id: String, in definition: CompositeDefinition) -> String {
        definition.body.nodes.first { $0.id == id }.map { $0.name.isEmpty ? $0.type : $0.name }
            ?? definition.body.instances.first { $0.id == id }.map { $0.name.isEmpty ? "Nested Block" : $0.name } ?? id
    }
    private func moveButton(_ id: String, to index: Int, symbol: String, disabled: Bool) -> some View {
        Button { context.perform(.move(id, index)) } label: {
            Image(systemName: symbol).font(.system(size: 10, weight: .semibold)).frame(width: 22, height: 22)
                .overlay(Rectangle().stroke(MLTheme.rule))
        }.buttonStyle(.plain).disabled(disabled).accessibilityLabel(symbol == "arrow.up" ? "Move child earlier" : "Move child later")
    }
}

enum ModuleCompositionPresentation {
    static func title(_ kind: CompositionPolicy.Kind) -> String {
        switch kind {
        case .graph: return "Graph"
        case .sequential: return "Sequence"
        case .orderedList: return "Ordered collection"
        case .namedMap: return "Named collection"
        }
    }
    static func summary(_ kind: CompositionPolicy.Kind) -> String {
        switch kind {
        case .graph: return "Connections decide what runs. Child order controls how the Block stores its parts."
        case .sequential: return "Each child passes its result to the next. Moving a child also updates the chain and boundary connections."
        case .orderedList: return "Children are stored by position. Moving a child changes storage order; the canvas connections still decide what runs."
        case .namedMap: return "Children are stored under unique keys. Keys label saved parameters; display names and connections stay separate."
        }
    }
    static func iconType(_ kind: CompositionPolicy.Kind?) -> String {
        switch kind {
        case .sequential: return "kwip.composition.sequential"
        case .orderedList: return "kwip.composition.orderedList"
        case .namedMap: return "kwip.composition.namedMap"
        case .graph, nil: return "kwip.block"
        }
    }
}
