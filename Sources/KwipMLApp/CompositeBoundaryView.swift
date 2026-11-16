import SwiftUI
import KwipMLCore

@MainActor protocol BlockBoundaryEditing: ObservableObject {
    var graph: GraphDocument { get }
    var registry: ComponentRegistry { get }
    var activeDefinition: CompositeDefinition? { get }
    var boundaryRevision: UUID { get }
    var boundaryContextIsActive: Bool { get }
    @discardableResult func releaseIntegerProduct(at index: Int, context: UUID) -> Bool
    func updateBoundary(context: UUID, _ action: (inout CompositeDefinition) -> Void)
    @discardableResult func setBoundaryValue(key: String, value: JSONValue, context: UUID) -> Bool
}

extension CompositeWorkspaceStore: BlockBoundaryEditing {
    var boundaryContextIsActive: Bool {
        editingDraft && projectID == workspace.activeProjectID && !workspace.isBusy
    }
}

struct CompositeBoundaryView<Store: BlockBoundaryEditing>: View {
    @ObservedObject var store: Store
    @State private var inputName = ""
    @State private var outputName = ""
    @State private var parameterName = ""
    private func ports(output: Bool) -> [(CompositePortReference, String)] {
        store.graph.nodes.flatMap { node in
            let definition = store.registry.component(type: node.type)
            return (output ? definition?.outputs ?? [] : definition?.inputs ?? []).map { (CompositePortReference(node.id, $0), node.name + " · " + $0) }
        }
    }
    private var fields: [(CompositeParameterTarget, ComponentField, String, JSONValue)] {
        store.graph.nodes.flatMap { node in (store.registry.component(type: node.type)?.fields ?? []).map { (CompositeParameterTarget(node.id, $0.key), $0, node.name + " · " + $0.title, node.config[$0.key] ?? $0.defaultValue) } }
    }
    var body: some View {
        if let definition = store.activeDefinition {
            let context = store.boundaryRevision
            VStack(alignment: .leading, spacing: 14) {
                if let adapter = store as? ModelCanvasAdapter {
                    ModuleCompositionInspectorView(adapter: adapter)
                }
                SectionLabel(title: "Exposed interface")
                TextField("Definition title", text: Binding(get: { definition.title }, set: { value in store.updateBoundary(context: context) { $0.title = String(value.prefix(80)) } })).flatField()
                Text("Inputs fan out to explicit internal ports. Outputs name one internal source.").font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(definition.inputs, id: \.id) { input in
                    VStack(alignment: .leading, spacing: 7) {
                        HStack { Text("Input · \(input.id)").font(.caption.weight(.semibold)); Spacer(); Button("Remove") { store.updateBoundary(context: context) { $0.inputs.removeAll { $0.id == input.id } } }.font(.caption) }
                        ForEach(input.targets, id: \.self) { target in
                            HStack { Text(label(target)).font(.system(size: 10)); Spacer(); Button { store.updateBoundary(context: context) { definition in guard let index=definition.inputs.firstIndex(where: { $0.id == input.id }) else { return }; definition.inputs[index].targets.removeAll { $0 == target } } } label: { Image(systemName: "minus") }.accessibilityLabel("Remove input binding") }
                        }
                        let choices = ports(output: false).filter { !input.targets.contains($0.0) }
                        FlatChoiceControl(title: "Bind input \(input.id)", selection: "", options: choices.enumerated().map { FlatChoiceOption(id: String($0.offset), title: $0.element.1) }, placeholder: "Add target port…") { value in
                            guard let option = Int(value), choices.indices.contains(option) else { return }; store.updateBoundary(context: context) { definition in guard let index=definition.inputs.firstIndex(where: { $0.id == input.id }) else { return }; definition.inputs[index].targets.append(choices[option].0) }
                        }
                    }
                }
                TextField("New exposed input name", text: $inputName).flatField()
                Button("Add exposed input") {
                    guard !inputName.isEmpty, !definition.inputs.contains(where: { $0.id == inputName }) else { return }; store.updateBoundary(context: context) { $0.inputs.append(CompositeInput(inputName, targets: [])) }; inputName = ""
                }
                Divider()
                ForEach(definition.outputs, id: \.id) { output in
                    HStack { Text("Output · \(output.id)").font(.caption.weight(.semibold)); Spacer(); Button("Remove") { store.updateBoundary(context: context) { $0.outputs.removeAll { $0.id == output.id } } }.font(.caption) }
                    let choices = ports(output: true)
                    FlatChoiceControl(title: "Source for \(output.id)", selection: choices.firstIndex(where: { $0.0 == output.source }).map(String.init) ?? "", options: choices.enumerated().map { FlatChoiceOption(id: String($0.offset), title: $0.element.1) }, placeholder: "Choose source port…") { value in
                        guard let option = Int(value), choices.indices.contains(option) else { return }; store.updateBoundary(context: context) { definition in guard let index=definition.outputs.firstIndex(where: { $0.id == output.id }) else { return }; definition.outputs[index].source = choices[option].0 }
                    }
                }
                TextField("New exposed output name", text: $outputName).flatField()
                Button("Add exposed output") {
                    guard !outputName.isEmpty, !definition.outputs.contains(where: { $0.id == outputName }), let source = ports(output: true).first?.0 else { return }
                    store.updateBoundary(context: context) { $0.outputs.append(CompositeOutput(outputName, source: source)) }; outputName = ""
                }
                Divider()
                if let products = definition.integerProducts, !products.isEmpty {
                    Text("Derived settings").font(.caption.weight(.semibold))
                    Text("Release a binding to edit its target directly. Its current default value is retained; future parameter changes will no longer update that target.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                    ForEach(Array(products.enumerated()), id: \.offset) { index, product in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(product.lhs + " × " + product.rhs).font(.caption.monospaced())
                            ForEach(product.targets, id: \.self) { target in
                                Text(fields.first(where: { $0.0 == target })?.2 ?? target.componentID + " · " + target.field)
                                    .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                            }
                            Button("Release binding · keep current value") { store.releaseIntegerProduct(at: index, context: context) }
                                .font(.caption)
                        }
                    }
                    Divider()
                }
                Text("Exposed parameters").font(.caption.weight(.semibold))
                ForEach(definition.parameters, id: \.field.key) { parameter in
                    VStack(alignment: .leading, spacing: 7) {
                        HStack { Text(parameter.field.title + " · " + parameter.field.key).font(.caption); Spacer(); Button("Remove") { store.updateBoundary(context: context) { $0.parameters.removeAll { $0.field.key == parameter.field.key } } }.font(.caption).disabled((definition.integerProducts ?? []).contains { $0.lhs == parameter.field.key || $0.rhs == parameter.field.key }) }
                        RegistryFieldControl(field: parameter.field, value: parameter.field.defaultValue, origin: context.uuidString + ":" + parameter.field.key, currentOrigin: {
                            guard store.boundaryContextIsActive else { return nil }; return store.boundaryRevision.uuidString + ":" + parameter.field.key
                        }) { store.setBoundaryValue(key: parameter.field.key, value: $0, context: context) }
                        if parameter.field.kind == .integer {
                            Text("Default · \(CanvasDisplayValue.dimension(parameter.field.min))–\(CanvasDisplayValue.dimension(parameter.field.max))").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                        } else if parameter.field.kind == .numberList {
                            Text("Number list · \(parameter.field.minItems ?? 1)–\(parameter.field.maxItems ?? 4096) values").font(.system(size:10)).foregroundStyle(MLTheme.muted)
                        } else if parameter.field.kind == .integerList {
                            Text("Integer list · \(parameter.field.minItems ?? 1)–\(parameter.field.maxItems ?? 64) values").font(.system(size:10)).foregroundStyle(MLTheme.muted)
                        } else if parameter.field.kind == .text {
                            Text("Text · \(parameter.field.minItems ?? 1)–\(parameter.field.maxItems ?? 128) characters").font(.system(size:10)).foregroundStyle(MLTheme.muted)
                        } else if parameter.field.kind == .choice {
                            Text("Choice · \(parameter.field.choices?.count ?? 0) options").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                        } else if parameter.field.kind == .number {
                            Text("Number · \(CanvasDisplayValue.dimension(parameter.field.min))–\(CanvasDisplayValue.dimension(parameter.field.max))").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                        } else { Text("Boolean default · on or off").font(.system(size: 10)).foregroundStyle(MLTheme.muted) }
                        ForEach(parameter.targets, id: \.self) { target in
                            HStack { Text(fields.first(where: { $0.0 == target })?.2 ?? target.componentID + " · " + target.field).font(.system(size: 10)); Spacer(); Button { store.updateBoundary(context: context) { definition in guard let index=definition.parameters.firstIndex(where: { $0.field.key == parameter.field.key }) else { return }; definition.parameters[index].targets.removeAll { $0 == target } } } label: { Image(systemName: "minus") }.accessibilityLabel("Remove parameter binding") }
                        }
                        let choices = fields.filter { item in RegistryFieldCompatibility.canExpose(parameter.field, to: item.1) && !definition.parameters.contains { $0.targets.contains(item.0) } && !(definition.integerProducts ?? []).contains { $0.targets.contains(item.0) } }
                        FlatChoiceControl(title: "Bind parameter \(parameter.field.title)", selection: "", options: choices.enumerated().map { FlatChoiceOption(id: String($0.offset), title: $0.element.2) }, placeholder: "Add target field…") { value in
                            guard let option = Int(value), choices.indices.contains(option) else { return }; store.updateBoundary(context: context) { definition in guard let index=definition.parameters.firstIndex(where: { $0.field.key == parameter.field.key }) else { return }; definition.parameters[index].targets.append(choices[option].0) }
                        }
                    }
                }
                TextField("New exposed parameter key", text: $parameterName).flatField()
                let choices = fields.filter { item in !definition.parameters.contains { $0.targets.contains(item.0) } && !(definition.integerProducts ?? []).contains { $0.targets.contains(item.0) } }
                FlatChoiceControl(title: "Expose field", selection: "", options: choices.enumerated().map { FlatChoiceOption(id: String($0.offset), title: $0.element.2) }, placeholder: "Choose a field to expose…") { value in
                    guard !parameterName.isEmpty, !definition.parameters.contains(where: { $0.field.key == parameterName }), let option = Int(value), choices.indices.contains(option) else { return }
                    let choice = choices[option]; var field = choice.1; field.key = parameterName; field.title = parameterName; field.defaultValue = choice.3
                    store.updateBoundary(context: context) { $0.parameters.append(CompositeParameter(field, targets: [choice.0])) }; parameterName = ""
                }.disabled(parameterName.isEmpty)
            }.id(context).disabled(!store.boundaryContextIsActive)
        }
    }
    private func label(_ reference: CompositePortReference) -> String {
        (store.graph.nodes.first { $0.id == reference.componentID }?.name ?? reference.componentID) + " · " + reference.portID
    }
}
