import SwiftUI
import KwipMLCore

struct CompositeWorkspaceView: View {
    @StateObject var store: CompositeWorkspaceStore
    init(workspace: WorkspaceStore) { _store = StateObject(wrappedValue: CompositeWorkspaceStore(workspace: workspace)) }
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Button("Root") { store.showRoot() }
                if let definition = store.activeDefinition {
                    Text(definition.title + " · " + (store.editingDraft ? "Draft revision" : CompositeWorkspaceStore.versionText(definition.reference.version))).font(.system(size: 12, weight: .medium)).lineLimit(1)
                    if store.editingDraft {
                        Button("Publish revision") { do { try store.publishRevision() } catch { store.statusMessage = error.localizedDescription } }.disabled(store.projectionError != nil)
                        Button("Discard draft") { store.discardDraft() }
                    } else { Button("Create draft revision") { store.beginRevision() } }
                } else { Text("Reusable architecture").font(.system(size: 12, weight: .medium)) }
                Spacer()
                Button("Undo") { store.undo() }.disabled(!store.history.canUndo)
                Button("Redo") { store.redo() }.disabled(!store.history.canRedo)
                Button("Save") { store.save() }
                Button("Reopen") { store.reopen() }
            }.font(.system(size: 11)).buttonStyle(.borderless).padding(11).background(MLTheme.panel).disabled(store.workspace.isBusy)
            if let error = store.projectionError { Text("Configuration unresolved: " + error).font(.caption).foregroundStyle(.orange).padding(8) }
            if store.workspace.showLibrary { ComponentRibbonView(entries:store.registry.components.map { definition in
                ComponentRibbonEntry(id:definition.id,title:definition.title,
                    shortTitle:ComponentStyle.shortTitle(definition.id,fallback:definition.title),
                    purpose:ComponentStyle.purpose(definition.id),category:definition.category,typeIcon:definition.id)
            },isBusy:store.isBusy,onAdd:{ store.addNode(type:$0,x:100,y:420) }) }
            HSplitView {
                if store.workspace.showInspector { inspector.frame(minWidth: 240, idealWidth: 300, maxWidth: 420).workspaceSplitLayout(store:store.workspace,scope:.compositeRibbonInspector) }
                GraphCanvasView(store: store).id(store.contextKey).frame(minWidth: 350)
            }
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(store.statusMessage).font(.system(size: 11)).lineLimit(3)
                    Text(store.workspace.compositeSourceStatus).font(.system(size: 10)).foregroundStyle(MLTheme.muted).lineLimit(2)
                }
                Spacer()
                Button("Validate active pins") { store.validateExpansion() }
                Button("Use expanded architecture") { store.useExpandedArchitecture() }
                    .disabled(store.state.activeReference != nil || store.workspace.isBusy)
                    .help("Replace the model with the validated active root pins and preserve an immutable source snapshot. Unpublished drafts are not applied.")
            }.font(.system(size: 11)).buttonStyle(.borderless).padding(11).background(MLTheme.panel)
        }.background(MLTheme.background).onDisappear { store.save() }
    }
    private var inspector: some View {
        ThemedScrollView {
            VStack(alignment: .leading, spacing: 14) {
                SectionLabel(title: store.activeDefinition == nil ? "Instance & primitive settings" : "Definition settings")
                if let id = store.selectedNodeID, let node = store.graph.nodes.first(where: { $0.id == id }) {
                    TextField("Block name", text: Binding(get: { node.name }, set: store.renameSelected)).flatField().disabled(store.isBusy)
                    if let instance = store.scope.instances.first(where: { $0.id == id }) {
                        Text("Instance · " + instance.definition.id).font(.caption).foregroundStyle(MLTheme.muted)
                        Text("Each instance has independent parameters. Its definition revision stays pinned.").font(.caption).foregroundStyle(MLTheme.muted)
                        Button("Open pinned definition") { store.openDefinition(instanceID: id) }.disabled(store.workspace.isBusy)
                        if store.state.activeReference == nil {
                            FlatChoiceControl(title: "Explicit definition upgrade", selection: CompositeWorkspaceStore.typeKey(instance.definition), options: store.project.definitions.filter { $0.reference.id == instance.definition.id }.map { FlatChoiceOption(id: CompositeWorkspaceStore.typeKey($0.reference), title: CompositeWorkspaceStore.versionText($0.reference.version)) }) { key in
                                guard let reference = store.project.definitions.first(where: { CompositeWorkspaceStore.typeKey($0.reference) == key })?.reference else { return }
                                do { try store.upgradeInstance(id, to: reference) } catch { store.statusMessage = error.localizedDescription }
                            }.disabled(store.workspace.isBusy)
                        }
                        Text("Exposed overrides").font(.caption.weight(.semibold))
                        if !instance.overrides.isEmpty { Button("Reset overrides to revision defaults") { store.resetSelectedOverrides() }.disabled(store.isBusy) }
                    } else { Text(store.registry.component(type: node.type)?.title ?? node.type).font(.caption).foregroundStyle(MLTheme.muted) }
                    ForEach(store.registry.component(type: node.type)?.fields(configuration:node.config) ?? [], id: \.key) { field in
                        VStack(alignment: .leading, spacing: 5) {
                            let derived = store.derivedParameter(nodeID: id, field: field.key)
                            let binding = store.controlledParameter(nodeID: id, field: field.key)
                            RegistryFieldControl(field: field, value: node.config[field.key] ?? field.defaultValue, origin: store.contextKey + ":" + id + ":" + field.key, currentOrigin: {
                                guard store.selectedNodeID == id, store.projectID == store.workspace.activeProjectID, !store.isBusy else { return nil }; return store.contextKey + ":" + id + ":" + field.key
                            }) { store.setConfigValue(field.key, value: $0, nodeID: id) }
                                .id(store.contextKey + ":" + id + ":" + field.key).disabled(store.isBusy || binding != nil || derived != nil)
                            if let derived { Text(store.projectionError == nil ? "Derived from \(derived.lhs) × \(derived.rhs). Edit the exposed values below." : "Derived value unresolved; repair the exposed values below.").font(.system(size: 10)).foregroundStyle(MLTheme.muted) }
                            if let binding { Text("Controlled by exposed “\(binding.field.title)” below.").font(.system(size: 10)).foregroundStyle(MLTheme.muted) }
                        }
                    }
                } else { Text("Select an instance to inspect its pin, or a primitive to configure it.").font(.caption).foregroundStyle(MLTheme.muted) }
                if store.activeDefinition != nil {
                    Divider()
                    CompositeBoundaryView(store: store)
                }
            }.padding(16)
        }
    }
}
