import Foundation
import KwipMLCore

/// Checks local interface integrity. Tensor compatibility across the enclosing model remains expansion's responsibility.
enum CompositeDraftValidation {
    static func validate(_ definition: CompositeDefinition, registry: ComponentRegistry) throws {
        guard definition.body.instances.isEmpty else { throw CompositeContractError(.invalidBoundary, "Native publication currently supports primitive-body definitions. Nested revisions remain pinned and inspectable.") }
        try CompositeDefinitionValidation.validate(definition, definitions: [], registry: registry)
    }

}

extension CompositeWorkspaceStore {
    func save() {
        guard projectID != nil else { statusMessage = "Save the model project before opening reusable blocks."; return }
        do {
            var current = state
            current.project = try CompositeMigration.toCurrent(current.project, registry: primitiveRegistry)
            if var draft = current.draft { draft.body = try CompositeMigration.toCurrentBody(draft.body, registry: primitiveRegistry); current.draft = draft }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(current)
            guard data.count <= 2_000_000 else { throw CompositeContractError(.resourceLimit, "Composite editor exceeds 2 MB.") }
            try FileManager.default.createDirectory(at: saveURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: saveURL, options: .atomic); statusMessage = "Composite source and revision draft saved."
            if projectID == workspace.activeProjectID {
                workspace.persistWorkspaceView()
                if let error = workspace.arrangementError { statusMessage = "Composite source saved; canvas view could not be saved: " + error }
            }
        } catch { statusMessage = error.localizedDescription }
    }
    func reopen() {
        guard !workspace.isBusy else { return }
        do {
            var value = try JSONDecoder().decode(CompositeEditorState.self, from: WorkspaceDocumentReader.read(saveURL))
            value.project = try CompositeMigration.toCurrent(value.project, registry: primitiveRegistry)
            if var draft = value.draft { draft.body = try CompositeMigration.toCurrentBody(draft.body, registry: primitiveRegistry); value.draft = draft }
            let bodies = [value.project.root] + value.project.definitions.map(\.body) + (value.draft.map { [$0.body] } ?? [])
            guard value.project.schemaVersion == 2, value.project.definitions.count <= 64, Set(value.project.definitions.map(\.reference)).count == value.project.definitions.count,
                  bodies.allSatisfy({ body in
                      let ids = body.nodes.map(\.id) + body.instances.map(\.id)
                      return ids.count <= 128 && Set(ids).count == ids.count && body.edges.count <= 256 && Set(body.edges.map(\.id)).count == body.edges.count && body.nodes.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.config.values.allSatisfy { value in value.isFiniteJSON } } && body.instances.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.overrides.values.allSatisfy { value in value.isFiniteJSON } }
                  }) else { throw CompositeContractError(.resourceLimit, "Composite editor has invalid identities or exceeds bounds.") }
            // Validate every retained primitive using its declared schema, preserving authored pins.
            for body in bodies { _ = try CompositeMigration.toCurrentBody(body, registry: primitiveRegistry) }
            historySessionRevision = UUID()
            state = value; history = EditHistory(); clearSelection(); canvasViewRevision = UUID(); statusMessage = "Composite source and draft reopened."
        } catch { statusMessage = error.localizedDescription }
    }
}
