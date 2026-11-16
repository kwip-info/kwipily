import Foundation
import KwipMLCore

enum ModelBlockPalette {
    static let type = "kwip.block"
    static var entry: ComponentRibbonEntry { .init(id: type, title: "Block", shortTitle: "Block", purpose: "A nested model canvas with named inputs, outputs and parameters", category: "Architecture", typeIcon: type) }
    static func make(x: Double, y: Double) -> (definition: CompositeDefinition, instance: CompositeInstance) {
        let reference = CompositeDefinitionReference(UUID().uuidString)
        let definition = CompositeDefinition(reference: reference, title: "Block", body: .init(), inputs: [], outputs: [])
        let instance = CompositeInstance(id: UUID().uuidString, definition: reference, name: "Block", x: x, y: y)
        return (definition, instance)
    }
}

extension WorkspaceStore {
    @discardableResult func addModelBlock(x: Double, y: Double) -> String? {
        guard !isBusy, !isLearningLessonActive, tutorialStep == nil else { return nil }
        if modelAuthoring == nil {
            guard saveManagedProject(), let id = activeProjectID, beginOwnedProject(id: id) else { return nil }
        }
        guard let owner = modelAuthoring else { return nil }
        let block = ModelBlockPalette.make(x: max(0, x), y: max(0, y))
        guard owner.insertBlock(definition: block.definition, instance: block.instance, lease: owner.lease()) else {
            reportModelEdit(false); return nil
        }
        if let canvas = modelCanvas {
            canvas.selectNode(block.instance.id, extending: false)
            // Promotion replaces the flat canvas. Seed its restoration state before
            // SwiftUI mounts the new editor, rather than waiting for onChange.
            var view = canvas.canvasViewState
            view.selectedNodeIDs = [block.instance.id]
            view.selectedEdgeIDs = []
            canvas.canvasViewState = view
        }
        reportModelEdit(true)
        return block.instance.id
    }
}
