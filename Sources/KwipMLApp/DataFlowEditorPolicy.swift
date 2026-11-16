import Foundation
import KwipMLCore

/// Immutable editor authority. Lesson persistence/execution belongs to its lesson adapter.
enum DataFlowEditorPolicy: Equatable {
    case preparation
    case normalizationLesson

    var fixtureColumns: [DataColumnSchema] {
        self == .normalizationLesson ? [DataColumnSchema(name:"x",dtype:"string",nullable:false),DataColumnSchema(name:"y",dtype:"string",nullable:false)] : []
    }
}

extension DataFlowWorkspaceStore {
    var allowedOperatorIDs: Set<String> { policy == .normalizationLesson ? NormalizationLessonTask.allowedOperators : Self.editableTypes }
    var allowsPreparationOperations: Bool { policy == .preparation && projectID == workspace.activeProjectID }
    @discardableResult func requirePreparationOperation() -> Bool {
        guard allowsPreparationOperations else {
            statusMessage = policy == .normalizationLesson ? "This lesson uses adapter-owned fixtures. Use the lesson controls to save or assess the construction." : "Return to this preparation's project before using its sources or artifacts."
            return false
        }
        return true
    }
    func requirePreparationAccess() throws {
        guard allowsPreparationOperations else { throw DataFlowContractError("Preparation sources, caches, artifacts and training handoff are unavailable in this lesson or inactive project.") }
    }
}
